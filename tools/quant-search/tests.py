#!/usr/bin/env python3
"""Run with: python3 tools/quant-search/tests.py build/bin/llama-quant-search"""

import csv
import io
import itertools
import math
from pathlib import Path
import re
import subprocess
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


BINARY = Path(sys.argv.pop(1)).resolve()
DENSE = "blk.0.attn_q.weight"
EXPERTS = "blk.0.ffn_up_exps.weight"


class QuantSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.dense = np.sin(np.arange(1024 * 160, dtype=np.float32)).reshape(160, 1024)
        self.experts = np.cos(np.arange(2 * 70 * 1024, dtype=np.float32)).reshape(2, 70, 1024)
        self.model = self.write_model("model.gguf", {
            DENSE: self.dense,
            EXPERTS: self.experts,
            "blk.0.attn_norm.weight": np.ones(1024, dtype=np.float32),
            "blk.0.ffn_gate_inp.weight": np.ones((2, 1024), dtype=np.float32),
        })

    def write_model(self, name, tensors, qtype=None, split=None):
        path = self.root / name
        writer = gguf.GGUFWriter(path, "llama")
        writer.add_context_length(32)
        writer.add_embedding_length(1024)
        writer.add_block_count(16)
        writer.add_feed_forward_length(1024)
        writer.add_head_count(16)
        writer.add_layer_norm_rms_eps(1e-5)

        if split is not None:
            writer.add_uint16("split.no", split[0])
            writer.add_uint16("split.count", split[1])

        for tensor_name, data in tensors.items():
            if qtype is not None:
                writer.add_tensor(tensor_name, gguf.quantize(data, qtype), raw_dtype=qtype)
            else:
                writer.add_tensor(tensor_name, data)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        return path

    def run_tool(self, *options, model=None, types="Q4_0,F16", success=True):
        result = subprocess.run([str(BINARY), "-t", "2", *map(str, options), str(model or self.model), types],
                                capture_output=True, text=True, timeout=60)

        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)

        return result

    def table(self, result):
        return list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))

    def recipe_choices(self, result, names):
        choices = {}
        for line in result.stdout.splitlines():
            pattern, qtype = line.split("=", 1)
            matches = [name for name in names if re.fullmatch(pattern, name)]
            self.assertEqual(len(matches), 1, line)
            self.assertNotIn(matches[0], choices)
            choices[matches[0]] = qtype

        self.assertEqual(choices.keys(), names.keys())
        return choices

    def test_allocation_against_exhaustive_search(self):
        shapes = [(3, 256), (5, 256), (4, 512), (7, 256), (2, 768)]
        tensors = {f"blk.{i}.attn_q.weight": np.sin(np.arange(r * c, dtype=np.float32) * (0.13 + i)).reshape(r, c)
                   for i, (r, c) in enumerate(shapes)}
        model = self.write_model("allocation.gguf", tensors)
        types = "Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,F16"
        rows = self.table(self.run_tool(model=model, types=types))
        groups = {name: [r for r in rows if r["tensor"] == name] for name in tensors}
        combinations = [(sum(int(r["bytes"]) for r in combination), sum(float(r["error"]) for r in combination))
                        for combination in itertools.product(*groups.values())]
        costs = sorted({cost for cost, _ in combinations})
        elements = sum(t.size for t in tensors.values())

        for cost in costs[::max(1, len(costs) // 12)]:
            bpw = str(8 * (cost + 0.5) / elements)
            result = self.run_tool("--target-bpw", bpw, model=model, types=types)
            choices = self.recipe_choices(result, groups)
            selected = [next(r for r in group if r["type"] == choices[name]) for name, group in groups.items()]
            actual_cost = sum(int(r["bytes"]) for r in selected)
            actual_error = sum(float(r["error"]) for r in selected)
            expected = min(error for size, error in combinations if size <= cost)

            self.assertLessEqual(actual_cost, cost)
            self.assertTrue(math.isclose(actual_error, expected, rel_tol=1e-11, abs_tol=1e-12), result.stderr)
            self.assertIn("(optimal)", result.stderr)

        result = self.run_tool("--target-bpw", "6.123", "--max-nodes", "1", model=model, types=types)
        choices = self.recipe_choices(result, groups)
        selected = [next(r for r in group if r["type"] == choices[name]) for name, group in groups.items()]
        budget = math.floor(6.123 * elements / 8)
        actual_error = sum(float(r["error"]) for r in selected)
        expected = min(error for size, error in combinations if size <= budget)
        match = re.search(r"lower bound: ([^,]+)", result.stderr)
        assert match is not None
        bound = float(match[1])

        self.assertLessEqual(sum(int(r["bytes"]) for r in selected), budget)
        self.assertLessEqual(bound, expected + 1e-10)
        self.assertGreaterEqual(actual_error + 1e-10, expected)
        self.assertIn("(node limit)", result.stderr)

    def test_allocation_accounting_and_missing_scores(self):
        name = DENSE.encode()
        path = self.root / "dense-only.dat"
        path.write_bytes(struct.pack("ii", 1, len(name)) + name + struct.pack("ii", 10, 1024) + np.ones(1024, dtype=np.float32).tobytes())

        result = self.run_tool("--target-bpw", "8", "--imatrix", path, success=False)
        self.assertIn("missing imatrix for searchable tensor: " + EXPERTS, result.stderr)
        self.assertEqual(result.stdout, "")
        self.run_tool("--target-bpw", "8", "--imatrix", path, "--exclude-bpw", "up_exps", success=False)

        # A fixed tensor needs no score, but its bytes still consume the budget.
        options = ("--imatrix", path, "--tensor-type", "up_exps=F16")
        self.run_tool("--target-bpw", "8", *options, success=False)
        result = self.run_tool("--target-bpw", "8", *options, "--exclude-bpw", "up_exps")
        self.assertIn("weighted variable error:", result.stderr)
        choices = self.recipe_choices(result, {DENSE: None, EXPERTS: None, "blk.0.attn_norm.weight": None, "blk.0.ffn_gate_inp.weight": None})
        self.assertEqual(choices[EXPERTS], "f16")

        # Exclusion from BPW leaves the reconstruction objective active.
        result = self.run_tool("--target-bpw", "8", "--exclude-bpw", "attn_q")
        choices = self.recipe_choices(result, {DENSE: None, EXPERTS: None, "blk.0.attn_norm.weight": None, "blk.0.ffn_gate_inp.weight": None})
        self.assertEqual(choices[DENSE], "f16")

        for options in (("--dry-run",), ("--exclude-bpw", ".*")):
            self.run_tool("--target-bpw", "8", *options, success=False)

        for bpw in ("0", "-1", "nan", "inf", "8x", "0.001"):
            self.run_tool("--target-bpw", bpw, success=False)

    def test_recipe_quantizer_round_trip(self):
        names = ["blk.1.attn_q.weight", "blk.10.attn_q.weight", "blk.2.a+[Z]= x.weight"]
        tensors = {name: self.dense[:2] for name in names}
        model = self.write_model("recipe-source.gguf", tensors)
        result = self.run_tool("--target-bpw", "8", "--tensor-type", "blk\\.1\\.=Q5_0", model=model)
        choices = self.recipe_choices(result, tensors)
        self.assertEqual(choices[names[0]], "q5_0")
        recipe = self.root / "recipe.txt"
        recipe.write_text(result.stdout)

        quantizer = BINARY.with_name("llama-quantize" + BINARY.suffix)
        if not quantizer.is_file():
            self.skipTest("build llama-quantize to test recipe application")

        output = self.root / "recipe-output.gguf"
        applied = subprocess.run([str(quantizer), "--tensor-type-file", str(recipe), str(model), str(output), "Q4_0", "2"],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
        reader = gguf.GGUFReader(output)
        actual = {t.name: t.tensor_type.name.lower() for t in reader.tensors}
        self.assertEqual(actual, choices)

        scores = self.table(self.run_tool("--tensor-type", "blk\\.1\\.=Q5_0", model=model))
        for tensor in reader.tensors:
            score = next(r for r in scores if r["tensor"] == tensor.name and r["type"] == choices[tensor.name])
            self.assertEqual(tensor.n_bytes, int(score["bytes"]))
            decoded = gguf.dequantize(tensor.data, tensor.tensor_type)
            error = np.square(tensors[tensor.name].astype(np.float64) - decoded.astype(np.float64)).sum()
            self.assertTrue(math.isclose(float(score["error"]), float(error), rel_tol=1e-7, abs_tol=1e-10))

    def test_sizes_and_independent_f16_error(self):
        result = self.run_tool()
        rows = self.table(result)
        dense = [r for r in rows if r["tensor"] == DENSE]

        self.assertEqual([int(r["bytes"]) for r in dense], [160 * 1024 // 32 * 18, 160 * 1024 * 2])

        expected = np.square(self.dense.astype(np.float64) - self.dense.astype(np.float16).astype(np.float64)).sum()
        self.assertAlmostEqual(float(dense[1]["error"]), expected, places=12)

        for row in rows:
            self.assertEqual(row["weighted_error"], "-")
            if row["tensor"].endswith("norm.weight") or "gate_inp" in row["tensor"]:
                self.assertEqual(row["type"], "f32")
                self.assertEqual(row["search"], "0")

    def test_slab_and_thread_invariance(self):
        small = self.table(self.run_tool("--max-buffer-size", "1", "-t", "1"))
        large = self.table(self.run_tool("--max-buffer-size", "4", "-t", "3"))

        for a, b in zip(small, large):
            self.assertEqual(a["bytes"], b["bytes"])
            self.assertTrue(math.isclose(float(a["error"]), float(b["error"]), rel_tol=1e-12, abs_tol=1e-12))

    def test_first_rule_fixed_and_exclusions(self):
        result = self.run_tool("--tensor-type", "attn_q=Q8_0", "--tensor-type", "attn_q=F16",
                               "--exclude-tensors", "up_exps", "--exclude-bpw", "attn_q")
        rows = self.table(result)

        dense = next(r for r in rows if r["tensor"] == DENSE)
        self.assertEqual((dense["type"], dense["search"], dense["count"]), ("q8_0", "0", "0"))

        expert = next(r for r in rows if r["tensor"] == EXPERTS)
        self.assertEqual((expert["type"], expert["count"]), ("f32", "1"))

    def test_bpw_exclusion_keeps_scoring(self):
        result = self.run_tool("--exclude-bpw", "attn_q")

        self.assertTrue(all(r["count"] == "0" and float(r["error"]) > 0
                            for r in self.table(result) if r["tensor"] == DENSE))

        counted = [r for r in self.table(result) if r["count"] == "1"]
        elements = sum(int(r["elements"]) for r in {r["tensor"]: r for r in counted}.values())
        self.assertIn(f"counted: {elements} elements", result.stderr)

    def test_invalid_candidates_and_empty_set(self):
        model = self.write_model("narrow.gguf", {DENSE: self.dense[:, :96]})

        result = self.run_tool(model=model, types="Q4_K,Q4_0")
        self.assertIn("skip q4_K", result.stderr)
        self.assertEqual(self.table(result)[0]["type"], "q4_0")

        result = self.run_tool(model=model, types="Q4_K", success=False)
        self.assertIn("no valid candidates", result.stderr)

    def test_dry_run_does_not_decode_weights(self):
        model = self.write_model("nan.gguf", {DENSE: np.full((2, 1024), np.nan, dtype=np.float32)})

        result = self.run_tool("--dry-run", model=model)
        self.assertTrue(all(r["error"] == "-" for r in self.table(result)))
        self.run_tool(model=model, success=False)

    def test_imatrix_expert_frequency_and_encoding(self):
        path = self.root / "imatrix.gguf"
        writer = gguf.GGUFWriter(path, "llama")

        writer.add_uint32("imatrix.chunk_count", 1)
        writer.add_uint32("imatrix.chunk_size", 100)
        writer.add_tensor(DENSE + ".in_sum2", np.full((1, 1024), 200, dtype=np.float32))
        writer.add_tensor(DENSE + ".counts", np.array([[100]], dtype=np.float32))
        writer.add_tensor(EXPERTS + ".in_sum2", np.repeat(np.array([[200], [20]], dtype=np.float32), 1024, axis=1))
        writer.add_tensor(EXPERTS + ".counts", np.array([[100], [10]], dtype=np.float32))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        rows = self.table(self.run_tool("--imatrix", path, types="F16,IQ2_XXS"))
        small = self.table(self.run_tool("--imatrix", path, "--max-buffer-size", "1", types="F16,IQ2_XXS"))
        for a, b in zip(rows, small):
            for key in ("error", "weighted_error"):
                if a[key] != "-":
                    self.assertTrue(math.isclose(float(a[key]), float(b[key]), rel_tol=1e-12, abs_tol=1e-12))

        dense = next(r for r in rows if r["tensor"] == DENSE and r["type"] == "f16")
        self.assertAlmostEqual(float(dense["weighted_error"]), 2 * float(dense["error"]), places=12)

        expert = next(r for r in rows if r["tensor"] == EXPERTS and r["type"] == "f16")
        errors = np.square(self.experts.astype(np.float64) - self.experts.astype(np.float16).astype(np.float64)).sum(axis=(1, 2))
        self.assertAlmostEqual(float(expert["weighted_error"]), 2 * errors[0] + 0.2 * errors[1], places=12)

        groups = {r["tensor"]: [s for s in rows if s["tensor"] == r["tensor"]] for r in rows}
        elements = sum(int(group[0]["elements"]) for group in groups.values())
        combinations = [sum(float(r["weighted_error"]) for r in combination if r["search"] == "1")
                        for combination in itertools.product(*groups.values())
                        if sum(int(r["bytes"]) for r in combination) <= 14 * elements // 8]
        result = self.run_tool("--imatrix", path, "--target-bpw", "14", types="F16,IQ2_XXS")
        choices = self.recipe_choices(result, groups)
        selected = [r for r in rows if r["type"] == choices[r["tensor"]] and r["search"] == "1"]
        self.assertAlmostEqual(sum(float(r["weighted_error"]) for r in selected), min(combinations), places=10)

    def test_requantization_and_bf16_source(self):
        for qtype in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.BF16):
            model = self.write_model(f"{qtype.name}.gguf", {DENSE: self.dense}, qtype=qtype)

            if qtype == gguf.GGMLQuantizationType.Q4_0:
                self.run_tool(model=model, types="Q8_0", success=False)

            result = self.run_tool("--allow-requantize", model=model, types="Q8_0")
            self.assertGreater(float(self.table(result)[0]["error"]), 0)

    def test_legacy_imatrix(self):
        name = DENSE.encode()
        sums = np.full(1024, 20, dtype=np.float32)
        path = self.root / "imatrix.dat"
        path.write_bytes(struct.pack("ii", 1, len(name)) + name + struct.pack("ii", 10, 1024) + sums.tobytes())

        rows = self.table(self.run_tool("--imatrix", path, types="F16"))
        dense = next(r for r in rows if r["tensor"] == DENSE)
        self.assertAlmostEqual(float(dense["weighted_error"]), 2 * float(dense["error"]), places=12)

    def test_incomplete_imatrix(self):
        path = self.root / "incomplete.gguf"
        writer = gguf.GGUFWriter(path, "llama")

        writer.add_uint32("imatrix.chunk_count", 1)
        writer.add_uint32("imatrix.chunk_size", 100)
        writer.add_tensor(EXPERTS + ".in_sum2", np.zeros((2, 1024), dtype=np.float32))
        writer.add_tensor(EXPERTS + ".counts", np.array([[100], [0]], dtype=np.float32))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        result = self.run_tool("--imatrix", path, success=False)
        self.assertIn("incomplete imatrix", result.stderr)

    def test_shards(self):
        first = self.write_model("split-00001-of-00002.gguf", {DENSE: self.dense}, split=(0, 2))
        self.write_model("split-00002-of-00002.gguf", {EXPERTS: self.experts}, split=(1, 2))

        rows = self.table(self.run_tool(model=first))
        self.assertEqual({r["tensor"] for r in rows}, {DENSE, EXPERTS})
        self.run_tool(model=self.root / "split-00002-of-00002.gguf", success=False)

    def test_missing_imatrix_and_bad_options(self):
        self.run_tool(types="IQ2_XXS", success=False)

        result = self.run_tool(types="Q4_0,IQ3_XXS")
        self.assertIn("skip iq3_xxs: requires imatrix data", result.stderr)
        self.assertFalse(any(r["type"] == "iq3_xxs" for r in self.table(result)))

        for args in (("--max-buffer-size", "0"), ("-t", "-1"), ("--tensor-type", "[=Q4_0")):
            self.run_tool(*args, success=False)

        for types in ("Q4_K_M", "Q8_1", "Q4_0,", "Q4_0,,F16"):
            self.run_tool(types=types, success=False)

    def test_row_over_buffer_limit(self):
        model = self.write_model("wide.gguf", {DENSE: np.ones((2, 262144), dtype=np.float32)})

        result = self.run_tool("--max-buffer-size", "1", model=model, success=False)
        self.assertRegex(result.stderr, r"one row needs \d+ buffer bytes")
        self.run_tool("--dry-run", "--max-buffer-size", "1", model=model)

    def test_truncated_input(self):
        data = self.model.read_bytes()
        self.model.write_bytes(data[:-100])

        result = self.run_tool("--dry-run", success=False)
        self.assertIn("outside file", result.stderr)


if __name__ == "__main__":
    unittest.main()
