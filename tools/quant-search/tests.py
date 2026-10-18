#!/usr/bin/env python3
"""Run with: python3 tools/quant-search/tests.py build/bin/llama-quant-search"""

import csv
import io
import math
from pathlib import Path
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
