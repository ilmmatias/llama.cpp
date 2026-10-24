#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
from tqdm import tqdm

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-merge-qwen4exp")


class MergePair(NamedTuple):
    name: str
    first: gguf.ReaderTensor
    second: gguf.ReaderTensor


def plan_merges(reader: gguf.GGUFReader) -> tuple[list[MergePair], dict[str, str]]:
    architecture = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if architecture is None or architecture.contents() != "qwen4exp":
        raise ValueError("only qwen4exp model GGUFs support these projection merges")
    model_type = reader.get_field(gguf.Keys.General.TYPE)
    if model_type is not None and model_type.contents() != gguf.GGUFType.MODEL:
        raise ValueError("projection merging requires a model, not an adapter GGUF")
    split_count = reader.get_field(gguf.Keys.Split.LLM_KV_SPLIT_COUNT)
    split_no = reader.get_field(gguf.Keys.Split.LLM_KV_SPLIT_NO)
    if ((split_count is not None and split_count.contents() != 1) or
            (split_no is not None and split_no.contents() != 0)):
        raise ValueError("merge the complete shard set first with llama-gguf-split --merge; partial shard rewrites are unsafe")

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    layers = sorted({int(match[1]) for name in tensors if (match := re.match(r"blk\.([0-9]+)\.", name))})
    merges = []
    kept = {}
    for bid in layers:
        for merged, parts, _linear_only in gguf.QWEN4EXP_SHARED_INPUT_MERGES:
            name = gguf.TENSOR_NAMES[merged].format(bid=bid) + ".weight"
            sources = [gguf.TENSOR_NAMES[part].format(bid=bid) + ".weight" for part in parts]
            if name in tensors:
                if any(source in tensors for source in sources):
                    raise ValueError(f"{name} exists alongside its separate projections")
                kept[name] = "already merged"
                continue
            if not any(source in tensors for source in sources):
                continue
            if not all(source in tensors for source in sources):
                kept[name] = "missing partner: " + ", ".join(source for source in sources if source not in tensors)
                continue

            first, second = (tensors[source] for source in sources)
            companions = [tensor_name for source in [*sources, name] for tensor_name in tensors
                          if tensor_name not in sources and tensor_name.startswith(source.removesuffix(".weight") + ".")]
            if companions:
                kept[name] = "separate scales, biases or adapters: " + ", ".join(companions)
            elif first.tensor_type != second.tensor_type:
                kept[name] = f"different types: {first.tensor_type.name}, {second.tensor_type.name}"
            elif (len(first.shape) != 2 or len(second.shape) != 2 or
                    int(first.shape[0]) != int(second.shape[0]) or
                    any(int(dim) <= 0 for tensor in (first, second) for dim in tensor.shape)):
                kept[name] = "incompatible matrix shapes"
            elif merged == gguf.MODEL_TENSOR.SSM_BETA_ALPHA_CAT and first.shape[1] != second.shape[1]:
                kept[name] = "beta and alpha must have the same row count"
            else:
                block_size, type_size = gguf.GGML_QUANT_SIZES[first.tensor_type]
                width = int(first.shape[0])
                row_bytes = width // block_size * type_size
                if width % block_size or any(tensor.n_bytes != int(tensor.shape[1]) * row_bytes for tensor in (first, second)):
                    kept[name] = "quantization blocks do not fit complete rows"
                else:
                    merges.append(MergePair(name, first, second))
    return merges, kept


def merge_qwen4exp(input_path: Path, output_path: Path | None = None, *, dry_run: bool = False) -> tuple[list[str], dict[str, str]]:
    """Write a new GGUF without dequantization, or report compatible pairs without reading weights."""
    if output_path is not None and input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")
    if not dry_run and output_path is None:
        raise ValueError("an output filename is required unless --dry-run is used")
    if not dry_run and output_path is not None and output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    reader = gguf.GGUFReader(input_path, 'r')
    merges, kept = plan_merges(reader)
    for pair in merges:
        logger.info("Merge %s + %s -> %s (%s, %d + %d rows)", pair.first.name, pair.second.name,
                    pair.name, pair.first.tensor_type.name, pair.first.shape[1], pair.second.shape[1])
    for name, reason in kept.items():
        logger.info("Keep %s separate/unchanged: %s", name, reason)
    logger.info("%d compatible pairs; %d unchanged pairs", len(merges), len(kept))
    if dry_run:
        return [pair.name for pair in merges], kept
    if not merges:
        raise ValueError("no compatible separate projection pairs; no output written")
    assert output_path is not None

    merged_by_first = {pair.first.name: pair for pair in merges}
    removed = {pair.second.name for pair in merges}
    output_tensors = []
    for tensor in reader.tensors:
        if tensor.name in removed:
            continue
        pair = merged_by_first.get(tensor.name)
        if pair is None:
            output_tensors.append((tensor.name, tensor.data, tensor.tensor_type))
        else:
            shape = (pair.first.data.shape[0] + pair.second.data.shape[0], pair.first.data.shape[1])
            chunks = [lambda part=part: part.data for part in (pair.first, pair.second)]
            data = gguf.LazyChunkedTensor(chunks, shape=shape, dtype=pair.first.data.dtype)
            output_tensors.append((pair.name, cast(np.ndarray, data), pair.first.tensor_type))

    # Publish only a complete output, without overwriting an existing file or symlink.
    with tempfile.TemporaryDirectory(prefix=".gguf-merge-", dir=output_path.parent) as temporary:
        staging = Path(temporary) / "merged.gguf"
        writer = gguf.GGUFWriter(staging, arch="qwen4exp", endianess=reader.endianess)
        writer.data_alignment = reader.alignment
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            value = field.contents()
            if field.name == gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT:
                value = len(output_tensors)
            sub_type = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, value, field.types[0], sub_type=sub_type)
        for name, data, tensor_type in output_tensors:
            writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=tensor_type)
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            with tqdm(desc="Writing", total=sum(data.nbytes for _, data, _ in output_tensors), unit="byte", unit_scale=True) as bar:
                for _, data, _ in output_tensors:
                    writer.write_tensor_data(data, tensor_endianess=reader.endianess)
                    bar.update(data.nbytes)
        finally:
            writer.close()
        os.link(staging, output_path)
    return [pair.name for pair in merges], kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Losslessly merge Qwen4Exp shared-input projections into a new GGUF, including compatible BF16 and quantized GGUFs.",
        epilog="No original HF weights or downloads are needed. Mixed-type pairs and pairs with separate scales/biases stay unchanged. "
               "No dequantization or requantization is performed. Existing output files are never overwritten. "
               "For shards, use llama-gguf-split --merge first, and optionally split the new output afterwards. "
               "Keep the original GGUF for LoRA adapters targeting separate projection names; those adapters do not target merged weights.")
    parser.add_argument("input", type=Path, help="Qwen4Exp GGUF input filename (never modified)")
    parser.add_argument("output", type=Path, nargs="?", help="new GGUF output filename; requires space for the complete model")
    parser.add_argument("--dry-run", action="store_true", help="report compatible and skipped pairs using metadata only; write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.output is None and not args.dry_run:
        parser.error("output is required unless --dry-run is used")
    try:
        merge_qwen4exp(args.input, args.output, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == '__main__':
    main()
