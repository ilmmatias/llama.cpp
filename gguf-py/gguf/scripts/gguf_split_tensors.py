#!/usr/bin/env python3
"""Losslessly split arbitrary tensors out of a GGUF file into a separate shard."""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Iterable

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{n_bytes} B"
        value /= 1024.0
    return f"{n_bytes} B"


def _match_tensors(
    tensors: Iterable[gguf.ReaderTensor], patterns: list[str]
) -> tuple[list[gguf.ReaderTensor], list[gguf.ReaderTensor]]:
    all_tensors = list(tensors)
    matched_names: set[str] = set()
    for pattern in patterns:
        hits = [t.name for t in all_tensors if fnmatch.fnmatchcase(t.name, pattern)]
        if not hits:
            raise ValueError(f"tensor pattern {pattern!r} matched no tensors")
        matched_names.update(hits)

    shard = [t for t in all_tensors if t.name in matched_names]
    backbone = [t for t in all_tensors if t.name not in matched_names]
    return shard, backbone


def _should_copy_kv(name: str, kv_mode: str) -> bool:
    if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF."):
        return False
    if kv_mode == "all":
        return True
    if kv_mode == "general":
        return name.startswith("general.")
    if kv_mode == "minimal":
        return False
    raise ValueError(f"unknown kv_mode: {kv_mode}")


def _write_gguf_subset(
    reader: gguf.GGUFReader,
    arch: str,
    out_path: Path,
    tensors: list[gguf.ReaderTensor],
    kv_mode: str,
    desc: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")

    writer = gguf.GGUFWriter(tmp_path, arch=arch, endianess=reader.endianess)
    writer.data_alignment = reader.alignment
    try:
        for field in reader.fields.values():
            if not _should_copy_kv(field.name, kv_mode):
                continue
            value = (
                len(tensors)
                if field.name == gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT
                else field.contents()
            )
            sub_type = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, value, field.types[0], sub_type=sub_type)

        for t in tensors:
            writer.add_tensor_info(
                t.name,
                t.data.shape,
                t.data.dtype,
                t.data.nbytes,
                raw_dtype=t.tensor_type,
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        total_bytes = sum(t.data.nbytes for t in tensors)
        bar = (
            tqdm(desc=desc, total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024)
            if tqdm is not None
            else None
        )
        try:
            for t in tensors:
                writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
                if bar is not None:
                    bar.update(t.data.nbytes)
        finally:
            if bar is not None:
                bar.close()
    except BaseException:
        writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    writer.close()
    tmp_path.replace(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Losslessly split arbitrary tensors out of a GGUF file without dequantizing or requantizing.",
        epilog=(
            "Examples:\n"
            "  %(prog)s model.gguf -t per_layer_token_embd.weight --shard-out ple.gguf --backbone-out backbone.gguf\n"
            "  %(prog)s model.gguf -t 'blk.0.*' --shard-out blk0.gguf --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input GGUF file (never modified)")
    parser.add_argument(
        "-t",
        "--tensor",
        dest="tensors",
        action="append",
        required=True,
        help="Tensor name or fnmatch glob pattern to extract into --shard-out (repeatable)",
    )
    parser.add_argument("--shard-out", type=Path, help="Output GGUF path for matched tensors")
    parser.add_argument("--backbone-out", type=Path, help="Output GGUF path for remaining tensors")
    parser.add_argument(
        "--shard-kv",
        choices=("general", "minimal", "all"),
        default="general",
        help="KV metadata to copy into --shard-out (default: general)",
    )
    parser.add_argument(
        "--backbone-kv",
        choices=("all", "general", "minimal"),
        default="all",
        help="KV metadata to copy into --backbone-out (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print split summary without writing files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.dry_run and args.shard_out is None and args.backbone_out is None:
        parser.error("at least one of --shard-out or --backbone-out is required unless --dry-run is set")

    if not args.input.is_file():
        parser.exit(1, f"error: input file not found: {args.input}\n")

    input_resolved = args.input.resolve()
    for out_opt in (args.shard_out, args.backbone_out):
        if out_opt is None:
            continue
        if out_opt.resolve() == input_resolved:
            parser.exit(1, f"error: output path cannot be the same as input: {out_opt}\n")
        if not args.dry_run and out_opt.exists() and not args.force:
            parser.exit(1, f"error: refusing to overwrite existing file (use --force): {out_opt}\n")

    if (
        args.shard_out is not None
        and args.backbone_out is not None
        and args.shard_out.resolve() == args.backbone_out.resolve()
    ):
        parser.exit(1, "error: --shard-out and --backbone-out must be different paths\n")

    reader = gguf.GGUFReader(args.input, "r")
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        parser.exit(1, f"error: missing {gguf.Keys.General.ARCHITECTURE} in {args.input}\n")
    arch = str(arch_field.contents())

    try:
        shard_tensors, backbone_tensors = _match_tensors(reader.tensors, args.tensors)
    except ValueError as err:
        parser.exit(1, f"error: {err}\n")

    shard_bytes = sum(t.data.nbytes for t in shard_tensors)
    backbone_bytes = sum(t.data.nbytes for t in backbone_tensors)
    print(f"Input:    {args.input} ({arch}, {len(reader.tensors)} tensors, {_format_bytes(shard_bytes + backbone_bytes)})")
    print(f"Shard:    {len(shard_tensors)} tensor(s), {_format_bytes(shard_bytes)}")
    for t in shard_tensors[:20]:
        print(f"  - {t.name} [{t.tensor_type.name}, shape={tuple(int(x) for x in t.shape)}, {_format_bytes(t.data.nbytes)}]")
    if len(shard_tensors) > 20:
        print(f"  ... and {len(shard_tensors) - 20} more")
    print(f"Backbone: {len(backbone_tensors)} tensor(s), {_format_bytes(backbone_bytes)}")

    if args.dry_run:
        return 0

    if args.shard_out is not None:
        _write_gguf_subset(reader, arch, args.shard_out, shard_tensors, args.shard_kv, f"Writing {args.shard_out.name}")
        print(f"Wrote shard:    {args.shard_out}")

    if args.backbone_out is not None:
        _write_gguf_subset(
            reader, arch, args.backbone_out, backbone_tensors, args.backbone_kv, f"Writing {args.backbone_out.name}"
        )
        print(f"Wrote backbone: {args.backbone_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
