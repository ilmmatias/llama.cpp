import numpy as np
import pytest

import gguf
from gguf.scripts.gguf_merge_qwen4exp import merge_qwen4exp


PAIRS = (
    ("hc_attn_down", "hc_attn_inject", "hc_attn_down_inject"),
    ("hc_ffn_down", "hc_ffn_inject", "hc_ffn_down_inject"),
    ("attn_qkv", "attn_gate", "attn_qkv_gate"),
    ("ssm_beta", "ssm_alpha", "ssm_beta_alpha"),
)


def raw_tensor(rows, width, tensor_type, seed=0):
    block_size, type_size = gguf.GGML_QUANT_SIZES[tensor_type]
    assert width % block_size == 0
    row_bytes = width // block_size * type_size
    return ((np.arange(rows * row_bytes, dtype=np.uint32) + seed) % 251).astype(np.uint8).reshape(rows, row_bytes)


def write_source(path, tensors, endian=gguf.GGUFEndian.LITTLE, arch="qwen4exp", split_count=None):
    writer = gguf.GGUFWriter(path, arch=arch, endianess=endian)
    writer.add_custom_alignment(64)
    writer.add_string("general.name", "synthetic Qwen4Exp")
    writer.add_uint64("custom.exact_integer", (1 << 63) + 123)
    writer.add_bool("custom.flag", True)
    writer.add_key_value("custom.empty_string", "", gguf.GGUFValueType.STRING)
    writer.add_key_value("custom.empty_array", [], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT16)
    writer.add_key_value("custom.words", ["first", "second"], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
    writer.add_key_value("custom.signed", [-3, 4], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT64)
    if split_count is not None:
        writer.add_uint16("split.no", 0)
        writer.add_uint16("split.count", split_count)
        writer.add_int32("split.tensors.count", len(tensors))
    for name, data, tensor_type in tensors:
        writer.add_tensor(name, data, raw_dtype=tensor_type, tensor_endianess=endian)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def metadata(reader):
    return {name: (field.types, field.contents()) for name, field in reader.fields.items()
            if not name.startswith("GGUF.")}


@pytest.mark.parametrize("tensor_type", [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                                        gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.Q4_0,
                                        gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.IQ2_XXS,
                                        gguf.GGMLQuantizationType.ZNQ3_M])
@pytest.mark.parametrize("endian", [gguf.GGUFEndian.LITTLE, gguf.GGUFEndian.BIG])
def test_raw_rows_and_metadata_preserved(tmp_path, tensor_type, endian):
    block_size, _ = gguf.GGML_QUANT_SIZES[tensor_type]
    width = block_size * 2
    tensors = []
    expected = {}
    for index, (first, second, merged) in enumerate(PAIRS):
        a = raw_tensor(3, width, tensor_type, index)
        b = raw_tensor(3 if first == "ssm_beta" else 2, width, tensor_type, index + 17)
        # Check that source file order does not determine the concatenation order.
        tensors.extend([(f"blk.0.{second}.weight", b, tensor_type), (f"blk.0.{first}.weight", a, tensor_type)])
        expected[f"blk.0.{merged}.weight"] = (a.tobytes() + b.tobytes(), a.shape[0] + b.shape[0])
    untouched = np.array([1.25, -2.5], dtype=">f4" if endian == gguf.GGUFEndian.BIG else "<f4")
    tensors.append(("output_norm.weight", untouched, gguf.GGMLQuantizationType.F32))
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    write_source(source, tensors, endian=endian)
    original = source.read_bytes()

    merged, kept = merge_qwen4exp(source, output)
    result = gguf.GGUFReader(output)
    assert set(merged) == set(expected)
    assert not kept
    assert result.endianess == endian
    assert result.alignment == 64
    assert metadata(result) == metadata(gguf.GGUFReader(source))
    assert result.get_field("custom.empty_array").types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT16]
    assert len(result.tensors) == 5
    for tensor in result.tensors:
        assert tensor.data_offset % 64 == 0
        if tensor.name == "output_norm.weight":
            assert tensor.data.tobytes() == untouched.tobytes()
        else:
            expected_bytes, rows = expected[tensor.name]
            assert tensor.tensor_type == tensor_type
            assert tensor.shape.tolist() == [width, rows]
            assert tensor.data.tobytes() == expected_bytes
    assert source.read_bytes() == original


@pytest.mark.parametrize("reason", ["types", "missing", "width", "rank", "scale", "input_scale", "bias", "lora_a", "merged_scale", "beta_rows"])
def test_incompatible_pairs_stay_separate(tmp_path, reason):
    qtype = gguf.GGMLQuantizationType.Q4_0
    first, second, merged = ("ssm_beta", "ssm_alpha", "ssm_beta_alpha") if reason == "beta_rows" else PAIRS[0]
    a = raw_tensor(3, 64, qtype)
    b_type = gguf.GGMLQuantizationType.Q8_0 if reason == "types" else qtype
    b = raw_tensor(2 if reason == "beta_rows" else 3, 128 if reason == "width" else 64, b_type, 37)
    if reason == "rank":
        b = b.reshape(1, *b.shape)
    tensors = [(f"blk.0.{first}.weight", a, qtype)]
    if reason != "missing":
        tensors.append((f"blk.0.{second}.weight", b, b_type))
    if reason in ("scale", "input_scale", "bias", "lora_a", "merged_scale"):
        name = f"blk.0.{merged}.scale" if reason == "merged_scale" else f"blk.0.{second}.{reason}"
        tensors.append((name, np.array([2.0], dtype=np.float32), gguf.GGMLQuantizationType.F32))
    good_a, good_b = raw_tensor(3, 64, qtype, 11), raw_tensor(2, 64, qtype, 23)
    tensors.extend([("blk.1.hc_ffn_down.weight", good_a, qtype), ("blk.1.hc_ffn_inject.weight", good_b, qtype)])
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    write_source(source, tensors)

    merged_names, kept = merge_qwen4exp(source, output)
    assert merged_names == ["blk.1.hc_ffn_down_inject.weight"]
    assert f"blk.0.{merged}.weight" in kept
    result = {tensor.name: tensor for tensor in gguf.GGUFReader(output).tensors}
    for name, data, tensor_type in tensors[:-2]:
        assert result[name].data.tobytes() == data.tobytes()
        assert result[name].tensor_type == tensor_type
    assert f"blk.0.{merged}.weight" not in result
    assert result[merged_names[0]].data.tobytes() == good_a.tobytes() + good_b.tobytes()


def test_dry_run_does_not_write(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    qtype = gguf.GGMLQuantizationType.BF16
    tensors = [("blk.0.hc_attn_down.weight", raw_tensor(3, 16, qtype), qtype),
               ("blk.0.hc_attn_inject.weight", raw_tensor(2, 16, qtype), qtype)]
    write_source(source, tensors)
    original = source.read_bytes()
    merged, _ = merge_qwen4exp(source, output, dry_run=True)
    assert merged == ["blk.0.hc_attn_down_inject.weight"]
    assert merge_qwen4exp(source, dry_run=True)[0] == merged
    assert not output.exists()
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


def test_missing_partner_without_other_merges_writes_nothing(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    qtype = gguf.GGMLQuantizationType.BF16
    write_source(source, [("blk.0.hc_attn_down.weight", raw_tensor(3, 16, qtype), qtype)])
    with pytest.raises(ValueError, match="no compatible"):
        merge_qwen4exp(source, output)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["same", "existing", "hardlink", "symlink"])
def test_existing_files_never_modified(tmp_path, mode):
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    qtype = gguf.GGMLQuantizationType.BF16
    write_source(source, [("blk.0.hc_attn_down.weight", raw_tensor(3, 16, qtype), qtype),
                          ("blk.0.hc_attn_inject.weight", raw_tensor(2, 16, qtype), qtype)])
    original = source.read_bytes()
    if mode == "same":
        output = source
    elif mode == "existing":
        output.write_bytes(b"existing data")
    elif mode == "hardlink":
        output.hardlink_to(source)
    else:
        output.symlink_to(tmp_path / "missing.gguf")
    with pytest.raises((ValueError, FileExistsError)):
        merge_qwen4exp(source, output)
    assert source.read_bytes() == original
    if mode == "existing":
        assert output.read_bytes() == b"existing data"
    if mode == "symlink":
        assert output.is_symlink()
        assert not (tmp_path / "missing.gguf").exists()
    assert not list(tmp_path.glob(".gguf-merge-*"))


@pytest.mark.parametrize("arch,split_count", [("llama", None), ("qwen4exp", 2)])
def test_wrong_architecture_and_partial_shards_rejected(tmp_path, arch, split_count):
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    qtype = gguf.GGMLQuantizationType.BF16
    write_source(source, [("blk.0.hc_attn_down.weight", raw_tensor(3, 16, qtype), qtype)], arch=arch, split_count=split_count)
    with pytest.raises(ValueError, match="only qwen4exp|complete shard set"):
        merge_qwen4exp(source, output, dry_run=True)
    assert not output.exists()


def test_single_shard_tensor_count_updated(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    qtype = gguf.GGMLQuantizationType.BF16
    write_source(source, [("blk.0.hc_attn_down.weight", raw_tensor(3, 16, qtype), qtype),
                          ("blk.0.hc_attn_inject.weight", raw_tensor(2, 16, qtype), qtype)], split_count=1)
    merge_qwen4exp(source, output)
    result = gguf.GGUFReader(output)
    assert result.get_field("split.no").contents() == 0
    assert result.get_field("split.count").contents() == 1
    assert result.get_field("split.tensors.count").contents() == 1


def test_merged_and_separate_duplicates_rejected(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "merged.gguf"
    qtype = gguf.GGMLQuantizationType.BF16
    write_source(source, [("blk.0.hc_attn_down.weight", raw_tensor(3, 16, qtype), qtype),
                          ("blk.0.hc_attn_inject.weight", raw_tensor(2, 16, qtype), qtype),
                          ("blk.0.hc_attn_down_inject.weight", raw_tensor(5, 16, qtype), qtype)])
    with pytest.raises(ValueError, match="exists alongside"):
        merge_qwen4exp(source, output)
    assert not output.exists()
