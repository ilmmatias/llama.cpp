# llama-quant-search

Measure candidate tensor types without loading the model weights into RAM or VRAM. This first stage reports costs and reconstruction errors; it does not select an allocation or write a quantized model.

```sh
llama-quant-search --imatrix imatrix.gguf model-bf16.gguf Q3_K,Q4_K,Q5_K > candidates.tsv
llama-quant-search --dry-run --tensor-type 'attn_v=Q6_K' --exclude-bpw 'token_embd' model-bf16.gguf Q3_K,Q4_K
```

Types are GGML tensor types, not model presets such as `Q4_K_M`. Rules use regex matching; the first matching `--tensor-type` rule replaces the default candidates. A single type fixes the tensor. `--exclude-tensors` preserves source types and takes precedence over type rules. `--exclude-bpw` only excludes bytes and elements from the BPW summary.

The tool follows `llama-quantize` tensor eligibility and reads GGUF shards starting with the first file. Invalid candidates are reported on stderr; a tensor with no valid candidates fails. There are no implicit type fallbacks.

The TSV contains payload bytes, BPW, search/accounting flags, squared weight error, and activation-weighted squared error. Missing measurements are `-`. Weighted error uses diagonal input second moments, with expert observation frequency retained for GGUF imatrices. It is a local reconstruction proxy, not a model-loss estimate.

`--max-buffer-size` caps the four tensor buffers, excluding metadata, imatrix data, quantizer tables, and thread state. A row that cannot fit fails. `--dry-run` does not read weight data. Use `--help` for the remaining options.
