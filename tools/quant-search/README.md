# llama-quant-search

Measure candidate tensor types and select an allocation without loading the model weights into RAM or VRAM.

Scoring progress is written to stderr, including the current type, completed rows, tensor work percentage, and elapsed time. Updates reuse one line in a terminal and use separate lines in redirected logs. Percentages count row/type evaluations, not estimated time.

```sh
llama-quant-search --imatrix imatrix.gguf model-bf16.gguf Q3_K,Q4_K,Q5_K > candidates.tsv
llama-quant-search --dry-run --tensor-type 'attn_v=Q6_K' --exclude-bpw 'token_embd' model-bf16.gguf Q3_K,Q4_K
llama-quant-search --imatrix imatrix.gguf --tensor-type '^token_embd\.weight$=Q8_0' --target-bpw 5 model-bf16.gguf Q3_K,Q4_K,Q5_K > types.txt
llama-quantize --imatrix imatrix.gguf --tensor-type-file types.txt model-bf16.gguf model-quant.gguf Q4_K
```

Types are GGML tensor types, not model presets such as `Q4_K_M`. Rules use regex matching; the first matching `--tensor-type` rule replaces the default candidates. A single type fixes the tensor. `--exclude-tensors` preserves source types and takes precedence over type rules. `--exclude-bpw` only excludes bytes and elements from the BPW summary.

The tool follows `llama-quantize` tensor eligibility and reads GGUF shards starting with the first file. Invalid candidates are reported on stderr; a tensor with no valid candidates fails. There are no implicit type fallbacks.

The TSV contains payload bytes, BPW, search/accounting flags, squared weight error, and activation-weighted squared error. Missing measurements are `-`. Weighted error uses diagonal input second moments, with expert observation frequency retained for GGUF imatrices. It is a local reconstruction proxy, not a model-loss estimate.

Partial expert coverage is reported on stderr. Unobserved experts use the quantizer's uniform encoding weights and contribute zero to the calibration-weighted error. Zero-count experts must have zero sums; invalid statistics and entries with no observations fail.

`--target-bpw` writes an exact-name recipe for `llama-quantize --tensor-type-file` instead of TSV. Apply it with `Q4_K` as the base type, the same imatrix, and `--allow-requantize` if needed. The limit counts tensor payload only, excludes GGUF metadata and alignment, and rounds down to whole bytes. Fixed tensors remain counted unless excluded with `--exclude-bpw`.

Allocation minimizes additive squared error, weighted when `--imatrix` is supplied. Every searchable tensor then requires calibration data; fix or exclude tensors with missing entries explicitly. Fixed tensors contribute no variable error and are not read for scoring in allocation mode. BPW exclusions do not remove tensors from the error objective.

The allocator uses a convex relaxation and deterministic branch-and-bound, retaining non-convex choices for discrete search. Stderr reports the feasible error, lower bound, gap, and search status. `optimal` means the additive problem is solved within floating-point arithmetic; `node limit` means a feasible recipe with a remaining bound. `--max-nodes` limits work (default: 100000); solver storage scales with candidate count, not model bytes or budget.

`--max-buffer-size` caps the four tensor buffers, excluding metadata, imatrix data, quantizer tables, and thread state. A row that cannot fit fails. `--dry-run` does not read weight data. Use `--help` for the remaining options.
