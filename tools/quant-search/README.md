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

Evaluate explicit recipes against the source model on calibration text:

```sh
llama-quant-search --imatrix imatrix.gguf --calibration-file calibration.txt \
    --eval-cache /path/on/disk/quant-search-cache --ctx-size 512 --batch-size 32 --chunks 4 \
    --eval-recipe manual.txt --eval-recipe searched.txt model-bf16.gguf BF16 > evaluation.tsv
```

Evaluation reports whole-model next-token KL divergence `KL(source || recipe)` and perplexity. It uses teacher forcing, resets attention and recurrent state for each context chunk, and scores every position with a following token. The source reference is evaluated once; full-vocabulary FP32 logits are streamed to disk and reused for all recipes.

Rules and BPW exclusions still apply. Explicit `--tensor-type` rules precede recipe rules; every tensor must resolve to one type. The positional type is the fallback for tensors not covered by a rule. Counts and bytes include only tensors selected for BPW accounting; inference and KL still include the complete model. Use the same fixed choices, exclusions, and imatrix as final quantization.

Weights use the normal CPU inference graph with caller-owned memory-mapped buffers, without GPU offloading or CPU weight repacking. Unchanged tensors refer to source files; converted tensor/type pairs are encoded once into a temporary disk cache and reused across recipes. No full model GGUF is written. Mapping keeps weights pageable; it does not impose a hard RSS limit. Context and compute buffers require additional RAM and are controlled separately by context and batch sizes.

`--eval-cache PREFIX` creates `PREFIX.weights` and `PREFIX.logits` exclusively and removes them on normal exit or a caught error. Use a new prefix on a disk with room for the union of converted tensor/type pairs and `scored_tokens * vocabulary_size * 4` logit bytes. These files can be large; avoid a RAM-backed filesystem. Reference results are reused within the invocation, not across runs. CPU evaluation can be much slower than GPU inference; start with a short calibration sample.

Experimental global refinement starts from a feasible recipe and writes a replacement recipe:

```sh
llama-quant-search --imatrix imatrix.gguf --global-search manual.txt --target-bpw 3.5 \
    --calibration-file calibration.txt --eval-cache /path/on/disk/global-cache \
    --ctx-size 128 --batch-size 32 --chunks 1 \
    --global-tensors '^blk\.0\.ffn_(gate_up|down)_exps\.weight$' \
    model-bf16.gguf ZNQ3_M,ZNQ4_M,ZNQ5_M,ZNQ6_M,ZNQ7_M,BF16 > refined.txt
```

`--global-tensors` is repeatable and restricts refinement to matching tensors; other tensors retain their initial recipe types. Without it, all variable tensors participate. Fixed type rules and source-preserving exclusions still apply. Initial types must be among the allowed candidates for tensors being refined, and the initial recipe must satisfy the counted byte budget. Preserve the same BPW exclusions used to define that budget.

The method evaluates each alternative in the complete initial model, records its logit change, and builds a coupled Fisher/Gauss-Newton approximation to global KL. It keeps the exact logit-space linear term and uses two fixed CountSketch maps for the Fisher products (`--response-rank`, default 64 buckets per scored token). The quadratic retains cross-tensor products rather than adding independent tensor errors. Conditional-gradient steps use the byte-constrained allocator as a linear oracle and an analytic line search; the best predicted discrete proposal is evaluated with actual full-vocabulary KL. A proposal is accepted only if calibration KL improves, otherwise the initial recipe is retained. Each invocation performs one response fit and at most one combined proposal evaluation. This is an approximate local method, not RCO or a global optimality certificate; calibration KL acceptance does not guarantee held-out perplexity or task accuracy.

Global refinement requires one reference evaluation, one initial-recipe evaluation, one evaluation per alternative, and at most one proposal evaluation. This can be much more expensive than local scoring. Start with a small tensor subset and short calibration sample. Encoded variants are reused within the run; `PREFIX.baseline` and `PREFIX.responses` additionally store initial logits and sketched responses. Response storage is approximately `alternatives * scored_tokens * response_rank * 4` bytes. Response working vectors are checked against `--max-buffer-size`; mapped weights and inference context/compute buffers remain outside that cap. Model weights, reference logits, and response records do not all need to fit in RAM.
