# Token generation performance troubleshooting

## Verifying that the model is running on the GPU with CUDA
Make sure you compiled llama with the correct env variables according to [this guide](/docs/build.md#cuda), so that llama accepts the `-ngl N` (or `--n-gpu-layers N`) flag. When running llama, you may configure `N` to be very large, and llama will offload the maximum possible number of layers to the GPU, even if it's less than the number you configured. For example:
```shell
./llama-cli -m "path/to/model.gguf" -ngl 200000 -p "Please sir, may I have some "
```

When running llama, before it starts the inference work, it will output diagnostic information that shows whether cuBLAS is offloading work to the GPU. Look for these lines:
```shell
llama_model_load_internal: [cublas] offloading 60 layers to GPU
llama_model_load_internal: [cublas] offloading output layer to GPU
llama_model_load_internal: [cublas] total VRAM used: 17223 MB
... rest of inference
```

If you see these lines, then the GPU is being used.

## Verifying that the CPU is not oversaturated
llama accepts a `-t N` (or `--threads N`) parameter. It's extremely important that this parameter is not too large. If your token generation is extremely slow, try setting this number to 1. If this significantly improves your token generation speed, then your CPU is being oversaturated and you need to explicitly set this parameter to the number of the physical CPU cores on your machine (even if you utilize a GPU). If in doubt, start with 1 and double the amount until you hit a performance bottleneck, then scale the number down.

## CPU/GPU handoffs

Mixed CPU/GPU graphs can spend time waiting for transfers even when the GPU kernels are fast. The scheduler groups compatible intermediate-tensor transfers from the preceding split without reordering user-input or fallback copies. GPU-to-CPU groups wait once for their results. CPU-to-GPU uploads, including single tensors, can continue into GPU computation on the same stream; completion is enforced before CPU storage reuse, fallback copies, host-weight callbacks, or return to the caller. User-input copies and unsupported transfer paths keep their existing behavior.

The expert cache uses the same CPU-miss/GPU-hit policy for native and repacked expert weights. Missing experts run on the CPU while ready cache hits run on the GPU; background workers copy or convert admitted experts without making the current token wait for a cache fill. ZNQ keeps its fused GPU FFN path. Other FFN layouts use projection-level hits so intervening activations and transforms keep their original graph semantics.

Use `--expert-cache-slots N` for a fixed number of slots per layer, `--expert-cache-slots auto` with `--fit` for automatic sizing, or `--moe-cache-mib N` for a weight-cache byte budget. Slot and byte settings are mutually exclusive. The byte budget is divided by host expert placement across devices; fitting also reserves compute and merge workspaces. Independent contexts own independent caches; a shared-model MTP context reuses its target's cache without a second reservation.

Supported expert-cache outputs stay on-device when their next consumers run on the cache GPU. A row-gather graph combines cached outputs with uploaded CPU misses before GPU weighting and reduction; fully cached batches need no CPU-output upload. This avoids the cached-output GPU-to-CPU round trip. CPU consumers and evaluation callbacks keep the host-side join.

Cache ownership is separate from batch eligibility. Batches larger than 128 tokens keep the CPU bulk-prefill path without cache callbacks or extra cache barriers. When CPU MoE reduction is supported, weighting and reduction run before upload so only the reduced output crosses to the GPU.

The merge workspace is shared across layers and grows with supported batch sizes rather than retaining a separate allocation for every layer. Fitting reserves its maximum size and the expert-compute workspace alongside cached weights, including loaded MTP layers. GPU events order workspace reuse without a host wait in steady-state merges.

CPU builds with AVX-512F/BW convert ZNQ3 CPU_REPACK bitplanes for eight expert rows at once during GPU cache admission. Other builds keep scalar conversion. Both paths produce the same native weight bytes and use the same cache policy.

CPU_REPACK `MUL_MAT` and `MUL_MAT_ID` accept F32 or the weight type's native vector-dot input type (Q8_0 or Q8_K). Prequantized inputs keep their tensor type and row strides; the CPU packs full GEMM groups without dequantizing them and reads GEMV tails directly. ZNQ expert-cache hits also accept native Q8_0 inputs, including combined gate/up weights, and pack them into the GPU's Q8_1 layout. Existing F32 graphs remain unchanged.

This is an operator capability, not automatic graph quantization. Callers must explicitly produce a supported quantized tensor before a GPU-to-CPU handoff. Measure transfer and packing costs against the F32 path before using it in a model graph.

Profile the active execution path before changing synchronization. Expert-cache configurations can bypass the routing-ID readback callback, and a backend transition without split inputs can still require a wait because scratch allocations are reused. Compare unprofiled `llama-bench` runs with the same context depth, offload settings, and thread count; profiler overhead can change the apparent bottleneck.

HIP ZNQ4 matrix-vector kernels expand four packed codes with a byte permutation before codebook lookup. This reduces decode instructions without changing the weight format or decoded values. The path is automatic; matrix-matrix kernels keep their existing decode path.

On a Ryzen 9 9950X and RX 6750 XT, the ZNQ3 admission conversion and HIP ZNQ4 byte permutation together increased Tinfield-1-ZNQ4-v2 generation throughput from 36.65 to 36.86 tokens/s (0.57%). This is a small end-to-end improvement, not the larger speedup seen in isolated conversion or matrix-vector measurements. The comparison used baseline/optimized/optimized/baseline runs with 10 repetitions each, 2048 prompt tokens, 256 generated tokens, 16 CPU threads, 32 expert-cache slots, an admission window of 1, and 2 cache workers; GPU offload, flash attention, and mlock loading were unchanged.

RDNA2 Q8_0 single-column grouped matrix-vector kernels use two warps for projections with fewer than 1024 output rows when the reduction can occupy both warps. Larger and very short projections keep their existing launch geometry. Other architectures, weight formats, and multi-column kernels keep their existing policies.

A separate optimized/baseline/optimized/baseline comparison with the same model, hardware, and runtime settings measured 36.39 tokens/s for the rebuilt baseline and 36.80 tokens/s with this Q8_0 launch policy (1.12%, 20 repetitions per variant). Isolated 2560-input projections improved from 6.24 to 5.53 microseconds for 512 output rows and from 6.18 to 5.40 microseconds for 320 output rows; the wider controls retained their existing launch geometry.

CUDA/HIP quantized `MUL_MAT` honors `GGML_HINT_EXACT_BATCH` by quantizing the activations once and using the single-column matrix-vector kernel for each output column. This keeps the single-token reduction order, including for batches larger than the normal matrix-vector limit. Fusions that would discard this hint are disabled. The hint trades batched throughput for batch-invariant results; unhinted dispatch keeps its existing policy.

## CPU sampling

With large vocabularies, CPU repetition, frequency and presence penalties can add noticeable sampling latency. Candidates indexed by token ID use direct updates for the tokens in the penalty window; other layouts retain the full scan when direct indexing is not valid. This follows the [sparse penalty update in strix-llama.cpp](https://github.com/asf0/strix-llama.cpp/commit/4202e1b56c401ef052b17c5c7bbd3e1dbdc855c6) and does not change penalty arithmetic or token-window eviction.

# Example of runtime flags effect on inference speed benchmark
These runs were tested on the following machine:
GPU: A6000 (48GB VRAM)
CPU: 7 physical cores
RAM: 32GB

Model: `TheBloke_Wizard-Vicuna-30B-Uncensored-GGML/Wizard-Vicuna-30B-Uncensored.q4_0.gguf` (30B parameters, 4bit quantization, GGML)

Run command: `./llama-cli -m "path/to/model.gguf" -p "An extremely detailed description of the 10 best ethnic dishes will follow, with recipes: " -n 1000 [additional benchmark flags]`

Result:

| command | tokens/second (higher is better) |
| - | - |
| -ngl 2000000 | N/A (less than 0.1) |
| -t 7 | 1.7 |
| -t 1 -ngl 2000000 | 5.5 |
| -t 7 -ngl 2000000 | 8.7 |
| -t 4 -ngl 2000000 | 9.1 |
