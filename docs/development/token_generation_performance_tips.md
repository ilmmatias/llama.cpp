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

Supported ZNQ expert-cache outputs stay on-device when their next consumers run on the cache GPU. A row-gather graph combines cached outputs with uploaded CPU misses before GPU weighting and reduction; fully cached batches need no CPU-output upload. This avoids the cached-output GPU-to-CPU round trip. CPU consumers and evaluation callbacks keep the host-side join, and batches larger than 128 tokens keep the CPU bulk-prefill path.

CPU_REPACK `MUL_MAT` and `MUL_MAT_ID` accept F32 or the weight type's native vector-dot input type (Q8_0 or Q8_K). Prequantized inputs keep their tensor type and row strides; the CPU packs full GEMM groups without dequantizing them and reads GEMV tails directly. ZNQ expert-cache hits also accept native Q8_0 inputs, including combined gate/up weights, and pack them into the GPU's Q8_1 layout. Existing F32 graphs remain unchanged.

This is an operator capability, not automatic graph quantization. Callers must explicitly produce a supported quantized tensor before a GPU-to-CPU handoff. Measure transfer and packing costs against the F32 path before using it in a model graph.

Profile the active execution path before changing synchronization. Expert-cache configurations can bypass the routing-ID readback callback, and a backend transition without split inputs can still require a wait because scratch allocations are reused. Compare unprofiled `llama-bench` runs with the same context depth, offload settings, and thread count; profiler overhead can change the apparent bottleneck.

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
