#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Backend-local fusion of a top-k selection mask with the existing sparse tile kernel.
bool ggml_cuda_flash_attn_ext_indices_supported(int device, const ggml_tensor * dst,
        const ggml_tensor * mask, const ggml_tensor * indices);
void ggml_cuda_flash_attn_ext_indices(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        ggml_tensor * mask, ggml_tensor * indices);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
