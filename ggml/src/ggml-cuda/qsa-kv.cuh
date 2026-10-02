#pragma once

#include "common.cuh"
#include "ggml-backend.h"

// Four contiguous cache rows form a page. RAM is authoritative; a negative
// mapping is also the correctness fallback when a selection exceeds capacity.
struct ggml_cuda_qsa_kv_view {
    const char * host       = nullptr;
    const char * cache      = nullptr;
    const int  * pages      = nullptr;
    uint32_t     page_shift = 0;
};

static __device__ __forceinline__ const char * ggml_cuda_qsa_kv_address(
        const char * address, const ggml_cuda_qsa_kv_view & view) {
    if (!view.pages) {
        return address;
    }

    const size_t offset    = address - view.host;
    const int    slot      = view.pages[offset >> view.page_shift];
    const size_t page_mask = (size_t(1) << view.page_shift) - 1;

    return slot < 0 ? address : view.cache + (size_t(slot) << view.page_shift) + (offset & page_mask);
}

bool ggml_backend_buft_is_cuda_qsa_kv(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_cuda_qsa_kv_buffer_type(ggml_backend_dev_t dev, uint32_t resident_tokens);
size_t ggml_backend_cuda_qsa_kv_tensor_device_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);

void * ggml_cuda_qsa_kv_device_ptr(const ggml_tensor * tensor);
bool ggml_cuda_qsa_kv_is_paged(const ggml_tensor * tensor);
// Counts all physical pages in the view, including partial pages at either end.
bool ggml_cuda_qsa_kv_fits(const ggml_tensor * tensor);

void ggml_cuda_qsa_kv_prepare(
        ggml_backend_cuda_context & ctx, const ggml_tensor * K, const ggml_tensor * V,
        const int32_t * indices, int n_indices, ggml_cuda_qsa_kv_view & K_cache, ggml_cuda_qsa_kv_view & V_cache);

void ggml_cuda_qsa_kv_invalidate_rows(
        ggml_backend_cuda_context & ctx, const ggml_tensor * tensor, const ggml_tensor * indices);

void ggml_cuda_qsa_kv_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor);
