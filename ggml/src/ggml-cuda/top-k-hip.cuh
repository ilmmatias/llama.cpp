#pragma once

#include "common.cuh"

bool ggml_cuda_top_k_hip(
        ggml_backend_cuda_context & ctx,
        const float * src,
        int * dst,
        int64_t ncols,
        int64_t nrows,
        int64_t k,
        ggml_cuda_pool & pool,
        cudaStream_t stream);
