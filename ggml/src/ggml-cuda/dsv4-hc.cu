#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

// Parallel 4x4 Sinkhorn kernel. Each 16-lane group owns one token and one
// lane owns one matrix element (idst + 4*isrc). The row/column reductions are
// then just xor-shuffles inside that 16-lane group. This mirrors the Vulkan
// implementation and, unlike the scalar one-thread-per-token path, exposes
// enough parallelism for prompt batches on wave32 GPUs.
static __global__ void dsv4_hc_comb_f32_wave(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;
    constexpr int lanes_per_token = DSV4_HC*DSV4_HC;
    constexpr int shuffle_width = lanes_per_token;

    ggml_cuda_pdl_lc();

    const int tid = threadIdx.x;
    const int token_in_block = tid / lanes_per_token;
    const int idx = tid % lanes_per_token;
    const int tokens_per_block = blockDim.x / lanes_per_token;
    const int64_t it = (int64_t) blockIdx.x * tokens_per_block + token_in_block;
    const bool in_range = it < n_tokens;

    // All lanes remain active through the shuffle sequence. Out-of-range token
    // groups compute harmless dummy values and simply skip the final store.
    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float v = 0.0f;
    if (in_range) {
        v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
    }

    // Softmax over destination lanes (idx bits 0..1).
    float vmax = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1, shuffle_width));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, 2, shuffle_width));
    v = expf(v - vmax);

    float sum = v + __shfl_xor_sync(0xffffffff, v, 1, shuffle_width);
    sum += __shfl_xor_sync(0xffffffff, sum, 2, shuffle_width);
    v = v / sum + eps;

    // Normalize columns (same idst, isrc differs: xor 4 and 8).
    sum = v + __shfl_xor_sync(0xffffffff, v, 4, shuffle_width);
    sum += __shfl_xor_sync(0xffffffff, sum, 8, shuffle_width);
    v /= sum + eps;

    for (int32_t i = 1; i < n_iter; ++i) {
        // Normalize rows.
        sum = v + __shfl_xor_sync(0xffffffff, v, 1, shuffle_width);
        sum += __shfl_xor_sync(0xffffffff, sum, 2, shuffle_width);
        v /= sum + eps;

        // Normalize columns.
        sum = v + __shfl_xor_sync(0xffffffff, v, 4, shuffle_width);
        sum += __shfl_xor_sync(0xffffffff, sum, 8, shuffle_width);
        v /= sum + eps;
    }

    if (in_range) {
        const int idst = idx & (DSV4_HC - 1);
        const int isrc = idx / DSV4_HC;
        dst[idst*sd0 + isrc*sd1 + it*sd2] = v;
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

// One block owns one embedding tile for one token. The four stream weights
// are uniform across the tile, so stage them once and reuse them for all 256
// embedding elements instead of issuing the same four weight loads per thread.
static __global__ void dsv4_hc_pre_f32_tiled(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sd0,
        int64_t sd1) {
    __shared__ float weights_s[DSV4_HC];

    ggml_cuda_pdl_lc();

    const int tid = threadIdx.x;
    const int64_t it = blockIdx.y;
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + tid;

    ggml_cuda_pdl_sync();

    if (tid < DSV4_HC) {
        weights_s[tid] = weights[tid*sw0 + it*sw1];
    }
    __syncthreads();

    if (i0 >= n_embd) {
        return;
    }

    float sum = 0.0f;
#pragma unroll
    for (int ih = 0; ih < DSV4_HC; ++ih) {
        sum = fmaf(x[i0*sx0 + ih*sx1 + it*sx2], weights_s[ih], sum);
    }

    dst[i0*sd0 + it*sd1] = sum;
}

// One block owns one embedding tile for one token. The 4 post coefficients
// and 4x4 combination matrix are uniform across that tile, so stage them once
// in shared memory. Each thread then loads x and the four residual streams once
// and produces all four destination streams. This mirrors the Vulkan kernel and
// avoids re-reading x/residual four times in separate destination blocks.
static __global__ void dsv4_hc_post_f32_tiled(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    __shared__ float post_s[DSV4_HC];
    __shared__ float comb_s[DSV4_HC*DSV4_HC];

    ggml_cuda_pdl_lc();

    const int tid = threadIdx.x;
    const int64_t it = blockIdx.y;
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + tid;

    ggml_cuda_pdl_sync();

    if (tid < DSV4_HC) {
        post_s[tid] = post[tid*sp0 + it*sp1];
    }
    if (tid < DSV4_HC*DSV4_HC) {
        const int idst = tid & (DSV4_HC - 1);
        const int isrc = tid / DSV4_HC;
        comb_s[tid] = comb[idst*sc0 + isrc*sc1 + it*sc2];
    }
    __syncthreads();

    if (i0 >= n_embd) {
        return;
    }

    const float xv = x[i0*sx0 + it*sx1];
    float r[DSV4_HC];
#pragma unroll
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        r[isrc] = residual[i0*sr0 + isrc*sr1 + it*sr2];
    }

#pragma unroll
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = xv * post_s[idst];
#pragma unroll
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum = fmaf(r[isrc], comb_s[idst + DSV4_HC*isrc], sum);
        }
        dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
    }
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    // 16 lanes per token; 128 threads expose 8 independent tokens per block.
    // This is enough work to fill an RDNA2 GPU even at ordinary prompt batch
    // sizes while keeping every 4x4 matrix entirely in registers.
    constexpr int block_size = 128;
    constexpr int lanes_per_token = DSV4_HC*DSV4_HC;
    constexpr int tokens_per_block = block_size / lanes_per_token;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + tokens_per_block - 1) / tokens_per_block, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32_wave, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc == DSV4_HC);

    constexpr int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, n_tokens, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32_tiled, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc == DSV4_HC);

    constexpr int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, n_tokens, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_f32_tiled, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}
