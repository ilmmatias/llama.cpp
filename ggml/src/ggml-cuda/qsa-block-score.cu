#include "common.cuh"
#include "qsa-block-score.cuh"


// Released Qwen4Exp indexer shape.  A Wave32 is split into four independent
// 8-lane groups.  Each group owns one indexer head and reduces a 128-element
// dot product.  The cached block key is staged once per wave and reused by all
// four heads, avoiding four identical global-memory reads.
static __global__ void qsa_block_score_f32_128x4_wave32(
        const float * q,
        const float * k,
        const int32_t * cells,
        const float * mask,
        float * dst,
        int64_t n_blocks,
        int64_t n_query,
        int64_t n_stream,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sk1,
        int64_t sc0,
        int64_t sc1,
        int64_t sm0,
        int64_t sm1,
        int64_t sm2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float scale) {
    constexpr int wave_size = 32;
    constexpr int n_embd = 128;
    constexpr int n_head = 4;
    constexpr int lanes_per_head = wave_size / n_head;
    constexpr int waves_per_block = 8; // 256 threads

    __shared__ float key_s[waves_per_block][n_embd];

    ggml_cuda_pdl_lc();

    const int tid = threadIdx.x;
    const int wave = tid / wave_size;
    const int lane = tid & (wave_size - 1);
    const int64_t ir = (int64_t) blockIdx.x * waves_per_block + wave;
    const int64_t nr = n_blocks*n_query*n_stream;
    const bool valid = ir < nr;

    int64_t ib = 0;
    int64_t iq = 0;
    int64_t is = 0;
    int32_t cell = 0;
    if (valid) {
        ib = ir % n_blocks;
        iq = (ir/n_blocks) % n_query;
        is = ir/(n_blocks*n_query);
        cell = cells[ib*sc0 + is*sc1];
    }

    // Every wave, including padding waves in the final block, participates in
    // the block barrier.  Invalid waves only stage zeroes and never store.
#pragma unroll
    for (int i = lane; i < n_embd; i += wave_size) {
        key_s[wave][i] = valid ? k[(int64_t) cell*sk1 + i] : 0.0f;
    }

    ggml_cuda_pdl_sync();
    __syncthreads();

    const int ih = lane / lanes_per_head;
    const int il = lane & (lanes_per_head - 1);

    float dot = 0.0f;
    if (valid) {
        const float * q_row = q + ih*sq1 + iq*sq2 + is*sq3;
#pragma unroll
        for (int i = il; i < n_embd; i += lanes_per_head) {
            dot = fmaf(q_row[i], key_s[wave][i], dot);
        }
    }

    // width=8 makes four independent reductions inside the Wave32.
    dot += __shfl_xor_sync(0xffffffff, dot, 4, lanes_per_head);
    dot += __shfl_xor_sync(0xffffffff, dot, 2, lanes_per_head);
    dot += __shfl_xor_sync(0xffffffff, dot, 1, lanes_per_head);

    // ReLU is per head, before the head sum.  Keep only the four subgroup
    // leaders.  Lane 0 fetches lanes 8/16/24 and adds in head order, matching
    // the generic graph's left-to-right four-head reduction as closely as the
    // different dot-product reduction permits.
    const float head_score = il == 0 ? fmaxf(dot, 0.0f) : 0.0f;
    const float h1 = __shfl_xor_sync(0xffffffff, head_score,  8, wave_size);
    const float h2 = __shfl_xor_sync(0xffffffff, head_score, 16, wave_size);
    const float h3 = __shfl_xor_sync(0xffffffff, head_score, 24, wave_size);

    if (valid && lane == 0) {
        const float score = ((head_score + h1) + h2) + h3;
        dst[ib*sd0 + iq*sd1 + is*sd2] =
            score*scale + mask[ib*sm0 + iq*sm1 + is*sm2];
    }
}


// Batched prefill path for the released 128x4 indexer shape. Each 256-thread
// workgroup covers an 8-block x 8-query tile. One wave stages one block key,
// then the eight waves reuse all staged keys while each wave owns one query.
// This keeps the head dimension fused without giving up the block/query reuse
// that makes the large-n_query path bandwidth-efficient.
static __global__ void qsa_block_score_f32_128x4_wave32_batched(
        const float * q,
        const float * k,
        const int32_t * cells,
        const float * mask,
        float * dst,
        int64_t n_blocks,
        int64_t n_query,
        int64_t n_stream,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sk1,
        int64_t sc0,
        int64_t sc1,
        int64_t sm0,
        int64_t sm1,
        int64_t sm2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float scale) {
    constexpr int wave_size = 32;
    constexpr int n_embd = 128;
    constexpr int n_head = 4;
    constexpr int lanes_per_head = wave_size / n_head;
    constexpr int waves_per_block = 8; // 256 threads
    constexpr int blocks_per_tile = 8;

    __shared__ float key_s[blocks_per_tile][n_embd];

    ggml_cuda_pdl_lc();

    const int tid = threadIdx.x;
    const int wave = tid / wave_size;
    const int lane = tid & (wave_size - 1);
    const int64_t ib0 = (int64_t) blockIdx.x * blocks_per_tile;
    const int64_t iq = (int64_t) blockIdx.y * waves_per_block + wave;
    const int64_t is = blockIdx.z;

    // Wave w stages key w. All eight waves then reuse the 8-key tile for their
    // own query, cutting repeated key traffic by the query-tile width.
    const int64_t key_ib = ib0 + wave;
    const bool valid_key = key_ib < n_blocks;
    const int32_t cell = valid_key ? cells[key_ib*sc0 + is*sc1] : 0;

#pragma unroll
    for (int i = lane; i < n_embd; i += wave_size) {
        key_s[wave][i] = valid_key ? k[(int64_t) cell*sk1 + i] : 0.0f;
    }

    ggml_cuda_pdl_sync();
    __syncthreads();

    const bool valid_query = iq < n_query;
    const int ih = lane / lanes_per_head;
    const int il = lane & (lanes_per_head - 1);

    float dot[blocks_per_tile] = {};
    if (valid_query) {
        const float * q_row = q + ih*sq1 + iq*sq2 + is*sq3;
#pragma unroll
        for (int i = il; i < n_embd; i += lanes_per_head) {
            const float qv = q_row[i];
#pragma unroll
            for (int jb = 0; jb < blocks_per_tile; ++jb) {
                dot[jb] = fmaf(qv, key_s[jb][i], dot[jb]);
            }
        }
    }

#pragma unroll
    for (int jb = 0; jb < blocks_per_tile; ++jb) {
        float v = dot[jb];
        v += __shfl_xor_sync(0xffffffff, v, 4, lanes_per_head);
        v += __shfl_xor_sync(0xffffffff, v, 2, lanes_per_head);
        v += __shfl_xor_sync(0xffffffff, v, 1, lanes_per_head);

        const float head_score = il == 0 ? fmaxf(v, 0.0f) : 0.0f;
        const float h1 = __shfl_xor_sync(0xffffffff, head_score,  8, wave_size);
        const float h2 = __shfl_xor_sync(0xffffffff, head_score, 16, wave_size);
        const float h3 = __shfl_xor_sync(0xffffffff, head_score, 24, wave_size);

        const int64_t ib = ib0 + jb;
        if (valid_query && ib < n_blocks && lane == 0) {
            const float score = ((head_score + h1) + h2) + h3;
            dst[ib*sd0 + iq*sd1 + is*sd2] =
                score*scale + mask[ib*sm0 + iq*sm1 + is*sm2];
        }
    }
}


// Portable fallback for unusual indexer shapes / non-Wave32 devices.  This is
// a correctness path; the performance target for this branch is the 128x4
// Wave32 kernels above.
static __global__ void qsa_block_score_f32_generic(
        const float * q,
        const float * k,
        const int32_t * cells,
        const float * mask,
        float * dst,
        int64_t n_embd,
        int64_t n_head,
        int64_t n_blocks,
        int64_t n_query,
        int64_t n_stream,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sk1,
        int64_t sc0,
        int64_t sc1,
        int64_t sm0,
        int64_t sm1,
        int64_t sm2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float scale) {
    ggml_cuda_pdl_lc();

    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_blocks*n_query*n_stream;
    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t ib = ir % n_blocks;
    const int64_t iq = (ir/n_blocks) % n_query;
    const int64_t is = ir/(n_blocks*n_query);
    const int32_t cell = cells[ib*sc0 + is*sc1];
    const float * k_row = k + (int64_t) cell*sk1;

    float score = 0.0f;
    for (int64_t ih = 0; ih < n_head; ++ih) {
        const float * q_row = q + ih*sq1 + iq*sq2 + is*sq3;
        float dot = 0.0f;
        for (int64_t i = 0; i < n_embd; ++i) {
            dot = fmaf(q_row[i], k_row[i], dot);
        }
        score += fmaxf(dot, 0.0f);
    }

    dst[ib*sd0 + iq*sd1 + is*sd2] =
        score*scale + mask[ib*sm0 + iq*sm1 + is*sm2];
}


void ggml_cuda_op_qsa_block_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * cells = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(cells->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(q));
    GGML_ASSERT(ggml_is_contiguous_rows(k));
    GGML_ASSERT(ggml_is_contiguous(cells));
    GGML_ASSERT(ggml_is_contiguous(mask));

    GGML_TENSOR_LOCALS(size_t, nbq, q,     nb);
    GGML_TENSOR_LOCALS(size_t, nbk, k,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, cells, nb);
    GGML_TENSOR_LOCALS(size_t, nbm, mask,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_embd   = q->ne[0];
    const int64_t n_head   = q->ne[1];
    const int64_t n_query  = q->ne[2];
    const int64_t n_stream = q->ne[3];
    const int64_t n_blocks = cells->ne[0];
    const float scale = ggml_get_op_params_f32(dst, 0);

    GGML_ASSERT(k->ne[0] == n_embd);
    GGML_ASSERT(cells->ne[1] == n_stream);
    GGML_ASSERT(mask->ne[0] == n_blocks);
    GGML_ASSERT(mask->ne[1] == n_query);
    GGML_ASSERT(mask->ne[2] == n_stream);
    GGML_ASSERT(dst->ne[0] == n_blocks);
    GGML_ASSERT(dst->ne[1] == n_query);
    GGML_ASSERT(dst->ne[2] == n_stream);

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    if (warp_size == 32 && n_embd == 128 && n_head == 4) {
        constexpr int block_size = 256;
        const dim3 block_dims(block_size, 1, 1);

        if (n_query >= 8) {
            constexpr int blocks_per_tile = 8;
            constexpr int queries_per_tile = block_size / 32;
            const dim3 grid_dims(
                    (n_blocks + blocks_per_tile - 1) / blocks_per_tile,
                    (n_query  + queries_per_tile - 1) / queries_per_tile,
                    n_stream);
            const ggml_cuda_kernel_launch_params launch_params =
                ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

            ggml_cuda_kernel_launch(qsa_block_score_f32_128x4_wave32_batched, launch_params,
                    (const float *) q->data,
                    (const float *) k->data,
                    (const int32_t *) cells->data,
                    (const float *) mask->data,
                    (float *) dst->data,
                    n_blocks, n_query, n_stream,
                    nbq1 / sizeof(float), nbq2 / sizeof(float), nbq3 / sizeof(float),
                    nbk1 / sizeof(float),
                    nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
                    nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
                    nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
                    scale);
            return;
        }

        constexpr int waves_per_block = block_size / 32;
        const int64_t nr = n_blocks*n_query*n_stream;
        const dim3 grid_dims((nr + waves_per_block - 1) / waves_per_block, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

        ggml_cuda_kernel_launch(qsa_block_score_f32_128x4_wave32, launch_params,
                (const float *) q->data,
                (const float *) k->data,
                (const int32_t *) cells->data,
                (const float *) mask->data,
                (float *) dst->data,
                n_blocks, n_query, n_stream,
                nbq1 / sizeof(float), nbq2 / sizeof(float), nbq3 / sizeof(float),
                nbk1 / sizeof(float),
                nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
                nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
                nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
                scale);
        return;
    }

    constexpr int block_size = 256;
    const int64_t nr = n_blocks*n_query*n_stream;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(qsa_block_score_f32_generic, launch_params,
            (const float *) q->data,
            (const float *) k->data,
            (const int32_t *) cells->data,
            (const float *) mask->data,
            (float *) dst->data,
            n_embd, n_head, n_blocks, n_query, n_stream,
            nbq1 / sizeof(float), nbq2 / sizeof(float), nbq3 / sizeof(float),
            nbk1 / sizeof(float),
            nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
            nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            scale);
}
