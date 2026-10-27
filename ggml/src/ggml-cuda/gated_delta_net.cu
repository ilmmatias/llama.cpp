#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include "unary.cuh"

#include <cstdint>

#if defined(GGML_USE_HIP)
constexpr int gdn_cols_per_warp = 1;
#else
constexpr int gdn_cols_per_warp = 4;
#endif
// On RDNA the generic warp_reduce_sum compiles to 5 dependent ds_bpermute round trips (~100+ cycles each),
// which dominates the per-token latency of the recurrence.
#if defined(GGML_USE_HIP) && defined(RDNA)
template <int mask>
static __device__ __forceinline__ float gdn_dpp_row_xmask(const float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
}

static __device__ __forceinline__ float gdn_permlanex16_swap(const float x) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false));
}

template <int width>
static __device__ __forceinline__ float gdn_warp_reduce_sum(float x) {
    static_assert(
        width == 1 || width == 2 || width == 4 ||
        width == 8 || width == 16 || width == 32,
        "unsupported reduction width");

    // XOR 16 crosses two 16-lane DPP rows.
    if constexpr (width >= 32) {
        x += gdn_permlanex16_swap(x);
    }

    // Remaining stages are entirely within a 16-lane DPP row.
    if constexpr (width >= 16) {
        x += gdn_dpp_row_xmask<8>(x);
    }
    if constexpr (width >= 8) {
        x += gdn_dpp_row_xmask<4>(x);
    }
    if constexpr (width >= 4) {
        x += gdn_dpp_row_xmask<2>(x);
    }
    if constexpr (width >= 2) {
        x += gdn_dpp_row_xmask<1>(x);
    }

    return x;
}

template <> __device__ __forceinline__ float warp_reduce_sum<1>(const float x) { return gdn_warp_reduce_sum<1>(x); }
template <> __device__ __forceinline__ float warp_reduce_sum<2>(const float x) { return gdn_warp_reduce_sum<2>(x); }
template <> __device__ __forceinline__ float warp_reduce_sum<4>(const float x) { return gdn_warp_reduce_sum<4>(x); }
template <> __device__ __forceinline__ float warp_reduce_sum<8>(const float x) { return gdn_warp_reduce_sum<8>(x); }
template <> __device__ __forceinline__ float warp_reduce_sum<16>(const float x) { return gdn_warp_reduce_sum<16>(x); }
template <> __device__ __forceinline__ float warp_reduce_sum<32>(const float x) { return gdn_warp_reduce_sum<32>(x); }
#endif // defined(GGML_USE_HIP) && defined(RDNA)

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_group = 1, int row_lanes = 0, int token_tile = 1>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * curr_state,
        float *       dst,
        float *       state,
        int64_t       H,
        int64_t       n_tokens,
        int64_t       n_seqs,
        int64_t       sq1,
        int64_t       sq2,
        int64_t       sq3,
        int64_t       sv1,
        int64_t       sv2,
        int64_t       sv3,
        int64_t       sb1,
        int64_t       sb2,
        int64_t       sb3,
        const uint3   neqk1_magic,
        const uint3   rq3_magic,
        float         scale,
        int64_t       state_slot_stride,
        int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    constexpr int physical_warp = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int warp_size = row_lanes ? row_lanes : physical_warp / gdn_cols_per_warp;

    // A physical warp can tile multiple value columns with independent row subgroups.
    const int lane = threadIdx.x % warp_size;
    const int col_base = ((blockIdx.z * blockDim.y + threadIdx.y) * (physical_warp / warp_size)
                         + threadIdx.x / warp_size) * cols_per_group;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col_base * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    static_assert(physical_warp % warp_size == 0, "row groups must partition the physical warp");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[cols_per_group][rows_per_lane];

    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int c = 0; c < cols_per_group; ++c) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[c][r] = curr_state[c * S_v + i];
        }
    }

    // Stage a short time tile before the dependent updates. State stays in
    // registers and the recurrence is still evaluated in token order.
    for (int first = 0; first < n_tokens; first += token_tile) {
        float k_tile[token_tile][rows_per_lane], q_tile[token_tile][rows_per_lane];
        float v_tile[token_tile][cols_per_group], beta_tile[token_tile], g_tile[token_tile];

#pragma unroll
        for (int j = 0; j < token_tile; ++j) {
            const int t = first + j;
            if constexpr (token_tile > 1) {
                if (t >= n_tokens) {
                    break;
                }
            }

            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            beta_tile[j] = beta[gb_offset];
            g_tile[j] = KDA ? 0.0f : expf(g[gb_offset]);

#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int i = r * warp_size + lane;
                k_tile[j][r] = k_t[i];
                q_tile[j][r] = q_t[i];
            }

#pragma unroll
            for (int c = 0; c < cols_per_group; ++c) {
                v_tile[j][c] = v_t[col_base + c];
            }
        }

#pragma unroll
        for (int j = 0; j < token_tile; ++j) {
            const int t = first + j;
            if constexpr (token_tile > 1) {
                if (t >= n_tokens) {
                    break;
                }
            }

            const float * g_t = g + (sequence * sb3 + t * sb2 + h_idx * sb1) * (KDA ? S_v : 1);
            const float beta_val = beta_tile[j];
            const float g_val    = g_tile[j];

            const float * k_reg = k_tile[j];
            const float * q_reg = q_tile[j];
            const float * v_reg = v_tile[j];

            float attn_reg[cols_per_group];

#pragma unroll
            for (int c = 0; c < cols_per_group; ++c) {
                const int col = col_base + c;
                if constexpr (!KDA) {
                    // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                    float kv_shard = 0.0f;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += s_shard[c][r] * k_reg[r];
                    }
                    float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                    // delta[col] = (v[col] - g * kv[col]) * beta
                    float delta_col = (v_reg[c] - g_val * kv_col) * beta_val;

                    // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                    // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                    float attn_partial = 0.0f;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        if constexpr (cols_per_group > 1) {
                            // Match the single-column kernel's FMA contraction.
                            s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * delta_col);
                        } else {
                            s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                        }
                        attn_partial += s_shard[c][r] * q_reg[r];
                    }

                    float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                    attn_reg[c] = attn_col * scale;
                } else {
                    // kv[col] = sum_i g[i] * S[i][col] * k[i]
                    float kv_shard = 0.0f;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        kv_shard += expf(g_t[i]) * s_shard[c][r] * k_reg[r];
                    }

                    float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                    // delta[col] = (v[col] - kv[col]) * beta
                    float delta_col = (v_reg[c] - kv_col) * beta_val;

                    // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
                    // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                    float attn_partial = 0.0f;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        s_shard[c][r]  = expf(g_t[i]) * s_shard[c][r] + k_reg[r] * delta_col;
                        attn_partial += s_shard[c][r] * q_reg[r];
                    }

                    float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                    attn_reg[c] = attn_col * scale;
                }

                if constexpr (keep_rs_t) {
                    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
                    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
                    const int target_slot = (int) n_tokens - 1 - t;
                    if (target_slot >= 0 && target_slot < K) {
                        float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            const int i = r * warp_size + lane;
                            curr_state[col * S_v + i] = s_shard[c][r];
                        }
                    }
                }
            }

            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < cols_per_group; ++c) {
                    attn_data[col_base + c] = attn_reg[c];
                }
            }

            attn_data += S_v * H;
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < cols_per_group; ++c) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                state[(col_base + c) * S_v + i] = s_shard[c][r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const auto & device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const int warp_size = device.warp_size;
    int num_warps = 4;
    const bool use_multi_col = gdn_cols_per_warp == 1 && !KDA && !keep_rs_t && S_v == 128 && H * n_seqs >= 32;
    const int cols_per_warp = use_multi_col ? (n_tokens >= 128 ? 8 : 4) : gdn_cols_per_warp;
    if (gdn_cols_per_warp > 1) {
        while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < device.nsm) {
            num_warps /= 2;
        }
    }
    const int cols_per_block = num_warps * cols_per_warp;
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_block - 1) / cols_per_block);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    if constexpr (!KDA && !keep_rs_t) {
        if (S_v == 128 && n_tokens >= 32 && H * n_seqs >= 32 && GGML_CUDA_CC_IS_RDNA2(device.cc)) {
            // RDNA2: two 16-lane row groups, four values/group, two tokens.
            // Selected by gfx1031 timing sweeps; other shapes use the fallback.
            constexpr int rows   = 16;
            constexpr int cols   = 4;
            constexpr int tokens = 2;

            const int tile_cols = num_warps * (warp_size / rows) * cols;
            const ggml_cuda_kernel_launch_params tiled(dim3(H, n_seqs, S_v / tile_cols), block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, false, cols, rows, tokens>, tiled,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            if constexpr (gdn_cols_per_warp == 1 && !KDA && !keep_rs_t) {
                if (cols_per_warp == 4) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, false, 4>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    break;
                }
                if (cols_per_warp == 8) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, false, 8>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (!keep_rs) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

template <typename T>
static __device__ __forceinline__ float gdn_ab_at(const T * w, int k) {
    return ggml_cuda_cast<float>(w[k]);
}

template <>
__device__ __forceinline__ float gdn_ab_at(const block_q8_0 * w, int k) {
    const block_q8_0 & b = w[k/QK8_0];
    return ggml_cuda_cast<float>(b.d)*b.qs[k % QK8_0];
}

template <typename T>
static __device__ __forceinline__ float gdn_ab_dot(const void * weights, const float * x, int K) {
    const T * w = (const T *) weights;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += gdn_ab_at(w, k)*x[k];
    }
    return sum;
}

template <int qk, int qr, dequantize_kernel_t dequantize>
static __device__ __forceinline__ float gdn_ab_dot_q(const void * w, const float * x, int K) {
    float sum = 0.0f;
    for (int k = 2*threadIdx.x; k < K; k += 2*blockDim.x) {
        const int ib = k/qk;
        const int iq = (k % qk)/qr;
        const int offset = qr == 1 ? 1 : qk/2;
        float2 v;
        dequantize(w, ib, iq, v);
        sum += v.x*x[ib*qk + iq] + v.y*x[ib*qk + iq + offset];
    }
    return sum;
}

template <dequantize_kq_t<float> dequantize, int threads, int subblock = QK_K>
static __device__ __forceinline__ float gdn_ab_dot_kq(const void * w, const float * x, int K, float * tile) {
    const int group = threadIdx.x/threads;
    const int lane = threadIdx.x % threads;
    const int groups = blockDim.x/threads;
    float * values = tile + group*QK_K;
    float sum = 0.0f;

    for (int base = 0; base < K; base += groups*QK_K) {
        const int k = base + group*QK_K;
        if (k < K && (subblock == QK_K || k + (lane % (QK_K/subblock))*subblock < K)) {
            dequantize(w, k/QK_K, values, lane);
        }
        __syncthreads();
        for (int j = lane; j < QK_K && k + j < K; j += threads) {
            sum += values[j]*x[k + j];
        }
        __syncthreads();
    }
    return sum;
}

static __device__ __forceinline__ float gdn_ab_dot_nvfp4(const void * weights, const float * x, int K) {
    const block_nvfp4 * w = (const block_nvfp4 *) weights;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < K/2; k += blockDim.x) {
        const int ib = k/(QK_NVFP4/2);
        const int sub = (k % (QK_NVFP4/2))/(QK_NVFP4_SUB/2);
        const int j = k % (QK_NVFP4_SUB/2);
        const float d = ggml_cuda_ue4m3_to_fp32(w[ib].d[sub]);
        const uint8_t q = w[ib].qs[sub*(QK_NVFP4_SUB/2) + j];
        const int i = ib*QK_NVFP4 + sub*QK_NVFP4_SUB + j;
        sum += d*(kvalues_mxfp4[q & 15]*x[i] + kvalues_mxfp4[q >> 4]*x[i + QK_NVFP4_SUB/2]);
    }
    return sum;
}

// One block per (head, projection, token).
template <int block_size, ggml_type weight_type = GGML_TYPE_COUNT>
static __global__ void k_gdn_ab_mul_mat(
        const float * __restrict__ x,
        const void * __restrict__ w_alpha,
        const void * __restrict__ w_beta,
        const ggml_type type_alpha, const ggml_type type_beta,
        const size_t stride_alpha, const size_t stride_beta,
        const float * __restrict__ dt,
        const float * __restrict__ ssm_a,
        float * __restrict__ gate,
        float * __restrict__ beta,
        int n_heads,
        int K) {
    const int  h       = blockIdx.x;
    const int  n       = blockIdx.y;
    const bool is_beta = blockIdx.z != 0;

    const void * w = is_beta ? (const char *) w_beta + h*stride_beta : (const char *) w_alpha + h*stride_alpha;
    const float * xn = x + int64_t(n)*K;

    float sum = 0.0f;
    if constexpr (weight_type == GGML_TYPE_F32) {
        sum = gdn_ab_dot<float>(w, xn, K);
    } else if constexpr (weight_type == GGML_TYPE_F16) {
        sum = gdn_ab_dot<half>(w, xn, K);
    } else if constexpr (weight_type == GGML_TYPE_BF16) {
        sum = gdn_ab_dot<nv_bfloat16>(w, xn, K);
    } else if constexpr (weight_type == GGML_TYPE_Q8_0) {
        sum = gdn_ab_dot<block_q8_0>(w, xn, K);
    } else {
        extern __shared__ float tile[];
        const ggml_type type = is_beta ? type_beta : type_alpha;
        switch (type) {
            case GGML_TYPE_F32:  sum = gdn_ab_dot<float>(w, xn, K);         break;
            case GGML_TYPE_F16:  sum = gdn_ab_dot<half>(w, xn, K);          break;
            case GGML_TYPE_BF16: sum = gdn_ab_dot<nv_bfloat16>(w, xn, K);   break;
            case GGML_TYPE_Q8_0: sum = gdn_ab_dot<block_q8_0>(w, xn, K);    break;
            case GGML_TYPE_Q1_0: sum = gdn_ab_dot_q<QK1_0, QR1_0, dequantize_q1_0>(w, xn, K); break;
            case GGML_TYPE_Q2_0: sum = gdn_ab_dot_q<QK2_0, QR2_0, dequantize_q2_0>(w, xn, K); break;
            case GGML_TYPE_Q4_0: sum = gdn_ab_dot_q<QK4_0, QR4_0, dequantize_q4_0>(w, xn, K); break;
            case GGML_TYPE_Q4_1: sum = gdn_ab_dot_q<QK4_1, QR4_1, dequantize_q4_1>(w, xn, K); break;
            case GGML_TYPE_Q5_0: sum = gdn_ab_dot_q<QK5_0, QR5_0, dequantize_q5_0>(w, xn, K); break;
            case GGML_TYPE_Q5_1: sum = gdn_ab_dot_q<QK5_1, QR5_1, dequantize_q5_1>(w, xn, K); break;
            case GGML_TYPE_ZNQ2_M: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<2, 16>>(w, xn, K); break;
            case GGML_TYPE_ZNQ2_L: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<2, 8>>(w, xn, K); break;
            case GGML_TYPE_ZNQ3_S: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<2, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ3_M: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<3, 16>>(w, xn, K); break;
            case GGML_TYPE_ZNQ3_L: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<3, 8>>(w, xn, K); break;
            case GGML_TYPE_ZNQ4_S: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<3, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ4_M: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<4, 16>>(w, xn, K); break;
            case GGML_TYPE_ZNQ4_L: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<4, 8>>(w, xn, K); break;
            case GGML_TYPE_ZNQ5_S: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<4, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ5_M: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<5, 16>>(w, xn, K); break;
            case GGML_TYPE_ZNQ5_L: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<5, 8>>(w, xn, K); break;
            case GGML_TYPE_ZNQ6_S: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<5, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ6_M: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<6, 16>>(w, xn, K); break;
            case GGML_TYPE_ZNQ6_L: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<6, 8>>(w, xn, K); break;
            case GGML_TYPE_ZNQ7_S: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<6, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ7_M: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<7, 16>>(w, xn, K); break;
            case GGML_TYPE_ZNQ7_L: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<7, 8>>(w, xn, K); break;
            case GGML_TYPE_ZNQ2_S: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<1, 4, 3>>(w, xn, K); break;
            case GGML_TYPE_ZNQ2_SM: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<1, 4, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ3_SM: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<2, 4, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ4_SM: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<3, 4, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ5_SM: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<4, 4, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ6_SM: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<5, 4, 4>>(w, xn, K); break;
            case GGML_TYPE_ZNQ7_SM: sum = gdn_ab_dot_q<QK_ZNQ, 1, dequantize_znq<6, 4, 4>>(w, xn, K); break;
            case GGML_TYPE_Q2_K: sum = gdn_ab_dot_kq<dequantize_q2_K, 64>(w, xn, K, tile); break;
            case GGML_TYPE_Q3_K: sum = gdn_ab_dot_kq<dequantize_q3_K, 64>(w, xn, K, tile); break;
            case GGML_TYPE_Q4_K: sum = gdn_ab_dot_kq<dequantize_q4_K, 32>(w, xn, K, tile); break;
            case GGML_TYPE_Q5_K: sum = gdn_ab_dot_kq<dequantize_q5_K, 64>(w, xn, K, tile); break;
            case GGML_TYPE_Q6_K: sum = gdn_ab_dot_kq<dequantize_q6_K, 64>(w, xn, K, tile); break;
            case GGML_TYPE_IQ2_XXS: sum = gdn_ab_dot_kq<dequantize_iq2_xxs, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ2_XS: sum = gdn_ab_dot_kq<dequantize_iq2_xs, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ2_S: sum = gdn_ab_dot_kq<dequantize_iq2_s, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ3_XXS: sum = gdn_ab_dot_kq<dequantize_iq3_xxs, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ3_S: sum = gdn_ab_dot_kq<dequantize_iq3_s, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ1_S: sum = gdn_ab_dot_kq<dequantize_iq1_s, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ1_M: sum = gdn_ab_dot_kq<dequantize_iq1_m, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ4_XS: sum = gdn_ab_dot_kq<dequantize_iq4_xs, 32>(w, xn, K, tile); break;
            case GGML_TYPE_IQ4_NL: sum = gdn_ab_dot_kq<dequantize_iq4_nl, 32, QK4_NL>(w, xn, K, tile); break;
            case GGML_TYPE_MXFP4: sum = gdn_ab_dot_kq<dequantize_mxfp4, 32, QK_MXFP4>(w, xn, K, tile); break;
            case GGML_TYPE_NVFP4: sum = gdn_ab_dot_nvfp4(w, xn, K); break;
            default: NO_DEVICE_CODE;
        }
    }

    __shared__ float s_sum[WARP_SIZE];
    sum = block_reduce<block_reduce_method::SUM, block_size>(sum, s_sum);

    if (threadIdx.x != 0) {
        return;
    }

    const int64_t i = int64_t(n)*n_heads + h;
    if (is_beta) {
        beta[i] = ggml_cuda_op_sigmoid_single(sum);
    } else {
        gate[i] = ggml_cuda_op_softplus_single(sum + dt[h])*ssm_a[h];
    }
}

bool ggml_cuda_gated_delta_net_ab_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_ZNQ2_M:
        case GGML_TYPE_ZNQ2_L:
        case GGML_TYPE_ZNQ3_S:
        case GGML_TYPE_ZNQ3_M:
        case GGML_TYPE_ZNQ3_L:
        case GGML_TYPE_ZNQ4_S:
        case GGML_TYPE_ZNQ4_M:
        case GGML_TYPE_ZNQ4_L:
        case GGML_TYPE_ZNQ5_S:
        case GGML_TYPE_ZNQ5_M:
        case GGML_TYPE_ZNQ5_L:
        case GGML_TYPE_ZNQ6_S:
        case GGML_TYPE_ZNQ6_M:
        case GGML_TYPE_ZNQ6_L:
        case GGML_TYPE_ZNQ7_S:
        case GGML_TYPE_ZNQ7_M:
        case GGML_TYPE_ZNQ7_L:
        case GGML_TYPE_ZNQ2_S:
        case GGML_TYPE_ZNQ2_SM:
        case GGML_TYPE_ZNQ3_SM:
        case GGML_TYPE_ZNQ4_SM:
        case GGML_TYPE_ZNQ5_SM:
        case GGML_TYPE_ZNQ6_SM:
        case GGML_TYPE_ZNQ7_SM:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_op_gated_delta_net_ab(ggml_backend_cuda_context & ctx, const ggml_cuda_gated_delta_net_ab & ab) {
    const int M = (int) ab.alpha_mm->ne[0];
    const int N = (int) ab.alpha_mm->ne[1];
    const int K = (int) ab.alpha_mm->src[0]->ne[0];

    constexpr int block_size = 256;
    auto kernel = k_gdn_ab_mul_mat<block_size>;
    if (ab.alpha_mm->src[0]->type == ab.beta_mm->src[0]->type) {
        switch (ab.alpha_mm->src[0]->type) {
            case GGML_TYPE_F32:  kernel = k_gdn_ab_mul_mat<block_size, GGML_TYPE_F32>;  break;
            case GGML_TYPE_F16:  kernel = k_gdn_ab_mul_mat<block_size, GGML_TYPE_F16>;  break;
            case GGML_TYPE_BF16: kernel = k_gdn_ab_mul_mat<block_size, GGML_TYPE_BF16>; break;
            case GGML_TYPE_Q8_0: kernel = k_gdn_ab_mul_mat<block_size, GGML_TYPE_Q8_0>; break;
            default: break;
        }
    }

    const auto uses_tile = [](ggml_type type) {
        return ggml_blck_size(type) == QK_K || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_MXFP4;
    };
    const size_t shared_mem = uses_tile(ab.alpha_mm->src[0]->type) || uses_tile(ab.beta_mm->src[0]->type) ?
        (block_size/32)*QK_K*sizeof(float) : 0;
    const ggml_cuda_kernel_launch_params params(dim3(M, N, 2), dim3(block_size, 1, 1), shared_mem, ctx.stream());
    ggml_cuda_kernel_launch(kernel, params,
        (const float *) ab.alpha_mm->src[1]->data,
        ab.alpha_mm->src[0]->data, ab.beta_mm->src[0]->data,
        ab.alpha_mm->src[0]->type, ab.beta_mm->src[0]->type,
        ab.alpha_mm->src[0]->nb[1], ab.beta_mm->src[0]->nb[1],
        (const float *) ab.dt->data,
        (const float *) ab.ssm_a->data,
        (float *)       ab.gate_out->data,
        (float *)       ab.beta_out->data,
        M, K);

    CUDA_CHECK(cudaGetLastError());
}
