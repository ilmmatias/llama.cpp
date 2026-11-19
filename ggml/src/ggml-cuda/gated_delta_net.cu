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

struct ggml_cuda_gdn_prep_dev {
    uint8_t       q_mode  = 0; // 0 = off, 1 = rms_norm+scale, 2 = l2_norm
    uint8_t       k_mode  = 0;
    uint8_t       fuse_gate = 0;
    uint8_t       fuse_beta = 0;
    float         q_eps   = 0.0f;
    float         q_scale = 1.0f;
    float         k_eps   = 0.0f;
    float         k_scale = 1.0f;
    const float * dt      = nullptr;
    const float * ssm_a   = nullptr;
    int64_t       sk1 = 0, sk2 = 0, sk3 = 0;
    int64_t       sg_h = 0, sg_t = 0, sg_s = 0;
    int64_t       sb_h = 0, sb_t = 0, sb_s = 0;
};

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_group = 1, int row_lanes = 0, int token_tile = 1, bool FUSE_PREP = false, bool G_IS_EXP = false, bool USE_LDS_QK = false>
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
        int           K,
        ggml_cuda_gdn_prep_dev prep = {}) {
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
    // wait after reading from global, or just before storing LDS below:
    // keep clean single-tile loop without stale reads across iterations!
#pragma unroll
    for (int c = 0; c < cols_per_group; ++c) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[c][r] = curr_state[c * S_v + i];
        }
    }

    __shared__ float lds_k[USE_LDS_QK ? token_tile * S_v : 1];
    __shared__ float lds_q[USE_LDS_QK ? token_tile * S_v : 1];

    // Stage a short time tile before the dependent updates. State stays in
    // registers and the recurrence is still evaluated in token order.
    for (int first = 0; first < n_tokens; first += token_tile) {
        float k_tile[USE_LDS_QK ? 1 : token_tile][rows_per_lane], q_tile[USE_LDS_QK ? 1 : token_tile][rows_per_lane];
        float v_tile[token_tile][cols_per_group], beta_tile[token_tile], g_tile[token_tile];

        if constexpr (USE_LDS_QK) {
            const int j = threadIdx.y;
            const int t = first + j;
            if (t < n_tokens) {
                const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
                const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
#pragma unroll
                for (int i = threadIdx.x; i < S_v; i += physical_warp) {
                    lds_k[j * S_v + i] = k_t[i];
                    lds_q[j * S_v + i] = q_t[i];
                }
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

            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + (FUSE_PREP
                ? (iq3 * prep.sk3 + t * prep.sk2 + iq1 * prep.sk1)
                : (iq3 * sq3 + t * sq2 + iq1 * sq1));
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

            if constexpr (FUSE_PREP) {
                const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
                if (prep.fuse_beta) {
                    const int64_t bo = sequence * prep.sb_s + t * prep.sb_t + h_idx * prep.sb_h;
                    beta_tile[j] = ggml_cuda_op_sigmoid_single(beta[bo]);
                } else {
                    beta_tile[j] = beta[gb_offset];
                }
                if constexpr (KDA) {
                    g_tile[j] = 0.0f;
                } else if (prep.fuse_gate) {
                    const int64_t go = sequence * prep.sg_s + t * prep.sg_t + h_idx * prep.sg_h;
                    float a_val = g[go];
                    if (prep.dt) {
                        a_val += prep.dt[h_idx];
                    }
                    const float sp = ggml_cuda_op_softplus_single(a_val);
                    g_tile[j] = expf(prep.ssm_a ? (sp * prep.ssm_a[h_idx]) : sp);
                } else {
                    g_tile[j] = expf(g[gb_offset]);
                }
            } else {
                const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
                beta_tile[j] = beta[gb_offset];
                if constexpr (KDA) {
                    g_tile[j] = 0.0f;
                } else if constexpr (G_IS_EXP) {
                    g_tile[j] = g[gb_offset];
                } else {
                    g_tile[j] = expf(g[gb_offset]);
                }
            }

            if constexpr (!USE_LDS_QK) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int i = r * warp_size + lane;
                    k_tile[j][r] = k_t[i];
                    q_tile[j][r] = q_t[i];
                }

                if constexpr (FUSE_PREP) {
                    if (prep.k_mode != 0) {
                        float k_ss = 0.0f;
#pragma unroll
                        for (int r = 0; r < rows_per_lane; ++r) {
                            k_ss += k_tile[j][r] * k_tile[j][r];
                        }
                        const float k_red = warp_reduce_sum<warp_size>(k_ss);
                        const float k_inv = (prep.k_mode == 1)
                            ? rsqrtf(k_red * (1.0f / (float) S_v) + prep.k_eps) * prep.k_scale
                            : rsqrtf(fmaxf(k_red, prep.k_eps * prep.k_eps));
#pragma unroll
                        for (int r = 0; r < rows_per_lane; ++r) {
                            k_tile[j][r] *= k_inv;
                        }
                    }
                    if (prep.q_mode != 0) {
                        float q_ss = 0.0f;
#pragma unroll
                        for (int r = 0; r < rows_per_lane; ++r) {
                            q_ss += q_tile[j][r] * q_tile[j][r];
                        }
                        const float q_red = warp_reduce_sum<warp_size>(q_ss);
                        const float q_inv = (prep.q_mode == 1)
                            ? rsqrtf(q_red * (1.0f / (float) S_v) + prep.q_eps) * prep.q_scale
                            : rsqrtf(fmaxf(q_red, prep.q_eps * prep.q_eps));
#pragma unroll
                        for (int r = 0; r < rows_per_lane; ++r) {
                            q_tile[j][r] *= q_inv;
                        }
                    }
                }
            }

#pragma unroll
            for (int c = 0; c < cols_per_group; ++c) {
                v_tile[j][c] = v_t[col_base + c];
            }
        }

        if constexpr (USE_LDS_QK) {
            __syncthreads();
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

            if constexpr (USE_LDS_QK) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; ++r) {
                    const int i = r * warp_size + lane;
                    k_tile[0][r] = lds_k[j * S_v + i];
                    q_tile[0][r] = lds_q[j * S_v + i];
                }
            }

            const float * k_reg = k_tile[USE_LDS_QK ? 0 : j];
            const float * q_reg = q_tile[USE_LDS_QK ? 0 : j];
            const float * v_reg = v_tile[j];

            float attn_reg[cols_per_group];

            if constexpr (!KDA) {
                float kv_part[cols_per_group];
#pragma unroll
                for (int c = 0; c < cols_per_group; ++c) {
                    // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
                    float kv_shard = 0.0f;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += s_shard[c][r] * k_reg[r];
                    }
                    kv_part[c] = kv_shard;
                }

                float delta_col[cols_per_group];
#pragma unroll
                for (int c = 0; c < cols_per_group; ++c) {
                    const float kv_col = warp_reduce_sum<warp_size>(kv_part[c]);
                    // delta[col] = (v[col] - g * kv[col]) * beta
                    delta_col[c] = (v_reg[c] - g_val * kv_col) * beta_val;
                }

                float attn_part[cols_per_group];
#pragma unroll
                for (int c = 0; c < cols_per_group; ++c) {
                    // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
                    // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
                    float attn_partial = 0.0f;
                    const float d_c = delta_col[c];
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        if constexpr (cols_per_group > 1) {
                            // Match the single-column kernel's FMA contraction.
                            s_shard[c][r] = fmaf(g_val, s_shard[c][r], k_reg[r] * d_c);
                        } else {
                            s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * d_c;
                        }
                        attn_partial += s_shard[c][r] * q_reg[r];
                    }
                    attn_part[c] = attn_partial;
                }

#pragma unroll
                for (int c = 0; c < cols_per_group; ++c) {
                    attn_reg[c] = warp_reduce_sum<warp_size>(attn_part[c]) * scale;
                }

                if constexpr (keep_rs_t) {
                    const int target_slot = (int) n_tokens - 1 - t;
                    if (target_slot >= 0 && target_slot < K) {
                        float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                        for (int c = 0; c < cols_per_group; ++c) {
                            const int col = col_base + c;
#pragma unroll
                            for (int r = 0; r < rows_per_lane; r++) {
                                const int i = r * warp_size + lane;
                                curr_state[col * S_v + i] = s_shard[c][r];
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (int c = 0; c < cols_per_group; ++c) {
                    const int col = col_base + c;
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
            }

            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < cols_per_group; ++c) {
                    attn_data[col_base + c] = attn_reg[c];
                }
            }

            attn_data += S_v * H;
        }
        if constexpr (USE_LDS_QK) {
            __syncthreads();
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

static __global__ void gdn_prep_kernel(
        const float * __restrict__ q_in,
        const float * __restrict__ k_in,
        const float * __restrict__ g_in,
        const float * __restrict__ b_in,
        const float * __restrict__ dt,
        const float * __restrict__ ssm_a,
        float * __restrict__ q_out,
        float * __restrict__ k_out,
        float * __restrict__ g_out,
        float * __restrict__ b_out,
        int64_t S_v,
        int64_t H,
        int64_t neqk1,
        int64_t n_tokens,
        int64_t neq3,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sk1, int64_t sk2, int64_t sk3,
        int64_t sg_h, int64_t sg_t, int64_t sg_s,
        int64_t sb_h, int64_t sb_t, int64_t sb_s,
        uint8_t q_mode, uint8_t k_mode,
        uint8_t fuse_gate, uint8_t fuse_beta,
        float q_eps, float q_scale,
        float k_eps, float k_scale) {
    constexpr int warp_sz = 32;
    const int w_id = (int) blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int64_t t   = blockIdx.y;
    const int64_t seq = blockIdx.z;

    const bool do_qk = (q_out != nullptr);
    const int qk_warps = do_qk ? (int) neqk1 : 0;
    const int gb_warp_base = 2 * qk_warps;

    if (w_id < gb_warp_base) {
        if (seq >= neq3) {
            return;
        }
        const bool is_q = (w_id < qk_warps);
        const int64_t h = is_q ? w_id : (w_id - qk_warps);
        const uint8_t mode = is_q ? q_mode : k_mode;
        const float * src = is_q
            ? (q_in + seq * sq3 + t * sq2 + h * sq1)
            : (k_in + seq * sk3 + t * sk2 + h * sk1);
        float * dst = (is_q ? q_out : k_out) + ((seq * n_tokens + t) * neqk1 + h) * S_v;

        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float ss = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = r * warp_sz + lane;
            if (i < S_v) {
                const float v = src[i];
                vals[r] = v;
                ss += v * v;
            }
        }

        float inv = 1.0f;
        if (mode != 0) {
            const float red = warp_reduce_sum<warp_sz>(ss);
            const float eps = is_q ? q_eps : k_eps;
            const float scl = is_q ? q_scale : k_scale;
            inv = (mode == 1)
                ? rsqrtf(red * (1.0f / (float) S_v) + eps) * scl
                : rsqrtf(fmaxf(red, eps * eps));
        }

#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = r * warp_sz + lane;
            if (i < S_v) {
                dst[i] = vals[r] * inv;
            }
        }
        return;
    }

    const int64_t h = (int64_t) (w_id - gb_warp_base) * warp_sz + lane;
    if (h >= H) {
        return;
    }

    const int64_t out_off = (seq * n_tokens + t) * H + h;
    const int64_t go = seq * sg_s + t * sg_t + h * sg_h;
    float g_raw = g_in[go];
    if (fuse_gate) {
        if (dt) {
            g_raw += dt[h];
        }
        const float sp = ggml_cuda_op_softplus_single(g_raw);
        g_raw = ssm_a ? (sp * ssm_a[h]) : sp;
    }
    g_out[out_off] = expf(g_raw);

    if (fuse_beta) {
        const int64_t bo = seq * sb_s + t * sb_t + h * sb_h;
        b_out[out_off] = ggml_cuda_op_sigmoid_single(b_in[bo]);
    }
}

template <bool KDA, bool keep_rs_t, bool FUSE_PREP = false, bool G_IS_EXP = false>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        ggml_cuda_gdn_prep_dev prep, cudaStream_t stream) {
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
            constexpr int rows   = 16;
            constexpr int cols   = 4;
            constexpr int tokens = FUSE_PREP ? 2 : 4;
            constexpr bool lds_qk = false;

            const int tile_cols = num_warps * (warp_size / rows) * cols;
            const ggml_cuda_kernel_launch_params tiled(dim3(H, n_seqs, S_v / tile_cols), block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, false, cols, rows, tokens, FUSE_PREP, G_IS_EXP, lds_qk>, tiled,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prep);
            return;
        }
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, 1, 0, 1, FUSE_PREP, G_IS_EXP>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prep);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, 1, 0, 1, FUSE_PREP, G_IS_EXP>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prep);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, 1, 0, 1, FUSE_PREP, G_IS_EXP>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prep);
            break;
        }
        case 128: {
            if constexpr (gdn_cols_per_warp == 1 && !KDA && !keep_rs_t) {
                if (cols_per_warp == 4) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, false, 4, 0, 1, FUSE_PREP, G_IS_EXP>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prep);
                    break;
                }
                if (cols_per_warp == 8) {
                    ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, false, 8, 0, 1, FUSE_PREP, G_IS_EXP>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prep);
                    break;
                }
            }
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, 1, 0, 1, FUSE_PREP, G_IS_EXP>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, prep);
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
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_cuda_gated_delta_net_fused_prep * prep) {
    ggml_tensor * src_q     = (prep && prep->q_norm.mode) ? const_cast<ggml_tensor *>(prep->q_norm.raw) : dst->src[0];
    ggml_tensor * src_k     = (prep && prep->k_norm.mode) ? const_cast<ggml_tensor *>(prep->k_norm.raw) : dst->src[1];
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
    const bool has_prep = prep != nullptr && !prep->empty();
    if (!has_prep) {
        GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    } else {
        GGML_ASSERT(neq0 == S_v && nek0 == S_v && src_q->nb[0] == sizeof(float) && src_k->nb[0] == sizeof(float));
    }
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_state));

    ggml_cuda_gdn_prep_dev prep_dev{};
    if (has_prep) {
        prep_dev.q_mode  = prep->q_norm.mode;
        prep_dev.k_mode  = prep->k_norm.mode;
        prep_dev.q_eps   = prep->q_norm.eps;
        prep_dev.q_scale = prep->q_norm.scale;
        prep_dev.k_eps   = prep->k_norm.eps;
        prep_dev.k_scale = prep->k_norm.scale;
        prep_dev.sk1     = (int64_t) (nbk1 / sizeof(float));
        prep_dev.sk2     = (int64_t) (nbk2 / sizeof(float));
        prep_dev.sk3     = (int64_t) (nbk3 / sizeof(float));

        if (prep->fuse_gate) {
            GGML_ASSERT(!kda && prep->alpha_raw != nullptr && prep->alpha_raw->type == GGML_TYPE_F32);
            GGML_ASSERT(prep->alpha_raw->nb[0] == sizeof(float));
            GGML_ASSERT(prep->alpha_raw->ne[0] == H && prep->alpha_raw->ne[1] == n_tokens && prep->alpha_raw->ne[2] == n_seqs);
            prep_dev.fuse_gate = 1;
            g_d                = (const float *) prep->alpha_raw->data;
            prep_dev.dt        = prep->dt ? (const float *) prep->dt->data : nullptr;
            prep_dev.ssm_a     = prep->ssm_a ? (const float *) prep->ssm_a->data : nullptr;
            prep_dev.sg_h      = 1;
            prep_dev.sg_t      = (int64_t) (prep->alpha_raw->nb[1] / sizeof(float));
            prep_dev.sg_s      = (int64_t) (prep->alpha_raw->nb[2] / sizeof(float));
        } else {
            GGML_ASSERT(ggml_is_contiguous(src_g));
        }

        if (prep->fuse_beta) {
            GGML_ASSERT(prep->beta_raw != nullptr && prep->beta_raw->type == GGML_TYPE_F32);
            GGML_ASSERT(prep->beta_raw->nb[0] == sizeof(float));
            prep_dev.fuse_beta = 1;
            b_d                = (const float *) prep->beta_raw->data;
            if (prep->beta_raw->ne[0] == 1 && prep->beta_raw->ne[1] == H) {
                prep_dev.sb_h = (int64_t) (prep->beta_raw->nb[1] / sizeof(float));
                prep_dev.sb_t = (int64_t) (prep->beta_raw->nb[2] / sizeof(float));
                prep_dev.sb_s = (int64_t) (prep->beta_raw->nb[3] / sizeof(float));
            } else {
                GGML_ASSERT(prep->beta_raw->ne[0] == H && prep->beta_raw->ne[1] == n_tokens && prep->beta_raw->ne[2] == n_seqs);
                prep_dev.sb_h = 1;
                prep_dev.sb_t = (int64_t) (prep->beta_raw->nb[1] / sizeof(float));
                prep_dev.sb_s = (int64_t) (prep->beta_raw->nb[2] / sizeof(float));
            }
        } else {
            GGML_ASSERT(ggml_is_contiguous(src_beta));
        }
    } else {
        GGML_ASSERT(ggml_is_contiguous(src_g));
        GGML_ASSERT(ggml_is_contiguous(src_beta));
    }

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

    if (!keep_rs && !has_prep) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
    }

    const bool use_parallel_prep = !kda && n_tokens <= 65535 &&
        ((has_prep && n_tokens >= 8) || (!has_prep && n_tokens >= 32));

    ggml_cuda_pool_alloc<float> prep_buf(ctx.pool());
    bool g_is_exp = false;
    bool exec_fuse_prep = has_prep;

    int64_t run_sq1 = sq1, run_sq2 = sq2, run_sq3 = sq3;
    int64_t run_sb1 = sb1, run_sb2 = sb2, run_sb3 = sb3;

    if (use_parallel_prep) {
        const bool need_qk = has_prep && (prep_dev.q_mode != 0 || prep_dev.k_mode != 0 ||
            sq1 != prep_dev.sk1 || sq2 != prep_dev.sk2 || sq3 != prep_dev.sk3);
        const bool need_beta = has_prep && prep_dev.fuse_beta != 0;
        const size_t qk_elems = need_qk ? (size_t) S_v * neqk1 * n_tokens * neq3 : 0;
        const size_t gb_elems = (size_t) H * n_tokens * n_seqs;
        const size_t total_elems = 2 * qk_elems + gb_elems + (need_beta ? gb_elems : 0);

        if (total_elems <= (64ull << 20)) {
            float * buf = prep_buf.alloc(total_elems);
            float * q_out = need_qk ? buf : nullptr;
            float * k_out = need_qk ? (buf + qk_elems) : nullptr;
            float * g_out = buf + 2 * qk_elems;
            float * b_out = need_beta ? (g_out + gb_elems) : nullptr;

            const int64_t g_in_s1 = prep_dev.fuse_gate ? prep_dev.sg_h : sb1;
            const int64_t g_in_s2 = prep_dev.fuse_gate ? prep_dev.sg_t : sb2;
            const int64_t g_in_s3 = prep_dev.fuse_gate ? prep_dev.sg_s : sb3;

            const int total_warps = (need_qk ? 2 * (int) neqk1 : 0) + (int) ((H + 31) / 32);
            const dim3 prep_block(32, 4, 1);
            const dim3 prep_grid((total_warps + 3) / 4, (unsigned) n_tokens, (unsigned) n_seqs);
            gdn_prep_kernel<<<prep_grid, prep_block, 0, stream>>>(
                q_d, k_d, g_d, b_d, prep_dev.dt, prep_dev.ssm_a,
                q_out, k_out, g_out, b_out,
                S_v, H, neqk1, n_tokens, neq3,
                sq1, sq2, sq3,
                prep_dev.sk1, prep_dev.sk2, prep_dev.sk3,
                g_in_s1, g_in_s2, g_in_s3,
                prep_dev.sb_h, prep_dev.sb_t, prep_dev.sb_s,
                prep_dev.q_mode, prep_dev.k_mode,
                prep_dev.fuse_gate, prep_dev.fuse_beta,
                prep_dev.q_eps, prep_dev.q_scale,
                prep_dev.k_eps, prep_dev.k_scale);

            if (need_qk) {
                q_d = q_out;
                k_d = k_out;
                run_sq1 = S_v;
                run_sq2 = S_v * neqk1;
                run_sq3 = S_v * neqk1 * n_tokens;
            }
            g_d = g_out;
            if (need_beta) {
                b_d = b_out;
            }
            run_sb1 = 1;
            run_sb2 = H;
            run_sb3 = H * n_tokens;
            g_is_exp = true;
            exec_fuse_prep = false;
        }
    }

    auto dispatch_mode = [&](auto kda_tag, auto krs_tag) {
        if (exec_fuse_prep) {
            launch_gated_delta_net<decltype(kda_tag)::value, decltype(krs_tag)::value, true, false>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, run_sq1, run_sq2, run_sq3, sv1, sv2, sv3,
                run_sb1, run_sb2, run_sb3, neqk1, rq3, scale, state_slot_stride, K, prep_dev, stream);
        } else if constexpr (!decltype(kda_tag)::value) {
            if (g_is_exp) {
                launch_gated_delta_net<false, decltype(krs_tag)::value, false, true>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, run_sq1, run_sq2, run_sq3, sv1, sv2, sv3,
                    run_sb1, run_sb2, run_sb3, neqk1, rq3, scale, state_slot_stride, K, prep_dev, stream);
            } else {
                launch_gated_delta_net<false, decltype(krs_tag)::value, false, false>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, run_sq1, run_sq2, run_sq3, sv1, sv2, sv3,
                    run_sb1, run_sb2, run_sb3, neqk1, rq3, scale, state_slot_stride, K, prep_dev, stream);
            }
        } else {
            launch_gated_delta_net<decltype(kda_tag)::value, decltype(krs_tag)::value, false, false>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, run_sq1, run_sq2, run_sq3, sv1, sv2, sv3,
                run_sb1, run_sb2, run_sb3, neqk1, rq3, scale, state_slot_stride, K, prep_dev, stream);
        }
    };

    if (kda) {
        if (keep_rs) {
            dispatch_mode(std::true_type{}, std::true_type{});
        } else {
            dispatch_mode(std::true_type{}, std::false_type{});
        }
    } else {
        if (keep_rs) {
            dispatch_mode(std::false_type{}, std::true_type{});
        } else {
            dispatch_mode(std::false_type{}, std::false_type{});
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_cuda_gated_delta_net_fused_prep * prep) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, cache, prep);
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
