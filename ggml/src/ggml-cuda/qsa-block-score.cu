#include "common.cuh"
#include "qsa-block-score.cuh"

#include <cstdint>
#include <type_traits>

static __global__ void qsa_mask(
        const int32_t * positions, const char * indices, float * dst,
        int32_t n_kv, int64_t width, size_t nb0, size_t nb1) {
    const int64_t column = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) {
        return;
    }

    const int32_t query = blockIdx.y;
    const int32_t cell = *(const int32_t *) (indices + column * nb0 + query * nb1);

    dst[query * width + column] = ggml_qsa_is_visible(positions, n_kv, cell, query) ? 0.0f : -INFINITY;
}

void ggml_cuda_op_qsa_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * indices = dst->src[1];
    const int64_t width = dst->ne[0];

    const ggml_cuda_kernel_launch_params launch_params(
            dim3((width + 255) / 256, dst->ne[1], 1), dim3(256, 1, 1), 0, ctx.stream());

    ggml_cuda_kernel_launch(qsa_mask, launch_params,
            (const int32_t *) dst->src[0]->data, (const char *) indices->data, (float *) dst->data,
            ggml_get_op_params_i32(dst, 0), width, indices->nb[0], indices->nb[1]);
}

// Keep QSA's ordered reductions in registers on RDNA instead of using
// LDS-backed shuffle instructions.
template <int mask>
static __device__ __forceinline__ float qsa_shuffle_xor(float x) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    if constexpr (mask < 16) {
        return __int_as_float(__builtin_amdgcn_update_dpp(
                0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
    } else {
        float swapped = __int_as_float(__builtin_amdgcn_permlanex16(
                __float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98, true, false));
        if constexpr (mask == 24) {
            return qsa_shuffle_xor<8>(swapped);
        }

        return swapped;
    }
#else
    return __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, x, mask, 32);
#endif
}

template <typename K_T>
static __device__ __forceinline__ float qsa_k_to_f32(K_T v) {
    if constexpr (std::is_same_v<K_T, half>) {
        return __half2float(v);
    } else {
        return v;
    }
}

#if defined(GGML_USE_HIP)
typedef _Float16 qsa_half2v __attribute__((ext_vector_type(2)));
#endif

static __device__ __forceinline__ void qsa_fdot2_u1(float & acc, uint32_t q_u, uint32_t k_u) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__) || \
    defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
    acc = __builtin_amdgcn_fdot2(
        *reinterpret_cast<const qsa_half2v *>(&q_u),
        *reinterpret_cast<const qsa_half2v *>(&k_u), acc, false);
#else
    ggml_cuda_mad(acc, *reinterpret_cast<const half2 *>(&q_u), *reinterpret_cast<const half2 *>(&k_u));
#endif
}

static __device__ __forceinline__ void qsa_fdot2_u4(float & acc, const uint4 qv, const uint4 kv) {
    qsa_fdot2_u1(acc, qv.x, kv.x);
    qsa_fdot2_u1(acc, qv.y, kv.y);
    qsa_fdot2_u1(acc, qv.z, kv.z);
    qsa_fdot2_u1(acc, qv.w, kv.w);
}

static __device__ __forceinline__ uint4 qsa_pack_f32_to_u4h2(const float4 a, const float4 b) {
    uint4 r;
    const half2 h0 = __floats2half2_rn(a.x, a.y);
    const half2 h1 = __floats2half2_rn(a.z, a.w);
    const half2 h2 = __floats2half2_rn(b.x, b.y);
    const half2 h3 = __floats2half2_rn(b.z, b.w);
    r.x = *reinterpret_cast<const uint32_t *>(&h0);
    r.y = *reinterpret_cast<const uint32_t *>(&h1);
    r.z = *reinterpret_cast<const uint32_t *>(&h2);
    r.w = *reinterpret_cast<const uint32_t *>(&h3);
    return r;
}

static __device__ __forceinline__ float qsa_eval_mask_val(
        const float * __restrict__ mask,
        const int32_t * __restrict__ bounds,
        int64_t ib,
        int64_t iq,
        int64_t is,
        int64_t sm0,
        int64_t sm1,
        int64_t sm2,
        int64_t sbo1,
        int64_t sbo2,
        int64_t nm_stream) {
    if (bounds != nullptr) {
        const int32_t * bnd = bounds + iq * sbo1 + is * sbo2;
        if (ib < bnd[0]) {
            return mask[ib * sm0 + (is % nm_stream) * sm2];
        }
        return ib < bnd[1] ? 1e9f : -INFINITY;
    }
    return mask[ib * sm0 + iq * sm1 + is * sm2];
}

// Zero-LDS Wave32 decode kernel for 128x4 indexer (n_query < 8).
// Each lane holds 4 query dims across all 4 heads in 8 VGPRs, reads F16 keys as
// one coalesced uint2 (8B) per key, runs v_dot2_f32_f16, and finishes with an 8-DPP reduce.
template <bool K_IS_F16, int KEYS_PER_WAVE, bool HAS_BOUNDS>
static __global__ void qsa_block_score_rdna2_decode_fdot2(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const int32_t * __restrict__ cells,
        const float * __restrict__ mask,
        const int32_t * __restrict__ bounds,
        float * __restrict__ dst,
        int64_t n_blocks,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1,
        int64_t sc0, int64_t sc1,
        int64_t sm0, int64_t sm1, int64_t sm2,
        int64_t sbo1, int64_t sbo2, int64_t nm_stream,
        int64_t sd0, int64_t sd1, int64_t sd2,
        float scale) {
    const int tid  = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;

    const int64_t ib_base = ((int64_t) blockIdx.x * 8 + wave) * KEYS_PER_WAVE;
    const int64_t iq = blockIdx.y;
    const int64_t is = blockIdx.z;

    if (ib_base >= n_blocks) {
        return;
    }

    int32_t b_lo = 0;
    int32_t b_hi = 0;
    int64_t sm_stream_off = 0;
    if constexpr (HAS_BOUNDS) {
        const int32_t * bnd = bounds + iq * sbo1 + is * sbo2;
        b_lo = bnd[0];
        b_hi = bnd[1];
        sm_stream_off = (is % nm_stream) * sm2;
    }

    uint32_t q0_u[4], q1_u[4];
    const char * q_base = q + (size_t) iq * nbq2 + (size_t) is * nbq3 + (size_t) lane * 16;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float4 qf = *reinterpret_cast<const float4 *>(q_base + (size_t) h * nbq1);
        const half2 h0 = __floats2half2_rn(qf.x, qf.y);
        const half2 h1 = __floats2half2_rn(qf.z, qf.w);
        q0_u[h] = *reinterpret_cast<const uint32_t *>(&h0);
        q1_u[h] = *reinterpret_cast<const uint32_t *>(&h1);
    }

    for (int j = 0; j < KEYS_PER_WAVE; ++j) {
        const int64_t ib = ib_base + j;
        if (ib >= n_blocks) {
            break;
        }
        float mv;
        if constexpr (HAS_BOUNDS) {
            if (ib >= b_hi) {
                if (lane == 0) {
                    dst[ib * sd0 + iq * sd1 + is * sd2] = -INFINITY;
                }
                continue;
            }
            mv = (ib < b_lo) ? mask[ib * sm0 + sm_stream_off] : 1e9f;
            if (mv == -INFINITY) {
                if (lane == 0) {
                    dst[ib * sd0 + iq * sd1 + is * sd2] = -INFINITY;
                }
                continue;
            }
        } else {
            mv = mask[ib * sm0 + iq * sm1 + is * sm2];
        }
        const int32_t cell = cells[ib * sc0 + is * sc1];
        const char * k_ptr = k + (size_t) cell * nbk1;
        uint32_t k0_u, k1_u;
        if constexpr (K_IS_F16) {
            const uint2 ku = *reinterpret_cast<const uint2 *>(k_ptr + (size_t) lane * 8);
            k0_u = ku.x;
            k1_u = ku.y;
        } else {
            const float4 kf = *reinterpret_cast<const float4 *>(k_ptr + (size_t) lane * 16);
            const half2 h0 = __floats2half2_rn(kf.x, kf.y);
            const half2 h1 = __floats2half2_rn(kf.z, kf.w);
            k0_u = *reinterpret_cast<const uint32_t *>(&h0);
            k1_u = *reinterpret_cast<const uint32_t *>(&h1);
        }

        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            qsa_fdot2_u1(d[h], q0_u[h], k0_u);
            qsa_fdot2_u1(d[h], q1_u[h], k1_u);
        }

        const float s0 = (lane & 1) ? d[0] : d[1];
        const float p0 = ((lane & 1) ? d[1] : d[0]) + qsa_shuffle_xor<1>(s0);
        const float s1 = (lane & 1) ? d[2] : d[3];
        const float p1 = ((lane & 1) ? d[3] : d[2]) + qsa_shuffle_xor<1>(s1);
        const float s2 = (lane & 2) ? p0 : p1;
        float v = ((lane & 2) ? p1 : p0) + qsa_shuffle_xor<2>(s2);
        v += qsa_shuffle_xor<4>(v);
        v += qsa_shuffle_xor<8>(v);
        v += qsa_shuffle_xor<16>(v);

        float r = (lane < 4) ? fmaxf(v, 0.0f) : 0.0f;
        r += qsa_shuffle_xor<1>(r);
        r += qsa_shuffle_xor<2>(r);
        if (lane == 0) {
            dst[ib * sd0 + iq * sd1 + is * sd2] = r * scale + mv;
        }
    }
}

// High-throughput RDNA2 v_dot2_f32_f16 2D tiled kernel for batched QSA block scoring (128x4).
// Reads K[cells[gk]] directly as F16/F32 and fuses (*scale + mask) into output with zero extra ops.
template <int BM, bool K_IS_F16, bool CHECK_INF, bool HAS_BOUNDS>
static __global__ void __launch_bounds__((BM == 64 ? 128 : 256), 2) qsa_block_score_rdna2_tiled_fdot2(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const int32_t * __restrict__ cells,
        const float * __restrict__ mask,
        const int32_t * __restrict__ bounds,
        float * __restrict__ dst,
        int64_t n_blocks,
        int64_t n_query,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1,
        int64_t sc0, int64_t sc1,
        int64_t sm0, int64_t sm1, int64_t sm2,
        int64_t sbo1, int64_t sbo2, int64_t nm_stream,
        int64_t sd0, int64_t sd1, int64_t sd2,
        float scale) {
    static_assert(BM == 32 || BM == 64, "BM must be 32 or 64");
    constexpr int BN           = 64;
    constexpr int NTHREADS     = (BM == 64) ? 128 : 256;
    constexpr int CK           = (BM == 64) ? 8 : 2;
    constexpr int WARP_K_COUNT = (BM == 64) ? 2 : 4;
    constexpr int WARP_K_WIDTH = BN / WARP_K_COUNT;
    constexpr int WARP_K_SHIFT = (BM == 64) ? 1 : 2;
    constexpr int WARP_K_MASK  = WARP_K_COUNT - 1;
    constexpr int STRIDE_Q     = (BM == 64) ? 8 : 4;
    constexpr int STRIDE_K     = (BM == 64) ? 4 : 8;
    constexpr int LOAD_ROWS    = NTHREADS / 8;

    const int64_t k0 = (int64_t) blockIdx.x * BN;
    const int64_t q0 = (int64_t) blockIdx.y * BM;
    const int64_t is = blockIdx.z;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_q = warp >> WARP_K_SHIFT;
    const int warp_k = warp & WARP_K_MASK;
    const int lane_q = (BM == 64) ? (lane & 7) : (lane & 3);
    const int lane_k = (BM == 64) ? (lane >> 3) : (lane >> 2);

    __shared__ union {
        struct {
            uint4 K_s[16][64];
            uint4 Q_s[16][BM];
        } smem;
        int any_active_s;
    } sh;

    if constexpr (CHECK_INF) {
        if (tid == 0) {
            sh.any_active_s = 0;
        }
        __syncthreads();

        bool any_active = false;
        if constexpr (HAS_BOUNDS) {
            int32_t b_lo[4] = {0, 0, 0, 0};
            int32_t b_hi[4] = {0, 0, 0, 0};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int64_t gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
                if (gq < n_query) {
                    const int32_t * bnd = bounds + gq * sbo1 + is * sbo2;
                    b_lo[r] = bnd[0];
                    b_hi[r] = bnd[1];
                }
            }
            const int64_t sm_stream_off = (is % nm_stream) * sm2;
#pragma unroll
            for (int c = 0; c < CK; ++c) {
                const int64_t gk = k0 + warp_k * WARP_K_WIDTH + lane_k + c * STRIDE_K;
                if (gk < n_blocks) {
                    float bb = -INFINITY;
                    bool loaded_bb = false;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        const int64_t gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
                        if (gq < n_query && gk < b_hi[r]) {
                            if (gk >= b_lo[r]) {
                                any_active = true;
                                break;
                            }
                            if (!loaded_bb) {
                                bb = mask[gk * sm0 + sm_stream_off];
                                loaded_bb = true;
                            }
                            if (bb != -INFINITY) {
                                any_active = true;
                                break;
                            }
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int64_t gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
                if (gq < n_query) {
#pragma unroll
                    for (int c = 0; c < CK; ++c) {
                        const int64_t gk = k0 + warp_k * WARP_K_WIDTH + lane_k + c * STRIDE_K;
                        if (gk < n_blocks) {
                            const float mv = mask[gk * sm0 + gq * sm1 + is * sm2];
                            if (mv != -INFINITY) {
                                any_active = true;
                            }
                        }
                    }
                }
            }
        }

        if (__any(any_active) && lane == 0) {
            atomicOr(&sh.any_active_s, 1);
        }
        __syncthreads();

        if (sh.any_active_s == 0) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int64_t gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
                if (gq < n_query) {
#pragma unroll
                    for (int c = 0; c < CK; ++c) {
                        const int64_t gk = k0 + warp_k * WARP_K_WIDTH + lane_k + c * STRIDE_K;
                        if (gk < n_blocks) {
                            dst[gk * sd0 + gq * sd1 + is * sd2] = -INFINITY;
                        }
                    }
                }
            }
            return;
        }
    }

    // Load BN=64 indexed keys (128 dims each) into K_s once for all queries and heads.
    {
        const int lk_base = tid >> 3;
        const int t8      = tid & 7;
#pragma unroll
        for (int iter = 0; iter < (BN / LOAD_ROWS); ++iter) {
            const int lk = iter * LOAD_ROWS + lk_base;
            const int64_t gk = k0 + lk;
            uint4 u0 = make_uint4(0, 0, 0, 0);
            uint4 u1 = make_uint4(0, 0, 0, 0);
            if (gk < n_blocks) {
                const int32_t cell = cells[gk * sc0 + is * sc1];
                const char * kr = k + (size_t) cell * nbk1;
                if constexpr (K_IS_F16) {
                    const uint4 * ku = reinterpret_cast<const uint4 *>(kr) + t8 * 2;
                    u0 = ku[0];
                    u1 = ku[1];
                } else {
                    const float4 * kf = reinterpret_cast<const float4 *>(kr) + t8 * 4;
                    u0 = qsa_pack_f32_to_u4h2(kf[0], kf[1]);
                    u1 = qsa_pack_f32_to_u4h2(kf[2], kf[3]);
                }
            }
            sh.smem.K_s[t8 * 2 + 0][lk ^ t8] = u0;
            sh.smem.K_s[t8 * 2 + 1][lk ^ t8] = u1;
        }
    }
    __syncthreads();

    float acc[4][CK];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < CK; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    const char * q_stream = q + (size_t) is * nbq3;
    const int lq_base = tid >> 3;
    const int t8      = tid & 7;

    for (int h = 0; h < 4; ++h) {
        if (h > 0) {
            __syncthreads();
        }

#pragma unroll
        for (int iter = 0; iter < (BM / LOAD_ROWS); ++iter) {
            const int lq = iter * LOAD_ROWS + lq_base;
            const int64_t gq = q0 + lq;
            if (gq < n_query) {
                const float4 * qf = reinterpret_cast<const float4 *>(
                    q_stream + (size_t) h * nbq1 + (size_t) gq * nbq2) + t8 * 4;
                sh.smem.Q_s[t8 * 2 + 0][lq ^ t8] = qsa_pack_f32_to_u4h2(qf[0], qf[1]);
                sh.smem.Q_s[t8 * 2 + 1][lq ^ t8] = qsa_pack_f32_to_u4h2(qf[2], qf[3]);
            } else {
                sh.smem.Q_s[t8 * 2 + 0][lq ^ t8] = make_uint4(0, 0, 0, 0);
                sh.smem.Q_s[t8 * 2 + 1][lq ^ t8] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        float dot[4][CK];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < CK; ++c) {
                dot[r][c] = 0.0f;
            }
        }

#pragma unroll 2
        for (int d4 = 0; d4 < 16; ++d4) {
            const int swiz = d4 >> 1;
            uint4 qv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qv[r] = sh.smem.Q_s[d4][(warp_q * (BM / 2) + lane_q + r * STRIDE_Q) ^ swiz];
            }
            uint4 kv_r[CK];
#pragma unroll
            for (int c = 0; c < CK; ++c) {
                kv_r[c] = sh.smem.K_s[d4][(warp_k * WARP_K_WIDTH + lane_k + c * STRIDE_K) ^ swiz];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int c = 0; c < CK; ++c) {
                    qsa_fdot2_u4(dot[r][c], qv[r], kv_r[c]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < CK; ++c) {
                if (dot[r][c] > 0.0f) {
                    acc[r][c] += dot[r][c];
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int64_t gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
        if (gq < n_query) {
            if constexpr (HAS_BOUNDS) {
                const int32_t * bnd = bounds + gq * sbo1 + is * sbo2;
                const int32_t b_lo = bnd[0];
                const int32_t b_hi = bnd[1];
                const int64_t sm_stream_off = (is % nm_stream) * sm2;
#pragma unroll
                for (int c = 0; c < CK; ++c) {
                    const int64_t gk = k0 + warp_k * WARP_K_WIDTH + lane_k + c * STRIDE_K;
                    if (gk < n_blocks) {
                        const float mv = (gk < b_lo) ? mask[gk * sm0 + sm_stream_off] : (gk < b_hi ? 1e9f : -INFINITY);
                        dst[gk * sd0 + gq * sd1 + is * sd2] = acc[r][c] * scale + mv;
                    }
                }
            } else {
#pragma unroll
                for (int c = 0; c < CK; ++c) {
                    const int64_t gk = k0 + warp_k * WARP_K_WIDTH + lane_k + c * STRIDE_K;
                    if (gk < n_blocks) {
                        dst[gk * sd0 + gq * sd1 + is * sd2] =
                            acc[r][c] * scale + mask[gk * sm0 + gq * sm1 + is * sm2];
                    }
                }
            }
        }
    }
}

// Released Qwen4Exp indexer shape.  A Wave32 is split into four independent
// 8-lane groups.  Each group owns one indexer head and reduces a 128-element
// dot product.  The cached block key is staged once per wave and reused by all
// four heads, avoiding four identical global-memory reads.
template <typename K_T>
static __global__ void qsa_block_score_f32_128x4_wave32(
        const float * q,
        const K_T * k,
        const int32_t * cells,
        const float * mask,
        const int32_t * bounds,
        float * dst,
        int64_t n_blocks,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sk1,
        int64_t sc0,
        int64_t sc1,
        int64_t sm0,
        int64_t sm1,
        int64_t sm2,
        int64_t sbo1,
        int64_t sbo2,
        int64_t nm_stream,
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
    const int64_t ib = (int64_t) blockIdx.x * waves_per_block + wave;
    const int64_t iq = blockIdx.y;
    const int64_t is = blockIdx.z;
    const bool valid = ib < n_blocks;

    int32_t cell = 0;
    if (valid) {
        cell = cells[ib*sc0 + is*sc1];
    }

    // Every wave, including padding waves in the final block, participates in
    // the block barrier.  Invalid waves only stage zeroes and never store.
#pragma unroll
    for (int i = lane; i < n_embd; i += wave_size) {
        key_s[wave][i] = valid ? qsa_k_to_f32(k[(int64_t) cell*sk1 + i]) : 0.0f;
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
    dot += qsa_shuffle_xor<4>(dot);
    dot += qsa_shuffle_xor<2>(dot);
    dot += qsa_shuffle_xor<1>(dot);

    // ReLU is per head, before the head sum.  Keep only the four subgroup
    // leaders.  Lane 0 fetches lanes 8/16/24 and adds in head order, matching
    // the generic graph's left-to-right four-head reduction as closely as the
    // different dot-product reduction permits.
    const float head_score = il == 0 ? fmaxf(dot, 0.0f) : 0.0f;
    const float h1 = qsa_shuffle_xor<8>(head_score);
    const float h2 = qsa_shuffle_xor<16>(head_score);
    const float h3 = qsa_shuffle_xor<24>(head_score);

    if (valid && lane == 0) {
        const float score = ((head_score + h1) + h2) + h3;
        dst[ib*sd0 + iq*sd1 + is*sd2] =
            score*scale + qsa_eval_mask_val(mask, bounds, ib, iq, is, sm0, sm1, sm2, sbo1, sbo2, nm_stream);
    }
}

// Batched prefill path for the released 128x4 indexer shape. Each 256-thread
// workgroup covers an 8-block x 8-query tile. One wave stages one block key,
// then the eight waves reuse all staged keys while each wave owns one query.
// This keeps the head dimension fused without giving up the block/query reuse
// that makes the large-n_query path bandwidth-efficient.
template <typename K_T>
static __global__ void qsa_block_score_f32_128x4_wave32_batched(
        const float * q,
        const K_T * k,
        const int32_t * cells,
        const float * mask,
        const int32_t * bounds,
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
        int64_t sbo1,
        int64_t sbo2,
        int64_t nm_stream,
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
        key_s[wave][i] = valid_key ? qsa_k_to_f32(k[(int64_t) cell*sk1 + i]) : 0.0f;
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

    float scores[blocks_per_tile];
#pragma unroll
    for (int jb = 0; jb < blocks_per_tile; ++jb) {
        float v = dot[jb];
        v += qsa_shuffle_xor<4>(v);
        v += qsa_shuffle_xor<2>(v);
        v += qsa_shuffle_xor<1>(v);

        const float head_score = il == 0 ? fmaxf(v, 0.0f) : 0.0f;
        const float h1 = qsa_shuffle_xor<8>(head_score);
        const float h2 = qsa_shuffle_xor<16>(head_score);
        const float h3 = qsa_shuffle_xor<24>(head_score);

        scores[jb] = ((head_score + h1) + h2) + h3;
    }

    if (valid_query && lane == 0) {
#pragma unroll
        for (int jb = 0; jb < blocks_per_tile; ++jb) {
            const int64_t ib = ib0 + jb;
            if (ib < n_blocks) {
                dst[ib*sd0 + iq*sd1 + is*sd2] =
                    scores[jb]*scale + qsa_eval_mask_val(mask, bounds, ib, iq, is, sm0, sm1, sm2, sbo1, sbo2, nm_stream);
            }
        }
    }
}


// Portable fallback for unusual indexer shapes / non-Wave32 devices.  This is
// a correctness path; the performance target for this branch is the 128x4
// Wave32 kernels above.
template <typename K_T>
static __global__ void qsa_block_score_f32_generic(
        const float * q,
        const K_T * k,
        const int32_t * cells,
        const float * mask,
        const int32_t * bounds,
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
        int64_t sbo1,
        int64_t sbo2,
        int64_t nm_stream,
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
    const K_T * k_row = k + (int64_t) cell*sk1;

    float score = 0.0f;
    for (int64_t ih = 0; ih < n_head; ++ih) {
        const float * q_row = q + ih*sq1 + iq*sq2 + is*sq3;
        float dot = 0.0f;
        for (int64_t i = 0; i < n_embd; ++i) {
            dot = fmaf(q_row[i], qsa_k_to_f32(k_row[i]), dot);
        }
        score += fmaxf(dot, 0.0f);
    }

    dst[ib*sd0 + iq*sd1 + is*sd2] =
        score*scale + qsa_eval_mask_val(mask, bounds, ib, iq, is, sm0, sm1, sm2, sbo1, sbo2, nm_stream);
}

template <typename K_T>
static void ggml_cuda_op_qsa_block_score_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q      = dst->src[0];
    const ggml_tensor * k      = dst->src[1];
    const ggml_tensor * cells  = dst->src[2];
    const ggml_tensor * mask   = dst->src[3];
    const ggml_tensor * bounds = dst->src[4];

    GGML_TENSOR_LOCALS(size_t, nbq, q,     nb);
    GGML_TENSOR_LOCALS(size_t, nbk, k,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, cells, nb);
    GGML_TENSOR_LOCALS(size_t, nbm, mask,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_embd    = q->ne[0];
    const int64_t n_head    = q->ne[1];
    const int64_t n_query   = q->ne[2];
    const int64_t n_stream  = q->ne[3];
    const int64_t n_blocks  = cells->ne[0];
    const int64_t nm_stream = mask->ne[2];
    const int64_t sbo1      = bounds ? (int64_t) (bounds->nb[1] / sizeof(int32_t)) : 0;
    const int64_t sbo2      = bounds ? (int64_t) (bounds->nb[2] / sizeof(int32_t)) : 0;
    const int32_t * bounds_d = bounds ? (const int32_t *) bounds->data : nullptr;
    const float scale = ggml_get_op_params_f32(dst, 0);

    constexpr bool K_IS_F16 = std::is_same_v<K_T, half>;
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

#if defined(GGML_USE_HIP)
    if (warp_size == 32 && !GGML_CUDA_CC_IS_RDNA3_5(cc) && n_embd == 128 && n_head == 4 && n_stream <= 65535) {
        const bool q_aligned16 = (((uintptr_t) q->data | nbq1 | nbq2 | nbq3) & 15) == 0 && nbq0 == sizeof(float);
        const bool k_aligned16 = (((uintptr_t) k->data | nbk1) & 15) == 0 && nbk0 == sizeof(K_T);
        const bool k_aligned8  = (((uintptr_t) k->data | nbk1) & 7) == 0 && nbk0 == sizeof(K_T);

        if (n_query >= 24 && q_aligned16 && k_aligned16) {
            const bool use_bm64 = (n_query >= 48) &&
                (((n_blocks + 63) / 64) * ((n_query + 63) / 64) * n_stream >= 80);
            const bool check_inf = bounds_d != nullptr || n_blocks <= 2 * n_query;
            const int BM = use_bm64 ? 64 : 32;
            const int nthreads = use_bm64 ? 128 : 256;
            const dim3 grid_dims(
                    (unsigned) ((n_blocks + 63) / 64),
                    (unsigned) ((n_query + BM - 1) / BM),
                    (unsigned) n_stream);

            auto launch_tiled = [&](auto bm_tag, auto chk_tag, auto bnd_tag) {
                qsa_block_score_rdna2_tiled_fdot2<
                        decltype(bm_tag)::value, K_IS_F16,
                        decltype(chk_tag)::value, decltype(bnd_tag)::value>
                    <<<grid_dims, nthreads, 0, ctx.stream()>>>(
                        (const char *) q->data, (const char *) k->data,
                        (const int32_t *) cells->data, (const float *) mask->data, bounds_d,
                        (float *) dst->data,
                        n_blocks, n_query,
                        nbq1, nbq2, nbq3, nbk1,
                        nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
                        nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
                        sbo1, sbo2, nm_stream,
                        nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
                        scale);
            };
            auto dispatch_bnd = [&](auto bm_tag, auto chk_tag) {
                if (bounds_d != nullptr) { launch_tiled(bm_tag, chk_tag, std::true_type{}); }
                else                     { launch_tiled(bm_tag, chk_tag, std::false_type{}); }
            };
            if (use_bm64) {
                if (check_inf) { dispatch_bnd(std::integral_constant<int, 64>{}, std::true_type{}); }
                else           { dispatch_bnd(std::integral_constant<int, 64>{}, std::false_type{}); }
            } else {
                if (check_inf) { dispatch_bnd(std::integral_constant<int, 32>{}, std::true_type{}); }
                else           { dispatch_bnd(std::integral_constant<int, 32>{}, std::false_type{}); }
            }
            return;
        }

        if (n_query < 8 && q_aligned16 && (K_IS_F16 ? k_aligned8 : k_aligned16)) {
            constexpr int block_size = 256;
            const int keys_per_wave = (n_blocks >= 512) ? 4 : 1;
            const dim3 grid_dims(
                    (unsigned) ((n_blocks + 8 * keys_per_wave - 1) / (8 * keys_per_wave)),
                    (unsigned) n_query,
                    (unsigned) n_stream);
            auto launch_decode = [&](auto kpw_tag, auto bnd_tag) {
                qsa_block_score_rdna2_decode_fdot2<K_IS_F16, decltype(kpw_tag)::value, decltype(bnd_tag)::value>
                    <<<grid_dims, block_size, 0, ctx.stream()>>>(
                        (const char *) q->data, (const char *) k->data,
                        (const int32_t *) cells->data, (const float *) mask->data, bounds_d,
                        (float *) dst->data,
                        n_blocks, nbq1, nbq2, nbq3, nbk1,
                        nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
                        nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
                        sbo1, sbo2, nm_stream,
                        nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
                        scale);
            };
            if (keys_per_wave == 4) {
                if (bounds_d != nullptr) { launch_decode(std::integral_constant<int, 4>{}, std::true_type{}); }
                else                     { launch_decode(std::integral_constant<int, 4>{}, std::false_type{}); }
            } else {
                if (bounds_d != nullptr) { launch_decode(std::integral_constant<int, 1>{}, std::true_type{}); }
                else                     { launch_decode(std::integral_constant<int, 1>{}, std::false_type{}); }
            }
            return;
        }
    }
#else
    GGML_UNUSED(cc);
#endif

    if (warp_size == 32 && n_embd == 128 && n_head == 4 && n_stream <= 65535) {
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

            ggml_cuda_kernel_launch(qsa_block_score_f32_128x4_wave32_batched<K_T>, launch_params,
                    (const float *) q->data,
                    (const K_T *) k->data,
                    (const int32_t *) cells->data,
                    (const float *) mask->data,
                    bounds_d,
                    (float *) dst->data,
                    n_blocks, n_query, n_stream,
                    nbq1 / sizeof(float), nbq2 / sizeof(float), nbq3 / sizeof(float),
                    nbk1 / sizeof(K_T),
                    nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
                    nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
                    sbo1, sbo2, nm_stream,
                    nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
                    scale);
            return;
        }

        constexpr int waves_per_block = block_size / 32;
        const dim3 grid_dims((n_blocks + waves_per_block - 1) / waves_per_block, n_query, n_stream);
        const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

        ggml_cuda_kernel_launch(qsa_block_score_f32_128x4_wave32<K_T>, launch_params,
                (const float *) q->data,
                (const K_T *) k->data,
                (const int32_t *) cells->data,
                (const float *) mask->data,
                bounds_d,
                (float *) dst->data,
                n_blocks,
                nbq1 / sizeof(float), nbq2 / sizeof(float), nbq3 / sizeof(float),
                nbk1 / sizeof(K_T),
                nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
                nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
                sbo1, sbo2, nm_stream,
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

    ggml_cuda_kernel_launch(qsa_block_score_f32_generic<K_T>, launch_params,
            (const float *) q->data,
            (const K_T *) k->data,
            (const int32_t *) cells->data,
            (const float *) mask->data,
            bounds_d,
            (float *) dst->data,
            n_embd, n_head, n_blocks, n_query, n_stream,
            nbq1 / sizeof(float), nbq2 / sizeof(float), nbq3 / sizeof(float),
            nbk1 / sizeof(K_T),
            nbc0 / sizeof(int32_t), nbc1 / sizeof(int32_t),
            nbm0 / sizeof(float), nbm1 / sizeof(float), nbm2 / sizeof(float),
            sbo1, sbo2, nm_stream,
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            scale);
}

void ggml_cuda_op_qsa_block_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q      = dst->src[0];
    const ggml_tensor * k      = dst->src[1];
    const ggml_tensor * cells  = dst->src[2];
    const ggml_tensor * mask   = dst->src[3];
    const ggml_tensor * bounds = dst->src[4];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    GGML_ASSERT(cells->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F32);
    GGML_ASSERT(!bounds || bounds->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(q));
    GGML_ASSERT(ggml_is_contiguous_rows(k));
    GGML_ASSERT(ggml_is_contiguous(cells));
    GGML_ASSERT(ggml_is_contiguous(mask));
    GGML_ASSERT(!bounds || ggml_is_contiguous(bounds));

    GGML_ASSERT(k->ne[0] == q->ne[0]);
    GGML_ASSERT(cells->ne[1] == q->ne[3]);
    GGML_ASSERT(mask->ne[0] == cells->ne[0]);
    if (bounds != nullptr) {
        GGML_ASSERT(mask->ne[1] == 1);
        GGML_ASSERT(mask->ne[2] == 1 || mask->ne[2] == q->ne[3]);
        GGML_ASSERT(bounds->ne[0] == 2);
        GGML_ASSERT(bounds->ne[1] == q->ne[2]);
        GGML_ASSERT(bounds->ne[2] == q->ne[3]);
    } else {
        GGML_ASSERT(mask->ne[1] == q->ne[2]);
        GGML_ASSERT(mask->ne[2] == q->ne[3]);
    }
    GGML_ASSERT(dst->ne[0] == cells->ne[0]);
    GGML_ASSERT(dst->ne[1] == q->ne[2]);
    GGML_ASSERT(dst->ne[2] == q->ne[3]);

    if (k->type == GGML_TYPE_F16) {
        ggml_cuda_op_qsa_block_score_impl<half>(ctx, dst);
    } else {
        ggml_cuda_op_qsa_block_score_impl<float>(ctx, dst);
    }
}
