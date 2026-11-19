#include "lightning-indexer-hip.cuh"

#include <cstdint>

namespace {

typedef _Float16 half2v __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ uint4 pack_f32_to_u4h2(
        const float4 a, const float4 b) {
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

// Non-volatile v_dot2_f32_f16 so LLVM schedules ds_read_b128 overlap cleanly.
static __device__ __forceinline__ void fdot2_u4(float & acc, const uint4 qv, const uint4 kv) {
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__) || \
    defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
    const half2v q0 = *reinterpret_cast<const half2v *>(&qv.x);
    const half2v k0 = *reinterpret_cast<const half2v *>(&kv.x);
    const half2v q1 = *reinterpret_cast<const half2v *>(&qv.y);
    const half2v k1 = *reinterpret_cast<const half2v *>(&kv.y);
    const half2v q2 = *reinterpret_cast<const half2v *>(&qv.z);
    const half2v k2 = *reinterpret_cast<const half2v *>(&kv.z);
    const half2v q3 = *reinterpret_cast<const half2v *>(&qv.w);
    const half2v k3 = *reinterpret_cast<const half2v *>(&kv.w);
    acc = __builtin_amdgcn_fdot2(q0, k0, acc, false);
    acc = __builtin_amdgcn_fdot2(q1, k1, acc, false);
    acc = __builtin_amdgcn_fdot2(q2, k2, acc, false);
    acc = __builtin_amdgcn_fdot2(q3, k3, acc, false);
#else
    const half2 q0 = *reinterpret_cast<const half2 *>(&qv.x);
    const half2 k0 = *reinterpret_cast<const half2 *>(&kv.x);
    const half2 q1 = *reinterpret_cast<const half2 *>(&qv.y);
    const half2 k1 = *reinterpret_cast<const half2 *>(&kv.y);
    const half2 q2 = *reinterpret_cast<const half2 *>(&qv.z);
    const half2 k2 = *reinterpret_cast<const half2 *>(&kv.z);
    const half2 q3 = *reinterpret_cast<const half2 *>(&qv.w);
    const half2 k3 = *reinterpret_cast<const half2 *>(&kv.w);
    ggml_cuda_mad(acc, q0, k0);
    ggml_cuda_mad(acc, q1, k1);
    ggml_cuda_mad(acc, q2, k2);
    ggml_cuda_mad(acc, q3, k3);
#endif
}

// RDNA2 2D register-tiled Lightning Indexer (BN=64 keys, Wave32).
// - BM=32: 256 threads (8 warps, 4x2 thread tile, 24KB LDS, 2 WGs/CU) for small nb/grid.
// - BM=64: 128 threads (4 warps, 4x8 thread tile, exact 32KB aliased LDS, 2 WGs/CU):
//   10.67 fdot2 per ds_read_b128 breaks the 4x4 LDS bandwidth ceiling (~1.33x higher arithmetic/LDS).
template <int BM, bool K_IS_F16>
static __global__ void __launch_bounds__((BM == 64 ? 128 : 256), 2) lightning_indexer_rdna2_tiled_fdot2(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ w,
        const char * __restrict__ m,
        char * __restrict__ dst,
        int nh,
        int kv,
        int nb,
        int nm,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3,
        size_t nbd1, size_t nbd3) {
    static_assert(BM == 32 || BM == 64, "BM must be 32 or 64");
    constexpr int BN            = 64;
    constexpr int NTHREADS      = (BM == 64) ? 128 : 256;
    constexpr int CK            = (BM == 64) ? 8 : 2;   // keys per thread
    constexpr int WARP_K_COUNT  = (BM == 64) ? 2 : 4;   // warps along K
    constexpr int WARP_K_WIDTH  = BN / WARP_K_COUNT;    // 32 (BM=64) or 16 (BM=32)
    constexpr int WARP_K_SHIFT  = (BM == 64) ? 1 : 2;
    constexpr int WARP_K_MASK   = WARP_K_COUNT - 1;
    constexpr int STRIDE_Q      = (BM == 64) ? 8 : 4;
    constexpr int STRIDE_K      = (BM == 64) ? 4 : 8;
    constexpr int LOAD_ROWS     = NTHREADS / 8;         // 16 (BM=64) or 32 (BM=32)

    const int k0 = blockIdx.x * BN;
    const int q0 = blockIdx.y * BM;
    const int is = blockIdx.z;
    const int is_m = is % nm;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_q = warp >> WARP_K_SHIFT;  // 0..1
    const int warp_k = warp & WARP_K_MASK;    // 0..1 (BM=64) or 0..3 (BM=32)
    const int lane_q = (BM == 64) ? (lane & 7) : (lane & 3);
    const int lane_k = (BM == 64) ? (lane >> 3) : (lane >> 2);

    // Exact 32KB (BM=64) or 24KB (BM=32) LDS with zero hidden __syncthreads_or overhead.
    __shared__ union {
        struct {
            uint4 K_s[16][64];
            uint4 Q_s[16][BM];
        } smem;
        int any_active_s;
    } sh;

    if (tid == 0) {
        sh.any_active_s = 0;
    }
    __syncthreads();

    half2 m_h2[4][CK / 2];
    bool any_active = false;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int rq = warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
        const int gq = q0 + rq;
        const half * m_row = (gq < nb)
            ? reinterpret_cast<const half *>(m + (size_t) gq * nbm1 + (size_t) is_m * nbm3)
            : nullptr;
#pragma unroll
        for (int c2 = 0; c2 < CK / 2; ++c2) {
            const int ck0 = warp_k * WARP_K_WIDTH + lane_k + (c2 * 2 + 0) * STRIDE_K;
            const int ck1 = warp_k * WARP_K_WIDTH + lane_k + (c2 * 2 + 1) * STRIDE_K;
            const int gk0 = k0 + ck0;
            const int gk1 = k0 + ck1;
            const half mh0 = (m_row && gk0 < kv) ? m_row[gk0] : __float2half(-INFINITY);
            const half mh1 = (m_row && gk1 < kv) ? m_row[gk1] : __float2half(-INFINITY);
            const float mv0 = __half2float(mh0);
            const float mv1 = __half2float(mh1);
            if (isfinite(mv0) || mv0 > -INFINITY || isfinite(mv1) || mv1 > -INFINITY) {
                any_active = true;
            }
            m_h2[r][c2] = __halves2half2(mh0, mh1);
        }
    }

    if (__any(any_active) && lane == 0) {
        atomicOr(&sh.any_active_s, 1);
    }
    __syncthreads();

    if (sh.any_active_s == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
            if (gq < nb) {
                float * dst_row = reinterpret_cast<float *>(dst + (size_t) gq * nbd1 + (size_t) is * nbd3);
#pragma unroll
                for (int c2 = 0; c2 < CK / 2; ++c2) {
                    const float2 mf = __half22float2(m_h2[r][c2]);
                    const int gk0 = k0 + warp_k * WARP_K_WIDTH + lane_k + (c2 * 2 + 0) * STRIDE_K;
                    const int gk1 = k0 + warp_k * WARP_K_WIDTH + lane_k + (c2 * 2 + 1) * STRIDE_K;
                    if (gk0 < kv) {
                        dst_row[gk0] = mf.x;
                    }
                    if (gk1 < kv) {
                        dst_row[gk1] = mf.y;
                    }
                }
            }
        }
        return;
    }

    // Load BN=64 keys (128 dims each) into K_s once for all queries and heads.
    {
        const char * k_stream = k + (size_t) is * nbk3;
        const int lk_base = tid >> 3;
        const int t8      = tid & 7;  // 0..7 (16 dims = 2 uint4s)
#pragma unroll
        for (int iter = 0; iter < (BN / LOAD_ROWS); ++iter) {
            const int lk = iter * LOAD_ROWS + lk_base;
            const int gk = k0 + lk;
            uint4 u0 = make_uint4(0, 0, 0, 0);
            uint4 u1 = make_uint4(0, 0, 0, 0);
            if (gk < kv) {
                const char * kr = k_stream + (size_t) gk * nbk2;
                if constexpr (K_IS_F16) {
                    const uint4 * ku = reinterpret_cast<const uint4 *>(kr) + t8 * 2;
                    u0 = ku[0];
                    u1 = ku[1];
                } else {
                    const float4 * kf = reinterpret_cast<const float4 *>(kr) + t8 * 4;
                    u0 = pack_f32_to_u4h2(kf[0], kf[1]);
                    u1 = pack_f32_to_u4h2(kf[2], kf[3]);
                }
            }
            sh.smem.K_s[t8 * 2 + 0][lk ^ t8] = u0;
            sh.smem.K_s[t8 * 2 + 1][lk ^ t8] = u1;
        }
    }
    // Synchronize after K_s write (also guarantees all threads finished reading sh.any_active_s).
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
    const char * w_stream = w + (size_t) is * nbw3;

    const int lq_base = tid >> 3;
    const int t8      = tid & 7;

    for (int h = 0; h < nh; ++h) {
        if (h > 0) {
            __syncthreads();
        }

#pragma unroll
        for (int iter = 0; iter < (BM / LOAD_ROWS); ++iter) {
            const int lq = iter * LOAD_ROWS + lq_base;
            const int gq = q0 + lq;
            if (gq < nb) {
                const float4 * qf = reinterpret_cast<const float4 *>(
                    q_stream + (size_t) h * nbq1 + (size_t) gq * nbq2) + t8 * 4;
                sh.smem.Q_s[t8 * 2 + 0][lq ^ t8] = pack_f32_to_u4h2(qf[0], qf[1]);
                sh.smem.Q_s[t8 * 2 + 1][lq ^ t8] = pack_f32_to_u4h2(qf[2], qf[3]);
            } else {
                sh.smem.Q_s[t8 * 2 + 0][lq ^ t8] = make_uint4(0, 0, 0, 0);
                sh.smem.Q_s[t8 * 2 + 1][lq ^ t8] = make_uint4(0, 0, 0, 0);
            }
        }

        // Preload head weights early so global/L2 latency is hidden under the d4 loop.
        float wh[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
            wh[r] = (gq < nb)
                ? reinterpret_cast<const float *>(w_stream + (size_t) gq * nbw1)[h]
                : 0.0f;
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
                    fdot2_u4(dot[r][c], qv[r], kv_r[c]);
                }
            }
        }

#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int c = 0; c < CK; ++c) {
                if (dot[r][c] > 0.0f) {
                    acc[r][c] = fmaf(wh[r], dot[r][c], acc[r][c]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int gq = q0 + warp_q * (BM / 2) + lane_q + r * STRIDE_Q;
        if (gq < nb) {
            float * dst_row = reinterpret_cast<float *>(dst + (size_t) gq * nbd1 + (size_t) is * nbd3);
#pragma unroll
            for (int c2 = 0; c2 < CK / 2; ++c2) {
                const float2 mf = __half22float2(m_h2[r][c2]);
                const int gk0 = k0 + warp_k * WARP_K_WIDTH + lane_k + (c2 * 2 + 0) * STRIDE_K;
                const int gk1 = k0 + warp_k * WARP_K_WIDTH + lane_k + (c2 * 2 + 1) * STRIDE_K;
                if (gk0 < kv) {
                    dst_row[gk0] = acc[r][c2 * 2 + 0] + mf.x;
                }
                if (gk1 < kv) {
                    dst_row[gk1] = acc[r][c2 * 2 + 1] + mf.y;
                }
            }
        }
    }
}

} // namespace

bool ggml_cuda_lightning_indexer_hip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#else
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (ggml_cuda_info().devices[ctx.device].warp_size != 32 || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    if (q->ne[0] != 128 || q->ne[2] < 8 ||
            q->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            m->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
            (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32)) {
        return false;
    }

    if (q->nb[0] != sizeof(float) || w->nb[0] != sizeof(float) ||
            m->nb[0] != ggml_type_size(GGML_TYPE_F16) || dst->nb[0] != sizeof(float) ||
            k->nb[0] != ggml_type_size(k->type)) {
        return false;
    }

    // 16-byte alignment check for vectorized float4 / uint4 global loads.
    if (((uintptr_t) q->data & 15) != 0 || ((uintptr_t) k->data & 15) != 0 ||
            (q->nb[1] & 15) != 0 || (q->nb[2] & 15) != 0 || (q->nb[3] & 15) != 0 ||
            (k->nb[2] & 15) != 0 || (k->nb[3] & 15) != 0) {
        return false;
    }

    const int nh = (int) q->ne[1];
    const int nb = (int) q->ne[2];
    const int ns = (int) q->ne[3];
    const int kv = (int) k->ne[2];
    const int nm = (int) m->ne[3];
    if (nh <= 0 || nb <= 0 || ns <= 0 || kv <= 0 || nm <= 0) {
        return false;
    }

    const bool use_bm64 = (nb >= 48) && (((kv + 63) / 64) * ((nb + 63) / 64) * ns >= 80);
    const int BM = use_bm64 ? 64 : 32;
    const dim3 grid((unsigned) ((kv + 63) / 64), (unsigned) ((nb + BM - 1) / BM), (unsigned) ns);

    auto launch = [&](auto bm_tag, auto k16_tag) {
        constexpr int nthreads = (decltype(bm_tag)::value == 64) ? 128 : 256;
        lightning_indexer_rdna2_tiled_fdot2<decltype(bm_tag)::value, decltype(k16_tag)::value>
            <<<grid, nthreads, 0, ctx.stream()>>>(
                (const char *) q->data, (const char *) k->data,
                (const char *) w->data, (const char *) m->data,
                (char *) dst->data,
                nh, kv, nb, nm,
                q->nb[1], q->nb[2], q->nb[3],
                k->nb[2], k->nb[3],
                w->nb[1], w->nb[3],
                m->nb[1], m->nb[3],
                dst->nb[1], dst->nb[3]);
    };

    if (use_bm64) {
        if (k->type == GGML_TYPE_F16) {
            launch(std::integral_constant<int, 64>{}, std::integral_constant<bool, true>{});
        } else {
            launch(std::integral_constant<int, 64>{}, std::integral_constant<bool, false>{});
        }
    } else {
        if (k->type == GGML_TYPE_F16) {
            launch(std::integral_constant<int, 32>{}, std::integral_constant<bool, true>{});
        } else {
            launch(std::integral_constant<int, 32>{}, std::integral_constant<bool, false>{});
        }
    }

    return true;
#endif
}
