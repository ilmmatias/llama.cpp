#include "fattn-common.cuh"
#include "fattn-qsa-hip.cuh"
#include "fattn-tile.cuh"

#include <cfloat>
#include <cstring>

// Dedicated RDNA2 (gfx103x, Wave32) FlashAttention kernels for QSA/GQA attention:
//  - Prefill: native GQA up to ncols2=12 (Tinfield-1 GQA=12 = 1x KV load vs 3x),
//    non-volatile __builtin_amdgcn_fdot2 on both Q*K^T and P*V, 0-bank-conflict
//    XOR-swizzled K_s & stripe-grouped V_s in <32 KB LDS (2 WGs/CU on D=256).
//  - Decode: 4-warp collaborative GQA head sharing (0 cross-warp O merge for
//    ncols2>=4) and single-lookup unified (K, V) QSA page table resolution.

namespace {

static __device__ __forceinline__ float qsa_fdot2(const half2 a, const half2 b, const float c) {
#if defined(V_DOT2_F32_F16_AVAILABLE)
    return __builtin_amdgcn_fdot2(a, b, c, false);
#else
    const float2 af = __half22float2(a);
    const float2 bf = __half22float2(b);
    return c + af.x * bf.x + af.y * bf.y;
#endif
}

template <int mask>
static __device__ __forceinline__ uint32_t qsa_shfl_xor_u32(uint32_t val) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    if constexpr (mask < 16) {
        return __builtin_amdgcn_update_dpp(0, val, 0x160 | mask, 0xf, 0xf, true);
    } else {
        const uint32_t swapped = __builtin_amdgcn_permlanex16(
                val, val, 0x76543210, 0xFEDCBA98, true, false);
        if constexpr (mask == 24) {
            return qsa_shfl_xor_u32<8>(swapped);
        }
        return swapped;
    }
#else
    return __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, val, mask, 32);
#endif
}

template <int mask>
static __device__ __forceinline__ float qsa_shfl_xor_32(float val) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    return __int_as_float(int(qsa_shfl_xor_u32<mask>(uint32_t(__float_as_int(val)))));
#else
    return ggml_cuda_shfl_xor_sync(val, mask);
#endif
}

static __device__ __forceinline__ float qsa_warp_sum_32(float val) {
    val += qsa_shfl_xor_32<16>(val);
    val += qsa_shfl_xor_32<8>(val);
    val += qsa_shfl_xor_32<4>(val);
    val += qsa_shfl_xor_32<2>(val);
    val += qsa_shfl_xor_32<1>(val);
    return val;
}

static __device__ __forceinline__ float qsa_warp_max_32(float val) {
    val = fmaxf(val, qsa_shfl_xor_32<16>(val));
    val = fmaxf(val, qsa_shfl_xor_32<8>(val));
    val = fmaxf(val, qsa_shfl_xor_32<4>(val));
    val = fmaxf(val, qsa_shfl_xor_32<2>(val));
    val = fmaxf(val, qsa_shfl_xor_32<1>(val));
    return val;
}

static __device__ __forceinline__ void qsa_resolve_kv_row_ptrs(
        const char * K_data, const char * V_data, int row, int ne11,
        int64_t nb11, int64_t nb21,
        const ggml_cuda_qsa_kv_view & K_cache, const ggml_cuda_qsa_kv_view & V_cache,
        bool use_sparse, bool use_paged, const char * & K_row_out, const char * & V_row_out) {
    if (row < 0 || (!use_sparse && row >= ne11)) {
        K_row_out = nullptr;
        V_row_out = nullptr;
        return;
    }

    const char * k_addr = K_data + int64_t(row) * nb11;
    const char * v_addr = V_data + int64_t(row) * nb21;
    if (!use_paged) {
        K_row_out = k_addr;
        V_row_out = v_addr;
        return;
    }

    if (K_cache.pages && K_cache.pages == V_cache.pages &&
            K_cache.page_bytes == V_cache.page_bytes &&
            (k_addr - K_cache.host) == (v_addr - V_cache.host)) {
        const size_t offset  = size_t(k_addr - K_cache.host);
        const size_t page    = K_cache.page_shift ? (offset >> K_cache.page_shift) : (offset / K_cache.page_bytes);
        const size_t in_page = offset - page * K_cache.page_bytes;
        const int    slot    = K_cache.pages[page];
        if (slot < 0) {
            K_row_out = k_addr;
            V_row_out = v_addr;
        } else {
            const size_t cache_off = size_t(slot) * K_cache.page_bytes + in_page;
            K_row_out = K_cache.cache + cache_off;
            V_row_out = V_cache.cache + cache_off;
        }
        return;
    }

    K_row_out = ggml_cuda_qsa_kv_address(k_addr, K_cache);
    V_row_out = ggml_cuda_qsa_kv_address(v_addr, V_cache);
}

static __device__ __forceinline__ float qsa_eval_mask_logit(
        float score, int row, int q_idx,
        const char * mask_b, int64_t nb31,
        const ggml_cuda_fattn_visibility & vis) {
    const int src_row = vis.row_map ? vis.row_map[row] : row;
    if (vis.positions) {
        return ggml_qsa_is_visible(vis.positions, vis.n_kv, src_row + vis.row_offset, q_idx) ? score : -INFINITY;
    }
    if (mask_b) {
        const half mh = *reinterpret_cast<const half *>(mask_b + int64_t(q_idx) * nb31 + size_t(src_row) * sizeof(half));
        const float mv = __half2float(mh);
        return isfinite(mv) ? (score + mv) : -INFINITY;
    }
    return score;
}

static __device__ __forceinline__ void qsa_load_q8_0_block(
        const char * bp, half & dh_out, uint32_t (&qs)[8]) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(bp);
    if ((addr & 3u) == 2u) {
        dh_out = *reinterpret_cast<const half *>(bp);
        const uint4 q0 = *reinterpret_cast<const uint4 *>(bp + 2);
        const uint4 q1 = *reinterpret_cast<const uint4 *>(bp + 18);
        qs[0] = q0.x; qs[1] = q0.y; qs[2] = q0.z; qs[3] = q0.w;
        qs[4] = q1.x; qs[5] = q1.y; qs[6] = q1.z; qs[7] = q1.w;
    } else if ((addr & 3u) == 0u) {
        const uint32_t h0 = *reinterpret_cast<const uint32_t *>(bp);
        const uint4    m0 = *reinterpret_cast<const uint4 *>(bp + 4);
        const uint2    m1 = *reinterpret_cast<const uint2 *>(bp + 20);
        const uint32_t m2 = *reinterpret_cast<const uint32_t *>(bp + 28);
        const uint16_t m3 = *reinterpret_cast<const uint16_t *>(bp + 32);
        const uint16_t dh_u16 = uint16_t(h0 & 0xffffu);
        memcpy(&dh_out, &dh_u16, sizeof(dh_out));
        qs[0] = (h0 >> 16) | (m0.x << 16);
        qs[1] = (m0.x >> 16) | (m0.y << 16);
        qs[2] = (m0.y >> 16) | (m0.z << 16);
        qs[3] = (m0.z >> 16) | (m0.w << 16);
        qs[4] = (m0.w >> 16) | (m1.x << 16);
        qs[5] = (m1.x >> 16) | (m1.y << 16);
        qs[6] = (m1.y >> 16) | (m2 << 16);
        qs[7] = (m2 >> 16) | (uint32_t(m3) << 16);
    } else {
        memcpy(&dh_out, bp, sizeof(half));
        memcpy(qs, bp + sizeof(half), sizeof(qs));
    }
}

static __device__ __forceinline__ void qsa_dequant_q8_4_to_h2(
        const half2 dh2, const uint32_t q4, half2 & h0, half2 & h1) {
    h0 = __hmul2(dh2, make_half2(float(int8_t(q4 & 0xff)),         float(int8_t((q4 >>  8) & 0xff))));
    h1 = __hmul2(dh2, make_half2(float(int8_t((q4 >> 16) & 0xff)), float(int8_t((q4 >> 24) & 0xff))));
}

template <ggml_type TK, int D, int NW>
static __device__ __forceinline__ void load_k_tile_to_smem(
        const char * const * K_ptrs, int warp, int lane, uint4 (*K_s4)[32]) {
    constexpr int KEYS_PER_WARP = 32 / NW;
    const int group   = lane >> 3;
    const int sublane = lane & 7;

    if constexpr (TK == GGML_TYPE_F16) {
#pragma unroll
        for (int batch = 0; batch < KEYS_PER_WARP; batch += 4) {
            const int k = warp * KEYS_PER_WARP + batch + group;
            const char * row_p = K_ptrs[k];
#pragma unroll
            for (int iter = 0; iter < D / 64; ++iter) {
                uint4 raw = make_uint4(0, 0, 0, 0);
                if (row_p) {
                    ggml_cuda_memcpy_1<16>(&raw, row_p + (iter * 64 + sublane * 8) * sizeof(half));
                }
                const int d8 = iter * 8 + sublane;
                K_s4[d8][k ^ (d8 & 31)] = raw;
            }
        }
    } else {
        static_assert(TK == GGML_TYPE_Q8_0 && D % QK8_0 == 0, "Q8_0 K requires D % 32 == 0");
        constexpr int NBLK = D / QK8_0;
        constexpr int KEY_LANES = 32 / NBLK; // 4 when D=256 (NBLK=8), 8 when D=128 (NBLK=4)
        const int kg  = lane / NBLK;         // 0..KEY_LANES-1
        const int blk = lane % NBLK;         // 0..NBLK-1

#pragma unroll
        for (int batch = 0; batch < KEYS_PER_WARP; batch += KEY_LANES) {
            const int k = warp * KEYS_PER_WARP + batch + kg;
            const char * row_p = K_ptrs[k];
            if (!row_p) {
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const int d8 = blk * 4 + w;
                    K_s4[d8][k ^ (d8 & 31)] = make_uint4(0, 0, 0, 0);
                }
                continue;
            }
            const char * bp = row_p + size_t(blk) * sizeof(block_q8_0);
            half d_h;
            uint32_t qs[8];
            qsa_load_q8_0_block(bp, d_h, qs);
            const half2 dh2 = __half2half2(d_h);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint4 out4;
                half2 * h2 = reinterpret_cast<half2 *>(&out4);
                qsa_dequant_q8_4_to_h2(dh2, qs[2 * w + 0], h2[0], h2[1]);
                qsa_dequant_q8_4_to_h2(dh2, qs[2 * w + 1], h2[2], h2[3]);
                const int d8 = blk * 4 + w;
                K_s4[d8][k ^ (d8 & 31)] = out4;
            }
        }
    }
}

template <ggml_type TV, int D, int NW>
static __device__ __forceinline__ void load_v_tile_to_smem(
        const char * const * V_ptrs, int tid, uint4 (*V_s2)[D / 4]) {
    constexpr int TH_PV    = D / 4; // 64 (D=256) or 32 (D=128)
    const int     kp_local = tid / (2 * TH_PV);
    const int     col      = (tid >> 1) & (TH_PV - 1);
    const int     k_bit    = tid & 1;

    const int blk = col >> 3;
    const int sub = col & 7;

#pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int kp    = b * 2 + kp_local;
        const int k     = kp * 2 + k_bit;
        const char * row_p = V_ptrs[k];
        uint2 v2 = make_uint2(0, 0);
        if (row_p) {
            if constexpr (TV == GGML_TYPE_F16) {
                ggml_cuda_memcpy_1<8>(&v2, row_p + size_t(col) * 8);
            } else {
                static_assert(TV == GGML_TYPE_Q8_0 && D % QK8_0 == 0);
                const char * bp = row_p + size_t(blk) * sizeof(block_q8_0);
                half dh;
                ggml_cuda_memcpy_1<2, 2>(&dh, bp);
                const half2 dh2 = __half2half2(dh);
                uint32_t q4;
                ggml_cuda_memcpy_1<4, 2>(&q4, bp + sizeof(half) + sub * 4);
                half2 * h2 = reinterpret_cast<half2 *>(&v2);
                qsa_dequant_q8_4_to_h2(dh2, q4, h2[0], h2[1]);
            }
        }
        const uint32_t peer_x = qsa_shfl_xor_u32<1>(v2.x);
        const uint32_t peer_y = qsa_shfl_xor_u32<1>(v2.y);
        if (k_bit == 0) {
            uint4 out4;
            out4.x = __byte_perm(v2.x, peer_x, 0x5410);
            out4.y = __byte_perm(v2.x, peer_x, 0x7632);
            out4.z = __byte_perm(v2.y, peer_y, 0x5410);
            out4.w = __byte_perm(v2.y, peer_y, 0x7632);
            V_s2[kp][col] = out4;
        }
    }
}

template <int D, int ncols1, int ncols2, ggml_type TK, ggml_type TV, bool use_sparse, bool use_paged>
__launch_bounds__(D, 2)
static __global__ void flash_attn_qsa_hip_prefill(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const char * __restrict__ sinks_ptr,
        const int  * __restrict__ KV_max_ptr,
        const int  * __restrict__ KV_rows_ptr,
        float      * __restrict__ dst_ptr,
        float2     * __restrict__ dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
        const int32_t nb11, const int32_t nb12, const int64_t nb13,
        const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
        const int32_t nb31, const int64_t nb32, const int64_t nb33,
        ggml_cuda_qsa_kv_view K_cache, ggml_cuda_qsa_kv_view V_cache,
        ggml_cuda_fattn_visibility visibility) {
    GGML_UNUSED_VARS(KV_rows_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31);

    constexpr int WARP  = 32;
    constexpr int NW    = D / 32;
    constexpr int D8    = D / 8;
    constexpr int BK    = 32;
    constexpr int M     = ncols1 * ncols2;
    constexpr int M_PER_WARP = (M + NW - 1) / NW;
    constexpr int NG_PV = 4;
    constexpr int TH_PV = D / 4;               // 64 (D=256) or 32 (D=128)
    constexpr int M_G   = (M + NG_PV - 1) / NG_PV;
    static_assert(ncols1 <= 8, "qmask_s uses 8-bit mask per query tile");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    __builtin_assume(lane >= 0 && lane < WARP);
    __builtin_assume(warp >= 0 && warp < NW);
    const int tid    = warp * WARP + lane;
    const int grp    = tid / TH_PV;
    const int tid_g  = tid % TH_PV;
    const int r_base = grp * M_G;

    const int col_Q_0   = blockIdx.x * ncols1;
    const int sequence  = blockIdx.z / (ne02 / ncols2);
    const int head0     = blockIdx.z * ncols2 - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * Q_base = Q_ptr + int64_t(sequence) * nb03 + int64_t(head0) * nb02;
    const char * K_data = K_ptr + int64_t(sequence) * nb13 + int64_t(head0 / gqa_ratio) * nb12;
    const char * V_data = V_ptr + int64_t(sequence) * nb23 + int64_t(head0 / gqa_ratio) * nb22;
    const char * mask_b = mask_ptr ? (mask_ptr + int64_t(sequence % ne33) * nb33 + int64_t(head0 % ne32) * nb32) : nullptr;

    __shared__ uint4 Q_s4[M][D8];
    __shared__ union {
        uint4 K_s4[D8][BK];
        uint4 V_s2[BK / 2][TH_PV];
    } kv_smem;
    __shared__ uint2        P_s2[M][BK / 4];
    __shared__ int4         q_pos_s[ncols1];
    __shared__ int          row_s[BK];
    __shared__ uint8_t      qmask_s[BK];
    __shared__ int          any_finite_s;
    __shared__ union {
        struct {
            const char * V[BK];
            union {
                const char * K[BK];
                float O_scale[M];
            };
        } step;
        struct {
            float KQ_max[M];
            float KQ_sum[M];
            float O_scale[M];
        } meta;
    } sh_aux;

    if (tid == 0) {
        any_finite_s = 0;
    }

    if (visibility.positions && tid < ncols1) {
        const int g_q = col_Q_0 + tid;
        if (g_q < int(ne01.z)) {
            ggml_cuda_memcpy_1<16>(&q_pos_s[tid], visibility.positions + 4 * (int64_t(visibility.n_kv) + g_q));
        } else {
            q_pos_s[tid] = make_int4(0, 0, 0, 0);
        }
    }

    // Load and prescale Q[0..M-1][0..D-1] into Q_s4 as packed uint4 (4 half2s each).
    for (int r = warp; r < M; r += NW) {
        const int j = r / ncols2;
        const int c = r % ncols2;
        if (col_Q_0 + j < int(ne01.z)) {
            const float * q_row = reinterpret_cast<const float *>(Q_base + int64_t(c) * nb02 + int64_t(col_Q_0 + j) * nb01);
            for (int d8 = lane; d8 < D8; d8 += WARP) {
                float4 a, b;
                ggml_cuda_memcpy_1<16>(&a, q_row + d8 * 8 + 0);
                ggml_cuda_memcpy_1<16>(&b, q_row + d8 * 8 + 4);
                uint4 q4;
                half2 * qh = reinterpret_cast<half2 *>(&q4);
                qh[0] = __float22half2_rn(make_float2(a.x * scale, a.y * scale));
                qh[1] = __float22half2_rn(make_float2(a.z * scale, a.w * scale));
                qh[2] = __float22half2_rn(make_float2(b.x * scale, b.y * scale));
                qh[3] = __float22half2_rn(make_float2(b.z * scale, b.w * scale));
                Q_s4[r][d8] = q4;
            }
        } else {
            for (int d8 = lane; d8 < D8; d8 += WARP) {
                Q_s4[r][d8] = make_uint4(0, 0, 0, 0);
            }
        }
    }

    float O_acc[M_G][4];
#pragma unroll
    for (int m = 0; m < M_G; ++m) {
        O_acc[m][0] = 0.0f;
        O_acc[m][1] = 0.0f;
        O_acc[m][2] = 0.0f;
        O_acc[m][3] = 0.0f;
    }

    float KQ_max[M_PER_WARP];
    float KQ_sum[M_PER_WARP];
#pragma unroll
    for (int u = 0; u < M_PER_WARP; ++u) {
        KQ_max[u] = -FLT_MAX / 2.0f;
        KQ_sum[u] = 0.0f;
    }

    __syncthreads();

    const int32_t * indices = nullptr;
    int k_VKQ_max;
    const uint8_t * sparse_qmask = visibility.sparse_qmask;
    if constexpr (use_sparse) {
        const int64_t list = int64_t(sequence % ne33) * gridDim.x + blockIdx.x;
        indices = KV_max_ptr + list * ne11;
        const int32_t * counts = KV_max_ptr + int64_t(ne33) * gridDim.x * ne11;
        k_VKQ_max = counts[list];
        if (sparse_qmask) {
            sparse_qmask += list * ne11;
        }
    } else {
        k_VKQ_max = KV_max_ptr ? KV_max_ptr[sequence * gridDim.x + blockIdx.x] : ne11;
    }

    for (int k_base = blockIdx.y * BK; k_base < k_VKQ_max; k_base += gridDim.y * BK) {
        if (warp == 0) {
            const int  k_idx         = k_base + lane;
            const bool valid         = k_idx < k_VKQ_max;
            const int  raw_row       = valid ? (use_sparse ? indices[k_idx] : k_idx) : -1;
            const bool row_in_bounds = (raw_row >= 0) && (use_sparse || raw_row < ne11);
            const int  src_row       = row_in_bounds
                ? (visibility.row_map ? visibility.row_map[raw_row] : raw_row)
                : -1;
            row_s[lane]   = src_row;
            qmask_s[lane] = row_in_bounds
                ? ((use_sparse && sparse_qmask) ? sparse_qmask[k_idx] : 0xffu)
                : 0u;

            const char * kr = nullptr;
            const char * vr = nullptr;
            qsa_resolve_kv_row_ptrs(
                    K_data, V_data, raw_row, ne11, nb11, nb21,
                    K_cache, V_cache, use_sparse, use_paged, kr, vr);
            sh_aux.step.K[lane] = kr;
            sh_aux.step.V[lane] = vr;
        }

        __syncthreads();

        load_k_tile_to_smem<TK, D, NW>(sh_aux.step.K, warp, lane, kv_smem.K_s4);

        __syncthreads();

        // S = Q @ K^T via 2x1 D-split outer product: cuts Q_s4 LDS reads in half and runs D8/2 steps.
        constexpr int D8_HALF = D8 / 2;
        const int d_half = lane >> 4;
        const int k_sub  = lane & 15;
        float s_even[M_PER_WARP];
        float s_odd[M_PER_WARP];
#pragma unroll
        for (int u = 0; u < M_PER_WARP; ++u) {
            s_even[u] = 0.0f;
            s_odd[u]  = 0.0f;
        }

#pragma unroll 1
        for (int d8_h = 0; d8_h < D8_HALF; ++d8_h) {
            const int d8 = d_half * D8_HALF + d8_h;
            const uint4 k0 = kv_smem.K_s4[d8][k_sub ^ (d8 & 31)];
            const uint4 k1 = kv_smem.K_s4[d8][(k_sub + 16) ^ (d8 & 31)];
            uint4 q_cur[M_PER_WARP];
#pragma unroll
            for (int u = 0; u < M_PER_WARP; ++u) {
                const int r = warp + u * NW;
                q_cur[u] = ((M % NW == 0) || (r < M)) ? Q_s4[r][d8] : make_uint4(0, 0, 0, 0);
            }

            const half2 * kh0 = reinterpret_cast<const half2 *>(&k0);
            const half2 * kh1 = reinterpret_cast<const half2 *>(&k1);
#pragma unroll
            for (int u = 0; u < M_PER_WARP; ++u) {
                const int r = warp + u * NW;
                if ((M % NW == 0) || (r < M)) {
                    const half2 * qh = reinterpret_cast<const half2 *>(&q_cur[u]);
                    s_even[u] = qsa_fdot2(qh[0], kh0[0], s_even[u]);
                    s_odd[u]  = qsa_fdot2(qh[0], kh1[0], s_odd[u]);
                    s_even[u] = qsa_fdot2(qh[1], kh0[1], s_even[u]);
                    s_odd[u]  = qsa_fdot2(qh[1], kh1[1], s_odd[u]);
                    s_even[u] = qsa_fdot2(qh[2], kh0[2], s_even[u]);
                    s_odd[u]  = qsa_fdot2(qh[2], kh1[2], s_odd[u]);
                    s_even[u] = qsa_fdot2(qh[3], kh0[3], s_even[u]);
                    s_odd[u]  = qsa_fdot2(qh[3], kh1[3], s_odd[u]);
                }
            }
        }

        float s_acc[M_PER_WARP];
#pragma unroll
        for (int u = 0; u < M_PER_WARP; ++u) {
            const float my_part  = d_half ? s_odd[u] : s_even[u];
            const float send_val = d_half ? s_even[u] : s_odd[u];
            s_acc[u] = my_part + qsa_shfl_xor_32<16>(send_val);
        }

        // Masking + warp-local online softmax across the 32 keys.
        const int     src_row   = row_s[lane];
        const uint8_t qm        = qmask_s[lane];
        const int     k_cell    = src_row + visibility.row_offset;
        const bool    k_cell_ok = (qm != 0) && (k_cell >= 0) && (k_cell < visibility.n_kv);
        int4          k_pos     = make_int4(-1, 0, 0, 0);
        if (visibility.positions && k_cell_ok) {
            ggml_cuda_memcpy_1<16>(&k_pos, visibility.positions + 4 * int64_t(k_cell));
        }

        int warp_has_finite = 0;

#pragma unroll
        for (int u = 0; u < M_PER_WARP; ++u) {
            const int r = warp + u * NW;
            if (!((M % NW == 0) || (r < M))) {
                continue;
            }
            const int j_q = r / ncols2;
            const int g_q = col_Q_0 + j_q;

            const bool valid_q = (qm & (1u << j_q)) && (g_q < int(ne01.z));
            float logit = -INFINITY;
            if (valid_q) {
                if (visibility.positions) {
                    const int4 qp = q_pos_s[j_q];
                    const bool vis_ok = k_cell_ok && k_pos.x >= 0 &&
                        (!(qp.w & 1) || k_pos.x <= qp.x) &&
                        (!(qp.w & 2) || k_pos.x != qp.x || k_pos.z < qp.z || (k_pos.z == qp.z && k_pos.y <= qp.y));
                    if (vis_ok) {
                        logit = s_acc[u];
                    }
                } else if (mask_b) {
                    const half mh = *reinterpret_cast<const half *>(mask_b + int64_t(g_q) * nb31 + size_t(src_row) * sizeof(half));
                    const float mv = __half2float(mh);
                    if (isfinite(mv)) {
                        logit = s_acc[u] + mv;
                    }
                } else {
                    logit = s_acc[u];
                }
            }

            const float tile_max   = qsa_warp_max_32(logit);
            warp_has_finite       |= int(isfinite(tile_max));
            const float m_new      = fmaxf(KQ_max[u], tile_max);
            const float scale_prev = (KQ_sum[u] > 0.0f) ? expf(KQ_max[u] - m_new) : 0.0f;
            const float p_f        = isfinite(logit) ? expf(logit - m_new) : 0.0f;
            const float tile_sum   = qsa_warp_sum_32(p_f);

            KQ_max[u] = m_new;
            KQ_sum[u] = KQ_sum[u] * scale_prev + tile_sum;

            if (lane == 0) {
                sh_aux.step.O_scale[r] = scale_prev;
            }

            const float p1 = qsa_shfl_xor_32<1>(p_f);
            if ((lane & 1) == 0) {
                reinterpret_cast<half2 *>(P_s2[r])[lane >> 1] = __float22half2_rn(make_float2(p_f, p1));
            }
        }

        if (lane == 0 && warp_has_finite) {
            any_finite_s = 1;
        }
        __syncthreads();
        if (!any_finite_s) {
            continue;
        }

        load_v_tile_to_smem<TV, D, NW>(sh_aux.step.V, tid, kv_smem.V_s2);

        __syncthreads();

#pragma unroll
        for (int m = 0; m < M_G; ++m) {
            const int r = r_base + m;
            if (r < M) {
                const float sc = sh_aux.step.O_scale[r];
                O_acc[m][0] *= sc;
                O_acc[m][1] *= sc;
                O_acc[m][2] *= sc;
                O_acc[m][3] *= sc;
            }
        }

        // Pairwise fdot2 P @ V: P_s2 holds half2 (2*kp, 2*kp+1) and V_s2 holds transposed half2 (2*kp, 2*kp+1).
#pragma unroll 1
        for (int nk4 = 0; nk4 < BK / 4; ++nk4) {
            const uint4 v0 = kv_smem.V_s2[2 * nk4 + 0][tid_g];
            const uint4 v1 = kv_smem.V_s2[2 * nk4 + 1][tid_g];
            const half2 * vh0 = reinterpret_cast<const half2 *>(&v0);
            const half2 * vh1 = reinterpret_cast<const half2 *>(&v1);
#pragma unroll
            for (int m = 0; m < M_G; ++m) {
                const int r = r_base + m;
                if (r < M) {
                    const uint2 p2 = P_s2[r][nk4];
                    const half2 * ph = reinterpret_cast<const half2 *>(&p2);
                    O_acc[m][0] = qsa_fdot2(ph[0], vh0[0], qsa_fdot2(ph[1], vh1[0], O_acc[m][0]));
                    O_acc[m][1] = qsa_fdot2(ph[0], vh0[1], qsa_fdot2(ph[1], vh1[1], O_acc[m][1]));
                    O_acc[m][2] = qsa_fdot2(ph[0], vh0[2], qsa_fdot2(ph[1], vh1[2], O_acc[m][2]));
                    O_acc[m][3] = qsa_fdot2(ph[0], vh0[3], qsa_fdot2(ph[1], vh1[3], O_acc[m][3]));
                }
            }
        }

        if (tid == 0) {
            any_finite_s = 0;
        }
        __syncthreads();
    }

#pragma unroll
    for (int u = 0; u < M_PER_WARP; ++u) {
        const int r = warp + u * NW;
        if ((M % NW == 0) || (r < M)) {
            float scale_prev = 1.0f;
            if (sinks_ptr && blockIdx.y == 0) {
                const int   c          = r % ncols2;
                const float sink       = reinterpret_cast<const float *>(sinks_ptr)[head0 + c];
                const float m_new      = fmaxf(KQ_max[u], sink);
                const float prev_scale = (KQ_sum[u] > 0.0f) ? expf(KQ_max[u] - m_new) : 0.0f;
                KQ_max[u]              = m_new;
                KQ_sum[u]              = KQ_sum[u] * prev_scale + (isfinite(sink) ? expf(sink - m_new) : 0.0f);
                scale_prev             = prev_scale;
            }
            if (lane == 0) {
                sh_aux.meta.O_scale[r] = scale_prev;
                sh_aux.meta.KQ_max[r]  = KQ_max[u];
                sh_aux.meta.KQ_sum[r]  = KQ_sum[u];
            }
        }
    }

    __syncthreads();

    const bool emit_meta = (dst_meta_ptr != nullptr);

#pragma unroll
    for (int m = 0; m < M_G; ++m) {
        const int r = r_base + m;
        if (r >= M) {
            continue;
        }
        const int j = r / ncols2;
        const int c = r % ncols2;
        if (col_Q_0 + j >= int(ne01.z)) {
            continue;
        }
        float sc = sh_aux.meta.O_scale[r];
        if (!emit_meta) {
            const float sum = sh_aux.meta.KQ_sum[r];
            sc = (sum > 0.0f) ? (sc / sum) : 0.0f;
        }
        const float4 out4 = make_float4(
                O_acc[m][0] * sc, O_acc[m][1] * sc, O_acc[m][2] * sc, O_acc[m][3] * sc);
        const int j_dst_unrolled = ((sequence * int(ne01.z) + col_Q_0 + j) * ne02 + head0 + c) * gridDim.y + blockIdx.y;
        ggml_cuda_memcpy_1<16>(dst_ptr + size_t(j_dst_unrolled) * D + size_t(tid_g) * 4, &out4);
        if (emit_meta && tid_g == 0) {
            dst_meta_ptr[j_dst_unrolled] = make_float2(sh_aux.meta.KQ_max[r], sh_aux.meta.KQ_sum[r]);
        }
    }
}

template <ggml_type TKV, int D, int E>
static __device__ __forceinline__ void decode_load_vec_h2(const char * row_p, int lane, half2 (&out)[E / 2]) {
    if constexpr (TKV == GGML_TYPE_F16) {
        if constexpr (D == 256) {
            uint4 u = make_uint4(0, 0, 0, 0);
            if (row_p) { ggml_cuda_memcpy_1<16>(&u, row_p + size_t(lane) * 16); }
            const half2 * h2 = reinterpret_cast<const half2 *>(&u);
#pragma unroll
            for (int i = 0; i < 4; ++i) { out[i] = h2[i]; }
        } else {
            static_assert(D == 128);
            float2 u = make_float2(0.0f, 0.0f);
            if (row_p) { ggml_cuda_memcpy_1<8>(&u, row_p + size_t(lane) * 8); }
            const half2 * h2 = reinterpret_cast<const half2 *>(&u);
#pragma unroll
            for (int i = 0; i < 2; ++i) { out[i] = h2[i]; }
        }
    } else {
        static_assert(TKV == GGML_TYPE_Q8_0 && D % QK8_0 == 0);
        if (!row_p) {
#pragma unroll
            for (int i = 0; i < E / 2; ++i) { out[i] = make_half2(0.0f, 0.0f); }
            return;
        }
        constexpr int SUB_DIV = (D == 256) ? 4 : 8;
        const int blk = lane / SUB_DIV;
        const int sb  = lane % SUB_DIV;
        const char * bp = row_p + size_t(blk) * sizeof(block_q8_0);
        half dh;
        ggml_cuda_memcpy_1<2, 2>(&dh, bp);
        const half2 dh2 = __half2half2(dh);

#pragma unroll
        for (int p = 0; p < E / 2; ++p) {
            uint16_t q2;
            ggml_cuda_memcpy_1<2, 2>(&q2, bp + sizeof(half) + sb * E + p * 2);
            const int8_t a = int8_t(q2 & 0xff);
            const int8_t b = int8_t((q2 >> 8) & 0xff);
            out[p] = __hmul2(dh2, make_half2(float(a), float(b)));
        }
    }
}

template <int D, int ncols2, ggml_type TK, ggml_type TV, bool use_sparse, bool use_paged>
__launch_bounds__(128, 4)
static __global__ void flash_attn_qsa_hip_decode(
        const char * __restrict__ Q_ptr,
        const char * __restrict__ K_ptr,
        const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr,
        const char * __restrict__ sinks_ptr,
        const int  * __restrict__ KV_max_ptr,
        const int  * __restrict__ KV_rows_ptr,
        float      * __restrict__ dst_ptr,
        float2     * __restrict__ dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
        const int32_t nb11, const int32_t nb12, const int64_t nb13,
        const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
        const int32_t nb31, const int64_t nb32, const int64_t nb33,
        ggml_cuda_qsa_kv_view K_cache, ggml_cuda_qsa_kv_view V_cache,
        ggml_cuda_fattn_visibility visibility) {
    GGML_UNUSED_VARS(KV_rows_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31);

    constexpr int NW = 4;
    constexpr int E  = D / 32; // 8 (D=256) or 4 (D=128)
    constexpr int E2 = E / 2;

    const int lane = threadIdx.x; // 0..31
    const int warp = threadIdx.y; // 0..3

    const int col_Q     = blockIdx.x;
    const int sequence  = blockIdx.z / (ne02 / ncols2);
    const int head0     = blockIdx.z * ncols2 - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    const char * Q_base = Q_ptr + int64_t(sequence) * nb03 + int64_t(head0) * nb02 + int64_t(col_Q) * nb01;
    const char * K_data = K_ptr + int64_t(sequence) * nb13 + int64_t(head0 / gqa_ratio) * nb12;
    const char * V_data = V_ptr + int64_t(sequence) * nb23 + int64_t(head0 / gqa_ratio) * nb22;
    const char * mask_b = mask_ptr ? (mask_ptr + int64_t(sequence % ne33) * nb33 + int64_t(head0 % ne32) * nb32) : nullptr;

    const int32_t * indices = nullptr;
    int k_VKQ_max;
    if constexpr (use_sparse) {
        const int64_t list = int64_t(sequence % ne33) * gridDim.x + blockIdx.x;
        indices = KV_max_ptr + list * ne11;
        const int32_t * counts = KV_max_ptr + int64_t(ne33) * gridDim.x * ne11;
        k_VKQ_max = counts[list];
    } else {
        k_VKQ_max = KV_max_ptr ? KV_max_ptr[sequence * gridDim.x + blockIdx.x] : ne11;
    }

    const bool emit_meta = (dst_meta_ptr != nullptr);
    constexpr int NBATCH_DEC = 32;

    if constexpr (ncols2 >= 4) {
        constexpr int HPW = ncols2 / NW; // 3 (GQA=12), 2 (GQA=8), 1 (GQA=4)

        half2 q_h2[HPW][E2];
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float * q_row = reinterpret_cast<const float *>(Q_base + int64_t(warp * HPW + h) * nb02);
            if constexpr (D == 256) {
                float4 a, b;
                ggml_cuda_memcpy_1<16>(&a, q_row + lane * 8 + 0);
                ggml_cuda_memcpy_1<16>(&b, q_row + lane * 8 + 4);
                q_h2[h][0] = __float22half2_rn(make_float2(a.x * scale, a.y * scale));
                q_h2[h][1] = __float22half2_rn(make_float2(a.z * scale, a.w * scale));
                q_h2[h][2] = __float22half2_rn(make_float2(b.x * scale, b.y * scale));
                q_h2[h][3] = __float22half2_rn(make_float2(b.z * scale, b.w * scale));
            } else {
                float4 a;
                ggml_cuda_memcpy_1<16>(&a, q_row + lane * 4);
                q_h2[h][0] = __float22half2_rn(make_float2(a.x * scale, a.y * scale));
                q_h2[h][1] = __float22half2_rn(make_float2(a.z * scale, a.w * scale));
            }
        }

        float O_acc[HPW][E];
        float KQ_max[HPW];
        float KQ_sum[HPW];
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            KQ_max[h] = -FLT_MAX / 2.0f;
            KQ_sum[h] = 0.0f;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                O_acc[h][e] = 0.0f;
            }
        }

        __shared__ half2 K_s[NW][32][E2];
        __shared__ half2 V_s[NW][32][E2];
        __shared__ int   row_s[NW];

        for (int k_chunk = blockIdx.y * NBATCH_DEC; k_chunk < k_VKQ_max; k_chunk += gridDim.y * NBATCH_DEC) {
            const int chunk_end = min(k_VKQ_max, k_chunk + NBATCH_DEC);
            for (int sub = k_chunk; sub < chunk_end; sub += NW) {
                const int k_idx   = sub + warp;
                const int raw_row = (k_idx < chunk_end) ? (use_sparse ? indices[k_idx] : k_idx) : -1;
                const char * kr = nullptr;
                const char * vr = nullptr;
                qsa_resolve_kv_row_ptrs(
                        K_data, V_data, raw_row, ne11, nb11, nb21,
                        K_cache, V_cache, use_sparse, use_paged, kr, vr);

                half2 k_loc[E2];
                half2 v_loc[E2];
                decode_load_vec_h2<TK, D, E>(kr, lane, k_loc);
                decode_load_vec_h2<TV, D, E>(vr, lane, v_loc);

                if (lane == 0) {
                    row_s[warp] = (raw_row >= 0 && (use_sparse || raw_row < ne11)) ? raw_row : -1;
                }
#pragma unroll
                for (int p = 0; p < E2; ++p) {
                    K_s[warp][lane][p] = k_loc[p];
                    V_s[warp][lane][p] = v_loc[p];
                }

                __syncthreads();

#pragma unroll
                for (int u = 0; u < NW; ++u) {
                    const int row = row_s[u];
                    if (row < 0) {
                        continue;
                    }
                    half2 kv_reg[E2];
                    half2 vv_reg[E2];
#pragma unroll
                    for (int p = 0; p < E2; ++p) {
                        kv_reg[p] = K_s[u][lane][p];
                        vv_reg[p] = V_s[u][lane][p];
                    }

#pragma unroll
                    for (int h = 0; h < HPW; ++h) {
                        float dot = 0.0f;
#pragma unroll
                        for (int p = 0; p < E2; ++p) {
                            dot = qsa_fdot2(q_h2[h][p], kv_reg[p], dot);
                        }
                        dot = qsa_warp_sum_32(dot);

                        const float logit = qsa_eval_mask_logit(dot, row, col_Q, mask_b, nb31, visibility);
                        const float m_new = fmaxf(KQ_max[h], logit);
                        const float scale_prev = (KQ_sum[h] > 0.0f) ? expf(KQ_max[h] - m_new) : 0.0f;
                        const float p_f = isfinite(logit) ? expf(logit - m_new) : 0.0f;
                        KQ_max[h] = m_new;
                        KQ_sum[h] = KQ_sum[h] * scale_prev + p_f;

#pragma unroll
                        for (int p = 0; p < E2; ++p) {
                            const float2 vf = __half22float2(vv_reg[p]);
                            O_acc[h][2 * p + 0] = O_acc[h][2 * p + 0] * scale_prev + p_f * vf.x;
                            O_acc[h][2 * p + 1] = O_acc[h][2 * p + 1] * scale_prev + p_f * vf.y;
                        }
                    }
                }

                __syncthreads();
            }
        }

#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const int c = warp * HPW + h;
            if (sinks_ptr && blockIdx.y == 0) {
                const float sink = reinterpret_cast<const float *>(sinks_ptr)[head0 + c];
                const float m_new = fmaxf(KQ_max[h], sink);
                const float prev_scale = (KQ_sum[h] > 0.0f) ? expf(KQ_max[h] - m_new) : 0.0f;
                KQ_max[h] = m_new;
                KQ_sum[h] = KQ_sum[h] * prev_scale + (isfinite(sink) ? expf(sink - m_new) : 0.0f);
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    O_acc[h][e] *= prev_scale;
                }
            }
            if (!emit_meta) {
                const float inv_s = (KQ_sum[h] > 0.0f) ? (1.0f / KQ_sum[h]) : 0.0f;
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    O_acc[h][e] *= inv_s;
                }
            }

            const int j_dst_unrolled = ((sequence * int(ne01.z) + col_Q) * ne02 + head0 + c) * gridDim.y + blockIdx.y;
            float * out_row = dst_ptr + size_t(j_dst_unrolled) * D + size_t(lane) * E;
            if constexpr (D == 256) {
                const float4 o0 = make_float4(O_acc[h][0], O_acc[h][1], O_acc[h][2], O_acc[h][3]);
                const float4 o1 = make_float4(O_acc[h][4], O_acc[h][5], O_acc[h][6], O_acc[h][7]);
                ggml_cuda_memcpy_1<16>(out_row + 0, &o0);
                ggml_cuda_memcpy_1<16>(out_row + 4, &o1);
            } else {
                const float4 o0 = make_float4(O_acc[h][0], O_acc[h][1], O_acc[h][2], O_acc[h][3]);
                ggml_cuda_memcpy_1<16>(out_row, &o0);
            }
            if (emit_meta && lane == 0) {
                dst_meta_ptr[j_dst_unrolled] = make_float2(KQ_max[h], KQ_sum[h]);
            }
        }
    } else {
        // ncols2 == 1 or 2: each warp scans keys stride 4 with zero loop barriers,
        // then reduces partial (O, max, sum) across the 4 warps once at the end.
        half2 q_h2[ncols2][E2];
#pragma unroll
        for (int c = 0; c < ncols2; ++c) {
            const float * q_row = reinterpret_cast<const float *>(Q_base + int64_t(c) * nb02);
            if constexpr (D == 256) {
                float4 a, b;
                ggml_cuda_memcpy_1<16>(&a, q_row + lane * 8 + 0);
                ggml_cuda_memcpy_1<16>(&b, q_row + lane * 8 + 4);
                q_h2[c][0] = __float22half2_rn(make_float2(a.x * scale, a.y * scale));
                q_h2[c][1] = __float22half2_rn(make_float2(a.z * scale, a.w * scale));
                q_h2[c][2] = __float22half2_rn(make_float2(b.x * scale, b.y * scale));
                q_h2[c][3] = __float22half2_rn(make_float2(b.z * scale, b.w * scale));
            } else {
                float4 a;
                ggml_cuda_memcpy_1<16>(&a, q_row + lane * 4);
                q_h2[c][0] = __float22half2_rn(make_float2(a.x * scale, a.y * scale));
                q_h2[c][1] = __float22half2_rn(make_float2(a.z * scale, a.w * scale));
            }
        }

        float O_acc[ncols2][E];
        float KQ_max[ncols2];
        float KQ_sum[ncols2];
#pragma unroll
        for (int c = 0; c < ncols2; ++c) {
            KQ_max[c] = -FLT_MAX / 2.0f;
            KQ_sum[c] = 0.0f;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                O_acc[c][e] = 0.0f;
            }
        }

        for (int k_chunk = blockIdx.y * NBATCH_DEC; k_chunk < k_VKQ_max; k_chunk += gridDim.y * NBATCH_DEC) {
            const int chunk_end = min(k_VKQ_max, k_chunk + NBATCH_DEC);
            for (int k_idx = k_chunk + warp; k_idx < chunk_end; k_idx += NW) {
                const int row = use_sparse ? indices[k_idx] : k_idx;
                const char * kr = nullptr;
                const char * vr = nullptr;
                qsa_resolve_kv_row_ptrs(
                        K_data, V_data, row, ne11, nb11, nb21,
                        K_cache, V_cache, use_sparse, use_paged, kr, vr);
                if (!kr) {
                    continue;
                }

                half2 kv_reg[E2];
                half2 vv_reg[E2];
                decode_load_vec_h2<TK, D, E>(kr, lane, kv_reg);
                decode_load_vec_h2<TV, D, E>(vr, lane, vv_reg);

#pragma unroll
                for (int c = 0; c < ncols2; ++c) {
                    float dot = 0.0f;
#pragma unroll
                    for (int p = 0; p < E2; ++p) {
                        dot = qsa_fdot2(q_h2[c][p], kv_reg[p], dot);
                    }
                    dot = qsa_warp_sum_32(dot);

                    const float logit = qsa_eval_mask_logit(dot, row, col_Q, mask_b, nb31, visibility);
                    const float m_new = fmaxf(KQ_max[c], logit);
                    const float scale_prev = (KQ_sum[c] > 0.0f) ? expf(KQ_max[c] - m_new) : 0.0f;
                    const float p_f = isfinite(logit) ? expf(logit - m_new) : 0.0f;
                    KQ_max[c] = m_new;
                    KQ_sum[c] = KQ_sum[c] * scale_prev + p_f;

#pragma unroll
                    for (int p = 0; p < E2; ++p) {
                        const float2 vf = __half22float2(vv_reg[p]);
                        O_acc[c][2 * p + 0] = O_acc[c][2 * p + 0] * scale_prev + p_f * vf.x;
                        O_acc[c][2 * p + 1] = O_acc[c][2 * p + 1] * scale_prev + p_f * vf.y;
                    }
                }
            }
        }

        __shared__ float  warp_O[NW][ncols2][D];
        __shared__ float2 warp_meta[NW][ncols2];

#pragma unroll
        for (int c = 0; c < ncols2; ++c) {
            if (lane == 0) {
                warp_meta[warp][c] = make_float2(KQ_max[c], KQ_sum[c]);
            }
#pragma unroll
            for (int e = 0; e < E; ++e) {
                warp_O[warp][c][lane * E + e] = O_acc[c][e];
            }
        }
        __syncthreads();

        if (warp == 0) {
#pragma unroll
            for (int c = 0; c < ncols2; ++c) {
                float m_all = warp_meta[0][c].x;
#pragma unroll
                for (int w = 1; w < NW; ++w) {
                    m_all = fmaxf(m_all, warp_meta[w][c].x);
                }
                if (sinks_ptr && blockIdx.y == 0) {
                    m_all = fmaxf(m_all, reinterpret_cast<const float *>(sinks_ptr)[head0 + c]);
                }

                float sum_all = 0.0f;
                float w_scales[NW];
#pragma unroll
                for (int w = 0; w < NW; ++w) {
                    const float2 wm = warp_meta[w][c];
                    const float sc = (wm.y > 0.0f) ? expf(wm.x - m_all) : 0.0f;
                    w_scales[w] = sc;
                    sum_all += wm.y * sc;
                }
                if (sinks_ptr && blockIdx.y == 0) {
                    const float sink = reinterpret_cast<const float *>(sinks_ptr)[head0 + c];
                    sum_all += isfinite(sink) ? expf(sink - m_all) : 0.0f;
                }

                const int j_dst_unrolled = ((sequence * int(ne01.z) + col_Q) * ne02 + head0 + c) * gridDim.y + blockIdx.y;
                const float inv_s = !emit_meta ? (sum_all > 0.0f ? 1.0f / sum_all : 0.0f) : 1.0f;

                float out_vals[E];
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    float acc = 0.0f;
                    const int d = lane * E + e;
#pragma unroll
                    for (int w = 0; w < NW; ++w) {
                        acc += warp_O[w][c][d] * w_scales[w];
                    }
                    out_vals[e] = acc * inv_s;
                }

                float * out_row = dst_ptr + size_t(j_dst_unrolled) * D + size_t(lane) * E;
                if constexpr (D == 256) {
                    const float4 o0 = make_float4(out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
                    const float4 o1 = make_float4(out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
                    ggml_cuda_memcpy_1<16>(out_row + 0, &o0);
                    ggml_cuda_memcpy_1<16>(out_row + 4, &o1);
                } else {
                    const float4 o0 = make_float4(out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
                    ggml_cuda_memcpy_1<16>(out_row, &o0);
                }
                if (emit_meta && lane == 0) {
                    dst_meta_ptr[j_dst_unrolled] = make_float2(m_all, sum_all);
                }
            }
        }
    }
}

template <int D, int ncols1, int ncols2, ggml_type TK, ggml_type TV, bool use_sparse, bool use_paged>
static void dispatch_prefill(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        bool need_f16_K, bool need_f16_V) {
    constexpr int nwarps = D / 32;
    launch_fattn<D, ncols1, ncols2>(
            ctx, dst,
            flash_attn_qsa_hip_prefill<D, ncols1, ncols2, TK, TV, use_sparse, use_paged>,
            nwarps, 0, 32, need_f16_K, need_f16_V, false, use_sparse, 32, use_paged);
}

template <int D, int ncols2, ggml_type TK, ggml_type TV, bool use_sparse, bool use_paged>
static void dispatch_decode(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        bool need_f16_K, bool need_f16_V) {
    launch_fattn<D, 1, ncols2>(
            ctx, dst,
            flash_attn_qsa_hip_decode<D, ncols2, TK, TV, use_sparse, use_paged>,
            4, 0, 32, need_f16_K, need_f16_V, false, use_sparse, 32, use_paged);
}

template <int D, int ncols2, ggml_type TK, ggml_type TV>
static void launch_mode(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        bool use_prefill, int ncols1_prefill, bool use_sparse, bool use_paged,
        bool need_f16_K, bool need_f16_V) {
    if (!use_prefill) {
        if (use_sparse) {
            if (use_paged) { dispatch_decode<D, ncols2, TK, TV, true,  true >(ctx, dst, need_f16_K, need_f16_V); }
            else           { dispatch_decode<D, ncols2, TK, TV, true,  false>(ctx, dst, need_f16_K, need_f16_V); }
        } else {
            if (use_paged) { dispatch_decode<D, ncols2, TK, TV, false, true >(ctx, dst, need_f16_K, need_f16_V); }
            else           { dispatch_decode<D, ncols2, TK, TV, false, false>(ctx, dst, need_f16_K, need_f16_V); }
        }
        return;
    }

    auto go_c1 = [&](auto c1_tag) {
        constexpr int c1 = decltype(c1_tag)::value;
        if (use_sparse) {
            if (use_paged) { dispatch_prefill<D, c1, ncols2, TK, TV, true,  true >(ctx, dst, need_f16_K, need_f16_V); }
            else           { dispatch_prefill<D, c1, ncols2, TK, TV, true,  false>(ctx, dst, need_f16_K, need_f16_V); }
        } else {
            if (use_paged) { dispatch_prefill<D, c1, ncols2, TK, TV, false, true >(ctx, dst, need_f16_K, need_f16_V); }
            else           { dispatch_prefill<D, c1, ncols2, TK, TV, false, false>(ctx, dst, need_f16_K, need_f16_V); }
        }
    };

    if constexpr (ncols2 == 12 || ncols2 == 8) {
        GGML_UNUSED(ncols1_prefill);
        go_c1(std::integral_constant<int, 2>{});
    } else if constexpr (ncols2 == 4) {
        if (ncols1_prefill == 4) { go_c1(std::integral_constant<int, 4>{}); }
        else                     { go_c1(std::integral_constant<int, 2>{}); }
    } else {
        if (ncols1_prefill == 8) { go_c1(std::integral_constant<int, 8>{}); }
        else                     { go_c1(std::integral_constant<int, 2>{}); }
    }
}

template <int D, int ncols2>
static void launch_types(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        ggml_type eff_k, ggml_type eff_v,
        bool use_prefill, int ncols1_prefill, bool use_sparse, bool use_paged,
        bool need_f16_K, bool need_f16_V) {
    if (eff_k == GGML_TYPE_F16 && eff_v == GGML_TYPE_F16) {
        launch_mode<D, ncols2, GGML_TYPE_F16, GGML_TYPE_F16>(
                ctx, dst, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V);
    } else if (eff_k == GGML_TYPE_Q8_0 && eff_v == GGML_TYPE_Q8_0) {
        launch_mode<D, ncols2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(
                ctx, dst, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V);
    } else if (eff_k == GGML_TYPE_Q8_0 && eff_v == GGML_TYPE_F16) {
        launch_mode<D, ncols2, GGML_TYPE_Q8_0, GGML_TYPE_F16>(
                ctx, dst, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V);
    } else {
        GGML_ASSERT(eff_k == GGML_TYPE_F16 && eff_v == GGML_TYPE_Q8_0);
        launch_mode<D, ncols2, GGML_TYPE_F16, GGML_TYPE_Q8_0>(
                ctx, dst, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V);
    }
}

template <int D>
static void launch_ncols2(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        int ncols2, ggml_type eff_k, ggml_type eff_v,
        bool use_prefill, int ncols1_prefill, bool use_sparse, bool use_paged,
        bool need_f16_K, bool need_f16_V) {
    switch (ncols2) {
        case 12: launch_types<D, 12>(ctx, dst, eff_k, eff_v, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V); break;
        case  8: launch_types<D,  8>(ctx, dst, eff_k, eff_v, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V); break;
        case  4: launch_types<D,  4>(ctx, dst, eff_k, eff_v, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V); break;
        case  2: launch_types<D,  2>(ctx, dst, eff_k, eff_v, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V); break;
        case  1: launch_types<D,  1>(ctx, dst, eff_k, eff_v, use_prefill, ncols1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V); break;
        default: GGML_ABORT("unsupported QSA HIP ncols2");
    }
}

static bool qsa_hip_supported_types(ggml_type t) {
    return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0;
}

static int qsa_hip_select_ncols2(const ggml_tensor * dst, float max_bias) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (max_bias != 0.0f || (mask && mask->ne[2] != 1)) {
        return 1;
    }
    if (gqa_ratio % 12 == 0) { return 12; }
    if (gqa_ratio %  8 == 0) { return  8; }
    if (gqa_ratio %  4 == 0) { return  4; }
    if (gqa_ratio %  2 == 0) { return  2; }
    return 1;
}

}  // namespace

bool ggml_cuda_flash_attn_ext_qsa_hip_supported(const ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(dst);
    return false;
#else
    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    if (warp_size != 32 || !GGML_CUDA_CC_IS_AMD(cc) || cc < GGML_CUDA_CC_RDNA2 || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (dst->src[7] != nullptr || K->ne[0] != V->ne[0] || (K->ne[0] != 256 && K->ne[0] != 128)) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (mask && mask->ne[2] != 1 && mask->ne[2] != Q->ne[2]) {
        return false;
    }

    const bool use_paged = ggml_cuda_fattn_use_paged(dst);
    const bool staged    = ggml_cuda_qsa_kv_is_paged(K) && !use_paged;
    const bool dense_m   = ggml_cuda_fattn_dense_mask(dst);

    if (use_paged) {
        return (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) ||
               (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0);
    }

    const int ncols2 = qsa_hip_select_ncols2(dst, max_bias);
    const int full_c1 = (ncols2 == 12 || ncols2 == 8) ? 2 : (ncols2 == 4 ? 4 : 8);
    const int sparse_c1 = (dst->src[5] == nullptr && ggml_cuda_flash_attn_ext_tile_shall_use_sparse(dst, 2)) ? 2 : full_c1;
    const bool use_sparse = (Q->ne[1] <= 8)
        ? ggml_cuda_flash_attn_ext_tile_shall_use_sparse(dst, 1)
        : ggml_cuda_flash_attn_ext_tile_shall_use_sparse(dst, sparse_c1);

    if (use_sparse) {
        return qsa_hip_supported_types(K->type) && qsa_hip_supported_types(V->type);
    }

    // Dense path: Q8_0/Q8_0 is read natively unless dense_mask falls back to F16;
    // avoid staged + Q8_0 + dense_mask assert by letting flash_attn_tile handle it.
    if (staged && dense_m && (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16)) {
        return false;
    }
    return true;
#endif
}

void ggml_cuda_flash_attn_ext_qsa_hip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const int  ncols2      = qsa_hip_select_ncols2(dst, max_bias);
    const bool use_prefill = Q->ne[1] >= 9;
    const int  full_c1     = (ncols2 == 12 || ncols2 == 8) ? 2 : (ncols2 == 4 ? 4 : 8);
    const int  c1_prefill  = !use_prefill ? 1 :
        ((dst->src[5] == nullptr && ggml_cuda_flash_attn_ext_tile_shall_use_sparse(dst, 2)) ? 2 : full_c1);

    const bool use_paged  = ggml_cuda_fattn_use_paged(dst);
    const bool use_sparse = ggml_cuda_flash_attn_ext_tile_shall_use_sparse(dst, use_prefill ? c1_prefill : 1);
    const bool dense_m    = ggml_cuda_fattn_dense_mask(dst);

    const bool q8_pair = K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0;
    const bool need_f16_K = !(use_sparse || (use_paged && q8_pair) || (q8_pair && !dense_m));
    const bool need_f16_V = need_f16_K;

    const ggml_type eff_k = need_f16_K ? GGML_TYPE_F16 : K->type;
    const ggml_type eff_v = need_f16_V ? GGML_TYPE_F16 : V->type;

    if (K->ne[0] == 256) {
        launch_ncols2<256>(ctx, dst, ncols2, eff_k, eff_v, use_prefill, c1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V);
    } else {
        launch_ncols2<128>(ctx, dst, ncols2, eff_k, eff_v, use_prefill, c1_prefill, use_sparse, use_paged, need_f16_K, need_f16_V);
    }
}
