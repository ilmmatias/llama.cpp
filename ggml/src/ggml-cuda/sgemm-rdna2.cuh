#pragma once

#include "common.cuh"

// Vectorized float4 TN SGEMM (64x64x16 with deterministic Split-K) for RDNA2
// where rocBLAS selects scalar MT64x64x8 GLVWA1 on small-tile / tall-K shapes.
template <int SPLIT_K>
static __global__ void __launch_bounds__(64, 4)
sgemm_tn_64x64x16_rdna2(
        const float * __restrict__ A,
        const float * __restrict__ B,
        float       * __restrict__ C_out,
        const int M,
        const int N,
        const int K,
        const int lda,
        const int ldb,
        const int ldc) {
    constexpr int KT = 16;
    __shared__ alignas(16) float sA[KT][68];
    __shared__ alignas(16) float sB[KT][68];

    const int tid = threadIdx.x;
    const int tx  = tid & 7;
    const int ty  = tid >> 3;

    const int bm = blockIdx.x * 64;
    const int bn = blockIdx.y * 64;
    const int sk = (SPLIT_K > 1) ? blockIdx.z : 0;

    const int k_tiles_total = (K + KT - 1) / KT;
    const int k_tiles_per   = (k_tiles_total + SPLIT_K - 1) / SPLIT_K;
    const int k0_start      = sk * k_tiles_per * KT;
    const int k0_end        = min(K, (sk + 1) * k_tiles_per * KT);

    if (k0_start >= k0_end) {
        if constexpr (SPLIT_K > 1) {
            float * base = C_out + (size_t) sk * (size_t) N * (size_t) M;
            for (int idx = tid; idx < 64 * 64; idx += 64) {
                const int lm = idx & 63;
                const int ln = idx >> 6;
                if (bm + lm < M && bn + ln < N) {
                    base[(size_t) (bn + ln) * M + bm + lm] = 0.0f;
                }
            }
        }
        return;
    }

    float acc00[4] = {}, acc01[4] = {}, acc02[4] = {}, acc03[4] = {};
    float acc10[4] = {}, acc11[4] = {}, acc12[4] = {}, acc13[4] = {};
    float acc20[4] = {}, acc21[4] = {}, acc22[4] = {}, acc23[4] = {};
    float acc30[4] = {}, acc31[4] = {}, acc32[4] = {}, acc33[4] = {};

    const int l_row = tid >> 2;
    const int l_k4  = (tid & 3) * 4;

    for (int k0 = k0_start; k0 < k0_end; k0 += KT) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int r  = step * 16 + l_row;
            const int gm = bm + r;
            const int gn = bn + r;
            float4 va = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 vb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gm < M && k0 + l_k4 + 3 < k0_end) {
                va = *reinterpret_cast<const float4 *>(&A[(size_t) gm * lda + k0 + l_k4]);
            } else if (gm < M) {
                va.x = (k0 + l_k4 + 0 < k0_end) ? A[(size_t) gm * lda + k0 + l_k4 + 0] : 0.0f;
                va.y = (k0 + l_k4 + 1 < k0_end) ? A[(size_t) gm * lda + k0 + l_k4 + 1] : 0.0f;
                va.z = (k0 + l_k4 + 2 < k0_end) ? A[(size_t) gm * lda + k0 + l_k4 + 2] : 0.0f;
                va.w = (k0 + l_k4 + 3 < k0_end) ? A[(size_t) gm * lda + k0 + l_k4 + 3] : 0.0f;
            }
            if (gn < N && k0 + l_k4 + 3 < k0_end) {
                vb = *reinterpret_cast<const float4 *>(&B[(size_t) gn * ldb + k0 + l_k4]);
            } else if (gn < N) {
                vb.x = (k0 + l_k4 + 0 < k0_end) ? B[(size_t) gn * ldb + k0 + l_k4 + 0] : 0.0f;
                vb.y = (k0 + l_k4 + 1 < k0_end) ? B[(size_t) gn * ldb + k0 + l_k4 + 1] : 0.0f;
                vb.z = (k0 + l_k4 + 2 < k0_end) ? B[(size_t) gn * ldb + k0 + l_k4 + 2] : 0.0f;
                vb.w = (k0 + l_k4 + 3 < k0_end) ? B[(size_t) gn * ldb + k0 + l_k4 + 3] : 0.0f;
            }
            sA[l_k4 + 0][r] = va.x; sA[l_k4 + 1][r] = va.y; sA[l_k4 + 2][r] = va.z; sA[l_k4 + 3][r] = va.w;
            sB[l_k4 + 0][r] = vb.x; sB[l_k4 + 1][r] = vb.y; sB[l_k4 + 2][r] = vb.z; sB[l_k4 + 3][r] = vb.w;
        }
        __syncthreads();

        const int m_off = tx * 4;
        const int n_off = ty * 4;
#pragma unroll
        for (int k = 0; k < KT; ++k) {
            const float4 a0 = *reinterpret_cast<const float4 *>(&sA[k][m_off]);
            const float4 a1 = *reinterpret_cast<const float4 *>(&sA[k][m_off + 32]);
            const float4 b0 = *reinterpret_cast<const float4 *>(&sB[k][n_off]);
            const float4 b1 = *reinterpret_cast<const float4 *>(&sB[k][n_off + 32]);

            acc00[0] = fmaf(a0.x, b0.x, acc00[0]); acc00[1] = fmaf(a0.y, b0.x, acc00[1]); acc00[2] = fmaf(a0.z, b0.x, acc00[2]); acc00[3] = fmaf(a0.w, b0.x, acc00[3]);
            acc01[0] = fmaf(a0.x, b0.y, acc01[0]); acc01[1] = fmaf(a0.y, b0.y, acc01[1]); acc01[2] = fmaf(a0.z, b0.y, acc01[2]); acc01[3] = fmaf(a0.w, b0.y, acc01[3]);
            acc02[0] = fmaf(a0.x, b0.z, acc02[0]); acc02[1] = fmaf(a0.y, b0.z, acc02[1]); acc02[2] = fmaf(a0.z, b0.z, acc02[2]); acc02[3] = fmaf(a0.w, b0.z, acc02[3]);
            acc03[0] = fmaf(a0.x, b0.w, acc03[0]); acc03[1] = fmaf(a0.y, b0.w, acc03[1]); acc03[2] = fmaf(a0.z, b0.w, acc03[2]); acc03[3] = fmaf(a0.w, b0.w, acc03[3]);

            acc10[0] = fmaf(a1.x, b0.x, acc10[0]); acc10[1] = fmaf(a1.y, b0.x, acc10[1]); acc10[2] = fmaf(a1.z, b0.x, acc10[2]); acc10[3] = fmaf(a1.w, b0.x, acc10[3]);
            acc11[0] = fmaf(a1.x, b0.y, acc11[0]); acc11[1] = fmaf(a1.y, b0.y, acc11[1]); acc11[2] = fmaf(a1.z, b0.y, acc11[2]); acc11[3] = fmaf(a1.w, b0.y, acc11[3]);
            acc12[0] = fmaf(a1.x, b0.z, acc12[0]); acc12[1] = fmaf(a1.y, b0.z, acc12[1]); acc12[2] = fmaf(a1.z, b0.z, acc12[2]); acc12[3] = fmaf(a1.w, b0.z, acc12[3]);
            acc13[0] = fmaf(a1.x, b0.w, acc13[0]); acc13[1] = fmaf(a1.y, b0.w, acc13[1]); acc13[2] = fmaf(a1.z, b0.w, acc13[2]); acc13[3] = fmaf(a1.w, b0.w, acc13[3]);

            acc20[0] = fmaf(a0.x, b1.x, acc20[0]); acc20[1] = fmaf(a0.y, b1.x, acc20[1]); acc20[2] = fmaf(a0.z, b1.x, acc20[2]); acc20[3] = fmaf(a0.w, b1.x, acc20[3]);
            acc21[0] = fmaf(a0.x, b1.y, acc21[0]); acc21[1] = fmaf(a0.y, b1.y, acc21[1]); acc21[2] = fmaf(a0.z, b1.y, acc21[2]); acc21[3] = fmaf(a0.w, b1.y, acc21[3]);
            acc22[0] = fmaf(a0.x, b1.z, acc22[0]); acc22[1] = fmaf(a0.y, b1.z, acc22[1]); acc22[2] = fmaf(a0.z, b1.z, acc22[2]); acc22[3] = fmaf(a0.w, b1.z, acc22[3]);
            acc23[0] = fmaf(a0.x, b1.w, acc23[0]); acc23[1] = fmaf(a0.y, b1.w, acc23[1]); acc23[2] = fmaf(a0.z, b1.w, acc23[2]); acc23[3] = fmaf(a0.w, b1.w, acc23[3]);

            acc30[0] = fmaf(a1.x, b1.x, acc30[0]); acc30[1] = fmaf(a1.y, b1.x, acc30[1]); acc30[2] = fmaf(a1.z, b1.x, acc30[2]); acc30[3] = fmaf(a1.w, b1.x, acc30[3]);
            acc31[0] = fmaf(a1.x, b1.y, acc31[0]); acc31[1] = fmaf(a1.y, b1.y, acc31[1]); acc31[2] = fmaf(a1.z, b1.y, acc31[2]); acc31[3] = fmaf(a1.w, b1.y, acc31[3]);
            acc32[0] = fmaf(a1.x, b1.z, acc32[0]); acc32[1] = fmaf(a1.y, b1.z, acc32[1]); acc32[2] = fmaf(a1.z, b1.z, acc32[2]); acc32[3] = fmaf(a1.w, b1.z, acc32[3]);
            acc33[0] = fmaf(a1.x, b1.w, acc33[0]); acc33[1] = fmaf(a1.y, b1.w, acc33[1]); acc33[2] = fmaf(a1.z, b1.w, acc33[2]); acc33[3] = fmaf(a1.w, b1.w, acc33[3]);
        }
        __syncthreads();
    }

    const int out_ld = (SPLIT_K > 1) ? M : ldc;
    float * base     = C_out + (size_t) sk * (size_t) N * (size_t) M;
    auto store4 = [&](int gm, int gn, const float v[4]) {
        if (gn < N && gm + 3 < M) {
            *reinterpret_cast<float4 *>(&base[(size_t) gn * out_ld + gm]) = make_float4(v[0], v[1], v[2], v[3]);
        } else if (gn < N) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (gm + r < M) {
                    base[(size_t) gn * out_ld + gm + r] = v[r];
                }
            }
        }
    };
    const int gm0 = bm + tx * 4;
    const int gm1 = gm0 + 32;
    const int gn0 = bn + ty * 4;
    const int gn1 = gn0 + 32;
    store4(gm0, gn0 + 0, acc00); store4(gm0, gn0 + 1, acc01); store4(gm0, gn0 + 2, acc02); store4(gm0, gn0 + 3, acc03);
    store4(gm1, gn0 + 0, acc10); store4(gm1, gn0 + 1, acc11); store4(gm1, gn0 + 2, acc12); store4(gm1, gn0 + 3, acc13);
    store4(gm0, gn1 + 0, acc20); store4(gm0, gn1 + 1, acc21); store4(gm0, gn1 + 2, acc22); store4(gm0, gn1 + 3, acc23);
    store4(gm1, gn1 + 0, acc30); store4(gm1, gn1 + 1, acc31); store4(gm1, gn1 + 2, acc32); store4(gm1, gn1 + 3, acc33);
}

template <int SPLIT_K>
static __global__ void sgemm_tn_reduce_split_k_f32(
        const float * __restrict__ tmp,
        float       * __restrict__ C,
        const int M,
        const int N,
        const int ldc) {
    const int m4 = ((int) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int n  = blockIdx.y;
    if (n >= N || m4 >= M) {
        return;
    }
    const size_t plane = (size_t) N * (size_t) M;
    if (m4 + 3 < M) {
        float4 s = *reinterpret_cast<const float4 *>(&tmp[(size_t) n * M + m4]);
#pragma unroll
        for (int k = 1; k < SPLIT_K; ++k) {
            const float4 p = *reinterpret_cast<const float4 *>(&tmp[(size_t) k * plane + (size_t) n * M + m4]);
            s.x += p.x; s.y += p.y; s.z += p.z; s.w += p.w;
        }
        *reinterpret_cast<float4 *>(&C[(size_t) n * ldc + m4]) = s;
    } else {
        for (int r = 0; r < 4 && m4 + r < M; ++r) {
            float s = tmp[(size_t) n * M + m4 + r];
#pragma unroll
            for (int k = 1; k < SPLIT_K; ++k) {
                s += tmp[(size_t) k * plane + (size_t) n * M + m4 + r];
            }
            C[(size_t) n * ldc + m4 + r] = s;
        }
    }
}

static bool ggml_cuda_mul_mat_f32_rdna2(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_RDNA2(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }

    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const int64_t K = src1->ne[0];
    if (N <= 8 || K < 512 || M > INT32_MAX || N > INT32_MAX || K > INT32_MAX) {
        return false;
    }

    const int64_t ntiles = ((M + 63) / 64) * ((N + 63) / 64);
    if (ntiles > 64) {
        return false;
    }

    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] < (size_t) K * sizeof(float) || src1->nb[1] < (size_t) K * sizeof(float) || dst->nb[1] < (size_t) M * sizeof(float)) {
        return false;
    }
    if (((uintptr_t) src0->data & 15) != 0 || ((uintptr_t) src1->data & 15) != 0 || ((uintptr_t) dst->data & 15) != 0) {
        return false;
    }
    if ((src0->nb[1] & 15) != 0 || (src1->nb[1] & 15) != 0 || (dst->nb[1] & 15) != 0) {
        return false;
    }

    const int lda = (int) (src0->nb[1] / sizeof(float));
    const int ldb = (int) (src1->nb[1] / sizeof(float));
    const int ldc = (int) (dst->nb[1] / sizeof(float));

    const float * A = (const float *) src0->data;
    const float * B = (const float *) src1->data;
    float       * C = (float *)       dst->data;

    cudaStream_t stream = ctx.stream();
    const bool use_s8 = (ntiles <= 32 && K >= 1024);
    const int SK = use_s8 ? 8 : 4;

    ggml_cuda_pool_alloc<float> split_tmp(ctx.pool(), (size_t) SK * (size_t) M * (size_t) N);
    const dim3 grid((unsigned) ((M + 63) / 64), (unsigned) ((N + 63) / 64), (unsigned) SK);
    const dim3 rgrid((unsigned) (((M + 3) / 4 + 63) / 64), (unsigned) N, 1);

    if (use_s8) {
        sgemm_tn_64x64x16_rdna2<8><<<grid, 64, 0, stream>>>(A, B, split_tmp.get(), (int) M, (int) N, (int) K, lda, ldb, ldc);
        sgemm_tn_reduce_split_k_f32<8><<<rgrid, 64, 0, stream>>>(split_tmp.get(), C, (int) M, (int) N, ldc);
    } else {
        sgemm_tn_64x64x16_rdna2<4><<<grid, 64, 0, stream>>>(A, B, split_tmp.get(), (int) M, (int) N, (int) K, lda, ldb, ldc);
        sgemm_tn_reduce_split_k_f32<4><<<rgrid, 64, 0, stream>>>(split_tmp.get(), C, (int) M, (int) N, ldc);
    }
    return true;
}
