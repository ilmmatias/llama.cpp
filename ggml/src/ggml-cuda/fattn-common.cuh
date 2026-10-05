#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"
#include "qsa-kv.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

struct ggml_cuda_fattn_visibility {
    const int32_t * positions = nullptr;
    int32_t n_kv = 0;
    int32_t row_offset = 0;
    const int32_t * row_map = nullptr;
};

// Bound temporary K/V transfer storage independently of the resident page window.
static inline size_t ggml_cuda_qsa_staging_bytes() {
    static const size_t bytes = []() {
        const char * env = getenv("GGML_CUDA_QSA_STAGING_MIB");
        return (env ? std::max(1, atoi(env)) : 32)*1024*1024;
    }();

    return bytes;
}

// A split kernel emits unnormalized sums only when gridDim.y is greater than one.
static constexpr int GGML_CUDA_QSA_STAGING_PARTS = 2;

static constexpr size_t GGML_CUDA_QSA_MASK_BYTES = 32 * 1024 * 1024;
static constexpr int GGML_CUDA_QSA_MASK_QUERY_PAD = 64;

static inline bool ggml_cuda_fattn_dense_mask(const ggml_tensor * dst, const ggml_tensor * selected = nullptr) {
    const ggml_tensor * Q        = dst->src[0];
    const ggml_tensor * K        = dst->src[1];
    if (!selected) {
        selected = dst->src[5];
    }

    if (Q->ne[1] <= 8 || Q->ne[3] != 1) {
        return false;
    }

#ifdef GGML_USE_HIP
    if (selected &&
            K->type == GGML_TYPE_Q8_0 && dst->src[2]->type == GGML_TYPE_Q8_0 &&
            ggml_cuda_qsa_kv_is_paged(K) && ggml_cuda_qsa_kv_is_paged(dst->src[2])) {
        return false;
    }
#endif

    if (!selected) {
        const size_t row_bytes = K->ne[1] * sizeof(half);
        return dst->src[6] && GGML_PAD(Q->ne[1], GGML_CUDA_QSA_MASK_QUERY_PAD) <= GGML_CUDA_QSA_MASK_BYTES / row_bytes;
    }

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    const bool use_gqa = gqa_ratio % 2 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;
    const int ncols1 = use_gqa ? 1 : 2;
    const int64_t sparse_min_rows = std::max<int64_t>(1024, 2 * ncols1 * selected->ne[0]);
    const size_t minimum_mask_row_bytes = GGML_CUDA_QSA_MASK_QUERY_PAD * sizeof(half);
    const bool mask_fits = K->ne[1] <= GGML_CUDA_QSA_MASK_BYTES / minimum_mask_row_bytes;

    return K->ne[1] < sparse_min_rows && mask_fits;
}

static inline int64_t ggml_cuda_fattn_mask_queries(const ggml_tensor * dst) {
    const size_t row_bytes = dst->src[1]->ne[1] * sizeof(half);
    const int64_t max_queries = GGML_CUDA_QSA_MASK_BYTES / row_bytes;
    const int64_t aligned_queries = max_queries / GGML_CUDA_QSA_MASK_QUERY_PAD * GGML_CUDA_QSA_MASK_QUERY_PAD;

    return std::min<int64_t>(dst->src[0]->ne[1], aligned_queries);
}

static inline bool ggml_cuda_fattn_use_paged(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (Q->ne[3] != 1 || K->ne[0] != 256 || V->ne[0] != 256 ||
            (mask && mask->ne[3] != 1) ||
            !ggml_cuda_qsa_kv_is_paged(K) || !ggml_cuda_qsa_kv_is_paged(V) ||
            ggml_cuda_fattn_dense_mask(dst)) {
        return false;
    }

    if (Q->ne[1] <= 8) {
        return true;
    }

    return ggml_cuda_qsa_kv_fits(K) && ggml_cuda_qsa_kv_fits(V);
}

static inline size_t ggml_cuda_fattn_row_bytes(const ggml_tensor * tensor) {
    return tensor->ne[0]*ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
}

static inline int64_t ggml_cuda_qsa_staging_rows(const ggml_tensor * K, const ggml_tensor * V) {
    const size_t row_bytes = ggml_cuda_fattn_row_bytes(K)*K->ne[2]*K->ne[3] + ggml_cuda_fattn_row_bytes(V)*V->ne[2]*V->ne[3];
    const int64_t rows = ggml_cuda_qsa_staging_bytes()/row_bytes;
    const int64_t aligned = std::max<int64_t>(FATTN_KQ_STRIDE, rows/FATTN_KQ_STRIDE*FATTN_KQ_STRIDE);

    return std::min<int64_t>(K->ne[1], aligned);
}

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
                            ggml_cuda_qsa_kv_view K_cache, ggml_cuda_qsa_kv_view V_cache,
                            ggml_cuda_fattn_visibility visibility);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t K_staging;
    uintptr_t V_staging;
    uintptr_t parts;
    uintptr_t parts_meta;
    uintptr_t stream_meta;
    uintptr_t indices;
    uintptr_t staging_needed;
    uintptr_t staging_row_to_slot;
    uintptr_t staging_slot_to_row;
    uintptr_t selection_mask;
    uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_extra_data ggml_cuda_flash_attn_ext_get_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V, uintptr_t workspace_begin = 0) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ggml_cuda_flash_attn_ext_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);
    if (workspace_begin) {
        data.end = workspace_begin;
    }

    const bool dense_mask = ggml_cuda_fattn_dense_mask(dst);
    const int64_t n_queries = dense_mask ? ggml_cuda_fattn_mask_queries(dst) : dst->src[0]->ne[1];
    if (dense_mask) {
        data.end = GGML_PAD(data.end, 128);
        data.selection_mask = data.end;
        data.end += K->ne[1] * GGML_PAD(n_queries, GGML_CUDA_QSA_MASK_QUERY_PAD) * sizeof(half);
    }

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    const ggml_tensor * Q = dst->src[0];
    const bool use_paged = !workspace_begin && ggml_cuda_fattn_use_paged(dst);

    if (!use_paged) {
        const int64_t staging_rows = ggml_cuda_qsa_staging_rows(K, V);

        if (ggml_cuda_qsa_kv_is_paged(K)) {
            data.end = GGML_PAD(data.end, 128);
            data.K_staging = data.end;
            data.end += staging_rows*
                ggml_cuda_fattn_row_bytes(K)*K->ne[2]*K->ne[3];
        }

        if (ggml_cuda_qsa_kv_is_paged(V)) {
            data.end = GGML_PAD(data.end, 128);
            data.V_staging = data.end;
            data.end += staging_rows*
                ggml_cuda_fattn_row_bytes(V)*V->ne[2]*V->ne[3];
        }

        if ((data.K_staging || data.V_staging) && staging_rows < K->ne[1]) {
            data.end = GGML_PAD(data.end, 128);
            data.parts = data.end;
            data.end += GGML_CUDA_QSA_STAGING_PARTS * ggml_nbytes(dst) / Q->ne[1] * n_queries;

            data.parts_meta = data.end;
            data.end += GGML_CUDA_QSA_STAGING_PARTS * ggml_nrows(dst) / Q->ne[1] * n_queries * sizeof(float2);

            data.stream_meta = data.end;
            data.end += ggml_nrows(dst) / Q->ne[1] * n_queries * sizeof(float2);

            const int32_t max_selected = ggml_get_op_params_i32(dst, 4);
            if (max_selected > 0 && !dense_mask) {
                const int union_queries = dst->src[5] ? 1 : 2;
                const int64_t width = std::min<int64_t>(K->ne[1], int64_t(union_queries) * max_selected);
                const int64_t n_streams = dst->src[3] ? dst->src[3]->ne[3] : 1;

                data.indices = data.end;
                data.end += (width + 1)*Q->ne[1]*n_streams*sizeof(int32_t);

                data.end = GGML_PAD(data.end, 128);
                data.staging_needed = data.end;
                data.end += ((staging_rows + 31)/32)*sizeof(uint32_t);

                data.end = GGML_PAD(data.end, 128);
                data.staging_row_to_slot = data.end;
                data.end += staging_rows*sizeof(int32_t);

                data.end = GGML_PAD(data.end, 128);
                data.staging_slot_to_row = data.end;
                data.end += staging_rows*sizeof(int32_t);
            }
        }
    }

    return data;
}

void ggml_cuda_flash_attn_ext_materialize_mask(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst, ggml_tensor * mask, int first, int n_queries);

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, ggml_cuda_shfl_xor_sync<32>(amax, mask));
        sum +=             ggml_cuda_shfl_xor_sync<32>(sum,  mask);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
#if defined(GGML_USE_HIP)
    // Keep this VMEM read close to its packed-byte dequantization. Hoisting it too far
    // increases VGPR pressure substantially in some FlashAttention vector kernels.
    __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
#endif // defined(GGML_USE_HIP)
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = ggml_cuda_vsubss4_nonnegative(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = ggml_cuda_vsubss4_nonnegative(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1, bool compact>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_visibility_to_KV_max(
        const half2 * mask_ptr,
        int * KV_max_ptr,
        const int ne30,
        const int64_t s31,
        const int64_t s33,
        const int n_queries,
        const ggml_cuda_fattn_visibility visibility) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    if constexpr (!compact) {
        mask += sequence*s33 + jt*ncols1*s31;
    }

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const int query = jt * ncols1 + j;
            if (query >= n_queries) {
                continue;
            }

            if constexpr (compact) {
                const int cell = visibility.row_offset + KV_max_sj + 2 * tid;
                const bool visible0 = ggml_qsa_is_visible(visibility.positions, visibility.n_kv, cell, query);
                const bool visible1 = ggml_qsa_is_visible(visibility.positions, visibility.n_kv, cell + 1, query);
                all_inf = all_inf && !visible0 && !visible1;
            } else {
                const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
                all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
            }
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream);

void ggml_cuda_flash_attn_ext_prepare_indices(
        ggml_cuda_pool & pool, const ggml_tensor * mask, const ggml_tensor * selected,
        int32_t * indices, int32_t * counts, int32_t n_queries, cudaStream_t stream);

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_denominator == 0.0f ? 0.0f : VKQ_numerator / VKQ_denominator;
}

template<int D>
static __global__ void flash_attn_merge_staged(
        const float * parts, const float2 * meta, float * dst, float2 * running,
        const int n_queries, const int n_heads, const bool first) {
    const int row = (blockIdx.z*n_queries + blockIdx.x)*n_heads + blockIdx.y;
    const int d = threadIdx.x;

    constexpr int n_parts = GGML_CUDA_QSA_STAGING_PARTS;

    __shared__ float2 previous_meta;
    if (d == 0) {
        previous_meta = first ? make_float2(-INFINITY, 0.0f) : running[row];
    }
    __syncthreads();

    const float2 previous = previous_meta;
    float maximum = previous.x;
    for (int i = 0; i < n_parts; ++i) {
        maximum = fmaxf(maximum, meta[row*n_parts + i].x);
    }

    const float previous_scale = previous.y > 0.0f ? expf(previous.x - maximum) : 0.0f;
    float sum = previous.y*previous_scale;
    float numerator = first ? 0.0f : dst[row*D + d]*sum;

    for (int i = 0; i < n_parts; ++i) {
        const float2 m = meta[row*n_parts + i];
        const float scale = m.y > 0.0f ? expf(m.x - maximum) : 0.0f;

        numerator += parts[(row*n_parts + i)*D + d]*scale;
        sum += m.y*scale;
    }

    dst[row*D + d] = sum > 0.0f ? numerator/sum : 0.0f;
    if (d == 0) {
        running[row] = make_float2(maximum, sum);
    }
}

static __global__ void flash_attn_staged_indices(
        const int32_t * selected, int32_t * indices, uint32_t * needed,
        const int width, const int n_lists, const int start, const int stop) {
    const int list = blockIdx.x;
    const int32_t * rows = selected + (int64_t) list*width;
    const int count = selected[(int64_t) n_lists*width + list];

    __shared__ int first;
    __shared__ int size;
    if (threadIdx.x == 0) {
        int lo = 0;
        int hi = count;
        while (lo < hi) {
            const int mid = lo + (hi - lo)/2;
            if (rows[mid] < start) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        first = lo;

        hi = count;
        while (lo < hi) {
            const int mid = lo + (hi - lo)/2;
            if (rows[mid] < stop) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        size = lo - first;
        indices[(int64_t) n_lists*width + list] = size;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const int row = i < size ? rows[first + i] - start : -1;
        indices[(int64_t) list*width + i] = row;
        if (row >= 0) {
            atomicOr(needed + (row >> 5), 1u << (row & 31));
        }
    }
}

static __global__ void flash_attn_staged_build_map(
        const uint32_t * needed, int32_t * row_to_slot, int32_t * slot_to_row, int rows) {
    __shared__ int scan[256];
    __shared__ int base;

    const int tid = threadIdx.x;
    const int words = (rows + 31)/32;
    if (tid == 0) {
        base = 0;
    }
    __syncthreads();

    for (int first = 0; first < words; first += blockDim.x) {
        const int word = first + tid;
        const uint32_t bits = word < words ? needed[word] : 0;
        scan[tid] = __popcll((unsigned long long) bits);
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            const int add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            if (tid >= offset) {
                scan[tid] += add;
            }
            __syncthreads();
        }

        const int word_offset = base + (tid > 0 ? scan[tid - 1] : 0);
        int local = 0;
        for (int bit = 0; bit < 32; ++bit) {
            const int row = word*32 + bit;
            if (row < rows && (bits & (1u << bit))) {
                const int slot = word_offset + local++;
                row_to_slot[row] = slot;
                slot_to_row[slot] = row;
            }
        }
        __syncthreads();

        if (tid == blockDim.x - 1) {
            base += scan[tid];
        }
        __syncthreads();
    }
}

static __global__ void flash_attn_stage_selected_rows(
        const char * src, char * dst, const uint32_t * needed,
        const int32_t * row_to_slot, int rows, int64_t start,
        size_t src_nb1, size_t src_nb2, size_t src_nb3, size_t width,
        size_t dst_pitch, size_t dst_stride, int n_heads, int n_streams) {
    const int row = blockIdx.x;
    const int hs  = blockIdx.y;
    if (row >= rows || !(needed[row >> 5] & (1u << (row & 31)))) {
        return;
    }

    const int slot = row_to_slot[row];
    const int s = hs/n_heads;
    const int h = hs - s*n_heads;
    if (s >= n_streams) {
        return;
    }

    const char * in = src + s*src_nb3 + h*src_nb2 + (start + row)*src_nb1;
    char * out = dst + s*dst_stride + slot*dst_pitch + h*width;
    for (size_t i = threadIdx.x*sizeof(uint4); i + sizeof(uint4) <= width; i += blockDim.x*sizeof(uint4)) {
        *reinterpret_cast<uint4 *>(out + i) = *reinterpret_cast<const uint4 *>(in + i);
    }
    for (size_t i = width/sizeof(uint4)*sizeof(uint4) + threadIdx.x; i < width; i += blockDim.x) {
        out[i] = in[i];
    }
}

static __global__ void flash_attn_staged_remap_indices(
        int32_t * indices, const int32_t * row_to_slot,
        int width, int n_lists) {
    const int list = blockIdx.x;
    const int count = indices[(int64_t) n_lists*width + list];
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int64_t offset = (int64_t) list*width + i;
        indices[offset] = row_to_slot[indices[offset]];
    }
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const bool use_sparse,
    const int warp_size = WARP_SIZE, const bool use_paged = false, const uintptr_t workspace_begin = 0,
    const int64_t scheduling_queries = 0
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    const ggml_cuda_flash_attn_ext_extra_data extra =
        ggml_cuda_flash_attn_ext_get_extra_data(KQV, need_f16_K, need_f16_V, workspace_begin);

    const bool staged = !use_paged && (extra.K_staging || extra.V_staging);
    GGML_ASSERT(!staged || !stream_k);

    const int64_t staging_rows = staged ? ggml_cuda_qsa_staging_rows(K, V) : 0;
    const bool split_staging = staged && staging_rows < K->ne[1];

    if (staged) {
        GGML_ASSERT(!need_f16_K || K->type == GGML_TYPE_F16);
        GGML_ASSERT(!need_f16_V || V->type == GGML_TYPE_F16);
    }

    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    ggml_cuda_qsa_kv_view K_cache;
    ggml_cuda_qsa_kv_view V_cache;

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (ggml_cuda_qsa_kv_is_paged(K)) {
        K_data = static_cast<const char *>(ggml_cuda_qsa_kv_device_ptr(K));

        if (!use_paged) {
            GGML_ASSERT(extra.K_staging != 0);
            K_data = reinterpret_cast<const char *>(extra.K_staging);
            const size_t row_bytes = ggml_cuda_fattn_row_bytes(K);
            nb11 = row_bytes*K->ne[2];
            nb12 = row_bytes;
            nb13 = staging_rows*nb11;
        }
    }

    if (ggml_cuda_qsa_kv_is_paged(V)) {
        V_data = static_cast<const char *>(ggml_cuda_qsa_kv_device_ptr(V));

        if (!use_paged) {
            GGML_ASSERT(extra.V_staging != 0);
            V_data = reinterpret_cast<const char *>(extra.V_staging);
            const size_t row_bytes = ggml_cuda_fattn_row_bytes(V);
            nb21 = row_bytes*V->ne[2];
            nb22 = row_bytes;
            nb23 = staging_rows*nb21;
        }
    }

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        GGML_ASSERT(extra.K != 0);
        half * K_f16 = (half *) extra.K;
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            GGML_ASSERT(extra.V != 0);
            half * V_f16 = (half *) extra.V;
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    if (extra.selection_mask) {
        GGML_ASSERT(!use_sparse && !use_paged);

        const int64_t n_queries = ggml_cuda_fattn_mask_queries(dst);
        ggml_tensor selection_mask = {};
        selection_mask.type  = GGML_TYPE_F16;
        selection_mask.ne[0] = K->ne[1];
        selection_mask.ne[1] = GGML_PAD(n_queries, GGML_CUDA_QSA_MASK_QUERY_PAD);
        selection_mask.ne[2] = 1;
        selection_mask.ne[3] = 1;
        selection_mask.nb[0] = sizeof(half);
        selection_mask.nb[1] = selection_mask.ne[0] * selection_mask.nb[0];
        selection_mask.nb[2] = selection_mask.ne[1] * selection_mask.nb[1];
        selection_mask.nb[3] = selection_mask.nb[2];
        selection_mask.data  = reinterpret_cast<void *>(extra.selection_mask);

        uintptr_t chunk_workspace = extra.selection_mask + ggml_nbytes(&selection_mask);
        ggml_tensor converted_K;
        ggml_tensor converted_V;

        if (extra.K) {
            converted_K = *K;
            converted_K.type  = GGML_TYPE_F16;
            converted_K.data  = (void *) K_data;
            converted_K.nb[0] = sizeof(half);
            converted_K.nb[1] = nb11;
            converted_K.nb[2] = nb12;
            converted_K.nb[3] = nb13;
            chunk_workspace = extra.K + ggml_nelements(K) * sizeof(half);
        }

        if (extra.V) {
            converted_V = *V;
            converted_V.type  = GGML_TYPE_F16;
            converted_V.data  = (void *) V_data;
            converted_V.nb[0] = sizeof(half);
            converted_V.nb[1] = nb21;
            converted_V.nb[2] = nb22;
            converted_V.nb[3] = nb23;
            if (!V_is_K_view) {
                chunk_workspace = extra.V + ggml_nelements(V) * sizeof(half);
            }
        }

        for (int64_t first = 0; first < Q->ne[1]; first += n_queries) {
            const int64_t count = std::min<int64_t>(n_queries, Q->ne[1] - first);
            ggml_cuda_flash_attn_ext_materialize_mask(ctx, dst, &selection_mask, first, count);

            ggml_tensor query = *Q;
            query.ne[1] = count;
            query.data  = (char *) Q->data + first * Q->nb[1];

            ggml_tensor chunk = *dst;
            chunk.ne[2] = count;
            chunk.data   = (char *) dst->data + first * dst->nb[2];
            chunk.src[0] = &query;
            chunk.src[1] = extra.K ? &converted_K : dst->src[1];
            chunk.src[2] = extra.V ? &converted_V : dst->src[2];
            chunk.src[3] = &selection_mask;
            chunk.src[5] = nullptr;
            chunk.src[6] = nullptr;
            ggml_flash_attn_ext_set_n_kv_max(&chunk, 0);

            launch_fattn<DV, ncols1, ncols2>(
                    ctx, &chunk, fattn_kernel, nwarps, nbytes_shared, nbatch_fa,
                    false, false, stream_k, false, warp_size, false, chunk_workspace, Q->ne[1]);
        }

        return;
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];
    const size_t n_lists = size_t(ntiles_x) * (mask ? mask->ne[3] : 1);

    // sparse: a query tile of ncols1 queries shares one index list, the union of the queries' visible columns
    int32_t n_kv_max = 0;
    if (use_sparse) {
        GGML_ASSERT(mask != nullptr || KQV->src[6] != nullptr);
        const int32_t n_kv_max_query = ggml_get_op_params_i32(KQV, 4);
        GGML_ASSERT(n_kv_max_query > 0);
        n_kv_max = std::min<int64_t>(K->ne[1], int64_t(ncols1)*n_kv_max_query);

        KV_max.alloc(size_t(n_kv_max)*n_lists + n_lists);
        // Explicit indices avoid materializing a context-sized selection mask.
        if (KQV->src[5]) {
            GGML_ASSERT(ncols1 == 1 && KQV->src[5]->ne[0] == n_kv_max);
            ggml_cuda_flash_attn_ext_prepare_indices(ctx.pool(), mask ? mask : KQV->src[6], KQV->src[5], KV_max.ptr,
                KV_max.ptr + size_t(n_kv_max)*n_lists, Q->ne[1], main_stream);
        } else {
            ggml_cuda_flash_attn_ext_compact_mask(mask, KV_max.ptr, KV_max.ptr + size_t(n_kv_max)*n_lists, Q->ne[1], ncols1, n_kv_max, main_stream);
        }
    }

    if (use_paged) {
        GGML_ASSERT(Q->ne[3] == 1 && (!mask || mask->ne[3] == 1));

        const int n_indices = use_sparse ? n_kv_max*ntiles_x : K->ne[1];
        const int32_t * indices = use_sparse ? KV_max.ptr : nullptr;

        ggml_cuda_qsa_kv_prepare(ctx, K, V, indices, n_indices, K_cache, V_cache);
    }

    const bool use_kv_bounds = !use_sparse && (mask || KQV->src[6]) &&
        K->ne[1] % FATTN_KQ_STRIDE == 0 &&
        (Q->ne[1] >= 1024 || Q->ne[3] > 1 || workspace_begin);
    if (use_kv_bounds) {
        KV_max.alloc(ntiles_x * Q->ne[3]);
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int64_t n_kv = use_sparse ? n_kv_max : staged ? staging_rows : K->ne[1];
    const int ntiles_KV = (n_kv + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        auto should_use_stream_k = [](const int cc, const int ntiles_dst, const int max_blocks, const int DKQ) {
            const int tiles_nwaves             = (ntiles_dst + max_blocks - 1) / max_blocks;
            const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

            if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return true;
            }
            if (amd_wmma_available(cc) && DKQ == 64) {
                return true; // TODO better configuration
            }
            return tiles_efficiency_percent < 75;
        };

        const int  max_blocks   = max_blocks_per_sm*nsm;
        const bool use_stream_k = should_use_stream_k(cc, ntiles_dst, max_blocks, Q->ne[0]);

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // Decode and speculative verify batches (n_q <= 8) must use a query-width-
        // independent KV split: ntiles_dst is a function of Q->ne[1]
        // (ntiles_x = ceil(Q->ne[1]/ncols1)), so n_q=3 and n_q=5 would otherwise
        // feed different partial sums into the online-softmax/PV combine, drift the
        // logits in the last bits and flip greedy near-ties (issue #25).  Evaluate
        // the heuristic as if n_q == 1 so every small batch agrees.
        // Keep the KV split unchanged across query slabs.
        const int64_t total_queries = scheduling_queries ? scheduling_queries : Q->ne[1];
        const int scheduling_tiles = (total_queries + ncols1 - 1) / ncols1;
        const int ntiles_dst_eff = (total_queries <= 8 ? 1 : scheduling_tiles) * ntiles_z_gqa * K->ne[2] * Q->ne[3];

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst_eff * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        if (split_staging) {
            parallel_blocks = GGML_CUDA_QSA_STAGING_PARTS;
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1 && !split_staging) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    const int64_t step = staged ? staging_rows : K->ne[1];
    for (int64_t start = 0; start < K->ne[1]; start += step) {
        const int64_t rows = staged ? std::min(step, K->ne[1] - start) : n_kv;

        if (use_kv_bounds) {
            const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
            const dim3 block_dim_KV_max(FATTN_KQ_STRIDE / 2, 1, 1);
            const ggml_cuda_kernel_launch_params bound_params(
                    blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
            const int iter_k = rows / FATTN_KQ_STRIDE;
            ggml_cuda_fattn_visibility visibility;
            visibility.n_kv = K->ne[1];
            visibility.row_offset = start;

            if (KQV->src[6]) {
                visibility.positions = (const int32_t *) KQV->src[6]->data;
                ggml_cuda_kernel_launch(flash_attn_visibility_to_KV_max<ncols1, true>, bound_params,
                        (const half2 *) nullptr, KV_max.ptr, iter_k, int64_t(0), int64_t(0),
                        (int) Q->ne[1], visibility);
            } else {
                const half2 * mask_data = (const half2 *) ((const char *) mask->data + start * mask->nb[0]);
                const int64_t query_stride = mask->nb[1] / sizeof(half2);
                const int64_t stream_stride = mask->nb[3] / sizeof(half2);
                ggml_cuda_kernel_launch(flash_attn_visibility_to_KV_max<ncols1, false>, bound_params,
                        mask_data, KV_max.ptr, iter_k, query_stride, stream_stride,
                        (int) Q->ne[1], visibility);
            }

            CUDA_CHECK(cudaGetLastError());
        }

        const int32_t * chunk_indices = KV_max.ptr;
        const int32_t * staging_row_map = nullptr;

        if (split_staging && use_sparse) {
            GGML_ASSERT(extra.indices != 0);
            GGML_ASSERT(extra.staging_needed != 0);
            GGML_ASSERT(extra.staging_row_to_slot != 0);
            GGML_ASSERT(extra.staging_slot_to_row != 0);

            CUDA_CHECK(cudaMemsetAsync((void *) extra.staging_needed, 0, ((rows + 31)/32)*sizeof(uint32_t), main_stream));

            const ggml_cuda_kernel_launch_params index_params(dim3(n_lists, 1, 1), dim3(256, 1, 1), 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_staged_indices, index_params,
                    KV_max.ptr, (int32_t *) extra.indices, (uint32_t *) extra.staging_needed,
                    n_kv_max, (int) n_lists, (int) start, (int) (start + rows));
            CUDA_CHECK(cudaGetLastError());

            const ggml_cuda_kernel_launch_params map_params(dim3(1, 1, 1), dim3(256, 1, 1), 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_staged_build_map, map_params,
                    (const uint32_t *) extra.staging_needed, (int32_t *) extra.staging_row_to_slot,
                    (int32_t *) extra.staging_slot_to_row, (int) rows);
            CUDA_CHECK(cudaGetLastError());

            chunk_indices = (const int32_t *) extra.indices;
            staging_row_map = (const int32_t *) extra.staging_slot_to_row;
        }

        auto stage = [&](const ggml_tensor * tensor, uintptr_t staging, const char * data, size_t row_stride) {
            if (!staging || !staged) {
                return data + (staged ? start*row_stride : 0);
            }

            const size_t width = ggml_cuda_fattn_row_bytes(tensor);
            const size_t pitch = width*tensor->ne[2];
            const size_t stride = step*pitch;

            if (split_staging && use_sparse) {
                const ggml_cuda_kernel_launch_params stage_params(
                        dim3(rows, tensor->ne[2]*tensor->ne[3], 1), dim3(128, 1, 1), 0, main_stream);
                ggml_cuda_kernel_launch(flash_attn_stage_selected_rows, stage_params,
                        (const char *) ggml_cuda_qsa_kv_device_ptr(tensor), (char *) staging,
                        (const uint32_t *) extra.staging_needed, (const int32_t *) extra.staging_row_to_slot,
                        rows, start, tensor->nb[1], tensor->nb[2], tensor->nb[3],
                        width, pitch, stride, tensor->ne[2], tensor->ne[3]);
                CUDA_CHECK(cudaGetLastError());

                return reinterpret_cast<const char *>(staging);
            }

            for (int64_t s = 0; s < tensor->ne[3]; ++s) {
                for (int64_t h = 0; h < tensor->ne[2]; ++h) {
                    CUDA_CHECK(cudaMemcpy2DAsync((void *) (staging + s*stride + h*width), pitch,
                            (const char *) tensor->data + s*tensor->nb[3] + h*tensor->nb[2] + start*tensor->nb[1],
                            tensor->nb[1], width, rows, cudaMemcpyHostToDevice, main_stream));
                }
            }

            return reinterpret_cast<const char *>(staging);
        };

        const char * chunk_K = stage(K, extra.K_staging, K_data, nb11);
        const char * chunk_V = stage(V, extra.V_staging, V_data, nb21);

        if (split_staging && use_sparse) {
            const ggml_cuda_kernel_launch_params remap_params(dim3(n_lists, 1, 1), dim3(256, 1, 1), 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_staged_remap_indices, remap_params,
                    (int32_t *) extra.indices, (const int32_t *) extra.staging_row_to_slot,
                    n_kv_max, (int) n_lists);
            CUDA_CHECK(cudaGetLastError());
        }

        const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
        ggml_cuda_kernel_launch(fattn_kernel, launch_params,
            (const char *) Q->data,
            chunk_K,
            chunk_V,
            mask ? ((const char *) mask->data + start*mask->nb[0]) : nullptr,
            sinks && start == 0 ? ((const char *) sinks->data) : nullptr,
            chunk_indices,
            split_staging ? (float *) extra.parts : !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data,
            split_staging ? (float2 *) extra.parts_meta : dst_tmp_meta.ptr,
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K->ne[0], use_sparse ? n_kv_max : rows, K->ne[2], K->ne[3], nb11, nb12, nb13,
            nb21, nb22, nb23,
            mask ? mask->ne[1] : Q->ne[1], mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1,
            mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
            K_cache, V_cache,
            ggml_cuda_fattn_visibility{KQV->src[6] ? (const int32_t *) KQV->src[6]->data : nullptr,
                (int32_t) K->ne[1], (int32_t) start, staging_row_map}
        );
        CUDA_CHECK(cudaGetLastError());

        if (split_staging) {
            const ggml_cuda_kernel_launch_params merge_params(
                    dim3(Q->ne[1], Q->ne[2], Q->ne[3]), dim3(DV, 1, 1), 0, main_stream);

            ggml_cuda_kernel_launch(flash_attn_merge_staged<DV>, merge_params,
                    (const float *) extra.parts, (const float2 *) extra.parts_meta,
                    (float *) KQV->data, (float2 *) extra.stream_meta,
                    (int) Q->ne[1], (int) Q->ne[2], start == 0);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (!split_staging && parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
