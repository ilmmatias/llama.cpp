#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB) || defined(GGML_USE_HIP)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#ifdef GGML_USE_HIP

struct hip_top_k_pair {
    float key;
    int   index;
};

static __device__ __forceinline__ bool hip_top_k_better(const hip_top_k_pair & a, const hip_top_k_pair & b) {
    if (a.index < 0) {
        return false;
    }
    if (b.index < 0) {
        return true;
    }
    if (a.key > b.key) {
        return true;
    }
    if (a.key < b.key) {
        return false;
    }
    return a.index < b.index;
}

// Reduce 1024 input candidates to k output candidates per workgroup. The first
// pass reads raw F32 scores and synthesizes their column indices. Later passes
// carry (score, original-index) pairs until only one workgroup remains per row.
// This is intentionally specialized for the DSV4 lightning-indexer top-512
// path: unlike segmented radix sort it never sorts elements that cannot survive
// the next reduction pass, and it is compatible with HIP stream capture.
template <int BLOCK_SIZE, int THREADS>
static __global__ void top_k_hip_reduce(
        const float * keys_in,
        const int *   indices_in,
        float *       keys_out,
        int *         indices_out,
        int *         dst,
        int            ncols,
        int            k,
        bool           first_pass,
        bool           last_pass) {
    static_assert(BLOCK_SIZE == 1024, "DSV4 HIP top-k expects 1024 candidates per block");

    __shared__ hip_top_k_pair candidates[BLOCK_SIZE];

    const int tid   = threadIdx.x;
    const int row   = blockIdx.y;
    const int group = blockIdx.x;
    const int base  = group * BLOCK_SIZE;

    for (int i = tid; i < BLOCK_SIZE; i += THREADS) {
        const int col = base + i;
        if (col < ncols) {
            const size_t pos = (size_t) row * ncols + col;
            candidates[i].key   = keys_in[pos];
            candidates[i].index = first_pass ? col : indices_in[pos];
        } else {
            candidates[i].key   = 0.0f;
            candidates[i].index = -1;
        }
    }
    __syncthreads();

    // Bitonic sort, best candidate first. Invalid sentinel entries always sort
    // after real entries, including when real scores are +/-inf or tied.
    for (int size = 2; size <= BLOCK_SIZE; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < BLOCK_SIZE; i += THREADS) {
                const int j = i ^ stride;
                if (j > i) {
                    const hip_top_k_pair a = candidates[i];
                    const hip_top_k_pair b = candidates[j];
                    const bool descending = (i & size) == 0;
                    const bool do_swap = descending ? hip_top_k_better(b, a) : hip_top_k_better(a, b);
                    if (do_swap) {
                        candidates[i] = b;
                        candidates[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < k; i += THREADS) {
        if (last_pass) {
            dst[(size_t) row * k + i] = candidates[i].index;
        } else {
            const int n_groups = gridDim.x;
            const size_t out = (size_t) row * n_groups * k + (size_t) group * k + i;
            keys_out[out]    = candidates[i].key;
            indices_out[out] = candidates[i].index;
        }
    }
}

static void top_k_hip_hierarchical(
        ggml_cuda_pool & pool,
        const float *    src,
        int *            dst,
        int              ncols,
        int              nrows,
        int              k,
        cudaStream_t     stream) {
    constexpr int BLOCK_SIZE = 1024;
    constexpr int THREADS    = 256;
    constexpr size_t TARGET_TEMP_BYTES = 128ull << 20;

    GGML_ASSERT(ncols > BLOCK_SIZE);
    GGML_ASSERT(k == 512);

    const int max_candidates = ((ncols + BLOCK_SIZE - 1) / BLOCK_SIZE) * k;
    const size_t temp_bytes_per_row = (size_t) max_candidates *
        2 * (sizeof(float) + sizeof(int));
    const int chunk_nrows = std::min<int64_t>(nrows,
        std::max<size_t>(1, TARGET_TEMP_BYTES / std::max<size_t>(1, temp_bytes_per_row)));

    const size_t n_temp = (size_t) max_candidates * chunk_nrows;
    ggml_cuda_pool_alloc<float> keys_a_alloc(pool, n_temp);
    ggml_cuda_pool_alloc<float> keys_b_alloc(pool, n_temp);
    ggml_cuda_pool_alloc<int>   indices_a_alloc(pool, n_temp);
    ggml_cuda_pool_alloc<int>   indices_b_alloc(pool, n_temp);

    float * keys_a = keys_a_alloc.get();
    float * keys_b = keys_b_alloc.get();
    int * indices_a = indices_a_alloc.get();
    int * indices_b = indices_b_alloc.get();

    for (int row0 = 0; row0 < nrows; row0 += chunk_nrows) {
        const int rows = std::min(chunk_nrows, nrows - row0);

        const float * keys_in = src + (size_t) row0 * ncols;
        const int * indices_in = nullptr;
        int current_ncols = ncols;
        bool first_pass = true;
        bool write_a = true;

        while (true) {
            const int n_groups = (current_ncols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const bool last_pass = n_groups == 1;

            float * keys_out = write_a ? keys_a : keys_b;
            int * indices_out = write_a ? indices_a : indices_b;

            const dim3 grid(n_groups, rows, 1);
            top_k_hip_reduce<BLOCK_SIZE, THREADS><<<grid, THREADS, 0, stream>>>(
                keys_in, indices_in,
                last_pass ? nullptr : keys_out,
                last_pass ? nullptr : indices_out,
                last_pass ? dst + (size_t) row0 * k : nullptr,
                current_ncols, k, first_pass, last_pass);
            CUDA_CHECK(cudaGetLastError());

            if (last_pass) {
                break;
            }

            current_ncols = n_groups * k;
            keys_in       = keys_out;
            indices_in    = indices_out;
            first_pass    = false;
            write_a       = !write_a;
        }
    }
}

bool ggml_cuda_top_k_hip_uses_radix(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    if (src0->ne[0] <= 1024) {
        return false;
    }

    // hipCUB is substantially faster on gfx1030 for prefill and ordinary
    // context lengths. At long contexts the hierarchical top-512 reducer also
    // wins for a few simultaneous decode rows: its crossover is approximately
    // ncols > nrows * 65536, with a hard floor at 131072 columns. Keep explicit
    // overrides so this measured workload policy can be retuned on other GPUs.
    if (getenv("GGML_HIP_TOPK_RADIX") != nullptr) {
        return true;
    }
    if (dst->ne[0] != 512) {
        return true;
    }
    if (getenv("GGML_HIP_TOPK_HIER") != nullptr) {
        return false;
    }

    const int64_t nrows = ggml_nrows(src0);
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA2(cc)) {
        return nrows != 1 || src0->ne[0] < 131072;
    }

    return src0->ne[0] < 131072 || src0->ne[0] <= nrows * 65536;
}

#endif // GGML_USE_HIP

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#elif defined(GGML_USE_HIP)
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;

    if (!use_bitonic && !ggml_cuda_top_k_hip_uses_radix(dst)) {
        top_k_hip_hierarchical(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

    const int    chunk_nrows    = argsort_f32_i32_cuda_hip_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int * tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);
        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_hip(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));
        src0_d += ncols * iter_nrows;
        dst_d  += k * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
