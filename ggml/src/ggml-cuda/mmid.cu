#include "common.cuh"
#include "mmid.cuh"

// Pad the compile-time route count to a power of two so one warp can process
// multiple tokens while keeping each token's route slots contiguous.
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
//
// Route IDs are not required to be unique within a token. This matters for hash-routed MoE models and for compact/pruned
// routing tables where multiple original experts can map to the same retained expert. The old helper compacted one row per
// (token, expert) pair, which left holes in ids_src1/ids_dst when a token selected the same expert more than once.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    // First pass: find the start of this expert's compact range. Counting every route slot (rather than every token)
    // makes the prefix ranges correct even when the same expert occurs multiple times in one token.
    int nex_prev_local = 0;
    if constexpr (n_expert_used_template == 0) {
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev_local += expert_used < expert;
            }
        }
    } else {
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        constexpr int tokens_per_warp = warp_size / neu_padded;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");

        for (int it0 = 0; it0 < n_tokens; it0 += tokens_per_warp) {
            const int it  = it0 + threadIdx.x / neu_padded;
            const int iex = threadIdx.x % neu_padded;

            if (it < n_tokens && iex < n_expert_used_template) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev_local += expert_used < expert;
            }
        }
    }
    const int nex_prev = warp_reduce_sum<warp_size>(nex_prev_local);

    // Second pass: rescan the small routing tensor and write one compact entry per route slot. A warp prefix scan gives
    // each duplicate occurrence a distinct compact row without any shared-memory storage proportional to the batch size.
    int it_compact = 0;
    if constexpr (n_expert_used_template == 0) {
        for (int it = 0; it < n_tokens; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const bool match = iex < n_expert_used && ids[it*si1 + iex] == expert;

                const int match_prefix = warp_prefix_inclusive_sum<int, warp_size>(match ? 1 : 0);
                const int match_count  = __shfl_sync(0xffffffff, match_prefix, warp_size - 1, warp_size);

                if (match) {
                    const int compact = nex_prev + it_compact + match_prefix - 1;
                    ids_dst[compact] = it*n_expert_used + iex;
                    if (write_inverse) {
                        ids_src1[it*n_expert_used + iex] = compact;
                    } else {
                        ids_src1[compact] = it*sis1 + iex % nchannels_y;
                    }
                }

                it_compact += match_count;
            }
        }
    } else {
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        constexpr int tokens_per_warp = warp_size / neu_padded;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");

        for (int it0 = 0; it0 < n_tokens; it0 += tokens_per_warp) {
            const int it  = it0 + threadIdx.x / neu_padded;
            const int iex = threadIdx.x % neu_padded;
            const bool match = it < n_tokens && iex < n_expert_used_template && ids[it*si1 + iex] == expert;

            // Lane order is token-major within the warp, so this preserves the same
            // flattened (token, route-slot) ordering as the generic path.

            const int match_prefix = warp_prefix_inclusive_sum<int, warp_size>(match ? 1 : 0);
            const int match_count  = __shfl_sync(0xffffffff, match_prefix, warp_size - 1, warp_size);

            if (match) {
                const int compact = nex_prev + it_compact + match_prefix - 1;
                ids_dst[compact] = it*n_expert_used + iex;
                if (write_inverse) {
                    ids_src1[it*n_expert_used + iex] = compact;
                } else {
                    ids_src1[compact] = it*sis1 + iex % nchannels_y;
                }
            }

            it_compact += match_count;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert == static_cast<int>(gridDim.x) - 1) {
        expert_bounds[gridDim.x] = nex_prev + it_compact;
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>(
       ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

#if defined(GGML_USE_HIP)
static __global__ void mm_ids_helper_histogram(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    __shared__ int counts[1024];
    __shared__ int cursors[1024];

    for (int expert = threadIdx.x; expert < n_experts; expert += blockDim.x) {
        counts[expert] = 0;
    }
    __syncthreads();

    const int n_routes = n_tokens*n_expert_used;
    for (int route = threadIdx.x; route < n_routes; route += blockDim.x) {
        const int token = route / n_expert_used;
        const int slot = route % n_expert_used;
        atomicAdd(counts + ids[token*si1 + slot], 1);
    }
    __syncthreads();

    if (threadIdx.x < warp_size) {
        const int chunk = (n_experts + warp_size - 1) / warp_size;
        const int begin = threadIdx.x*chunk;
        const int end = min(begin + chunk, n_experts);
        int count = 0;
        for (int expert = begin; expert < end; ++expert) {
            count += counts[expert];
        }
        const int prefix = warp_prefix_inclusive_sum<int, warp_size>(count);
        int offset = prefix - count;
        for (int expert = begin; expert < end; ++expert) {
            expert_bounds[expert] = offset;
            cursors[expert] = offset;
            offset += counts[expert];
        }
        if (threadIdx.x == warp_size - 1) {
            expert_bounds[n_experts] = prefix;
        }
    }
    __syncthreads();

    for (int route = threadIdx.x; route < n_routes; route += blockDim.x) {
        const int token = route / n_expert_used;
        const int slot = route % n_expert_used;
        const int compact = atomicAdd(cursors + ids[token*si1 + slot], 1);
        ids_dst[compact] = route;
        if (write_inverse) {
            ids_src1[route] = compact;
        } else {
            ids_src1[compact] = token*sis1 + slot % nchannels_y;
        }
    }
}
#endif

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    if (n_experts <= 1024 && n_tokens >= 128) {
        mm_ids_helper_histogram<<<1, 256, 0, stream>>>(
            ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
        return;
    }
#endif

    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
