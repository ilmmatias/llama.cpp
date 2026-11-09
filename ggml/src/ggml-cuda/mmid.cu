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
// The first (n_tokens*n_expert_used + tile_width - 1) / tile_width elements of tile_spec contain the specification for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
// Route IDs are not required to be unique within a token.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse) {
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

    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), it_compact);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + it_compact);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
    }
}
