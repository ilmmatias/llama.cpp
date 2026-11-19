#include "top-k-hip.cuh"

#include <algorithm>
#include <cstdint>

namespace {

static __device__ __forceinline__ uint32_t top_k_hip_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

// Single-workgroup per slice 3-pass (11+11+10 bit) radix select using 8KB LDS + float4 loads.
// - Wave-uniform __all optimization coalesces 128 identical floats (e.g. -inf / ties) into 1 LDS atomic.
// - Early-exits remaining radix passes whenever the boundary bucket fits exactly (rem_rank == bin_cnt).
// - Deterministic packed-scan gather preserves ascending column tie-breaking for Split-K and test determinism.
template <bool WRITE_VALS, bool HAS_MAP>
static __global__ void __launch_bounds__(256, 4) top_k_rdna2_11_11_10_kernel(
        const float * __restrict__ src,
        const int * __restrict__ idx_map_all,
        float * __restrict__ dst_vals,
        int * __restrict__ dst_idxs,
        int64_t ncols,
        int64_t k,
        int parts_per_row) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int NBINS      = 2048;

    const int64_t bid = (int64_t) blockIdx.x;
    const int64_t row = (parts_per_row == 1) ? bid : (bid / parts_per_row);
    const int part    = (parts_per_row == 1) ? 0   : (int) (bid % parts_per_row);

    const int64_t col_begin = (parts_per_row == 1) ? 0     : (ncols * part) / parts_per_row;
    const int64_t col_end   = (parts_per_row == 1) ? ncols : (ncols * (part + 1)) / parts_per_row;
    const int64_t len       = col_end - col_begin;
    const float * ptr       = src + row * ncols + col_begin;
    const int * idx_map     = HAS_MAP ? (idx_map_all + row * ncols + col_begin) : nullptr;

    float * out_v = WRITE_VALS ? (dst_vals + bid * k) : nullptr;
    int *   out_i = dst_idxs + bid * k;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (len <= k) {
        for (int64_t c = tid; c < len; c += BLOCK_SIZE) {
            if constexpr (WRITE_VALS) {
                out_v[c] = ptr[c];
            }
            out_i[c] = HAS_MAP ? idx_map[c] : (int) (col_begin + c);
        }
        return;
    }

    int head = ((16 - (int) ((uintptr_t) ptr & 15)) >> 2) & 3;
    if (head > len) {
        head = (int) len;
    }
    const int64_t body_len  = len - head;
    const int64_t n4        = body_len >> 2;
    const float4 * ptr4     = reinterpret_cast<const float4 *>(ptr + head);
    const int tail          = (int) (body_len & 3);
    const int64_t tail_base = head + n4 * 4;

    __shared__ uint32_t hist[NBINS];
    __shared__ uint32_t warp_sums[8];
    __shared__ uint32_t warp_offs[8];
    __shared__ uint32_t win_bin_s;
    __shared__ uint32_t win_gt_cnt_s;
    __shared__ uint32_t win_bin_cnt_s;
    __shared__ uint32_t gt_written_s;
    __shared__ uint32_t eq_written_s;
    __shared__ uint32_t tile_gt_base_s;
    __shared__ uint32_t tile_eq_base_s;

    uint32_t rem_rank    = (uint32_t) k;
    uint32_t prefix      = 0;
    uint32_t thresh_gt   = 0;
    bool     need_ties   = false;
    int      greater_total = (int) k;

    for (int pass = 0; pass < 3; ++pass) {
        const int shift      = (pass == 0) ? 21 : ((pass == 1) ? 10 : 0);
        const int mask_shift = (pass == 1) ? 21 : 10;
        const int n_active   = (pass == 2) ? 1024 : 2048;
        const uint32_t target_parent = prefix;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = tid + j * BLOCK_SIZE;
            if (idx < n_active) {
                hist[idx] = 0;
            }
        }
        __syncthreads();

        if (tid < head) {
            const uint32_t ord = top_k_hip_float_to_ordered(ptr[tid]);
            if (pass == 0 || (ord >> mask_shift) == target_parent) {
                const uint32_t bin = (ord >> shift) & (uint32_t) (n_active - 1);
                atomicAdd(&hist[bin], 1U);
            }
        }

        for (int64_t i4 = tid; i4 < n4; i4 += BLOCK_SIZE) {
            const float4 v = ptr4[i4];
            const uint32_t o0 = top_k_hip_float_to_ordered(v.x);
            const uint32_t o1 = top_k_hip_float_to_ordered(v.y);
            const uint32_t o2 = top_k_hip_float_to_ordered(v.z);
            const uint32_t o3 = top_k_hip_float_to_ordered(v.w);
            const bool full_wave = (i4 - lane + 31) < n4;

            if (pass == 0) {
                const uint32_t b0 = o0 >> 21;
                const uint32_t b1 = o1 >> 21;
                const uint32_t b2 = o2 >> 21;
                const uint32_t b3 = o3 >> 21;
                const uint32_t r0 = __shfl(b0, 0);
                if (full_wave && __all(b0 == r0 && b1 == r0 && b2 == r0 && b3 == r0)) {
                    if (lane == 0) {
                        atomicAdd(&hist[r0], 128U);
                    }
                } else {
                    atomicAdd(&hist[b0], 1U);
                    atomicAdd(&hist[b1], 1U);
                    atomicAdd(&hist[b2], 1U);
                    atomicAdd(&hist[b3], 1U);
                }
            } else {
                const bool m0 = (o0 >> mask_shift) == target_parent;
                const bool m1 = (o1 >> mask_shift) == target_parent;
                const bool m2 = (o2 >> mask_shift) == target_parent;
                const bool m3 = (o3 >> mask_shift) == target_parent;
                if (full_wave && __all(!m0 && !m1 && !m2 && !m3)) {
                    continue;
                }
                const uint32_t bin_mask = (uint32_t) (n_active - 1);
                if (m0) { atomicAdd(&hist[(o0 >> shift) & bin_mask], 1U); }
                if (m1) { atomicAdd(&hist[(o1 >> shift) & bin_mask], 1U); }
                if (m2) { atomicAdd(&hist[(o2 >> shift) & bin_mask], 1U); }
                if (m3) { atomicAdd(&hist[(o3 >> shift) & bin_mask], 1U); }
            }
        }

        if (tid < tail) {
            const uint32_t ord = top_k_hip_float_to_ordered(ptr[tail_base + tid]);
            if (pass == 0 || (ord >> mask_shift) == target_parent) {
                const uint32_t bin = (ord >> shift) & (uint32_t) (n_active - 1);
                atomicAdd(&hist[bin], 1U);
            }
        }
        __syncthreads();

        // Descending bin scan: tid owns 8 descending bins (or 4 for pass == 2).
        const int bpt = (pass == 2) ? 4 : 8;
        uint32_t c[8] = {};
        uint32_t t_sum = 0;
        const int top_bin = n_active - 1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < bpt) {
                const uint32_t cnt = hist[top_bin - (tid * bpt + j)];
                c[j] = cnt;
                t_sum += cnt;
            }
        }

        const uint32_t w_inc = warp_prefix_inclusive_sum<uint32_t, 32>(t_sum);
        if (lane == 31) {
            warp_sums[warp] = w_inc;
        }
        __syncthreads();

        if (warp == 0) {
            const uint32_t ws = (lane < 8) ? warp_sums[lane] : 0U;
            uint32_t ws_inc = ws;
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                const uint32_t t = ggml_cuda_shfl_up(ws_inc, off, 8);
                if (lane >= off) {
                    ws_inc += t;
                }
            }
            if (lane < 8) {
                warp_offs[lane] = ws_inc - ws;
            }
        }
        __syncthreads();

        const uint32_t base = warp_offs[warp] + (w_inc - t_sum);
        if (base < rem_rank && base + t_sum >= rem_rank) {
            uint32_t run = base;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < bpt && run + c[j] >= rem_rank) {
                    win_bin_s     = (uint32_t) (top_bin - (tid * bpt + j));
                    win_gt_cnt_s  = run;
                    win_bin_cnt_s = c[j];
                    break;
                }
                run += c[j];
            }
        }
        __syncthreads();

        const int bits_this_pass = (pass == 2) ? 10 : 11;
        prefix = (prefix << bits_this_pass) | win_bin_s;
        rem_rank -= win_gt_cnt_s;

        if (rem_rank == win_bin_cnt_s) {
            thresh_gt     = prefix << shift;
            need_ties     = false;
            greater_total = (int) k;
            break;
        }
        if (pass == 2) {
            thresh_gt     = prefix;
            need_ties     = true;
            greater_total = (int) k - (int) rem_rank;
        }
    }

    if (tid == 0) {
        gt_written_s = 0;
        eq_written_s = 0;
    }
    __syncthreads();

    auto emit_chunk = [&](const float vals[4], const int64_t lcols[4], int n_valid) {
        bool is_gt[4] = {false, false, false, false};
        bool is_eq[4] = {false, false, false, false};
        int c_gt = 0;
        int c_eq = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < n_valid) {
                const uint32_t ord = top_k_hip_float_to_ordered(vals[j]);
                const bool gt = need_ties ? (ord > thresh_gt) : (ord >= thresh_gt);
                const bool eq = need_ties && (ord == thresh_gt);
                is_gt[j] = gt;
                is_eq[j] = eq;
                c_gt += gt ? 1 : 0;
                c_eq += eq ? 1 : 0;
            }
        }

        const uint32_t packed = ((uint32_t) c_gt << 16) | (uint32_t) c_eq;
        const uint32_t w_inc  = warp_prefix_inclusive_sum<uint32_t, 32>(packed);
        if (lane == 31) {
            warp_sums[warp] = w_inc;
        }
        __syncthreads();

        if (warp == 0) {
            const uint32_t ws = (lane < 8) ? warp_sums[lane] : 0U;
            uint32_t ws_inc = ws;
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                const uint32_t t = ggml_cuda_shfl_up(ws_inc, off, 8);
                if (lane >= off) {
                    ws_inc += t;
                }
            }
            if (lane < 8) {
                warp_offs[lane] = ws_inc - ws;
            }
            if (lane == 7) {
                tile_gt_base_s  = gt_written_s;
                tile_eq_base_s  = eq_written_s;
                gt_written_s   += (ws_inc >> 16);
                eq_written_s   += (ws_inc & 0xFFFFU);
            }
        }
        __syncthreads();

        const uint32_t my_base = warp_offs[warp] + (w_inc - packed);
        int gt_pos = (int) tile_gt_base_s + (int) (my_base >> 16);
        int eq_pos = (int) tile_eq_base_s + (int) (my_base & 0xFFFFU);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (is_gt[j]) {
                if (gt_pos < greater_total) {
                    const int64_t lc = lcols[j];
                    if constexpr (WRITE_VALS) {
                        out_v[gt_pos] = vals[j];
                    }
                    out_i[gt_pos] = HAS_MAP ? idx_map[lc] : (int) (col_begin + lc);
                }
                ++gt_pos;
            } else if (is_eq[j]) {
                if (eq_pos < (int) rem_rank) {
                    const int out_slot = greater_total + eq_pos;
                    const int64_t lc = lcols[j];
                    if constexpr (WRITE_VALS) {
                        out_v[out_slot] = vals[j];
                    }
                    out_i[out_slot] = HAS_MAP ? idx_map[lc] : (int) (col_begin + lc);
                }
                ++eq_pos;
            }
        }
    };

    if (head > 0) {
        float   vals[4]  = {};
        int64_t lcols[4] = {};
        int     n_valid  = 0;
        if (tid < head) {
            vals[0]  = ptr[tid];
            lcols[0] = tid;
            n_valid  = 1;
        }
        emit_chunk(vals, lcols, n_valid);
        if (gt_written_s >= (uint32_t) greater_total && (!need_ties || eq_written_s >= rem_rank)) {
            return;
        }
    }

    for (int64_t b4 = 0; b4 < n4; b4 += BLOCK_SIZE) {
        const int64_t i4 = b4 + tid;
        float   vals[4]  = {};
        int64_t lcols[4] = {};
        int     n_valid  = 0;
        if (i4 < n4) {
            const float4 v = ptr4[i4];
            const int64_t c0 = (int64_t) head + i4 * 4;
            vals[0] = v.x; vals[1] = v.y; vals[2] = v.z; vals[3] = v.w;
            lcols[0] = c0; lcols[1] = c0 + 1; lcols[2] = c0 + 2; lcols[3] = c0 + 3;
            n_valid = 4;
        }
        emit_chunk(vals, lcols, n_valid);
        if (gt_written_s >= (uint32_t) greater_total && (!need_ties || eq_written_s >= rem_rank)) {
            return;
        }
    }

    if (tail > 0) {
        float   vals[4]  = {};
        int64_t lcols[4] = {};
        int     n_valid  = 0;
        if (tid < tail) {
            vals[0]  = ptr[tail_base + tid];
            lcols[0] = tail_base + tid;
            n_valid  = 1;
        }
        emit_chunk(vals, lcols, n_valid);
    }
}

} // namespace

bool ggml_cuda_top_k_hip(
        ggml_backend_cuda_context & ctx,
        const float * src,
        int * dst,
        int64_t ncols,
        int64_t nrows,
        int64_t k,
        ggml_cuda_pool & pool,
        cudaStream_t stream) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_UNUSED(ncols);
    GGML_UNUSED(nrows);
    GGML_UNUSED(k);
    GGML_UNUSED(pool);
    GGML_UNUSED(stream);
    return false;
#else
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    if (warp_size != 32 || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }

    if (k <= 1 || ncols <= 1024 || k >= ncols || nrows <= 0) {
        return false;
    }

    constexpr int BLOCK_SIZE = 256;

    // Small-nrows decode/low-batch Split-K (2 launches total instead of 6-8 nary launches).
    if (nrows < 40 && ncols >= 4096) {
        const int64_t min_slice = std::max<int64_t>(2 * k, 1024);
        const int64_t max_parts = ncols / min_slice;
        const int64_t target_p  = std::max<int64_t>(1, 160 / nrows);
        const int     parts     = (int) std::min<int64_t>({32LL, max_parts, target_p});

        if (parts >= 4) {
            const int64_t tmp_elems = nrows * parts * k;
            ggml_cuda_pool_alloc<float> tmp_vals(pool, tmp_elems);
            ggml_cuda_pool_alloc<int>   tmp_idxs(pool, tmp_elems);

            const int64_t stage1_blocks = nrows * parts;
            top_k_rdna2_11_11_10_kernel<true, false><<<(unsigned) stage1_blocks, BLOCK_SIZE, 0, stream>>>(
                src, nullptr, tmp_vals.get(), tmp_idxs.get(), ncols, k, parts);

            const int64_t stage2_ncols = (int64_t) parts * k;
            top_k_rdna2_11_11_10_kernel<false, true><<<(unsigned) nrows, BLOCK_SIZE, 0, stream>>>(
                tmp_vals.get(), tmp_idxs.get(), nullptr, dst, stage2_ncols, k, 1);
            return true;
        }
    }

    // Single-launch row-per-block radix select for prefill / high-nrows / small-ratio slices.
    if (nrows >= 16 || ncols >= 2048) {
        top_k_rdna2_11_11_10_kernel<false, false><<<(unsigned) nrows, BLOCK_SIZE, 0, stream>>>(
            src, nullptr, nullptr, dst, ncols, k, 1);
        return true;
    }

    return false;
#endif
}
