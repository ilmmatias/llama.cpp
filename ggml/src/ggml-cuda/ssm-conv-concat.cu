#include "ssm-conv-concat.cuh"
#include "cpy.cuh"

#include <algorithm>
#include <cstdint>

namespace {

static bool ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const uintptr_t ab = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t ae = ab + ggml_nbytes(a);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b->data);
    const uintptr_t be = bb + ggml_nbytes(b);
    return ab < be && bb < ae;
}

static bool ssm_conv_concat_can_vec4(
        const ggml_tensor * src1, const float * bias_d, const ggml_tensor * dst) {
    return (dst->ne[0] % 4 == 0) &&
           src1->nb[0] == sizeof(float) &&
           ((uintptr_t) src1->data & 15) == 0 &&
           (src1->nb[1] & 15) == 0 && (src1->nb[2] & 15) == 0 &&
           ((uintptr_t) dst->data & 15) == 0 &&
           (dst->nb[1] & 15) == 0 && (dst->nb[2] & 15) == 0 &&
           (!bias_d || ((uintptr_t) bias_d & 15) == 0);
}

static bool ssm_conv_concat_try_write_state(
        const ggml_cuda_concat_ssm_conv_plan & plan,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        int K,
        bool can_vec4,
        char ** state_dst_out,
        uint64_t * st_s_col_out,
        uint64_t * st_s_seq_out) {
    const int64_t C     = dst->ne[0];
    const int64_t T_out = dst->ne[1];
    const int64_t S     = dst->ne[2];
    if (!can_vec4 || K < 2 || K > 8 || T_out != 1 || plan.n_cpys != 1 ||
            src0->ne[1] != K - 1 || src1->ne[1] != 1 || plan.cpy_tok0[0] != 1 ||
            plan.cpy_src_tmpl[0].ne[1] != src0->ne[1] || plan.cpy_src_tmpl[0].ne[2] != S) {
        return false;
    }
    ggml_tensor * cpy_dst     = plan.cpys[0]->src[1];
    const bool shape_2d       = cpy_dst->ne[0] == C * src0->ne[1] && cpy_dst->ne[1] == S && cpy_dst->ne[2] == 1 && cpy_dst->ne[3] == 1;
    const bool shape_3d       = cpy_dst->ne[0] == C && cpy_dst->ne[1] == src0->ne[1] && cpy_dst->ne[2] == S && cpy_dst->ne[3] == 1;
    const uint64_t col_stride = shape_2d ? (uint64_t) C * sizeof(float) : (uint64_t) cpy_dst->nb[1];
    const uint64_t seq_stride = shape_2d ? (uint64_t) cpy_dst->nb[1]     : (uint64_t) cpy_dst->nb[2];
    if ((!shape_2d && !shape_3d) || cpy_dst->type != GGML_TYPE_F32 || cpy_dst->nb[0] != sizeof(float) ||
            col_stride < (uint64_t) C * sizeof(float) || (S > 1 && seq_stride < (uint64_t) src0->ne[1] * col_stride) ||
            ((uintptr_t) cpy_dst->data & 15) != 0 || (col_stride & 15) != 0 ||
            (S > 1 && (seq_stride & 15) != 0)) {
        return false;
    }
    if (state_dst_out) {
        *state_dst_out = (char *) cpy_dst->data;
    }
    if (st_s_col_out) {
        *st_s_col_out = col_stride;
    }
    if (st_s_seq_out) {
        *st_s_seq_out = seq_stride;
    }
    return true;
}

template <int K, int BT, bool HAS_BIAS, bool HAS_SILU, bool WRITE_STATE>
static __global__ void __launch_bounds__(128) ssm_conv_concat_dim1_vec4_f32(
        // src0 may alias dst (when T_out=1..BT inplace reuse) or state_dst (in-place state shift);
        // keep those pointers non-__restrict__ so loads before stores are preserved.
        const char * src0,
        const char * __restrict__ src1,
        const char * __restrict__ weight,
        const float * __restrict__ bias,
        char * dst,
        char * state_dst,
        int64_t C,
        int64_t T0,
        int64_t T1,
        int64_t T_out,
        uint64_t s00, uint64_t s01, uint64_t s02,
        uint64_t s11, uint64_t s12,
        uint64_t sw0, uint64_t sw1,
        uint64_t sd1, uint64_t sd2,
        uint64_t st_s_col, uint64_t st_s_seq) {
    static_assert(K >= 1 && K <= 8, "K must be in [1, 8]");

    const int64_t c4  = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t c0  = c4 * 4;
    if (c0 >= C) {
        return;
    }

    const int64_t t_base = (int64_t) blockIdx.y * BT;
    const int64_t seq    = blockIdx.z;

    const char * s0_seq  = src0 + (uint64_t) seq * s02;
    const char * s1_seq  = src1 + (uint64_t) seq * s12;
    char       * dst_seq = dst  + (uint64_t) seq * sd2;

    float4 w_tap[K];
    if constexpr (K == 4) {
        if (sw0 == sizeof(float) && sw1 == 4 * sizeof(float) && (((uintptr_t) weight & 15) == 0)) {
            const char * wp = weight + (uint64_t) c0 * sw1;
            const float4 r0 = *reinterpret_cast<const float4 *>(wp + 0 * sw1);
            const float4 r1 = *reinterpret_cast<const float4 *>(wp + 1 * sw1);
            const float4 r2 = *reinterpret_cast<const float4 *>(wp + 2 * sw1);
            const float4 r3 = *reinterpret_cast<const float4 *>(wp + 3 * sw1);
            w_tap[0] = make_float4(r0.x, r1.x, r2.x, r3.x);
            w_tap[1] = make_float4(r0.y, r1.y, r2.y, r3.y);
            w_tap[2] = make_float4(r0.z, r1.z, r2.z, r3.z);
            w_tap[3] = make_float4(r0.w, r1.w, r2.w, r3.w);
        } else {
#pragma unroll
            for (int j = 0; j < K; ++j) {
                w_tap[j] = make_float4(
                    *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 0) * sw1 + (uint64_t) j * sw0),
                    *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 1) * sw1 + (uint64_t) j * sw0),
                    *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 2) * sw1 + (uint64_t) j * sw0),
                    *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 3) * sw1 + (uint64_t) j * sw0));
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < K; ++j) {
            w_tap[j] = make_float4(
                *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 0) * sw1 + (uint64_t) j * sw0),
                *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 1) * sw1 + (uint64_t) j * sw0),
                *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 2) * sw1 + (uint64_t) j * sw0),
                *reinterpret_cast<const float *>(weight + (uint64_t) (c0 + 3) * sw1 + (uint64_t) j * sw0));
        }
    }

    float4 b4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (HAS_BIAS) {
        b4 = *reinterpret_cast<const float4 *>(bias + c0);
    }

    auto load_tok4 = [&](int64_t u) -> float4 {
        if (u < T0) {
            const char * p = s0_seq + (uint64_t) u * s01 + (uint64_t) c0 * s00;
            if (s00 == sizeof(float) && (((uintptr_t) p & 15) == 0)) {
                return *reinterpret_cast<const float4 *>(p);
            }
            return make_float4(
                *reinterpret_cast<const float *>(p + 0 * s00),
                *reinterpret_cast<const float *>(p + 1 * s00),
                *reinterpret_cast<const float *>(p + 2 * s00),
                *reinterpret_cast<const float *>(p + 3 * s00));
        }
        const int64_t u1 = u - T0;
        if (u1 >= T1) {
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return *reinterpret_cast<const float4 *>(s1_seq + (uint64_t) u1 * s11 + c0 * sizeof(float));
    };

    float4 win[K > 1 ? K - 1 : 1];
#pragma unroll
    for (int j = 0; j < K - 1; ++j) {
        win[j] = load_tok4(t_base + j);
    }

    const int64_t t_end = min(T_out, t_base + BT);
    for (int64_t t = t_base; t < t_end; ++t) {
        const float4 x_last = load_tok4(t + K - 1);

        float4 acc = b4;
#pragma unroll
        for (int j = 0; j < K - 1; ++j) {
            acc.x = fmaf(win[j].x, w_tap[j].x, acc.x);
            acc.y = fmaf(win[j].y, w_tap[j].y, acc.y);
            acc.z = fmaf(win[j].z, w_tap[j].z, acc.z);
            acc.w = fmaf(win[j].w, w_tap[j].w, acc.w);
        }
        acc.x = fmaf(x_last.x, w_tap[K - 1].x, acc.x);
        acc.y = fmaf(x_last.y, w_tap[K - 1].y, acc.y);
        acc.z = fmaf(x_last.z, w_tap[K - 1].z, acc.z);
        acc.w = fmaf(x_last.w, w_tap[K - 1].w, acc.w);

        if constexpr (HAS_SILU) {
            acc.x = acc.x / (1.0f + expf(-acc.x));
            acc.y = acc.y / (1.0f + expf(-acc.y));
            acc.z = acc.z / (1.0f + expf(-acc.z));
            acc.w = acc.w / (1.0f + expf(-acc.w));
        }

        *reinterpret_cast<float4 *>(dst_seq + (uint64_t) t * sd1 + c0 * sizeof(float)) = acc;

        if constexpr (WRITE_STATE) {
            // Decode single-token step: win[1..K-2] and x_last form the updated state_cols rows.
            char * st_seq = state_dst + (uint64_t) seq * st_s_seq + (uint64_t) c0 * sizeof(float);
#pragma unroll
            for (int j = 0; j < K - 2; ++j) {
                *reinterpret_cast<float4 *>(st_seq + (uint64_t) j * st_s_col) = win[j + 1];
            }
            *reinterpret_cast<float4 *>(st_seq + (uint64_t) (K - 2) * st_s_col) = x_last;
        } else if constexpr (K > 1) {
#pragma unroll
            for (int j = 0; j < K - 2; ++j) {
                win[j] = win[j + 1];
            }
            win[K - 2] = x_last;
        }
    }
}

template <int BT>
static __global__ void __launch_bounds__(128) ssm_conv_concat_dim1_scalar_f32(
        const char * __restrict__ src0,
        const char * __restrict__ src1,
        const char * __restrict__ weight,
        const float * __restrict__ bias,
        char * __restrict__ dst,
        int K,
        int64_t C,
        int64_t T0,
        int64_t T1,
        int64_t T_out,
        uint64_t s00, uint64_t s01, uint64_t s02,
        uint64_t s10, uint64_t s11, uint64_t s12,
        uint64_t sw0, uint64_t sw1,
        uint64_t sd1, uint64_t sd2,
        bool has_silu) {
    const int64_t c = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }

    const int64_t t_base = (int64_t) blockIdx.y * BT;
    const int64_t seq    = blockIdx.z;

    const char * s0_seq  = src0 + (uint64_t) seq * s02 + (uint64_t) c * s00;
    const char * s1_seq  = src1 + (uint64_t) seq * s12 + (uint64_t) c * s10;
    const char * w_ch    = weight + (uint64_t) c * sw1;
    char       * dst_seq = dst  + (uint64_t) seq * sd2 + c * sizeof(float);
    const float  bv      = bias ? bias[c] : 0.0f;

    const int64_t t_end = min(T_out, t_base + BT);
    for (int64_t t = t_base; t < t_end; ++t) {
        float acc = bv;
        for (int j = 0; j < K; ++j) {
            const int64_t u = t + j;
            const float xv = (u < T0)
                ? *reinterpret_cast<const float *>(s0_seq + (uint64_t) u * s01)
                : *reinterpret_cast<const float *>(s1_seq + (uint64_t) (u - T0) * s11);
            const float wv = *reinterpret_cast<const float *>(w_ch + (uint64_t) j * sw0);
            acc = fmaf(xv, wv, acc);
        }
        if (has_silu) {
            acc = acc / (1.0f + expf(-acc));
        }
        *reinterpret_cast<float *>(dst_seq + (uint64_t) t * sd1) = acc;
    }
}

template <int K>
static void dispatch_k(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * w,
        const float * bias_d,
        ggml_tensor * dst,
        bool has_silu,
        char * state_dst,
        uint64_t st_s_col,
        uint64_t st_s_seq,
        cudaStream_t stream) {
    const int64_t C     = dst->ne[0];
    const int64_t T_out = dst->ne[1];
    const int64_t S     = dst->ne[2];
    const int64_t T0    = src0->ne[1];
    const int64_t T1    = src1->ne[1];
    constexpr int BT    = 32;
    const int64_t nc4   = C / 4;
    const dim3 grid((unsigned) ((nc4 + 127) / 128), (unsigned) ((T_out + BT - 1) / BT), (unsigned) S);

    auto launch = [&](auto hb, auto hs, auto ws) {
        ssm_conv_concat_dim1_vec4_f32<K, BT, decltype(hb)::value, decltype(hs)::value, decltype(ws)::value><<<grid, 128, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (const char *) w->data,
            bias_d, (char *) dst->data, state_dst,
            C, T0, T1, T_out,
            src0->nb[0], src0->nb[1], src0->nb[2],
            src1->nb[1], src1->nb[2],
            w->nb[0], w->nb[1],
            dst->nb[1], dst->nb[2],
            st_s_col, st_s_seq);
    };

    auto launch_ws = [&](auto hb, auto hs) {
        if (state_dst != nullptr && K >= 2 && T_out == 1) {
            launch(hb, hs, std::true_type{});
        } else {
            launch(hb, hs, std::false_type{});
        }
    };

    if (bias_d && has_silu) {
        launch_ws(std::true_type{}, std::true_type{});
    } else if (bias_d) {
        launch_ws(std::true_type{}, std::false_type{});
    } else if (has_silu) {
        launch_ws(std::false_type{}, std::true_type{});
    } else {
        launch_ws(std::false_type{}, std::false_type{});
    }
}

} // namespace

bool ggml_cuda_match_concat_ssm_conv(
        ggml_backend_cuda_context & ctx,
        ggml_cgraph *               cgraph,
        int                         start_idx,
        ggml_cuda_concat_ssm_conv_plan & plan) {
    const ggml_tensor * concat_node = cgraph->nodes[start_idx];
    if (concat_node->op != GGML_OP_CONCAT || concat_node->type != GGML_TYPE_F32 ||
            ((const int32_t *) concat_node->op_params)[0] != 1 || concat_node->ne[3] != 1) {
        return false;
    }

    const ggml_tensor * src0 = concat_node->src[0];
    const ggml_tensor * src1 = concat_node->src[1];
    if (!src0 || !src1 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
            src0->ne[3] != 1 || src1->ne[3] != 1 ||
            src0->ne[0] != concat_node->ne[0] || src1->ne[0] != concat_node->ne[0] ||
            src0->ne[2] != concat_node->ne[2] || src1->ne[2] != concat_node->ne[2]) {
        return false;
    }

    constexpr int MAX_SPAN = 28;
    const ggml_tensor * aliases[MAX_SPAN];
    const ggml_tensor * cont_src[MAX_SPAN];
    int n_aliases = 0;
    aliases[n_aliases]  = concat_node;
    cont_src[n_aliases] = nullptr;
    n_aliases++;

    auto find_alias = [&](const ggml_tensor * t) -> int {
        if (!t) {
            return -1;
        }
        for (int a = 0; a < n_aliases; ++a) {
            if (aliases[a] == t) {
                return a;
            }
        }
        return -1;
    };
    auto is_alias = [&](const ggml_tensor * t) -> bool {
        return find_alias(t) >= 0;
    };

    plan = {};
    plan.concat_node = concat_node;

    int node_idxs[MAX_SPAN];
    enum ggml_op ops[MAX_SPAN];
    int intervening_idxs[MAX_SPAN];
    int out_idxs[12];
    int count = 0;
    int n_intervening = 0;
    int n_outs = 0;
    bool has_cpy_src0_overlap = false;

    node_idxs[count] = start_idx;
    ops[count]       = GGML_OP_CONCAT;
    count++;

    const int limit = std::min(cgraph->n_nodes, start_idx + MAX_SPAN);
    int ssm_pos = -1;
    for (int j = start_idx + 1; j < limit; ++j) {
        ggml_tensor * cur = cgraph->nodes[j];
        if (count >= MAX_SPAN) {
            return false;
        }
        if (cur->op == GGML_OP_VIEW || cur->op == GGML_OP_RESHAPE) {
            const int a0 = find_alias(cur->src[0]);
            const int av = find_alias(cur->view_src);
            if (a0 >= 0 || av >= 0) {
                const int p = (av >= 0) ? av : a0;
                aliases[n_aliases]  = cur;
                cont_src[n_aliases] = cont_src[p];
                n_aliases++;
                node_idxs[count] = j;
                ops[count]       = cur->op;
                count++;
                continue;
            }
        }
        if (cur->op == GGML_OP_CONT && cur->src[0] && (is_alias(cur->src[0]) || is_alias(cur->src[0]->view_src))) {
            aliases[n_aliases]  = cur;
            cont_src[n_aliases] = cur->src[0];
            n_aliases++;
            node_idxs[count] = j;
            ops[count]       = GGML_OP_CONT;
            count++;
            continue;
        }
        if (cur->op == GGML_OP_CPY && cur->src[0]) {
            const int a0 = find_alias(cur->src[0]);
            const int av = find_alias(cur->src[0]->view_src);
            if ((a0 >= 0 || av >= 0) && cur->src[1] != nullptr) {
                if (plan.n_cpys >= 4) {
                    return false;
                }
                const int pa = (a0 >= 0) ? a0 : av;
                const ggml_tensor * slice_t = cont_src[pa] ? cont_src[pa] : cur->src[0];
                const size_t offs = (slice_t == concat_node) ? 0 : slice_t->view_offs;
                if (slice_t->nb[0] != sizeof(float) || slice_t->ne[0] != concat_node->ne[0] || slice_t->ne[3] != 1) {
                    return false;
                }
                if ((slice_t->ne[1] > 1 && slice_t->nb[1] != concat_node->nb[1]) ||
                    (slice_t->ne[2] > 1 && slice_t->nb[2] != concat_node->nb[2]) ||
                    (concat_node->nb[1] == 0) || (offs % concat_node->nb[1] != 0)) {
                    return false;
                }
                const int64_t tok0 = (int64_t) (offs / concat_node->nb[1]);
                if (tok0 < 0 || tok0 + slice_t->ne[1] > concat_node->ne[1] || slice_t->ne[2] > src1->ne[2]) {
                    return false;
                }
                if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float)) {
                    return false;
                }
                if (ranges_overlap(cur->src[1], src1)) {
                    return false;
                }
                if (ranges_overlap(cur->src[1], src0)) {
                    if (tok0 >= src0->ne[1]) {
                        return false;
                    }
                    has_cpy_src0_overlap = true;
                }
                if (tok0 < src0->ne[1]) {
                    const ggml_tensor * cpy_dst = cur->src[1];
                    const int64_t cols = slice_t->ne[1];
                    const int64_t seqs = slice_t->ne[2];
                    const int64_t C    = concat_node->ne[0];
                    const bool shape_2d = cpy_dst->ne[0] == C * cols && cpy_dst->ne[1] == seqs && cpy_dst->ne[2] == 1 && cpy_dst->ne[3] == 1;
                    const bool shape_3d = cpy_dst->ne[0] == C && cpy_dst->ne[1] == cols && cpy_dst->ne[2] == seqs && cpy_dst->ne[3] == 1;
                    if (cpy_dst->type != GGML_TYPE_F32 || cpy_dst->nb[0] != sizeof(float) || (!shape_2d && !shape_3d)) {
                        return false;
                    }
                    if (shape_2d && seqs > 1 &&
                            (src0->nb[1] != (size_t) C * sizeof(float) || src1->nb[1] != (size_t) C * sizeof(float))) {
                        return false;
                    }
                }
                plan.cpys[plan.n_cpys]         = cur;
                plan.cpy_tok0[plan.n_cpys]     = tok0;
                plan.cpy_src_tmpl[plan.n_cpys] = *slice_t;
                plan.n_cpys++;
                node_idxs[count]   = j;
                ops[count]         = GGML_OP_CPY;
                out_idxs[n_outs++] = j;
                count++;
                continue;
            }
        }
        if (cur->op == GGML_OP_SSM_CONV && cur->src[0]) {
            const int a0 = find_alias(cur->src[0]);
            const int av = find_alias(cur->src[0]->view_src);
            const bool direct_concat = (a0 >= 0 && cont_src[a0] == nullptr) ||
                (av >= 0 && cont_src[av] == nullptr && cur->src[0]->view_offs == 0 && ggml_are_same_shape(cur->src[0], concat_node));
            if (direct_concat) {
                if (cur->src[1] == nullptr ||
                        cur->type != GGML_TYPE_F32 || cur->src[1]->type != GGML_TYPE_F32 ||
                        ggml_ssm_conv_get_layout(cur) != GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR) {
                    return false;
                }
                const ggml_tensor * w = cur->src[1];
                if (w->ne[0] < 1 || w->ne[0] > 8 || w->ne[1] != concat_node->ne[0] ||
                        cur->ne[0] != concat_node->ne[0] || cur->ne[2] != concat_node->ne[2] || cur->ne[3] != 1 ||
                        src0->ne[1] + src1->ne[1] != cur->ne[1] + w->ne[0] - 1) {
                    return false;
                }
                plan.ssm_conv = cur;
                ssm_pos = j;
                node_idxs[count] = j;
                ops[count]       = GGML_OP_SSM_CONV;
                count++;
                break;
            }
        }

        // Allow unrelated intervening nodes (e.g. build_rs state fetch/scale) between CONCAT and SSM_CONV.
        if (n_intervening >= 16) {
            return false;
        }
        intervening_idxs[n_intervening++] = j;
    }

    if (ssm_pos < 0 || !plan.ssm_conv) {
        return false;
    }

    int last_pos = ssm_pos;
    if (last_pos + 1 < limit && cgraph->nodes[last_pos + 1]->op == GGML_OP_ADD && count < MAX_SPAN) {
        ggml_tensor * add = cgraph->nodes[last_pos + 1];
        if (add->type == GGML_TYPE_F32 && (add->src[0] == plan.ssm_conv || add->src[1] == plan.ssm_conv)) {
            const ggml_tensor * bias = (add->src[0] == plan.ssm_conv) ? add->src[1] : add->src[0];
            if (bias && bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias) &&
                    ggml_nelements(bias) == plan.ssm_conv->ne[0] && bias->ne[0] == plan.ssm_conv->ne[0]) {
                plan.bias_add = add;
                last_pos++;
                node_idxs[count] = last_pos;
                ops[count]       = GGML_OP_ADD;
                count++;
            }
        }
    }

    ggml_tensor * silu_in = plan.bias_add ? plan.bias_add : plan.ssm_conv;
    if (last_pos + 1 < limit && cgraph->nodes[last_pos + 1]->op == GGML_OP_UNARY && count < MAX_SPAN) {
        ggml_tensor * u = cgraph->nodes[last_pos + 1];
        if (ggml_get_unary_op(u) == GGML_UNARY_OP_SILU && u->type == GGML_TYPE_F32 && u->src[0] == silu_in) {
            plan.silu_dst = u;
            last_pos++;
            node_idxs[count] = last_pos;
            ops[count]       = GGML_OP_UNARY;
            count++;
        }
    }

    ggml_tensor * final_out = plan.silu_dst ? plan.silu_dst : (plan.bias_add ? plan.bias_add : plan.ssm_conv);
    out_idxs[n_outs++] = last_pos;

    if (ggml_cuda_fusion_stream_end(ctx, cgraph, start_idx, last_pos) != last_pos + 1) {
        return false;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, node_idxs, count, ops, out_idxs, n_outs)) {
        return false;
    }

    const ggml_tensor * w    = plan.ssm_conv->src[1];
    const ggml_tensor * bias = plan.bias_add ? ((plan.bias_add->src[0] == plan.ssm_conv) ? plan.bias_add->src[1] : plan.bias_add->src[0]) : nullptr;

    // Verify no intervening node depends on skipped nodes or writes to any tensor used/produced at ssm_pos.
    for (int iv = 0; iv < n_intervening; ++iv) {
        const ggml_tensor * m = cgraph->nodes[intervening_idxs[iv]];
        if ((m->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                m->op == GGML_OP_VIEW || m->op == GGML_OP_RESHAPE ||
                m->op == GGML_OP_PERMUTE || m->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (is_alias(m->src[s]) || (m->src[s] && is_alias(m->src[s]->view_src))) {
                return false;
            }
        }
        if (m->view_src && is_alias(m->view_src)) {
            return false;
        }
        // Intervening node runs before ssm_pos; reject if its write target aliases inputs/outputs at ssm_pos.
        if (ranges_overlap(m, src0) || ranges_overlap(m, src1) || ranges_overlap(m, w) ||
                ranges_overlap(m, bias) || ranges_overlap(m, final_out)) {
            return false;
        }
        for (int k = 0; k < plan.n_cpys; ++k) {
            if (ranges_overlap(m, plan.cpys[k]->src[1])) {
                return false;
            }
        }
    }

    const float * bias_d  = bias ? (const float *) bias->data : nullptr;
    const bool can_vec4   = ssm_conv_concat_can_vec4(src1, bias_d, final_out);
    uint64_t st_s_col     = 0;
    uint64_t st_s_seq     = 0;
    const bool write_state = ssm_conv_concat_try_write_state(
        plan, src0, src1, final_out, (int) w->ne[0], can_vec4, nullptr, &st_s_col, &st_s_seq);

    if (has_cpy_src0_overlap) {
        // Only safe when the single fused WRITE_STATE kernel loads src0 into registers per-channel before overwriting it.
        const int64_t S = final_out->ne[2];
        const ggml_tensor * cpy_dst = plan.cpys[0]->src[1];
        if (!write_state ||
                (const void *) cpy_dst->data != (const void *) src0->data ||
                src0->nb[0] != sizeof(float) ||
                st_s_col != (uint64_t) src0->nb[1] ||
                (S > 1 && st_s_seq != (uint64_t) src0->nb[2])) {
            return false;
        }
    }

    if (ranges_overlap(final_out, src1) || ranges_overlap(final_out, w) || ranges_overlap(final_out, bias)) {
        return false;
    }
    if (ranges_overlap(final_out, src0)) {
        // Safe when gallocr reuses dead src0 starting at exact base pointer for final_out (S == 1):
        // by == 0 loads src0 into registers per channel before writing, by >= 1 only touches src1 and t >= 32 > K-1,
        // and any trailing Step 2 copy from src0 starts at tok0 >= T_out (disjoint from [0 .. T_out)).
        const int K = (int) w->ne[0];
        bool trailing_safe = true;
        if (!write_state) {
            for (int k = 0; k < plan.n_cpys; ++k) {
                if (plan.cpy_tok0[k] < src0->ne[1] && plan.cpy_tok0[k] < final_out->ne[1]) {
                    trailing_safe = false;
                    break;
                }
            }
        }
        if (!can_vec4 || !trailing_safe ||
                final_out->ne[2] != 1 || K < 2 || K > 8 || src0->ne[1] != K - 1 ||
                (const void *) final_out->data != (const void *) src0->data ||
                src0->nb[0] != sizeof(float) || final_out->nb[0] != sizeof(float) ||
                final_out->nb[1] != src0->nb[1] ||
                (uint64_t) src0->nb[1] * 32u < ggml_nbytes(src0)) {
            return false;
        }
    }
    for (int k = 0; k < plan.n_cpys; ++k) {
        if (ranges_overlap(plan.cpys[k]->src[1], final_out)) {
            return false;
        }
    }

    plan.ssm_pos  = ssm_pos;
    plan.last_pos = last_pos;
    plan.n_skip_idxs = 0;
    for (int k = 0; k < count; ++k) {
        if (node_idxs[k] < ssm_pos && plan.n_skip_idxs < 28) {
            plan.skip_idxs[plan.n_skip_idxs++] = node_idxs[k];
        }
    }
    return true;
}

int ggml_cuda_concat_ssm_conv_scheduler::try_step(
        ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    for (size_t p = 0; p < pending.size(); ++p) {
        if (pending[p].ssm_pos == i) {
            ggml_cuda_exec_concat_ssm_conv(ctx, pending[p]);
            const int extra = pending[p].last_pos - i;
            pending.erase(pending.begin() + p);
            return extra;
        }
    }

    if (cgraph->nodes[i]->op == GGML_OP_CONCAT) {
        ggml_cuda_concat_ssm_conv_plan plan;
        if (ggml_cuda_match_concat_ssm_conv(ctx, cgraph, i, plan)) {
            for (int k = 0; k < plan.n_skip_idxs; ++k) {
                const int idx = plan.skip_idxs[k];
                if (idx >= 0 && idx < (int) skip.size()) {
                    skip[idx] = 1;
                }
            }
            if (plan.ssm_pos == i) {
                ggml_cuda_exec_concat_ssm_conv(ctx, plan);
                return plan.last_pos - i;
            }
            pending.push_back(plan);
            return 0;
        }
    }

    return -1;
}

void ggml_cuda_exec_concat_ssm_conv(
        ggml_backend_cuda_context &            ctx,
        const ggml_cuda_concat_ssm_conv_plan & plan) {
    const ggml_tensor * concat_node = plan.concat_node;
    const ggml_tensor * src0        = concat_node->src[0];
    const ggml_tensor * src1        = concat_node->src[1];
    const ggml_tensor * w           = plan.ssm_conv->src[1];
    const ggml_tensor * bias = plan.bias_add
        ? ((plan.bias_add->src[0] == plan.ssm_conv) ? plan.bias_add->src[1] : plan.bias_add->src[0])
        : nullptr;
    ggml_tensor * dst = plan.silu_dst ? plan.silu_dst : (plan.bias_add ? plan.bias_add : plan.ssm_conv);

    const float * bias_d = bias ? (const float *) bias->data : nullptr;
    const bool has_silu  = plan.silu_dst != nullptr;
    cudaStream_t stream  = ctx.stream();

    // 1. Run SSM_CONV(+BIAS+SILU) first reading original src0 and src1 before any CPY updates state.
    const int64_t C     = dst->ne[0];
    const int64_t T_out = dst->ne[1];
    const int64_t S     = dst->ne[2];
    const int     K     = (int) w->ne[0];

    const bool can_vec4 = ssm_conv_concat_can_vec4(src1, bias_d, dst);

    char *   state_dst = nullptr;
    uint64_t st_s_col  = 0;
    uint64_t st_s_seq  = 0;
    const bool fused_state_write = ssm_conv_concat_try_write_state(
        plan, src0, src1, dst, K, can_vec4, &state_dst, &st_s_col, &st_s_seq);

    if (can_vec4 && K >= 1 && K <= 8) {
        switch (K) {
            case 1: dispatch_k<1>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            case 2: dispatch_k<2>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            case 3: dispatch_k<3>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            case 4: dispatch_k<4>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            case 5: dispatch_k<5>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            case 6: dispatch_k<6>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            case 7: dispatch_k<7>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            case 8: dispatch_k<8>(src0, src1, w, bias_d, dst, has_silu, state_dst, st_s_col, st_s_seq, stream); break;
            default: break;
        }
    } else {
        constexpr int BT = 32;
        const dim3 grid((unsigned) ((C + 127) / 128), (unsigned) ((T_out + BT - 1) / BT), (unsigned) S);
        ssm_conv_concat_dim1_scalar_f32<BT><<<grid, 128, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (const char *) w->data,
            bias_d, (char *) dst->data,
            K, C, src0->ne[1], src1->ne[1], T_out,
            src0->nb[0], src0->nb[1], src0->nb[2],
            src1->nb[0], src1->nb[1], src1->nb[2],
            w->nb[0], w->nb[1],
            dst->nb[1], dst->nb[2],
            has_silu);
    }

    if (fused_state_write) {
        return;
    }

    // 2. Run trailing CPY state updates directly from src0/src1 without allocating/writing concat.
    for (int k = 0; k < plan.n_cpys; ++k) {
        const ggml_tensor * cpy = plan.cpys[k];
        const int64_t tok0      = plan.cpy_tok0[k];
        if (tok0 >= src0->ne[1]) {
            ggml_tensor src_slice   = plan.cpy_src_tmpl[k];
            const int64_t tok_off   = tok0 - src0->ne[1];
            src_slice.view_src      = const_cast<ggml_tensor *>(src1);
            src_slice.view_offs     = (size_t) tok_off * src1->nb[1] + (src1->view_src ? src1->view_offs : 0);
            src_slice.data          = (char *) src1->data + (size_t) tok_off * src1->nb[1];
            src_slice.nb[0]         = src1->nb[0];
            src_slice.nb[1]         = src1->nb[1];
            src_slice.nb[2]         = src1->nb[2];
            src_slice.nb[3]         = src1->nb[3];
            ggml_cuda_cpy(ctx, &src_slice, cpy->src[1]);
            continue;
        }

        const int64_t cols_total = plan.cpy_src_tmpl[k].ne[1];
        const int64_t seqs       = plan.cpy_src_tmpl[k].ne[2];
        const int64_t cols_0     = std::min(cols_total, src0->ne[1] - tok0);
        const int64_t cols_1     = cols_total - cols_0;
        ggml_tensor * cpy_dst    = cpy->src[1];
        const bool dst_2d        = cpy_dst->ne[0] == C * cols_total;

        auto copy_part = [&](const ggml_tensor * base_src, int64_t src_tok, int64_t dst_col, int64_t cols) {
            if (cols <= 0) {
                return;
            }
            ggml_tensor s_part = plan.cpy_src_tmpl[k];
            s_part.view_src    = const_cast<ggml_tensor *>(base_src);
            s_part.view_offs   = (size_t) src_tok * base_src->nb[1] + (base_src->view_src ? base_src->view_offs : 0);
            s_part.data        = (char *) base_src->data + (size_t) src_tok * base_src->nb[1];
            if (dst_2d) {
                s_part.ne[0] = C * cols;
                s_part.ne[1] = seqs;
                s_part.ne[2] = 1;
                s_part.nb[0] = sizeof(float);
                s_part.nb[1] = base_src->nb[2];
                s_part.nb[2] = base_src->nb[2] * seqs;
                s_part.nb[3] = s_part.nb[2];
            } else {
                s_part.ne[1] = cols;
                s_part.nb[0] = base_src->nb[0];
                s_part.nb[1] = base_src->nb[1];
                s_part.nb[2] = base_src->nb[2];
                s_part.nb[3] = base_src->nb[3];
            }

            ggml_tensor d_part = *cpy_dst;
            const size_t dst_off = dst_2d
                ? (size_t) dst_col * (size_t) C * sizeof(float)
                : (size_t) dst_col * cpy_dst->nb[1];
            d_part.ne[0]     = dst_2d ? C * cols : C;
            d_part.ne[1]     = dst_2d ? seqs : cols;
            d_part.view_src  = cpy_dst->view_src ? cpy_dst->view_src : cpy_dst;
            d_part.view_offs = (cpy_dst->view_src ? cpy_dst->view_offs : 0) + dst_off;
            d_part.data      = (char *) cpy_dst->data + dst_off;
            ggml_cuda_cpy(ctx, &s_part, &d_part);
        };

        copy_part(src0, tok0, 0, cols_0);
        copy_part(src1, 0, cols_0, cols_1);
    }
}
