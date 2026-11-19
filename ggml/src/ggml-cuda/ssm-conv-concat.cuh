#pragma once

#include "common.cuh"

#include <vector>

struct ggml_cuda_concat_ssm_conv_plan {
    const ggml_tensor * concat_node = nullptr;
    ggml_tensor *       ssm_conv    = nullptr;
    ggml_tensor *       bias_add    = nullptr;
    ggml_tensor *       silu_dst    = nullptr;
    ggml_tensor *       cpys[4]     = {};
    int64_t             cpy_tok0[4] = {};
    ggml_tensor         cpy_src_tmpl[4] = {};
    int                 n_cpys      = 0;
    int                 ssm_pos     = -1;
    int                 last_pos    = -1;
    int                 skip_idxs[28] = {};
    int                 n_skip_idxs = 0;
};

struct ggml_cuda_concat_ssm_conv_scheduler {
    std::vector<uint8_t> skip;
    std::vector<ggml_cuda_concat_ssm_conv_plan> pending;

    void reset(int n_nodes) {
        if ((int) skip.size() < n_nodes) {
            skip.assign(n_nodes, 0);
        } else {
            std::fill(skip.begin(), skip.begin() + n_nodes, 0);
        }
        pending.clear();
    }

    bool should_skip(int i) const {
        return i < (int) skip.size() && skip[i] != 0;
    }

    // Returns >= 0 as extra nodes to advance `i` after handling node `i`, or -1 if unchanged.
    int try_step(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
};

// Matches CONCAT(dim=1) -> {VIEW|CPY|independent}* -> SSM_CONV(channels_major) (+ ADD)? (+ SILU)?
// when CONCAT has no external consumers and all trailing CPY slices lie in src1.
bool ggml_cuda_match_concat_ssm_conv(
        ggml_backend_cuda_context & ctx,
        ggml_cgraph *               cgraph,
        int                         start_idx,
        ggml_cuda_concat_ssm_conv_plan & plan);

void ggml_cuda_exec_concat_ssm_conv(
        ggml_backend_cuda_context &            ctx,
        const ggml_cuda_concat_ssm_conv_plan & plan);
