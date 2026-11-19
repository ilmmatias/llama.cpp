#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

struct ggml_cuda_gdn_norm_spec {
    uint8_t             mode  = 0; // 0 = off, 1 = rms_norm+scale, 2 = l2_norm
    float               eps   = 0.0f;
    float               scale = 1.0f;
    const ggml_tensor * raw   = nullptr;
};

struct ggml_cuda_gated_delta_net_fused_prep {
    ggml_cuda_gdn_norm_spec q_norm;
    ggml_cuda_gdn_norm_spec k_norm;

    bool                fuse_gate = false;
    const ggml_tensor * alpha_raw = nullptr;
    const ggml_tensor * dt        = nullptr;
    const ggml_tensor * ssm_a     = nullptr;

    bool                fuse_beta = false;
    const ggml_tensor * beta_raw  = nullptr;

    bool empty() const {
        return q_norm.mode == 0 && k_norm.mode == 0 && !fuse_gate && !fuse_beta;
    }
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gated_delta_net_fused_cache * cache,
                                        const ggml_cuda_gated_delta_net_fused_prep * prep);

#define GDN_AB_FUSION_MAX_TOKENS 8 // Max. number of tokens for which to fuse the GDN alpha/beta projections.

// the alpha/beta projections that feed gated_delta_net
struct ggml_cuda_gated_delta_net_ab {
    const ggml_tensor * alpha_mm;
    const ggml_tensor * beta_mm;
    const ggml_tensor * dt;
    const ggml_tensor * ssm_a;
    ggml_tensor *       gate_out;
    ggml_tensor *       beta_out;
};

bool ggml_cuda_gated_delta_net_ab_supported_type(ggml_type type);

// runs both projections and their activation tails in one kernel
void ggml_cuda_op_gated_delta_net_ab(ggml_backend_cuda_context & ctx, const ggml_cuda_gated_delta_net_ab & ab);
