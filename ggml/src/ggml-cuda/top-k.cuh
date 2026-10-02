#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
// mask is an F16 mask bank or I32 QSA visibility metadata.
void ggml_cuda_qsa_refine(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * scores,
        const ggml_tensor * cells,
        const ggml_tensor * mask,
        ggml_tensor * dst);
#endif
