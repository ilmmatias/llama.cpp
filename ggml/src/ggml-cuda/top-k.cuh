#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_USE_HIP
bool ggml_cuda_top_k_hip_uses_radix(const ggml_tensor * dst);
#endif
