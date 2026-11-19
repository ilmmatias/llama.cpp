#include "common.cuh"

#define CUDA_IM2COL_BLOCK_SIZE 256

void ggml_cuda_op_im2col(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_conv_1d_dw_fused(ggml_backend_cuda_context & ctx, ggml_tensor * im2col_node, ggml_tensor * mul_mat_node);
