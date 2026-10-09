#pragma once

struct ggml_backend_cuda_context;
struct ggml_cgraph;
struct ggml_cuda_rtc_fusion;

bool ggml_cuda_rtc_fusion_enabled();
int ggml_cuda_rtc_fusion_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int node_idx);
void ggml_cuda_rtc_fusion_free(ggml_backend_cuda_context & ctx);
