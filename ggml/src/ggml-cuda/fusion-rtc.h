#pragma once

struct ggml_backend_cuda_context;
struct ggml_cgraph;
struct ggml_cuda_rtc_fusion;
struct ggml_cuda_graph;
struct ggml_cuda_rtc_plan;

bool ggml_cuda_rtc_fusion_enabled();

bool ggml_cuda_rtc_fusion_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph,
                                  ggml_cuda_graph * capture_graph, bool graph_compatible);
void ggml_cuda_rtc_fusion_record(ggml_backend_cuda_context & ctx, ggml_cuda_graph * capture_graph);

int ggml_cuda_rtc_fusion_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int node_idx, bool allow_compile);

void ggml_cuda_rtc_fusion_free(ggml_backend_cuda_context & ctx);
