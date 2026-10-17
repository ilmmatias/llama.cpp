#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cpu_fusion_plan;

// Prepared before dispatch; bindings are immutable until all workers finish.
GGML_BACKEND_API struct ggml_cpu_fusion_plan * ggml_cpu_fusion_plan_create(const struct ggml_cgraph * graph);
GGML_BACKEND_API void                          ggml_cpu_fusion_plan_free  (struct ggml_cpu_fusion_plan * plan);
GGML_BACKEND_API int                           ggml_cpu_fusion_compute    (const struct ggml_cpu_fusion_plan * plan, int node, int ith, int nth);

#ifdef __cplusplus
}
#endif
