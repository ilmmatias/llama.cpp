#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_backend_cpu_expert_cache_shadow_configure(
        uint32_t slots,
        uint32_t admit_window,
        ggml_backend_dev_t device);

void ggml_backend_cpu_expert_cache_hybrid_configure(
        uint32_t slots,
        uint32_t admit_window,
        uint32_t convert_workers,
        ggml_backend_dev_t device);

void ggml_backend_cpu_expert_cache_shadow_route(
        const struct ggml_tensor * weights,
        const struct ggml_tensor * ids);

// CPU_REPACK hybrid hooks. begin() is called by thread 0 before a MUL_MAT_ID
// is grouped; its return value is a route-position bit mask to omit from the
// CPU work. end() is called by thread 0 after its share of CPU work completes.
uint64_t ggml_backend_cpu_expert_cache_hybrid_begin(struct ggml_tensor * op);
void     ggml_backend_cpu_expert_cache_hybrid_end  (struct ggml_tensor * op);

#ifdef __cplusplus
}
#endif
