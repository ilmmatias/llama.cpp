#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_backend_cpu_expert_cache_configure(
        uint32_t slots,
        uint32_t admit_window,
        uint32_t convert_workers,
        ggml_backend_dev_t device);

// CPU_REPACK cache hooks. begin() freezes the ready cache-hit set, launches
// cached expert work on the GPU, and returns the route-position bit mask to
// omit from CPU MUL_MAT_ID. end() joins the GPU result after CPU work.
uint64_t ggml_backend_cpu_expert_cache_begin(struct ggml_tensor * op);
void     ggml_backend_cpu_expert_cache_end  (struct ggml_tensor * op);

#ifdef __cplusplus
}
#endif
