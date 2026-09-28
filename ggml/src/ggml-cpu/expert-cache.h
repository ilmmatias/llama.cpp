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

// begin() freezes the batch's ready hits and launches cached expert work.
// Returns one route-position mask per ids token, or NULL for CPU-only work.
// Masks remain valid until end(), which joins GPU results after CPU work.
const uint64_t * ggml_backend_cpu_expert_cache_begin(struct ggml_tensor * op);
void             ggml_backend_cpu_expert_cache_end  (struct ggml_tensor * op);

#ifdef __cplusplus
}
#endif
