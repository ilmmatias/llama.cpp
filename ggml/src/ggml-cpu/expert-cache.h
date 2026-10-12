#pragma once

#include "ggml-cpu.h"
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
// Masks remain valid until end(). CPU consumers keep the host join.
const uint64_t * ggml_backend_cpu_expert_cache_begin(struct ggml_tensor * op);
void             ggml_backend_cpu_expert_cache_end  (struct ggml_tensor * op);

ggml_backend_dev_t ggml_backend_cpu_expert_cache_device(const struct ggml_tensor * op);
void ggml_backend_cpu_expert_cache_prepare(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
bool ggml_backend_cpu_expert_cache_copy(ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
