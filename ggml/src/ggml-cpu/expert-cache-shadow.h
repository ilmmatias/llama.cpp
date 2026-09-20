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

void ggml_backend_cpu_expert_cache_shadow_route(
        const struct ggml_tensor * weights,
        const struct ggml_tensor * ids);

#ifdef __cplusplus
}
#endif
