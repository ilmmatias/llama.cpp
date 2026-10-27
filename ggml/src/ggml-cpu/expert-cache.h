#pragma once

#include "ggml-cpu.h"

enum { GGML_CPU_EXPERT_CACHE_MAX_TOKENS = 128 };

static inline bool ggml_cpu_expert_cache_supports_ids(const struct ggml_tensor * ids) {
    return ids != NULL && ids->type == GGML_TYPE_I32 &&
        ids->ne[0] > 0 && ids->ne[0] <= 64 && ids->ne[1] > 0 && ids->ne[1] <= GGML_CPU_EXPERT_CACHE_MAX_TOKENS &&
        ids->ne[2] == 1 && ids->ne[3] == 1;
}

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_cpu_expert_cache_unpack(const struct ggml_tensor * tensor, int32_t expert, void * dst, size_t size);

#ifdef __cplusplus
}
#endif
