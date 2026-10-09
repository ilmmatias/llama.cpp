#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>

struct ggml_cgraph;

static constexpr int GGML_FUSION_MAX_VALUES = 16;
static constexpr int GGML_FUSION_MAX_INPUTS = 32;
static constexpr int GGML_FUSION_MAX_OUTPUTS = 8;
static constexpr int GGML_FUSION_MAX_PARAMS = 32;

struct ggml_fusion_access {
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {};
    size_t nb[4] = {};
    uint8_t repeat = 0;
    uint8_t domain = 0;
};

struct ggml_fusion_value {
    ggml_op op = GGML_OP_NONE;
    ggml_unary_op unary = GGML_UNARY_OP_NEG;
    ggml_type type = GGML_TYPE_F32;
    int src[2] = {0, 0};
    int param = -1;
    uint8_t domain = 0;
};

struct ggml_fusion_reduction {
    int value = -1;
    uint8_t axes = 0;
    ggml_type accumulator = GGML_TYPE_F32;
};

struct ggml_fusion_program {
    int count = 0;
    int inputs = 0;
    int outputs = 0;
    int params = 0;
    int64_t ne[4] = {};
    bool indexed = false;
    ggml_fusion_value values[GGML_FUSION_MAX_VALUES];
    ggml_fusion_access accesses[GGML_FUSION_MAX_INPUTS];
    ggml_fusion_access stores[GGML_FUSION_MAX_OUTPUTS];
    int roots[GGML_FUSION_MAX_OUTPUTS] = {};
    ggml_fusion_reduction reduction;
};

// Tensor identities and runtime scalars never enter a compiled program.
struct ggml_fusion_region {
    int count = 0;
    int members[GGML_FUSION_MAX_VALUES] = {};
    int uses[GGML_FUSION_MAX_VALUES] = {};
    int output_indices[GGML_FUSION_MAX_OUTPUTS] = {};
    const ggml_tensor * inputs[GGML_FUSION_MAX_INPUTS] = {};
    const ggml_tensor * outputs[GGML_FUSION_MAX_OUTPUTS] = {};
    const ggml_tensor * parameters[GGML_FUSION_MAX_PARAMS] = {};
    float params[GGML_FUSION_MAX_PARAMS] = {};
    uint64_t n = 0;
};

enum ggml_fusion_schedule_kind {
    GGML_FUSION_SCHEDULE_POINTWISE_FLAT,
    GGML_FUSION_SCHEDULE_POINTWISE_INDEXED,
    GGML_FUSION_SCHEDULE_REDUCE_ROWS,
    GGML_FUSION_SCHEDULE_REDUCE_ALL_TWO_STAGE,
};

struct ggml_fusion_schedule {
    ggml_fusion_schedule_kind kind = GGML_FUSION_SCHEDULE_POINTWISE_FLAT;
    uint32_t threads = 256;
    uint32_t wave = 0;
    uint32_t tile = 0;
    uint32_t stages = 1;
    uint64_t scratch = 0;
};

struct ggml_fusion_cache_key {
    size_t size = 0;
    uint8_t bytes[8192] = {};
};

bool ggml_fusion_build(const ggml_cgraph * graph, int start, ggml_fusion_region & region, ggml_fusion_program & program);
bool ggml_fusion_make_key(const ggml_fusion_program & program, const ggml_fusion_schedule & schedule, ggml_fusion_cache_key & key);
