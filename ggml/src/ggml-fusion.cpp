#include "ggml-fusion.h"
#include "ggml-impl.h"

#include <climits>
#include <cstring>

static bool fusion_reduces(const ggml_tensor * node) {
    return node->op == GGML_OP_SUM_ROWS || node->op == GGML_OP_MEAN || node->op == GGML_OP_SUM;
}

static int fusion_sources(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL: return 2;
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_SUM: return 1;
        case GGML_OP_UNARY: {
            const auto unary = ggml_get_unary_op(node);
            return unary == GGML_UNARY_OP_NEG || unary == GGML_UNARY_OP_RELU || unary == GGML_UNARY_OP_SILU ? 1 : 0;
        }
        default: return 0;
    }
}

static bool fusion_layout(const ggml_tensor * tensor, uint64_t & n) {
    if (!tensor || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16)) {
        return false;
    }
    const size_t element = ggml_type_size(tensor->type);

    n = 1;
    size_t span = element;
    int order[4] = {0, 1, 2, 3};
    for (int d = 0; d < 4; ++d) {
        if (tensor->ne[d] <= 0 || n > uint64_t(INT64_MAX) / uint64_t(tensor->ne[d]) ||
            tensor->nb[d] % element || (tensor->ne[d] > 1 && !tensor->nb[d]) ||
            uint64_t(tensor->ne[d] - 1) > (SIZE_MAX - span) / (tensor->nb[d] ? tensor->nb[d] : 1)) {
            return false;
        }
        n *= uint64_t(tensor->ne[d]);
        span += size_t(tensor->ne[d] - 1) * tensor->nb[d];
    }
    if (n > SIZE_MAX / element) {
        return false;
    }

    for (int d = 1; d < 4; ++d) {
        for (int j = d; j > 0 && tensor->nb[order[j]] < tensor->nb[order[j - 1]]; --j) {
            const int previous = order[j - 1];
            order[j - 1] = order[j];
            order[j] = previous;
        }
    }
    span = element;
    for (int d : order) {
        if (tensor->ne[d] > 1) {
            if (tensor->nb[d] < span) {
                return false;
            }
            span += size_t(tensor->ne[d] - 1) * tensor->nb[d];
        }
    }
    return true;
}

static bool fusion_dense(const ggml_tensor * tensor, const ggml_tensor * shape, uint64_t & n) {
    return fusion_layout(tensor, n) && ggml_are_same_shape(tensor, shape) && ggml_is_contiguous(tensor);
}

static ggml_fusion_access fusion_access(const ggml_tensor * tensor) {
    ggml_fusion_access access;
    access.type = tensor->type;
    for (int d = 0; d < 4; ++d) {
        access.ne[d] = tensor->ne[d];
        access.nb[d] = tensor->nb[d];
    }
    return access;
}

static bool fusion_source(const ggml_tensor * source, const ggml_tensor * node, int slot, uint64_t & n) {
    if (!fusion_layout(source, n)) {
        return false;
    }
    if ((node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) && slot == 1) {
        for (int d = 0; d < 4; ++d) {
            if (node->ne[d] % source->ne[d]) {
                return false;
            }
        }
        return true;
    }
    return ggml_are_same_shape(source, node) &&
        ((node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) || ggml_is_contiguous(source));
}

static bool fusion_types(const ggml_tensor * node) {
    const auto a = node->src[0]->type;
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        const auto b = node->src[1]->type;
        return (a == GGML_TYPE_F32 && b != GGML_TYPE_BF16 && node->type == GGML_TYPE_F32) ||
            (a == GGML_TYPE_F16 && (b == GGML_TYPE_F16 || b == GGML_TYPE_F32) &&
                (node->type == GGML_TYPE_F16 || (b == GGML_TYPE_F32 && node->type == GGML_TYPE_F32))) ||
            (a == GGML_TYPE_BF16 && (b == GGML_TYPE_BF16 || b == GGML_TYPE_F32) && node->type == GGML_TYPE_BF16);
    }
    return node->type == a && (node->op != GGML_OP_SCALE || a != GGML_TYPE_F16);
}

static int fusion_member(const ggml_cgraph * graph, int start, int count, const ggml_tensor * tensor) {
    for (int i = 0; i < count; ++i) {
        if (graph->nodes[start + i] == tensor) {
            return i;
        }
    }
    return -1;
}

struct fusion_builder {
    const ggml_cgraph * graph;
    int start;
    int count;
    ggml_fusion_region & region;
    ggml_fusion_program & program;
    int references[GGML_FUSION_MAX_VALUES];
    bool visiting[GGML_FUSION_MAX_VALUES] = {};
    const uint8_t * domains = nullptr;

    int visit(int member) {
        if (references[member] >= 0) {
            return references[member];
        }
        if (visiting[member]) {
            return -1;
        }
        visiting[member] = true;

        const auto * node = graph->nodes[start + member];
        ggml_fusion_value value;
        value.op = node->op;
        value.type = node->type;
        value.domain = domains ? domains[member] : 0;
        if (node->op == GGML_OP_UNARY) {
            value.unary = ggml_get_unary_op(node);
        }

        for (int s = 0; s < fusion_sources(node); ++s) {
            const auto * source = node->src[s];
            const int internal = fusion_member(graph, start, count, source);
            if (internal >= 0) {
                if (internal >= member) {
                    return -1;
                }
                value.src[s] = visit(internal);
                if (value.src[s] < 0) {
                    return -1;
                }
                continue;
            }

            int depth = 0;
            for (const auto * ancestor = source->view_src; ancestor; ancestor = ancestor->view_src) {
                if (++depth > 64 || fusion_member(graph, start, count, ancestor) >= 0) {
                    return -1;
                }
            }

            auto access = fusion_access(source);
            access.domain = fusion_reduces(node) ? 1 : value.domain;
            const auto * shape = fusion_reduces(node) ? source : node;
            for (int d = 0; d < 4; ++d) {
                if (source->ne[d] != shape->ne[d]) {
                    access.repeat |= 1 << d;
                }
            }

            int slot = 0;
            while (slot < program.inputs &&
                   (region.inputs[slot] != source || program.accesses[slot].repeat != access.repeat ||
                    program.accesses[slot].domain != access.domain)) {
                ++slot;
            }
            if (slot == program.inputs) {
                if (slot == GGML_FUSION_MAX_INPUTS) {
                    return -1;
                }
                region.inputs[slot] = source;
                program.accesses[slot] = access;
                program.indexed |= access.repeat || !ggml_is_contiguous(source);
                ++program.inputs;
            }
            value.src[s] = -1 - slot;
        }

        if (node->op == GGML_OP_SCALE) {
            if (program.params + 2 > GGML_FUSION_MAX_PARAMS) {
                return -1;
            }
            value.param = program.params;
            for (int p = 0; p < 2; ++p) {
                region.parameters[program.params] = node;
                region.params[program.params++] = ggml_get_op_params_f32(node, p);
            }
        }

        const int reference = program.count++;
        program.values[reference] = value;
        references[member] = reference;
        visiting[member] = false;
        return reference;
    }
};

static bool fusion_connected(const ggml_cgraph * graph, int start, int count) {
    bool reached[GGML_FUSION_MAX_VALUES] = {};
    reached[0] = true;
    for (int pass = 0; pass < count; ++pass) {
        for (int i = 0; i < count; ++i) {
            if (!reached[i]) {
                continue;
            }
            const auto * a = graph->nodes[start + i];
            for (int j = 0; j < count; ++j) {
                const auto * b = graph->nodes[start + j];
                for (int s = 0; s < fusion_sources(a); ++s) {
                    if (a->src[s] == b) {
                        reached[j] = true;
                    }
                    for (int t = 0; t < fusion_sources(b); ++t) {
                        if (b->src[t] == a || b->src[t] == a->src[s]) {
                            reached[j] = true;
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < count; ++i) {
        if (!reached[i]) {
            return false;
        }
    }
    return true;
}

static bool fusion_liveness(const ggml_fusion_program & program) {
    int input_last[GGML_FUSION_MAX_INPUTS] = {};
    int value_last[GGML_FUSION_MAX_VALUES] = {};
    for (int i = 0; i < program.count; ++i) {
        const auto & value = program.values[i];
        const int sources = value.op == GGML_OP_ADD || value.op == GGML_OP_MUL ? 2 : 1;
        value_last[i] = i;
        for (int s = 0; s < sources; ++s) {
            const int reference = value.src[s];
            if (reference < 0) {
                input_last[-1 - reference] = i;
            } else {
                value_last[reference] = i;
            }
        }
    }
    for (int i = 0; i < program.outputs; ++i) {
        value_last[program.roots[i]] = program.count;
    }

    if (program.inputs > 24) {
        return false;
    }
    for (int i = 0; i < program.count; ++i) {
        int live = 1;
        for (int k = 0; k < program.inputs; ++k) {
            live += input_last[k] >= i;
        }
        for (int j = 0; j < i; ++j) {
            live += value_last[j] >= i;
        }
        if (live > 24) {
            return false;
        }
    }
    return true;
}

static bool fusion_mark_domain(const ggml_cgraph * graph, int start, int count, int member,
                               int reduction, uint8_t domain, uint8_t * domains) {
    if (member == reduction && domain == 2) {
        domains[member] = 2;
        return true;
    }
    if (domains[member]) {
        return domains[member] == domain;
    }
    domains[member] = domain;
    const auto * node = graph->nodes[start + member];
    for (int s = 0; s < fusion_sources(node); ++s) {
        const int source = fusion_member(graph, start, count, node->src[s]);
        if (source >= member || (source >= 0 && !fusion_mark_domain(graph, start, count, source, reduction, domain, domains))) {
            return false;
        }
    }
    return true;
}

bool ggml_fusion_build(const ggml_cgraph * graph, int start, ggml_fusion_region & region, ggml_fusion_program & program) {
    region = {};
    program = {};
    if (!graph || start < 0 || start >= graph->n_nodes) {
        return false;
    }

    ggml_op ops[GGML_FUSION_MAX_VALUES];
    int count = 0;
    uint64_t n;
    const ggml_tensor * shape = fusion_reduces(graph->nodes[start]) ? graph->nodes[start]->src[0] : graph->nodes[start];
    int reduction = -1;
    while (count < GGML_FUSION_MAX_VALUES && count < graph->n_nodes - start) {
        const auto * node = graph->nodes[start + count];
        const int sources = fusion_sources(node);
        if (fusion_reduces(node)) {
            uint64_t elements;
            if (reduction >= 0 || node->type != GGML_TYPE_F32 || !node->src[0] ||
                node->src[0]->type != GGML_TYPE_F32 || !fusion_dense(node->src[0], shape, elements) ||
                (node->op == GGML_OP_SUM ? elements > 16777216 : node->src[0]->ne[0] > 4096)) {
                break;
            }
            if (fusion_sources(node->src[0]) && fusion_member(graph, start, count, node->src[0]) < 0) {
                break;
            }
            reduction = count;
            shape = node;
        }
        if (!sources || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || node->view_src ||
            !fusion_dense(node, shape, n)) {
            break;
        }

        bool valid = true;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (s < sources) {
                valid = valid && (fusion_reduces(node) ? s == 0 : fusion_source(node->src[s], node, s, n));
            } else {
                valid = valid && node->src[s] == nullptr;
            }
        }
        if (!valid || !fusion_types(node)) {
            break;
        }
        ops[count++] = node->op;
    }

    for (; count >= 2; --count) {
        const int last = start + count - 1;
        if (!fusion_connected(graph, start, count)) {
            continue;
        }

        region = {};
        program = {};
        region.count = count;
        for (int i = 0; i < count; ++i) {
            region.members[i] = start + i;
        }

        bool valid = true;
        for (int i = 0; i < count; ++i) {
            const auto * node = graph->nodes[start + i];
            for (int j = i + 1; j < count; ++j) {
                const auto * consumer = graph->nodes[start + j];
                for (int s = 0; s < fusion_sources(consumer); ++s) {
                    region.uses[i] += consumer->src[s] == node;
                }
            }
            if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                ggml_node_get_use_count(graph, start + i) > region.uses[i] || region.uses[i] == 0) {
                if (program.outputs == GGML_FUSION_MAX_OUTPUTS) {
                    valid = false;
                    break;
                }
                const int slot = program.outputs++;
                region.outputs[slot] = node;
                region.output_indices[slot] = start + i;
                program.stores[slot] = fusion_access(node);
            }
        }
        uint8_t domains[GGML_FUSION_MAX_VALUES] = {};
        const int reduced = reduction < count ? reduction : -1;
        if (valid && reduced >= 0) {
            const auto * reduce = graph->nodes[start + reduced];
            const int producer = fusion_member(graph, start, count, reduce->src[0]);
            valid = producer < 0 || fusion_mark_domain(graph, start, count, producer, reduced, 1, domains);
            for (int k = 0; valid && k < program.outputs; ++k) {
                valid = fusion_mark_domain(graph, start, count, region.output_indices[k] - start, reduced, 2, domains);
                program.stores[k].domain = 2;
            }
            int producers = 0;
            for (int i = 0; valid && i < count; ++i) {
                const auto * node = graph->nodes[start + i];
                valid = domains[i] && node->type == GGML_TYPE_F32;
                if (domains[i] == 1) {
                    ++producers;
                    valid = valid && !(node->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                        ggml_node_get_use_count(graph, start + i) == region.uses[i] && region.uses[i] > 0;
                }
            }
            uint64_t elements;
            valid = valid && fusion_dense(reduce->src[0], reduce->src[0], elements);
            if (valid && reduce->op == GGML_OP_SUM && elements > 4096 && producers < 2) {
                valid = false;
            }
        }
        if (!valid || !ggml_can_fuse_subgraph_ext(graph, region.members, count, ops,
                                                region.output_indices, program.outputs)) {
            continue;
        }
        const auto * domain_shape = reduced >= 0 ? graph->nodes[start + reduced]->src[0] : graph->nodes[last];
        fusion_dense(domain_shape, domain_shape, region.n);
        for (int d = 0; d < 4; ++d) {
            program.ne[d] = domain_shape->ne[d];
        }

        fusion_builder builder {graph, start, count, region, program, {}, {}, reduced >= 0 ? domains : nullptr};
        for (int & reference : builder.references) {
            reference = -1;
        }
        for (int k = 0; k < program.outputs; ++k) {
            const int root = builder.visit(region.output_indices[k] - start);
            if (root < 0) {
                valid = false;
                break;
            }
            program.roots[k] = root;
        }
        if (valid && reduced >= 0) {
            program.reduction.value = builder.references[reduced];
            program.reduction.axes = graph->nodes[start + reduced]->op == GGML_OP_SUM ? 15 : 1;
            for (int k = 0; k < program.inputs; ++k) {
                const auto & access = program.accesses[k];
                if (access.type != GGML_TYPE_F32 ||
                    (access.domain == 1 && access.ne[0] > 1 && access.nb[0] != sizeof(float))) {
                    valid = false;
                }
            }
        }
        if (valid && program.count == count && fusion_liveness(program)) {
            return true;
        }
    }
    return false;
}

bool ggml_fusion_make_key(const ggml_fusion_program & program, const ggml_fusion_schedule & schedule, ggml_fusion_cache_key & key) {
    key = {};
    bool valid = true;
    auto put = [&](uint64_t value, size_t bytes) {
        if (bytes > sizeof(key.bytes) - key.size) {
            valid = false;
            return;
        }
        for (size_t i = 0; i < bytes; ++i) {
            key.bytes[key.size++] = uint8_t(value >> (8 * i));
        }
    };

    if (program.count < 2 || program.count > GGML_FUSION_MAX_VALUES ||
        program.inputs < 0 || program.inputs > GGML_FUSION_MAX_INPUTS ||
        program.outputs < 1 || program.outputs > GGML_FUSION_MAX_OUTPUTS ||
        program.params < 0 || program.params > GGML_FUSION_MAX_PARAMS) {
        return false;
    }

    put(2, 4);
    put(program.count, 4);
    put(program.inputs, 4);
    put(program.outputs, 4);
    put(program.params, 4);
    put(program.indexed, 1);
    if (program.indexed) {
        for (int d = 0; d < 4; ++d) {
            put(program.ne[d], 8);
        }
    }
    for (int i = 0; i < program.count; ++i) {
        const auto & value = program.values[i];
        put(value.op, 4);
        if (value.op == GGML_OP_UNARY) {
            put(value.unary, 4);
        }
        put(value.type, 4);
        put(value.src[0], 4);
        if (value.op == GGML_OP_ADD || value.op == GGML_OP_MUL) {
            put(value.src[1], 4);
        }
        if (value.op == GGML_OP_SCALE) {
            put(value.param, 4);
        }
        put(value.domain, 1);
    }
    for (int i = 0; i < program.inputs; ++i) {
        const auto & access = program.accesses[i];
        put(access.type, 4);
        put(access.repeat, 1);
        put(access.domain, 1);
        if (program.indexed) {
            for (int d = 0; d < 4; ++d) {
                put(access.ne[d], 8);
                put(access.nb[d], 8);
            }
        }
    }
    for (int i = 0; i < program.outputs; ++i) {
        put(program.roots[i], 4);
        put(program.stores[i].type, 4);
    }

    put(program.reduction.value, 4);
    put(program.reduction.axes, 1);
    put(program.reduction.accumulator, 4);

    put(schedule.kind, 4);
    put(schedule.threads, 4);
    put(schedule.wave, 4);
    put(schedule.tile, 4);
    put(schedule.stages, 4);
    put(schedule.scratch, 8);
    return valid;
}
