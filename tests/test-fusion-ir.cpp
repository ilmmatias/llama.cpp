#include "ggml-fusion.h"
#include "ggml-impl.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1); \
        } \
    } while (0)

static void test_materialization_and_extents(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    auto * u = ggml_add(ctx, a, b);
    auto * v = ggml_sqr(ctx, u);
    auto * out = ggml_scale_bias(ctx, v, 0.5f, -1.0f);

    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_fusion_region region;
    ggml_fusion_program program;
    REQUIRE(ggml_fusion_build(graph, 0, region, program));
    REQUIRE(program.count == 3 && region.n == 513);
    REQUIRE(program.values[1].src[0] == 0 && program.values[2].src[0] == 1);

    ggml_set_output(u);
    REQUIRE(ggml_fusion_build(graph, 1, region, program));
    REQUIRE(region.inputs[0] == u && program.count == 2);

    auto slice = ggml_graph_view(graph, 0, 2);
    REQUIRE(ggml_fusion_build(&slice, 0, region, program));
    REQUIRE(program.outputs == 2 && region.outputs[0] == u && region.outputs[1] == v);

    u->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    REQUIRE(ggml_fusion_build(&slice, 0, region, program));
    REQUIRE(region.outputs[0] == v);

    const auto saved = a->ne[0];
    a->ne[0] = -1;
    REQUIRE(!ggml_fusion_build(graph, 0, region, program));

    a->ne[0] = INT64_MAX;
    a->ne[1] = 2;
    REQUIRE(!ggml_fusion_build(graph, 0, region, program));

    a->ne[0] = saved;
    a->ne[1] = 1;
    REQUIRE(ggml_fusion_build(graph, 0, region, program));
}

static void test_dependencies_and_keys(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_fusion_region region;
    ggml_fusion_program program;

    auto * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
    auto * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
    auto * left = ggml_add(ctx, a, b);
    auto * right = ggml_mul(ctx, c, d);
    auto * join = ggml_add(ctx, left, right);

    auto * dag = ggml_new_graph(ctx);
    ggml_build_forward_expand(dag, join);
    REQUIRE(ggml_fusion_build(dag, 0, region, program));

    ggml_fusion_cache_key first;
    ggml_fusion_cache_key second;
    REQUIRE(ggml_fusion_make_key(program, {}, first));

    std::swap(dag->nodes[0], dag->nodes[1]);
    REQUIRE(ggml_fusion_build(dag, 0, region, program));
    REQUIRE(ggml_fusion_make_key(program, {}, second));
    REQUIRE(first.size == second.size && std::memcmp(first.bytes, second.bytes, first.size) == 0);

    auto * consumer = ggml_sqr(ctx, left);
    ggml_build_forward_expand(dag, consumer);
    auto dag_slice = ggml_graph_view(dag, 0, 3);
    REQUIRE(ggml_fusion_build(&dag_slice, 0, region, program));
    REQUIRE(program.outputs == 2 && region.outputs[0] == left && region.outputs[1] == join);

    auto * disconnected = ggml_new_graph(ctx);
    ggml_build_forward_expand(disconnected, ggml_neg(ctx, a));
    ggml_build_forward_expand(disconnected, ggml_sqr(ctx, c));
    REQUIRE(!ggml_fusion_build(disconnected, 0, region, program));
}

static void test_indexed_layouts(ggml_context * ctx) {
    ggml_fusion_region region;
    ggml_fusion_program program;
    ggml_fusion_cache_key first;
    ggml_fusion_cache_key second;

    auto * dense = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 10, 6, 4, 3);
    auto * repeated = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 2, 2, 1);

    auto * indexed = ggml_new_graph(ctx);
    ggml_build_forward_expand(indexed, ggml_sqr(ctx, ggml_add(ctx, dense, repeated)));
    REQUIRE(ggml_fusion_build(indexed, 0, region, program));
    REQUIRE(program.indexed && program.accesses[1].repeat == 15);
    REQUIRE(ggml_fusion_make_key(program, {}, first));

    const size_t plane_stride = repeated->nb[2];
    repeated->nb[2] *= 2;
    REQUIRE(ggml_fusion_build(indexed, 0, region, program));
    REQUIRE(ggml_fusion_make_key(program, {}, second));
    REQUIRE(first.size == second.size && std::memcmp(first.bytes, second.bytes, first.size) != 0);

    repeated->nb[2] = plane_stride;
    const size_t row_stride = repeated->nb[1];
    const size_t element_stride = repeated->nb[0];

    for (size_t stride : {size_t(0), size_t(1), SIZE_MAX - 3}) {
        repeated->nb[0] = stride;
        REQUIRE(!ggml_fusion_build(indexed, 0, region, program));
    }

    repeated->nb[0] = element_stride;
    repeated->nb[1] = element_stride;
    REQUIRE(!ggml_fusion_build(indexed, 0, region, program));

    repeated->nb[1] = row_stride;
    repeated->ne[0] = 0;
    REQUIRE(!ggml_fusion_build(indexed, 0, region, program));

    repeated->ne[0] = 5;
    REQUIRE(ggml_fusion_build(indexed, 0, region, program));
}

static void test_reduction_boundaries(ggml_context * ctx) {
    ggml_fusion_region region;
    ggml_fusion_program program;

    for (int64_t columns : {1, 257, 4096, 4097}) {
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, 3);
        auto * producer = ggml_neg(ctx, ggml_sqr(ctx, input));
        auto * reduced = ggml_sum_rows(ctx, producer);
        auto * result = ggml_neg(ctx, reduced);

        auto * reduction_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(reduction_graph, result);
        REQUIRE(ggml_fusion_build(reduction_graph, 0, region, program));

        if (columns <= 4096) {
            REQUIRE(program.reduction.value >= 0);
            REQUIRE(program.values[0].domain == 1 && program.values[1].domain == 1);
            REQUIRE(program.values[program.reduction.value].domain == 2);
            REQUIRE(region.outputs[0] == result);

            ggml_set_output(producer);
            REQUIRE(ggml_fusion_build(reduction_graph, 0, region, program));
            REQUIRE(program.reduction.value < 0 && region.outputs[0] == producer);
            REQUIRE(!ggml_fusion_build(reduction_graph, 2, region, program));
        } else {
            REQUIRE(program.reduction.value < 0 && region.outputs[0] == producer);
            REQUIRE(!ggml_fusion_build(reduction_graph, 2, region, program));
        }
    }

    auto * unit = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 3);
    auto * shared = ggml_sqr(ctx, unit);
    auto * mixed_domains = ggml_new_graph(ctx);
    ggml_build_forward_expand(mixed_domains, ggml_add(ctx, ggml_sum_rows(ctx, shared), shared));
    REQUIRE(!ggml_fusion_build(mixed_domains, 0, region, program));

    auto * shared_external = ggml_new_graph(ctx);
    ggml_build_forward_expand(shared_external, ggml_add(ctx, ggml_sum_rows(ctx, unit), unit));
    REQUIRE(ggml_fusion_build(shared_external, 0, region, program));
    REQUIRE(program.inputs == 2 && region.inputs[0] == unit && region.inputs[1] == unit);
    REQUIRE(program.accesses[0].domain != program.accesses[1].domain);
}

int main() {
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    REQUIRE(ctx);

    auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
    auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);

    test_materialization_and_extents(ctx, a, b);
    test_dependencies_and_keys(ctx, a, b);
    test_indexed_layouts(ctx);
    test_reduction_boundaries(ctx);

    ggml_free(ctx);
    std::puts("fusion IR: dependency, materialization, extent and indexed-layout boundaries passed");
}
