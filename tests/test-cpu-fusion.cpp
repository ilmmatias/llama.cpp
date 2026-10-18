#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-cpu/fusion.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1); \
        } \
    } while (0)

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using fusion_ptr  = std::unique_ptr<ggml_cpu_fusion_plan, decltype(&ggml_cpu_fusion_plan_free)>;

static context_ptr make_context(size_t bytes = 4 * 1024 * 1024) {
    return context_ptr(ggml_init({bytes, nullptr, false}), ggml_free);
}

static void compute(ggml_cgraph * graph, int threads, bool ref = false, ggml_threadpool_t pool = nullptr) {
    auto plan = ggml_graph_plan(graph, threads, pool);
    std::vector<uint8_t> work(plan.work_size);

    plan.work_data = work.data();
    plan.use_ref   = ref;

    REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
}

static void equal(const float * actual, const std::vector<float> & expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(expected[i])) {
            REQUIRE(std::isnan(actual[i]));
        } else if (expected[i] == 0 || std::isinf(expected[i])) {
            REQUIRE(actual[i] == expected[i] && std::signbit(actual[i]) == std::signbit(expected[i]));
        } else {
            REQUIRE(std::fabs(actual[i] - expected[i]) <= 2e-6f * std::max(1.0f, std::fabs(expected[i])));
        }
    }
}

static void test_chain(int64_t n, int threads) {
    auto ctx = make_context();

    auto * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    auto * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);

    for (int64_t i = 0; i < n; ++i) {
        static_cast<float *>(a->data)[i] = float(i % 37) * 0.125f - 2;
        static_cast<float *>(b->data)[i] = float(i % 11) * 0.1f   + 1;
    }

    auto * sum = ggml_add(ctx.get(), a, b);
    ggml_set_output(sum);

    auto * out   = ggml_scale_bias(ctx.get(), ggml_sqr(ctx.get(), sum), 0.5f, -1.0f);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    compute(graph, threads, true);
    std::vector<float> expected(static_cast<float *>(out->data), static_cast<float *>(out->data) + n);

    fusion_ptr fusion(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);
    REQUIRE(fusion);

    // Empty ranges must return the same skip count as the other workers.
    for (int ith = 0; ith < threads; ++ith) {
        REQUIRE(ggml_cpu_fusion_compute(fusion.get(), 0, ith, threads) == 2);
    }

    equal(static_cast<float *>(out->data), expected);

    for (int64_t i = 0; i < n; ++i) {
        REQUIRE(static_cast<float *>(sum->data)[i] == static_cast<float *>(a->data)[i] + static_cast<float *>(b->data)[i]);
    }

    // Same compiled program, new scalar values and input contents.
    ggml_set_op_params_f32(out, 0, 0.25f);
    ggml_set_op_params_f32(out, 1, 2.0f);
    static_cast<float *>(a->data)[0] = 3;

    compute(graph, threads, true);
    std::copy_n(static_cast<float *>(out->data), n, expected.begin());

    auto params = ggml_threadpool_params_default(threads);
    auto pool   = ggml_threadpool_new(&params);

    for (int repeat = 0; repeat < 3; ++repeat) {
        std::fill_n(static_cast<float *>(out->data), n, -1000.0f);
        compute(graph, threads, false, pool);
        equal(static_cast<float *>(out->data), expected);
    }

    ggml_threadpool_free(pool);
}

static void test_numerics() {
    auto ctx = make_context();

    const float values[] = {0.0f, -0.0f, 1, -1, 3.25f, -5, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};

    auto * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 81);

    for (int i = 0; i < 81; ++i) {
        static_cast<float *>(a->data)[i] = values[i % 9];
    }

    for (int mode = 0; mode < 5; ++mode) {
        auto * x = ggml_neg(ctx.get(), a);

        switch (mode) {
            case 0: x = ggml_abs(ctx.get(), x);                break;
            case 1: x = ggml_relu(ctx.get(), x);               break;
            case 2: x = ggml_sqrt(ctx.get(), x);               break;
            case 3: x = ggml_scale(ctx.get(), x, 0.5f);        break;
            case 4: x = ggml_clamp(ctx.get(), x, -1.0f, 1.0f); break;
        }

        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, x);

        compute(graph, 3, true);
        std::vector<float> expected(static_cast<float *>(x->data), static_cast<float *>(x->data) + 81);

        fusion_ptr fusion(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);
        REQUIRE(ggml_cpu_fusion_compute(fusion.get(), 0, 0, 1) == 1);
        equal(static_cast<float *>(x->data), expected);
    }
}

static void test_fallbacks() {
    auto ctx = make_context();

    auto * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 128);
    ggml_set_f32(a, 2);

    auto * out   = ggml_sqr(ctx.get(), ggml_neg_inplace(ctx.get(), a));
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    fusion_ptr alias(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);
    REQUIRE(ggml_cpu_fusion_compute(alias.get(), 0, 0, 1) == 0);
    compute(graph, 4);
    REQUIRE(static_cast<float *>(out->data)[127] == 4);

    auto * scalar = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_set_f32(scalar, 3);

    out   = ggml_sqr(ctx.get(), ggml_add(ctx.get(), a, scalar));
    graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    fusion_ptr broadcast(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);
    REQUIRE(ggml_cpu_fusion_compute(broadcast.get(), 0, 0, 1) == 0);
    compute(graph, 4);
    REQUIRE(static_cast<float *>(out->data)[127] == 1);

    out   = ggml_sum(ctx.get(), ggml_sqr(ctx.get(), a));
    graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    fusion_ptr reduction(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);
    REQUIRE(ggml_cpu_fusion_compute(reduction.get(), 0, 0, 1) == 0);
    compute(graph, 4);
    REQUIRE(static_cast<float *>(out->data)[0] == 512);
}

static void test_boundaries_and_aliases() {
    auto ctx = make_context();

    auto * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 65);
    auto * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 65);

    ggml_set_f32(a, 2);
    ggml_set_f32(b, 3);

    auto * sum    = ggml_add(ctx.get(), a, b);
    auto * square = ggml_sqr(ctx.get(), sum);
    auto * total  = ggml_sum(ctx.get(), square);
    auto * other  = ggml_sum(ctx.get(), sum);
    auto * graph  = ggml_new_graph(ctx.get());

    ggml_build_forward_expand(graph, total);
    ggml_build_forward_expand(graph, other);

    fusion_ptr fusion(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);

    // A consumer after the reduction still needs the intermediate ADD result.
    REQUIRE(ggml_cpu_fusion_compute(fusion.get(), 0, 0, 1) == 1);
    REQUIRE(static_cast<float *>(sum->data)[64] == 5);
    REQUIRE(static_cast<float *>(square->data)[64] == 25);

    compute(graph, 4);
    REQUIRE(static_cast<float *>(total->data)[0] == 1625);
    REQUIRE(static_cast<float *>(other->data)[0] == 325);

    graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, square);

    void * saved = square->data;
    square->data = a->data;
    fusion.reset(ggml_cpu_fusion_plan_create(graph));
    REQUIRE(ggml_cpu_fusion_compute(fusion.get(), 0, 0, 1) == 1);
    REQUIRE(static_cast<float *>(a->data)[64] == 25);

    square->data = static_cast<float *>(a->data) + 1;
    fusion.reset(ggml_cpu_fusion_plan_create(graph));
    REQUIRE(ggml_cpu_fusion_compute(fusion.get(), 0, 0, 1) == 0);

    square->data = saved;
}

static void test_arithmetic_and_abort() {
    auto ctx = make_context();

    auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 33, 17);
    auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 33, 17);

    ggml_set_f32(a, 5);
    ggml_set_f32(b, 2);

    auto * out   = ggml_cont(ctx.get(), ggml_mul(ctx.get(), ggml_div(ctx.get(), ggml_sub(ctx.get(), a, b), b), a));
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    fusion_ptr fusion(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);
    REQUIRE(ggml_cpu_fusion_compute(fusion.get(), 0, 0, 1) == 3);
    REQUIRE(static_cast<float *>(out->data)[560] == 7.5f);

    ggml_set_f32(out, -1);

    auto plan = ggml_graph_plan(graph, 4, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();

    int calls = 0;
    plan.abort_callback      = [](void * data) { return ++*static_cast<int *>(data) == 2; };
    plan.abort_callback_data = &calls;

    REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_ABORTED);
    REQUIRE(calls == 2 && static_cast<float *>(out->data)[560] == -1);
}

static void test_backend_allocation() {
    context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);

    auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 33, 17);
    auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 33, 17);

    ggml_set_input(a);
    ggml_set_input(b);

    auto * out = ggml_scale(ctx.get(), ggml_sqr(ctx.get(), ggml_add(ctx.get(), a, b)), 0.25f);
    ggml_set_output(out);

    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    auto allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    REQUIRE(ggml_gallocr_alloc_graph(allocator, graph));

    ggml_set_f32(a, 1);
    ggml_set_f32(b, 3);

    fusion_ptr fusion(ggml_cpu_fusion_plan_create(graph), ggml_cpu_fusion_plan_free);
    REQUIRE(ggml_cpu_fusion_compute(fusion.get(), 0, 0, 1) == 2);
    REQUIRE(static_cast<float *>(out->data)[560] == 4);

    ggml_set_f32(a, 1);
    ggml_set_f32(b, 3);

    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    REQUIRE(static_cast<float *>(out->data)[560] == 4);

    ggml_backend_free(backend);
    ggml_gallocr_free(allocator);
}

static void benchmark() {
    const int64_t n = 1024 * 1024;
    auto ctx = make_context(32 * 1024 * 1024);

    auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1024, n / 1024);
    auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1024, n / 1024);

    ggml_set_f32(a, 0.5f);
    ggml_set_f32(b, 1.5f);

    auto * out   = ggml_scale_bias(ctx.get(), ggml_sqr(ctx.get(), ggml_add(ctx.get(), a, b)), 0.5f, 1);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    auto params = ggml_threadpool_params_default(4);
    auto pool   = ggml_threadpool_new(&params);

    for (bool ref : {true, false}) {
        compute(graph, 4, ref, pool);

        const auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < 100; ++i) {
            compute(graph, 4, ref, pool);
        }

        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 100;
        std::printf("%s: %.2f us\n", ref ? "unfused" : "LLVM fused", us);
    }

    ggml_threadpool_free(pool);
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "cache") == 0) {
        int hits = 0;
        ggml_log_set([](ggml_log_level, const char * text, void * data) {
            if (std::strstr(text, "CPU fusion: loaded object ")) {
                ++*static_cast<int *>(data);
            }
        }, &hits);

        const bool warm = std::strcmp(argv[2], "warm") == 0;
        if (warm) {
            test_arithmetic_and_abort();
            test_chain(513, 4);
        } else {
            test_chain(31, 1);
            test_arithmetic_and_abort();
        }

        REQUIRE(hits == (warm ? 2 : 0));
        ggml_log_set(nullptr, nullptr);
        return 0;
    }

    if (argc == 2 && std::strcmp(argv[1], "bench") == 0) {
        benchmark();
        return 0;
    }

    for (int threads : {1, 4}) {
        for (int64_t n : {1, 7, 8, 9, 31, 513, 4096}) {
            test_chain(n, threads);
        }
    }

    test_numerics();
    test_fallbacks();
    test_boundaries_and_aliases();
    test_arithmetic_and_abort();
    test_backend_allocation();

    // Independent graph bindings may use the same cached code concurrently.
    std::thread first ([] { test_chain(1025, 2); });
    std::thread second([] { test_chain(2049, 2); });

    first.join();
    second.join();

    std::puts("CPU fusion tests passed");
}
