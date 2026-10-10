#pragma once

#include <algorithm>
#include <chrono>
#include <string>

struct fusion_bench_log {
    ggml_log_callback previous;
    void * previous_data;
    long long source_us = 0;
    long long compile_us = 0;
    int registers = -1;
    int local = -1;
    int shared = -1;
    int blocks = -1;

    static void callback(ggml_log_level level, const char * text, void * data) {
        auto & log = *static_cast<fusion_bench_log *>(data);
        long long source, compile;
        const char * timing = std::strstr(text, "source_us=");
        if (timing && std::sscanf(timing, "source_us=%lld compile_us=%lld", &source, &compile) == 2) {
            log.source_us += source;
            log.compile_us += compile;
        }

        int registers, local, shared, threads, blocks;
        const char * resources = std::strstr(text, "registers=");
        if (resources && std::sscanf(resources, "registers=%d local=%d shared=%d max_threads=%d blocks=%d",
                                     &registers, &local, &shared, &threads, &blocks) == 5) {
            log.registers = std::max(log.registers, registers);
            log.local = std::max(log.local, local);
            log.shared = std::max(log.shared, shared);
            log.blocks = log.blocks < 0 ? blocks : std::min(log.blocks, blocks);
        }

        if (log.previous) {
            log.previous(level, text, log.previous_data);
        } else {
            std::fputs(text, stderr);
        }
    }
};

static ggml_cgraph * fusion_bench_build_graph(ggml_context * ctx, const std::string & name, ggml_type type,
                                            const std::array<int64_t, 4> & ne, std::vector<ggml_tensor *> & outputs) {
    auto * graph = ggml_new_graph(ctx);
    auto * a = name == "rms_gate_strided" ? ggml_permute(ctx, ggml_new_tensor_4d(ctx, type, ne[0], ne[2], ne[1], ne[3]), 0, 2, 1, 3) : ggml_new_tensor(ctx, type, 4, ne.data());
    ggml_tensor * b;
    if (name == "repeat") {
        b = ggml_new_tensor_4d(ctx, type, 5, 2, 2, 1);
    } else if (name == "padded") {
        b = ggml_new_tensor_4d(ctx, type, ne[0] * 3, ne[1] * 2, ne[2], ne[3]);
        b = ggml_view_4d(ctx, b, ne[0], ne[1], ne[2], ne[3], b->nb[1], b->nb[2], b->nb[3], 0);
    } else if (name == "permuted") {
        b = ggml_new_tensor_4d(ctx, type, ne[1], ne[2], ne[0], ne[3]);
        b = ggml_permute(ctx, b, 1, 2, 0, 3);
    } else if (name == "gather" || name == "tiny_indexed" || name == "large_stride") {
        const int stride = name == "large_stride" ? 64 : 2;
        b = ggml_new_tensor_4d(ctx, type, ne[0] * stride, ne[1], ne[2], ne[3]);
        b = ggml_view_4d(ctx, b, ne[0], ne[1], ne[2], ne[3], b->nb[1], b->nb[2], b->nb[3], 0);
        b->nb[0] = stride * ggml_type_size(type);
    } else {
        b = ggml_new_tensor(ctx, type, 4, ne.data());
    }

    ggml_tensor * out;
    if (name == "wide" || name == "many_output") {
        std::array<ggml_tensor *, 8> branches;
        for (int i = 0; i < 8; ++i) {
            auto * input = i == 0 ? b : ggml_new_tensor(ctx, type, 4, ne.data());
            branches[i] = ggml_add(ctx, a, input);
            if (name == "many_output") {
                ggml_set_output(branches[i]);
                outputs.push_back(branches[i]);
            }

            ggml_build_forward_expand(graph, branches[i]);
        }

        out = branches[0];
        for (int i = 1; i < 8; ++i) {
            out = ggml_add(ctx, out, branches[i]);
        }
    } else if (name == "rms_gate" || name == "rms_gate_strided") {
        auto * gamma = ggml_new_tensor_2d(ctx, type, ne[0], ne[1]);
        out = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, a, 1e-6f), gamma), ggml_sigmoid(ctx, b));
    } else {
        auto * u = ggml_add(ctx, a, b);
        if (name == "tiny_indexed") {
            out = ggml_neg(ctx, u);
        } else if (name == "basic" || name == "diamond") {
            auto * c = ggml_new_tensor(ctx, type, 4, ne.data());
            auto * v = ggml_mul(ctx, u, c);
            if (name == "diamond") {
                out = ggml_add(ctx, v, ggml_sqr(ctx, ggml_neg(ctx, u)));
            } else {
                out = ggml_neg(ctx, type == GGML_TYPE_F16 ? v : ggml_scale_bias(ctx, v, 0.5f, -1.0f));
            }
        } else if (name == "row" || name == "full" || name == "silu_row") {
            ggml_tensor * producer;
            if (name == "silu_row") {
                producer = u;
                for (int i = 0; i < 8; ++i) {
                    producer = ggml_silu(ctx, producer);
                }
            } else {
                producer = ggml_add(ctx, ggml_mul(ctx, u, a), ggml_neg(ctx, u));
            }

            out = name == "full" ? ggml_sum(ctx, producer) : ggml_sum_rows(ctx, producer);
        } else {
            out = ggml_sqr(ctx, ggml_neg(ctx, ggml_mul(ctx, u, b)));
        }
    }

    ggml_set_output(out);
    outputs.push_back(out);
    ggml_build_forward_expand(graph, out);

    return graph;
}

static void fusion_bench_init_inputs(ggml_context * ctx, ggml_type type) {
    uint32_t state = 1;
    for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (tensor->view_src || tensor->op != GGML_OP_NONE) {
            continue;
        }

        const size_t n = ggml_nelements(tensor);
        std::vector<float> values(n);
        for (float & value : values) {
            state = state * 1664525U + 1013904223U;
            value = float(int(state >> 24) - 128) / 128.0f;
        }

        if (type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(tensor, values.data(), 0, n * sizeof(float));
        } else if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> storage(n);
            ggml_fp32_to_fp16_row(values.data(), storage.data(), n);
            ggml_backend_tensor_set(tensor, storage.data(), 0, n * sizeof(ggml_fp16_t));
        } else {
            std::vector<ggml_bf16_t> storage(n);
            ggml_fp32_to_bf16_row_ref(values.data(), storage.data(), n);
            ggml_backend_tensor_set(tensor, storage.data(), 0, n * sizeof(ggml_bf16_t));
        }
    }
}

static std::vector<std::vector<unsigned char>> fusion_bench_reference(ggml_backend_t backend, ggml_cgraph * graph,
                                                                    const std::vector<ggml_tensor *> & outputs) {
    // Single-node views cannot compile a fusion region and provide the ordinary HIP reference.
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto node = ggml_graph_view(graph, i, i + 1);
        REQUIRE(ggml_backend_graph_compute(backend, &node) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);

    std::vector<std::vector<unsigned char>> reference;
    for (auto * tensor : outputs) {
        reference.emplace_back(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, reference.back().data(), 0, reference.back().size());
    }

    return reference;
}

static void fusion_bench_check_outputs(const std::vector<ggml_tensor *> & outputs,
                                     const std::vector<std::vector<unsigned char>> & reference,
                                     double & max_error, double & max_nmse) {
    for (size_t j = 0; j < outputs.size(); ++j) {
        auto * tensor = outputs[j];
        std::vector<unsigned char> actual(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());

        double squared_error = 0;
        double squared_reference = 0;
        for (size_t i = 0; i < size_t(ggml_nelements(tensor)); ++i) {
            const size_t offset = i * ggml_type_size(tensor->type);
            const double expected = decode_scalar(reference[j].data() + offset, tensor->type);
            const double value = decode_scalar(actual.data() + offset, tensor->type);
            REQUIRE(std::isfinite(value));
            const double error = value - expected;
            max_error = std::max(max_error, std::abs(error));
            squared_error += error * error;
            squared_reference += expected * expected;
        }

        const double nmse = squared_error / std::max(squared_reference, 1e-30);
        max_nmse = std::max(max_nmse, nmse);
        REQUIRE(nmse <= 1e-7);
    }
}

static void fusion_bench_case(const std::string & name, ggml_type type, std::array<int64_t, 4> ne) {
    auto backend = ggml_backend_cuda_init(0);
    REQUIRE(backend);

    auto * ctx = ggml_init({4 << 20, nullptr, true});
    REQUIRE(ctx);

    std::vector<ggml_tensor *> outputs;
    auto * graph = fusion_bench_build_graph(ctx, name, type, ne, outputs);

    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer);

    fusion_bench_init_inputs(ctx, type);
    const auto reference = fusion_bench_reference(backend, graph, outputs);

    double max_error = 0;
    double max_nmse = 0;

    const bool observed = dlsym(RTLD_DEFAULT, "hiprtc_test_snapshot") != nullptr;
    uint64_t before[HIPRTC_TEST_COUNTER_COUNT] = {};
    uint64_t cold[HIPRTC_TEST_COUNTER_COUNT] = {};
    uint64_t warm_before[HIPRTC_TEST_COUNTER_COUNT] = {};
    uint64_t warm_after[HIPRTC_TEST_COUNTER_COUNT] = {};

    auto snapshot = [&](uint64_t (&counters)[HIPRTC_TEST_COUNTER_COUNT]) {
        if (observed) {
            observer_snapshot(counters);
        }
    };

    fusion_bench_log log;
    ggml_log_get(&log.previous, &log.previous_data);
    ggml_log_set(fusion_bench_log::callback, &log);

    snapshot(before);
    const auto first_start = std::chrono::steady_clock::now();
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    const double first_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - first_start).count();
    snapshot(cold);

    fusion_bench_check_outputs(outputs, reference, max_error, max_nmse);

    for (int i = 0; i < 20; ++i) {
        REQUIRE(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);

    auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    hipEvent_t start, end;
    CUDA_CHECK(hipEventCreate(&start));
    CUDA_CHECK(hipEventCreate(&end));

    std::array<double, 5> gpu;
    snapshot(warm_before);
    for (double & us : gpu) {
        CUDA_CHECK(hipEventRecord(start, cuda_ctx.stream()));
        for (int i = 0; i < 100; ++i) {
            REQUIRE(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
        }

        CUDA_CHECK(hipEventRecord(end, cuda_ctx.stream()));
        CUDA_CHECK(hipEventSynchronize(end));
        float ms;
        CUDA_CHECK(hipEventElapsedTime(&ms, start, end));
        us = ms * 10.0;
    }

    snapshot(warm_after);

    std::array<double, 5> wall;
    wall.fill(-1);

    // Run without the observer to measure synchronized host execution without interposition overhead.
    if (!observed) {
        for (double & us : wall) {
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < 100; ++i) {
                REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                ggml_backend_synchronize(backend);
            }
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / 100.0;
        }
    }

    fusion_bench_check_outputs(outputs, reference, max_error, max_nmse);
    ggml_log_set(log.previous, log.previous_data);

    std::sort(gpu.begin(), gpu.end());
    std::sort(wall.begin(), wall.end());

    auto count = [&](const uint64_t * from, const uint64_t * to, hiprtc_test_counter counter) -> long long {
        return observed ? static_cast<long long>(to[counter] - from[counter]) : -1;
    };

    const char * admission = "unobserved";
    if (observed) {
        admission = cold[HIPRTC_TEST_MODULE_LAUNCH] > before[HIPRTC_TEST_MODULE_LAUNCH] ? "rtc" : "ordinary";
    }

    std::printf(
        "%s,%s,%lldx%lldx%lldx%lld,%s,"
        "%lld,%lld,%.3f,%.3f,"
        "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
        "%lld,%lld,%lld,%lld,%lld,"
        "%lld,%lld,%lld,%lld,"
        "%lld,%lld,%lld,"
        "%d,%d,%d,%d,%.9g,%.9g\n",
        name.c_str(), ggml_type_name(type), (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], admission,
        log.source_us, log.compile_us, observed ? (cold[HIPRTC_TEST_LOAD_NS] - before[HIPRTC_TEST_LOAD_NS]) / 1000.0 : -1.0,
        first_us, gpu[2], gpu[0], gpu[4], wall[2], wall[0], wall[4],
        count(before, cold, HIPRTC_TEST_COMPILE), count(before, cold, HIPRTC_TEST_LOAD), count(before, cold, HIPRTC_TEST_MODULE_LAUNCH),
        count(before, cold, HIPRTC_TEST_CAPTURE_BEGIN), count(before, cold, HIPRTC_TEST_GRAPH_LAUNCH),
        count(warm_before, warm_after, HIPRTC_TEST_COMPILE), count(warm_before, warm_after, HIPRTC_TEST_LOAD),
        count(warm_before, warm_after, HIPRTC_TEST_MODULE_LAUNCH), count(warm_before, warm_after, HIPRTC_TEST_GRAPH_LAUNCH),
        count(before, warm_after, HIPRTC_TEST_CAPTURE_BEGIN), count(warm_before, warm_after, HIPRTC_TEST_CAPTURE_BEGIN),
        count(before, warm_after, HIPRTC_TEST_SETUP_DURING_CAPTURE),
        log.registers, log.local, log.shared, log.blocks, max_error, max_nmse);
    std::fflush(stdout);

    CUDA_CHECK(hipEventDestroy(start));
    CUDA_CHECK(hipEventDestroy(end));

    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void bench() {
    std::puts(
        "case,type,shape,admission,"
        "source_us,compile_us,load_us,first_us,"
        "gpu_us,gpu_min_us,gpu_max_us,sync_us,sync_min_us,sync_max_us,"
        "cold_compile,cold_load,cold_launch,cold_capture,cold_graph,"
        "warm_compile,warm_load,warm_launch,warm_graph,total_capture,warm_capture,capture_setup,"
        "registers,local_bytes,shared_bytes,blocks,max_abs_error,nmse");

    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (const char * name : {"basic", "diamond"}) {
            for (int64_t n : {256, 4096, 1048576}) {
                fusion_bench_case(name, type, {n, 1, 1, 1});
            }
        }
        for (int64_t n : {240, 983040}) {
            fusion_bench_case("repeat", type, {n / 24, 6, 2, 2});
        }
        for (const char * name : {"padded", "permuted", "gather"}) {
            for (int64_t n : {257, 1048576}) {
                fusion_bench_case(name, type, {n, 1, 1, 1});
            }
        }
    }
    for (const char * name : {"wide", "many_output"}) {
        for (int64_t n : {4096, 1048576}) {
            fusion_bench_case(name, GGML_TYPE_F32, {n, 1, 1, 1});
        }
    }
    fusion_bench_case("tiny_indexed", GGML_TYPE_F32, {257, 1, 1, 1});
    fusion_bench_case("large_stride", GGML_TYPE_F32, {4096, 1, 1, 1});
    for (auto shape : {std::array<int64_t, 4>{32, 64, 1, 1}, {257, 64, 1, 1}, {4096, 1, 1, 1}, {4096, 64, 1, 1}}) {
        fusion_bench_case("row", GGML_TYPE_F32, shape);
    }
    fusion_bench_case("silu_row", GGML_TYPE_F32, {4096, 1, 1, 1});
    for (int64_t n : {4096, 4097, 8193, 1048576}) {
        fusion_bench_case("full", GGML_TYPE_F32, {n, 1, 1, 1});
    }

    for (auto shape : {std::array<int64_t, 4>{128, 48, 1, 1}, {128, 48, 1024, 1}, {2560, 4, 1024, 1}, {257, 3, 2, 2}}) {
        fusion_bench_case("rms_gate", GGML_TYPE_F32, shape);
    }

    fusion_bench_case("rms_gate_strided", GGML_TYPE_F32, {128, 48, 1024, 1});
}
