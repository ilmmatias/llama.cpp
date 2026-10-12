#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void test_round_trip(ggml_backend_dev_t dev, bool parallel, bool mixed_sources, int64_t n) {
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t other = mixed_sources ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(gpu && cpu && (!mixed_sources || other));
    ggml_backend_cpu_set_n_threads(cpu, 4);

    ggml_backend_t backends[] = { gpu, mixed_sources ? other : cpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, mixed_sources ? 3 : 2, 128, parallel, true);
    GGML_ASSERT(sched);

    ggml_context * inputs = ggml_init({ 2*ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * x = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, n + 8);
    ggml_tensor * y = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, n + 8);
    ggml_set_input(x);
    ggml_set_input(y);

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    GGML_ASSERT(input_buffer);

    ggml_context * ctx = ggml_init({ 32*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
    ggml_tensor * a = ggml_scale(ctx, x, 1.25f);
    ggml_tensor * b = ggml_scale(ctx, y, -0.75f);
    ggml_backend_sched_set_tensor_backend(sched, a, gpu);
    ggml_backend_sched_set_tensor_backend(sched, b, mixed_sources ? other : gpu);

    a = ggml_view_1d(ctx, a, n, 3*sizeof(float));
    b = ggml_view_1d(ctx, b, n, 5*sizeof(float));

    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_tensor * d = ggml_sub(ctx, a, b);
    ggml_backend_sched_set_tensor_backend(sched, c, cpu);
    ggml_backend_sched_set_tensor_backend(sched, d, cpu);

    ggml_tensor * e = ggml_mul(ctx, c, d);
    ggml_tensor * f = ggml_add(ctx, c, d);
    ggml_backend_sched_set_tensor_backend(sched, e, gpu);
    ggml_backend_sched_set_tensor_backend(sched, f, mixed_sources ? other : gpu);

    ggml_tensor * g = ggml_add(ctx, e, f);
    ggml_tensor * h = ggml_sub(ctx, e, f);
    ggml_backend_sched_set_tensor_backend(sched, g, cpu);
    ggml_backend_sched_set_tensor_backend(sched, h, cpu);

    ggml_tensor * out = ggml_add(ctx, g, h);
    ggml_backend_sched_set_tensor_backend(sched, out, gpu);
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));

    std::vector<float> xv(n + 8), yv(n + 8), actual(n);
    float error = 0;
    for (int iteration = 0; iteration < 12; ++iteration) {
        for (int64_t i = 0; i < n + 8; ++i) {
            xv[i] = float((i + iteration*3) % 31 - 15)/16;
            yv[i] = float((i*7 + iteration) % 37 - 18)/32;
        }

        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(y, yv.data(), 0, ggml_nbytes(y));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);

        // Overwrite user inputs across several queued evaluations and copy-ring wraparound.
        if (iteration % 6 != 5) {
            continue;
        }

        ggml_backend_sched_synchronize(sched);
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));

        for (int64_t i = 0; i < n; ++i) {
            const float av = 1.25f*xv[i + 3];
            const float bv = -0.75f*yv[i + 5];
            const float expected = 2*(av + bv)*(av - bv);

            GGML_ASSERT(std::isfinite(actual[i]));
            error = std::max(error, std::abs(actual[i] - expected));
        }

        GGML_ASSERT(error <= 3e-6f);
    }

    printf("%s: round trip n=%lld parallel=%d mixed_sources=%d PASS (max error %g)\n",
            ggml_backend_dev_name(dev), (long long) n, parallel, mixed_sources, error);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(inputs);
    ggml_backend_free(cpu);
    if (other) {
        ggml_backend_free(other);
    }
    ggml_backend_free(gpu);
}

static bool copy_selected_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst,
        ggml_cgraph * graph, void *) {
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }

    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    int32_t ids[2];
    ggml_backend_tensor_get(node->src[2], ids, 0, sizeof(ids));

    for (int32_t id : ids) {
        GGML_ASSERT(id >= 0 && id < src->ne[2]);
        const size_t offset = id*src->nb[2];
        ggml_backend_tensor_set_async(backend, dst, static_cast<const char *>(src->data) + offset, offset, src->nb[2]);
    }

    return true;
}

static void test_expert_callback(ggml_backend_dev_t dev, bool parallel) {
    constexpr int k = 128, m = 64, n_experts = 4;
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(gpu && cpu);
    ggml_backend_cpu_set_n_threads(cpu, 4);

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 128, parallel, true);
    ggml_backend_sched_set_copy_callback(sched, copy_selected_experts, nullptr);

    ggml_context * weights = ggml_init({ ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * w = ggml_new_tensor_3d(weights, GGML_TYPE_F32, k, m, n_experts);
    ggml_backend_buffer_t weight_buffer = ggml_backend_alloc_ctx_tensors(weights, cpu);
    GGML_ASSERT(weight_buffer);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> wv(k*m*n_experts);
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < k; ++col) {
                wv[(expert*m + row)*k + col] = float((expert*3 + row + col) % 17 - 8)/32;
            }
        }
    }

    ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));

    ggml_context * inputs = ggml_init({ 2*ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * x = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, k, 1, 1);
    ggml_tensor * scores = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, n_experts, 1);
    ggml_set_input(x);
    ggml_set_input(scores);

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    GGML_ASSERT(input_buffer);

    ggml_context * ctx = ggml_init({ 32*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
    ggml_tensor * activation = ggml_scale(ctx, x, 0.5f);
    ggml_tensor * ids = ggml_cont(ctx, ggml_top_k(ctx, scores, 2));
    ggml_backend_sched_set_tensor_backend(sched, activation, cpu);
    ggml_backend_sched_set_tensor_backend(sched, ids->src[0], cpu);
    ggml_backend_sched_set_tensor_backend(sched, ids, cpu);

    ggml_tensor * out = ggml_mul_mat_id(ctx, w, activation, ids);
    ggml_backend_sched_set_tensor_backend(sched, out, gpu);
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));

    float xv[k], sv[n_experts], actual[m*2];
    float error = 0;
    for (int iteration = 0; iteration < 12; ++iteration) {
        for (int col = 0; col < k; ++col) {
            xv[col] = float((col + iteration*3) % 23 - 11)/16;
        }

        for (int expert = 0; expert < n_experts; ++expert) {
            sv[expert] = float((expert + iteration) % n_experts);
        }

        ggml_backend_tensor_set(x, xv, 0, sizeof(xv));
        ggml_backend_tensor_set(scores, sv, 0, sizeof(sv));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, actual, 0, sizeof(actual));

        int32_t expected_ids[2];
        ggml_backend_tensor_get(ids, expected_ids, 0, sizeof(expected_ids));

        for (int slot = 0; slot < 2; ++slot) {
            const int expert = expected_ids[slot];
            GGML_ASSERT(expert >= 0 && expert < n_experts && sv[expert] >= 2);

            for (int row = 0; row < m; ++row) {
                float expected = 0;
                for (int col = 0; col < k; ++col) {
                    expected += wv[(expert*m + row)*k + col]*xv[col]*0.5f;
                }

                GGML_ASSERT(std::isfinite(actual[slot*m + row]));
                error = std::max(error, std::abs(actual[slot*m + row] - expected));
            }
        }
        GGML_ASSERT(error <= 3e-6f);
    }

    printf("%s: selected expert callback parallel=%d PASS (max error %g)\n", ggml_backend_dev_name(dev), parallel, error);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(inputs);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weights);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
}

static void test_upload_lifetime(ggml_backend_dev_t dev, bool parallel, bool user_input) {
    constexpr int n = 65536, iterations = 12;
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(gpu && cpu);
    ggml_backend_cpu_set_n_threads(cpu, 4);

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 128, parallel, true);
    GGML_ASSERT(sched);

    ggml_context * inputs = ggml_init({ 2*ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * x = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, n);
    ggml_tensor * y = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_set_input(y);

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    GGML_ASSERT(input_buffer);

    ggml_context * ctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
    ggml_tensor * a = ggml_scale(ctx, x, 2);
    ggml_tensor * b = ggml_scale(ctx, y, 3);
    ggml_backend_sched_set_tensor_backend(sched, a, cpu);
    ggml_backend_sched_set_tensor_backend(sched, b, cpu);

    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_backend_sched_set_tensor_backend(sched, out, gpu);
    if (user_input) {
        out = ggml_add(ctx, out, y);
        ggml_backend_sched_set_tensor_backend(sched, out, gpu);
        out = ggml_add(ctx, out, x);
        ggml_backend_sched_set_tensor_backend(sched, out, gpu);

        ggml_tensor * c = ggml_scale(ctx, x, 4);
        ggml_backend_sched_set_tensor_backend(sched, c, cpu);
        out = ggml_add(ctx, out, c);
        ggml_backend_sched_set_tensor_backend(sched, out, gpu);
    }

    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));

    std::vector<float> values(n), actual(n*iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::fill(values.begin(), values.end(), float(iteration + 1));
        ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
        std::fill(values.begin(), values.end(), float(iteration + 2));
        ggml_backend_tensor_set(y, values.data(), 0, ggml_nbytes(y));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);

        // Preserve each result before the next CPU split overwrites its intermediate sources.
        ggml_backend_tensor_get_async(gpu, out, actual.data() + iteration*n, 0, ggml_nbytes(out));
    }

    ggml_backend_sched_synchronize(sched);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float expected = user_input ? 11*iteration + 15 : 5*iteration + 8;
        for (int i = 0; i < n; ++i) {
            GGML_ASSERT(actual[iteration*n + i] == expected);
        }
    }

    printf("%s: queued CPU-first uploads parallel=%d user_input=%d PASS\n", ggml_backend_dev_name(dev), parallel, user_input);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(inputs);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
}

struct activation_merge {
    ggml_backend_t backend;
    const ggml_tensor * source;
    ggml_tensor * misses;
    ggml_tensor * result;
    ggml_cgraph * graph;
};

static bool merge_activation(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph *, void * user_data) {
    auto & merge = *static_cast<activation_merge *>(user_data);
    if (src != merge.source) {
        return false;
    }

    ggml_backend_tensor_set_async(merge.backend, merge.misses, src->data, 0, ggml_nbytes(src));
    GGML_ASSERT(ggml_backend_graph_compute_async(merge.backend, merge.graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy_async(merge.backend, backend, merge.result, dst);

    return true;
}

static void test_activation_merge(ggml_backend_dev_t dev, bool parallel, bool mixed_sources) {
    constexpr int n = 65536, iterations = 12;
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cached = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t other = mixed_sources ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(gpu && cached && cpu && (!mixed_sources || other));
    ggml_backend_cpu_set_n_threads(cpu, 4);

    ggml_backend_t backends[] = { gpu, mixed_sources ? other : cpu, cpu };
    auto sched = ggml_backend_sched_new(backends, nullptr, mixed_sources ? 3 : 2, 128, parallel, true);

    auto inputs = ggml_init({ 2*ggml_tensor_overhead(), nullptr, true });
    auto x = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, n);
    auto y = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_set_input(y);
    auto input_buffer = ggml_backend_alloc_ctx_tensors(inputs, cpu);
    GGML_ASSERT(input_buffer);

    auto merge_ctx = ggml_init({ 8*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true });
    auto misses = ggml_new_tensor_1d(merge_ctx, GGML_TYPE_F32, n);
    auto resident = ggml_new_tensor_1d(merge_ctx, GGML_TYPE_F32, n);
    auto result = ggml_add(merge_ctx, misses, ggml_scale(merge_ctx, resident, -0.5f));
    auto merge_graph = ggml_new_graph_custom(merge_ctx, 16, false);
    ggml_build_forward_expand(merge_graph, result);
    auto merge_buffer = ggml_backend_alloc_ctx_tensors(merge_ctx, cached);
    GGML_ASSERT(merge_buffer);

    std::vector<float> values(n), actual(n*iterations);
    for (int i = 0; i < n; ++i) {
        values[i] = float(i % 17 - 8);
    }

    ggml_backend_tensor_set(resident, values.data(), 0, ggml_nbytes(resident));

    auto ctx = ggml_init({ 16*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
    auto a = ggml_scale(ctx, x, 2);
    ggml_backend_sched_set_tensor_backend(sched, a, cpu);

    auto y_gpu = mixed_sources ? ggml_scale(ctx, y, 1) : y;
    if (mixed_sources) {
        ggml_backend_sched_set_tensor_backend(sched, y_gpu, other);
    }

    auto out = ggml_add(ctx, a, y_gpu);
    ggml_backend_sched_set_tensor_backend(sched, out, gpu);

    auto b = ggml_scale(ctx, out, 3);
    ggml_backend_sched_set_tensor_backend(sched, b, cpu);

    out = ggml_add(ctx, b, y);
    ggml_backend_sched_set_tensor_backend(sched, out, gpu);

    ggml_set_output(out);
    auto graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);

    activation_merge merge = { cached, a, misses, result, merge_graph };
    ggml_backend_sched_set_copy_callback(sched, merge_activation, &merge);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::fill(values.begin(), values.end(), float(iteration + 1));
        ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
        std::fill(values.begin(), values.end(), float(iteration + 2));
        ggml_backend_tensor_set(y, values.data(), 0, ggml_nbytes(y));
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);

        ggml_backend_tensor_get_async(gpu, out, actual.data() + iteration*n, 0, ggml_nbytes(out));
    }

    ggml_backend_sched_synchronize(sched);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int i = 0; i < n; ++i) {
            const float expected = 6*(iteration + 1) + 4*(iteration + 2) - 1.5f*(i % 17 - 8);
            GGML_ASSERT(actual[iteration*n + i] == expected);
        }
    }

    printf("%s: cross-stream activation merge parallel=%d mixed_sources=%d PASS\n", ggml_backend_dev_name(dev), parallel, mixed_sources);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_synchronize(cached);
    ggml_backend_buffer_free(merge_buffer);
    ggml_free(merge_ctx);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(inputs);
    ggml_backend_free(cpu);
    ggml_backend_free(cached);
    ggml_backend_free(gpu);
    if (other != nullptr) {
        ggml_backend_free(other);
    }
}

static bool copy_cached_activation(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph *, void * data) {
    auto copy = *static_cast<ggml_backend_cpu_expert_cache_copy_t *>(data);
    return copy(backend, src, dst);
}

static void test_expert_cache_reuse(ggml_backend_dev_t dev, bool parallel) {
    auto cpu = ggml_backend_cpu_init();
    auto cpu_dev = ggml_backend_get_device(cpu);
    auto reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto extras = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    auto configure = (ggml_backend_cpu_expert_cache_configure_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_configure");
    auto prepare = (ggml_backend_cpu_expert_cache_prepare_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_prepare");
    auto copy = (ggml_backend_cpu_expert_cache_copy_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_copy");
    ggml_backend_buffer_type_t buft = nullptr;
    if (extras) {
        for (auto p = extras(cpu_dev); p && *p; ++p) {
            if (strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) {
                buft = *p;
            }
        }
    }
    if (!buft || !configure || !prepare || !copy) {
        ggml_backend_free(cpu);
        return;
    }

    auto gpu = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(gpu);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    constexpr int dim = 256, experts = 4, used = 2, layers = 2;
    auto weights = ggml_init({ 3*layers*ggml_tensor_overhead(), nullptr, true });
    ggml_tensor * w[layers][3];
    const char * parts[] = { "gate", "up", "down" };
    for (int layer = 0; layer < layers; ++layer) {
        for (int part = 0; part < 3; ++part) {
            w[layer][part] = ggml_new_tensor_3d(weights, GGML_TYPE_ZNQ4, dim, dim, experts);
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", layer, parts[part]);
            ggml_set_name(w[layer][part], name);
        }
    }
    auto weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights, buft);
    GGML_ASSERT(weight_buffer);
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values(dim*dim*experts), importance(dim, 1);
    for (int layer = 0; layer < layers; ++layer) {
        for (int part = 0; part < 3; ++part) {
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.02f*std::sin(float(i*13 + part*7 + layer*19));
            }
            std::vector<unsigned char> quant(ggml_nbytes(w[layer][part]));
            ggml_quantize_chunk(GGML_TYPE_ZNQ4, values.data(), quant.data(), 0, dim*experts, dim, importance.data());
            ggml_backend_tensor_set(w[layer][part], quant.data(), 0, quant.size());
        }
    }

    auto run = [&](int tokens, int pattern, int mode, int repeats) {
        ggml_backend_t backends[] = { gpu, cpu };
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, 128, parallel, true);
        ggml_backend_sched_set_copy_callback(sched, copy_cached_activation, &copy);
        auto ctx = ggml_init({ 64*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
        auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 1, tokens);
        auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
        auto scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, used, tokens);
        ggml_set_input(x);
        ggml_set_input(ids);
        ggml_set_input(scales);
        ggml_tensor * outputs[layers];
        for (int layer = 0; layer < layers; ++layer) {
            auto gate = ggml_mul_mat_id(ctx, w[layer][0], x, ids);
            auto up = ggml_mul_mat_id(ctx, w[layer][1], x, ids);
            auto act = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
            auto down = ggml_mul_mat_id(ctx, w[layer][2], act, ids);
            for (auto t : { gate, up, act, down }) {
                ggml_backend_sched_set_tensor_backend(sched, t, cpu);
            }
            auto weighted = ggml_mul(ctx, down, scales);
            ggml_backend_sched_set_tensor_backend(sched, weighted, mode == 1 ? cpu : gpu);
            if (mode == 2) {
                ggml_set_output(down);
            }
            auto a = ggml_view_2d(ctx, weighted, dim, tokens, weighted->nb[2], 0);
            auto b = ggml_view_2d(ctx, weighted, dim, tokens, weighted->nb[2], weighted->nb[1]);
            outputs[layer] = ggml_add(ctx, a, b);
            ggml_backend_sched_set_tensor_backend(sched, outputs[layer], gpu);
        }
        auto out = ggml_add(ctx, outputs[0], outputs[1]);
        ggml_backend_sched_set_tensor_backend(sched, out, gpu);
        ggml_set_output(out);
        auto graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, out);
        prepare(sched, graph);
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input(dim*tokens), factors(used*tokens), output(dim*tokens*repeats);
        std::vector<int32_t> routes(used*tokens);
        for (int t = 0; t < tokens; ++t) {
            routes[t*used] = pattern == 2 ? 2 : 0;
            routes[t*used + 1] = pattern == 0 ? 1 : pattern == 1 ? 2 : 3;
            factors[t*used] = 0.25f;
            factors[t*used + 1] = 0.75f;
        }
        for (int iteration = 0; iteration < repeats; ++iteration) {
            for (int t = 0; t < tokens; ++t) {
                for (int i = 0; i < dim; ++i) {
                    input[t*dim + i] = std::cos(float(i*3 + t*5 + iteration*7))*0.25f;
                }
            }
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
            ggml_backend_tensor_set(scales, factors.data(), 0, ggml_nbytes(scales));
            GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get_async(gpu, out, output.data() + iteration*dim*tokens, 0, ggml_nbytes(out));
        }
        ggml_backend_sched_synchronize(sched);
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return output;
    };

    const int sizes[] = { 1, 3, 8, 9, 128, 1 };
    std::vector<std::vector<float>> expected;
    configure(0, 0, 1, nullptr);
    for (int tokens : sizes) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            expected.push_back(run(tokens, pattern, 0, 4));
        }
    }
    double worst = 0;
    for (int mode = 0; mode < 3; ++mode) {
        configure(2, 0, 1, dev);
        size_t scenario = 0;
        for (int tokens : sizes) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                run(1, 0, 0, 30);
                auto actual = run(tokens, pattern, mode, 4);
                double error = 0, norm = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    GGML_ASSERT(std::isfinite(actual[i]));
                    const double delta = actual[i] - expected[scenario][i];
                    error += delta*delta;
                    norm += double(expected[scenario][i])*expected[scenario][i];
                }
                const double nmse = error/std::max(norm, 1e-30);
                GGML_ASSERT(nmse < 0.001);
                worst = std::max(worst, nmse);
                ++scenario;
            }
        }
        configure(0, 0, 1, nullptr);
    }
    printf("%s: expert cache reuse parallel=%d PASS (maximum nmse %g)\n", ggml_backend_dev_name(dev), parallel, worst);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weights);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
}

int main() {
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        for (bool parallel : { false, true }) {
            test_round_trip(dev, parallel, false, 2048);
            test_round_trip(dev, parallel, false, 65536);
            test_round_trip(dev, parallel, true, 2048);
            test_expert_callback(dev, parallel);
            test_upload_lifetime(dev, parallel, false);
            test_upload_lifetime(dev, parallel, true);
            test_activation_merge(dev, parallel, false);
            test_activation_merge(dev, parallel, true);
            test_expert_cache_reuse(dev, parallel);
        }
    }

    return 0;
}
