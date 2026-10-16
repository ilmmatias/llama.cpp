#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
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

struct expert_cache_copy_data {
    ggml_backend_cpu_expert_cache_t cache;
    ggml_backend_cpu_expert_cache_copy_t copy;
    uint64_t copies = 0;
};

static bool copy_cached_activation(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph *, void * data) {
    auto & state = *static_cast<expert_cache_copy_data *>(data);
    if (!state.copy(state.cache, backend, src, dst)) {
        return false;
    }
    ++state.copies;
    return true;
}

struct expert_cache_result {
    std::vector<float> output;
    std::vector<float> observed;
    uint64_t copies = 0;
};

static void test_expert_cache_reuse(ggml_backend_dev_t dev, bool parallel, ggml_type type, bool relu_squared) {
    auto cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);
    auto cpu_dev = ggml_backend_get_device(cpu);
    auto reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto extras = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    auto create = (ggml_backend_cpu_expert_cache_create_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_create");
    auto free_cache = (ggml_backend_cpu_expert_cache_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_free");
    auto add_device = (ggml_backend_cpu_expert_cache_add_device_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_add_device");
    auto register_weights = (ggml_backend_cpu_expert_cache_register_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_register");
    auto set_cache = (ggml_backend_cpu_set_expert_cache_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_expert_cache");
    auto prepare = (ggml_backend_cpu_expert_cache_prepare_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_prepare");
    auto copy = (ggml_backend_cpu_expert_cache_copy_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_copy");
    auto cache_device = (ggml_backend_cpu_expert_cache_device_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_device");
    auto offload_device = (ggml_backend_cpu_expert_cache_device_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_offload_device");
    auto memory = (ggml_backend_cpu_expert_cache_memory_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_memory");
    auto hits = (ggml_backend_cpu_expert_cache_hits_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_expert_cache_hits");
    GGML_ASSERT(create && free_cache && add_device && register_weights && set_cache && prepare && copy && cache_device && offload_device && memory && hits);
    const bool repacked = type == GGML_TYPE_ZNQ2 || type == GGML_TYPE_ZNQ3 || type == GGML_TYPE_ZNQ4;
    auto buft = ggml_backend_cpu_buffer_type();
    if (repacked) {
        buft = nullptr;
        if (extras) {
            for (auto p = extras(cpu_dev); p && *p; ++p) {
                if (strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) {
                    buft = *p;
                }
            }
        }
        if (!buft) {
            printf("%s: expert cache type=%s SKIP (CPU_REPACK unavailable)\n", ggml_backend_dev_name(dev), ggml_type_name(type));
            ggml_backend_free(cpu);
            return;
        }
    }

    auto gpu = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(gpu);
    ggml_backend_cpu_set_n_threads(cpu, 4);
    constexpr int dim = 256, experts = 4, used = 2, layers = 2;
    const int first_part = relu_squared ? 1 : 0;
    ggml_context * weights[3] = {};
    ggml_backend_buffer_t weight_buffers[3] = {};
    ggml_tensor * w[layers][3] = {};
    const char * parts[] = { "gate", "up", "down" };
    std::vector<float> values(dim*dim*experts), importance(dim, 1);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distribution(-0.02f, 0.02f);
    for (int part = first_part; part < 3; ++part) {
        // Registration must join a layer even when its projections have different host buffers.
        weights[part] = ggml_init({ layers*ggml_tensor_overhead(), nullptr, true });
        for (int layer = 0; layer < layers; ++layer) {
            w[layer][part] = ggml_new_tensor_3d(weights[part], type, dim, dim, experts);
            char name[64];
            snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", layer, parts[part]);
            ggml_set_name(w[layer][part], name);
        }
        weight_buffers[part] = ggml_backend_alloc_ctx_tensors_from_buft(weights[part], buft);
        GGML_ASSERT(weight_buffers[part]);
        ggml_backend_buffer_set_usage(weight_buffers[part], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int layer = 0; layer < layers; ++layer) {
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = distribution(rng);
            }
            if (type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(w[layer][part], values.data(), 0, ggml_nbytes(w[layer][part]));
            } else {
                std::vector<unsigned char> quant(ggml_nbytes(w[layer][part]));
                GGML_ASSERT(ggml_quantize_chunk(type, values.data(), quant.data(), 0, dim*experts, dim, importance.data()) == quant.size());
                ggml_backend_tensor_set(w[layer][part], quant.data(), 0, quant.size());
            }
        }
    }

    auto start_cache = [&](ggml_backend_t backend) {
        auto cache = create();
        GGML_ASSERT(cache);
        GGML_ASSERT(add_device(cache, 2, 0, type == GGML_TYPE_ZNQ3 ? 2 : 1, dev));
        for (int layer = 0; layer < layers; ++layer) {
            for (int part = first_part; part < 3; ++part) {
                register_weights(cache, dev, w[layer][part]);
            }
        }
        GGML_ASSERT(memory(cache, dev) == 0);
        GGML_ASSERT(hits(cache) == 0);
        set_cache(backend, cache);
        return cache;
    };
    auto stop_cache = [&](ggml_backend_t backend, ggml_backend_cpu_expert_cache_t cache) {
        set_cache(backend, nullptr);
        free_cache(cache);
    };

    auto run = [&](ggml_backend_cpu_expert_cache_t cache, ggml_backend_t cpu_backend, int tokens, int pattern, int mode, int repeats) {
        ggml_backend_t backends[] = { gpu, cpu_backend };
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, 128, parallel, true);
        GGML_ASSERT(sched);
        expert_cache_copy_data copy_data { cache, copy };
        if (cache) {
            ggml_backend_sched_set_copy_callback(sched, copy_cached_activation, &copy_data);
        }
        auto ctx = ggml_init({ 64*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
        auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 1, tokens);
        auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
        auto scales = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, used, tokens);
        ggml_set_input(x);
        ggml_set_input(ids);
        ggml_set_input(scales);
        ggml_tensor * outputs[layers];
        ggml_tensor * down_ops[layers];
        ggml_tensor * up_ops[layers];
        for (int layer = 0; layer < layers; ++layer) {
            auto up = ggml_mul_mat_id(ctx, w[layer][1], x, ids);
            up_ops[layer] = up;
            ggml_backend_sched_set_tensor_backend(sched, up, cpu_backend);
            ggml_tensor * act;
            if (relu_squared) {
                auto relu = ggml_relu(ctx, up);
                ggml_backend_sched_set_tensor_backend(sched, relu, cpu_backend);
                act = ggml_sqr(ctx, relu);
            } else {
                auto gate = ggml_mul_mat_id(ctx, w[layer][0], x, ids);
                ggml_backend_sched_set_tensor_backend(sched, gate, cpu_backend);
                act = ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU);
            }
            // Down consumes a different activation for each selected expert, not broadcast input.
            GGML_ASSERT(act->ne[1] == used && act->ne[2] == tokens);
            auto down = ggml_mul_mat_id(ctx, w[layer][2], act, ids);
            down_ops[layer] = down;
            ggml_backend_sched_set_tensor_backend(sched, act, cpu_backend);
            ggml_backend_sched_set_tensor_backend(sched, down, cpu_backend);
            if (cache) {
                GGML_ASSERT(cache_device(cache, down) == dev);
                GGML_ASSERT(offload_device(cache, down) == (tokens <= 128 ? dev : nullptr));
            }
            auto weighted = ggml_mul(ctx, down, scales);
            ggml_backend_sched_set_tensor_backend(sched, weighted, mode == 1 ? cpu_backend : gpu);
            if (mode == 2) {
                ggml_set_output(up);
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
        if (cache) {
            prepare(cache, sched, graph);
        }
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
        std::vector<float> input(dim*tokens), factors(used*tokens);
        std::vector<int32_t> routes(used*tokens);
        expert_cache_result result;
        result.output.resize(dim*tokens*repeats);
        if (mode == 2) {
            result.observed.resize(2*dim*used*tokens*layers*repeats);
        }
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
            ggml_backend_tensor_get_async(gpu, out, result.output.data() + iteration*dim*tokens, 0, ggml_nbytes(out));
            if (mode == 2) {
                for (int layer = 0; layer < layers; ++layer) {
                    const size_t offset = 2*(iteration*layers + layer)*dim*used*tokens;
                    ggml_backend_tensor_get_async(cpu_backend, up_ops[layer],
                            result.observed.data() + offset, 0, ggml_nbytes(up_ops[layer]));
                    ggml_backend_tensor_get_async(cpu_backend, down_ops[layer],
                            result.observed.data() + offset + dim*used*tokens, 0, ggml_nbytes(down_ops[layer]));
                }
            }
        }
        ggml_backend_sched_synchronize(sched);
        result.copies = copy_data.copies;
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return result;
    };

    double worst = 0;
    auto compare = [&](const expert_cache_result & actual, const expert_cache_result & expected) {
        auto check = [&](const std::vector<float> & a, const std::vector<float> & e) {
            GGML_ASSERT(a.size() == e.size());
            double error = 0, norm = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                GGML_ASSERT(std::isfinite(a[i]) && std::isfinite(e[i]));
                const double delta = a[i] - e[i];
                error += delta*delta;
                norm += double(e[i])*e[i];
            }
            const double nmse = error/std::max(norm, 1e-30);
            if (nmse >= 0.001) {
                fprintf(stderr, "expert cache mismatch: type=%s activation=%s count=%zu nmse=%g error=%g norm=%g actual[0]=%g expected[0]=%g\n",
                        ggml_type_name(type), relu_squared ? "relu_squared" : "swiglu", a.size(), nmse, error, norm, a[0], e[0]);
            }
            GGML_ASSERT(nmse < 0.001);
            worst = std::max(worst, nmse);
        };
        check(actual.output, expected.output);
        if (!actual.observed.empty()) {
            check(actual.observed, expected.observed);
        }
    };
    const int sizes[] = { 1, 3, 8, 9, 128, 129, 1 };
    std::vector<expert_cache_result> expected;
    for (int tokens : sizes) {
        for (int pattern = 0; pattern < 3; ++pattern) {
            expected.push_back(run(nullptr, cpu, tokens, pattern, 2, 4));
        }
    }
    const expert_cache_result warm_expected[] = {
        run(nullptr, cpu, 1, 0, 0, 30),
        run(nullptr, cpu, 1, 2, 0, 30),
    };
    auto warm = [&](ggml_backend_cpu_expert_cache_t cache, ggml_backend_t backend, int pattern) {
        // Allocation may first happen at down; let asynchronous conversion/upload finish.
        const uint64_t minimum = uint64_t(30)*used*layers*(relu_squared ? 2 : 1);
        for (int attempt = 0; attempt < 8; ++attempt) {
            const uint64_t before = hits(cache);
            compare(run(cache, backend, 1, pattern, 0, 30), warm_expected[pattern == 2]);
            if (hits(cache) - before >= minimum) {
                GGML_ASSERT(memory(cache, dev) > 0);
                return;
            }
        }
        GGML_ABORT("expert cache did not execute GPU hits for %s", ggml_type_name(type));
    };

    uint64_t verified_hits = 0;
    uint64_t cached_copies = 0;
    for (int mode = 0; mode < 3; ++mode) {
        auto cache = start_cache(cpu);
        size_t scenario = 0;
        for (int tokens : sizes) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                warm(cache, cpu, 0);
                const uint64_t before = hits(cache);
                auto actual = run(cache, cpu, tokens, pattern, mode, 4);
                compare(actual, expected[scenario]);
                cached_copies += actual.copies;
                if (tokens > 128) {
                    GGML_ASSERT(hits(cache) == before);
                } else if (pattern == 0) {
                    // ReLU² must offload both up and expert-axis down, not just one projection.
                    const uint64_t minimum = uint64_t(tokens)*used*layers*4*(relu_squared ? 2 : 1);
                    GGML_ASSERT(hits(cache) - before >= minimum);
                } else if (pattern == 1) {
                    GGML_ASSERT(hits(cache) > before);
                } else {
                    // Fill both slots with different experts, then revisit the evicted routes.
                    warm(cache, cpu, 2);
                    compare(run(cache, cpu, tokens, 0, mode, 4), expected[scenario - 2]);
                }
                ++scenario;
            }
        }
        verified_hits += hits(cache);
        stop_cache(cpu, cache);
    }
    if (repacked && !relu_squared) {
        GGML_ASSERT(cached_copies > 0);
    }

    // Two simultaneously live roots/backends must retain independent cache ownership.
    auto other_cpu = ggml_backend_cpu_init();
    GGML_ASSERT(other_cpu);
    ggml_backend_cpu_set_n_threads(other_cpu, 4);
    auto first_cache = start_cache(cpu);
    auto second_cache = start_cache(other_cpu);
    warm(first_cache, cpu, 0);
    warm(second_cache, other_cpu, 2);
    const uint64_t survivor_hits = hits(second_cache);
    const size_t survivor_memory = memory(second_cache, dev);
    verified_hits += hits(first_cache);
    stop_cache(cpu, first_cache);
    ggml_backend_free(cpu);
    GGML_ASSERT(hits(second_cache) == survivor_hits);
    GGML_ASSERT(memory(second_cache, dev) == survivor_memory);
    compare(run(second_cache, other_cpu, 1, 2, 0, 4), expected[2]);
    GGML_ASSERT(hits(second_cache) > survivor_hits);
    verified_hits += hits(second_cache);
    stop_cache(other_cpu, second_cache);
    ggml_backend_free(other_cpu);

    printf("%s: expert cache reuse parallel=%d type=%s activation=%s PASS (maximum nmse %g, GPU hits %llu, cached copies %llu)\n",
            ggml_backend_dev_name(dev), parallel, ggml_type_name(type), relu_squared ? "relu_squared" : "swiglu",
            worst, (unsigned long long) verified_hits, (unsigned long long) cached_copies);
    for (int part = first_part; part < 3; ++part) {
        ggml_backend_buffer_free(weight_buffers[part]);
        ggml_free(weights[part]);
    }
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
            for (ggml_type type : {
                    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
                    GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
                    GGML_TYPE_ZNQ2, GGML_TYPE_ZNQ3, GGML_TYPE_ZNQ4 }) {
                test_expert_cache_reuse(dev, parallel, type, false);
                test_expert_cache_reuse(dev, parallel, type, true);
            }
        }
    }

    return 0;
}
