#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
        const float expected = user_input ? 10*iteration + 14 : 5*iteration + 8;
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
        }
    }
    return 0;
}
