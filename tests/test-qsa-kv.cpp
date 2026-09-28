#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int dim   = 256;
constexpr int heads = 2;
constexpr int rows  = 4096;
constexpr int width = dim * heads;

struct cache {
    ggml_context *        ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor *         k;
    ggml_tensor *         v;

    explicit cache(ggml_backend_buffer_type_t buft, int n_rows = rows) {
        ctx    = ggml_init({ 16 * ggml_tensor_overhead(), nullptr, true });
        k      = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, n_rows, 1);
        v      = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, n_rows, 1);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        GGML_ASSERT(buffer);
    }

    ~cache() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    void restore(int version) {
        const int64_t n_values = ggml_nelements(k);
        std::vector<ggml_fp16_t> values(n_values);

        for (int64_t i = 0; i < n_values; ++i) {
            values[i] = ggml_fp32_to_fp16(std::sin(float(i % 997 + version * 17) * 0.03125f));
        }

        ggml_backend_tensor_set(k, values.data(), 0, ggml_nbytes(k));
        std::reverse(values.begin(), values.end());
        ggml_backend_tensor_set(v, values.data(), 0, ggml_nbytes(v));
    }
};

static std::vector<float> attention(
        ggml_backend_t backend, cache & kv, int n_queries, int first_page, int n_pages, int write_row,
        int n_rows = rows, int n_seqs = 1, int row_offset = 0, bool alias_v = false) {
    ggml_context * ctx =
        ggml_init({ 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true });
    ggml_cgraph * graph  = ggml_new_graph_custom(ctx, 128, false);
    ggml_tensor * update = nullptr;
    ggml_tensor * index  = nullptr;

    if (write_row >= 0) {
        update = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        index  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ggml_build_forward_expand(graph, ggml_set_rows(ctx, kv.k, update, index));
        ggml_build_forward_expand(graph, ggml_set_rows(ctx, kv.v, update, index));
    }

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, n_queries, 24, n_seqs);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_rows, n_queries, 1, n_seqs);
    const auto    view = [&](ggml_tensor * tensor, int extra_offset = 0) {
        return ggml_view_4d(ctx, tensor, dim, n_rows, heads, n_seqs,
                width*sizeof(ggml_fp16_t), dim*sizeof(ggml_fp16_t),
                size_t(n_rows)*width*sizeof(ggml_fp16_t), size_t(row_offset + extra_offset)*width*sizeof(ggml_fp16_t));
    };
    ggml_tensor * k = view(kv.k);
    ggml_tensor * v = alias_v ? view(kv.k, 4) : view(kv.v);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / 16, 0, 0);
    ggml_flash_attn_ext_set_n_kv_max(out, 512);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    std::vector<float> queries(ggml_nelements(q));

    for (size_t i = 0; i < queries.size(); ++i) {
        queries[i] = std::cos(float(i % 251) * 0.0625f);
    }

    ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));

    std::vector<ggml_fp16_t> masks(n_rows*n_queries*n_seqs, ggml_fp32_to_fp16(-INFINITY));

    for (int query = 0; query < n_queries*n_seqs; ++query) {
        for (int page = 0; page < n_pages; ++page) {
            const int base = 4*((first_page + page + 3*query) % (n_rows/4));

            for (int j = 0; j < 4; ++j) {
                masks[query*n_rows + base + j] = ggml_fp32_to_fp16(-0.0625f*(j + query % 3));
            }
        }
    }

    ggml_backend_tensor_set(mask, masks.data(), 0, ggml_nbytes(mask));

    if (update) {
        const int64_t      row = write_row;
        std::vector<float> values(width);

        for (int i = 0; i < width; ++i) {
            values[i] = float((i + write_row) % 19 - 9) * 0.125f;
        }

        ggml_backend_tensor_set(update, values.data(), 0, ggml_nbytes(update));
        ggml_backend_tensor_set(index, &row, 0, sizeof(row));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    if (update) {
        std::vector<ggml_fp16_t> saved(width);
        for (ggml_tensor * tensor : {kv.k, kv.v}) {
            ggml_backend_tensor_get(tensor, saved.data(), size_t(write_row)*width*sizeof(ggml_fp16_t),
                saved.size()*sizeof(ggml_fp16_t));

            for (int i = 0; i < width; ++i) {
                const float expected = float((i + write_row) % 19 - 9)*0.125f;
                GGML_ASSERT(saved[i] == ggml_fp32_to_fp16(expected));
            }
        }
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    return result;
}

static void compare(const std::vector<float> & expected, const std::vector<float> & actual, const char * scenario) {
    GGML_ASSERT(expected.size() == actual.size());

    float error = 0;

    for (size_t i = 0; i < expected.size(); ++i) {
        GGML_ASSERT(std::isfinite(actual[i]));
        error = std::max(error, std::abs(expected[i] - actual[i]));
    }

    if (error > 2e-6f) {
        fprintf(stderr, "%s: maximum output error %g\n", scenario, error);
        GGML_ABORT("paged KV attention differs from resident KV");
    }

    printf("%s: PASS (max error %g)\n", scenario, error);
}

enum class mutation {
    copy,
    shift,
    rope_set_rows,
    rms_norm_rope_set_rows,
};

static void mutate(ggml_backend_t backend, cache & kv, mutation kind) {
    ggml_context * ctx = ggml_init({
        32*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true,
    });
    ggml_cgraph * graph     = ggml_new_graph_custom(ctx, 32, false);
    ggml_tensor * input     = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * indices   = nullptr;
    ggml_tensor * scale     = nullptr;
    ggml_tensor * out;

    if (kind == mutation::copy) {
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
        out   = ggml_cpy(ctx, input, kv.k);
    } else if (kind == mutation::shift) {
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
        out       = ggml_rope_ext_inplace(ctx, ggml_reshape_3d(ctx, kv.k, dim, heads, rows),
                positions, nullptr, 64, GGML_ROPE_TYPE_NEOX, rows, 10000, 1, 0, 1, 32, 1);
    } else {
        input     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, 1);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        indices   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        out       = input;

        if (kind == mutation::rms_norm_rope_set_rows) {
            scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            out   = ggml_mul(ctx, ggml_rms_norm(ctx, out, 1e-6f), scale);
        }

        out = ggml_rope_ext(ctx, out, positions, nullptr,
                64, GGML_ROPE_TYPE_NEOX, rows, 10000, 1, 0, 1, 32, 1);
        out = ggml_set_rows(ctx, kv.k, ggml_view_2d(ctx, out, width, 1, width*sizeof(float), 0), indices);
    }

    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    if (input) {
        std::vector<float> values(ggml_nelements(input));

        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = std::cos(float(i % 137)*0.125f);
        }

        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    }

    if (positions) {
        std::vector<int32_t> values(ggml_nelements(positions), -7);
        ggml_backend_tensor_set(positions, values.data(), 0, ggml_nbytes(positions));
    }

    if (indices) {
        const int64_t row = 3;
        ggml_backend_tensor_set(indices, &row, 0, sizeof(row));
    }

    if (scale) {
        std::vector<float> values(dim, 0.75f);
        ggml_backend_tensor_set(scale, values.data(), 0, ggml_nbytes(scale));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test(ggml_backend_dev_t dev, ggml_backend_buffer_type_t paged_type) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);

    {
        cache resident(ggml_backend_dev_buffer_type(dev));
        cache paged(paged_type);
        resident.restore(0);
        paged.restore(0);

        const auto run = [&](const char * label, int nq, int page, int count, int write_row = -1) {
            compare(attention(backend, resident, nq, page, count, write_row),
                    attention(backend, paged, nq, page, count, write_row), label);
        };

        run("cold pages", 1, 0, 8);
        run("resident pages", 1, 0, 8);
        run("eviction", 1, 64, 8);
        run("revisit evicted pages", 1, 0, 8);
        run("overwrite a resident row", 1, 0, 8, 3);
        run("partial page survives overwrite", 2, 0, 8);
        run("selection union exceeds capacity", 8, 128, 48);
        run("prefill staging writes cache", 33, 0, 8, 2);
        run("decode after prefill", 1, 0, 8);

        compare(attention(backend, resident, 1, 0, 8, -1, 512),
                attention(backend, paged,    1, 0, 8, -1, 512), "short-context attention");
        compare(attention(backend, resident, 2, 0, 8, -1, 512, 1, 5),
                attention(backend, paged,    2, 0, 8, -1, 512, 1, 5), "unaligned cache view offset");
        compare(attention(backend, resident, 1, 0, 8, -1, rows - 256, 1, 0, true),
                attention(backend, paged,    1, 0, 8, -1, rows - 256, 1, 0, true), "aliased K/V cache views");
        compare(attention(backend, resident, 2, 0, 8, -1, rows/2, 2),
                attention(backend, paged,    2, 0, 8, -1, rows/2, 2), "multiple sequences use staging");

        resident.restore(1);
        paged.restore(1);
        run("host state restore invalidates pages", 1, 0, 8);

        const struct {
            mutation kind;
            const char * label;
        } mutations[] = {
            {mutation::copy,                   "device copy invalidates pages"},
            {mutation::shift,                  "K-shift invalidates pages"},
            {mutation::rope_set_rows,          "rotary row overwrite invalidates pages"},
            {mutation::rms_norm_rope_set_rows,  "normalized rotary row overwrite invalidates pages"},
        };

        for (const auto & test_case : mutations) {
            mutate(backend, resident, test_case.kind);
            mutate(backend, paged, test_case.kind);
            run(test_case.label, 1, 0, 8);
        }

        ggml_backend_buffer_clear(resident.buffer, 0);
        ggml_backend_buffer_clear(paged.buffer, 0);
        run("clear invalidates pages", 1, 0, 8);
    }

    ggml_backend_free(backend);
}

static void test_clock_wrap(ggml_backend_dev_t dev, ggml_backend_buffer_type_t paged_type) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);

    {
        constexpr int n_rows = 8192;
        cache resident(ggml_backend_dev_buffer_type(dev), n_rows);
        cache paged(paged_type, n_rows);
        resident.restore(0);
        paged.restore(0);

        // A 1025-slot cache crosses the cooperative sweep boundary and wraps
        // in the middle of a block. Fill it, evict its first pages, then revisit.
        for (int page = 0; page < n_rows/4; page += 128) {
            const auto actual = attention(backend, paged, 1, page, 128, -1, n_rows);

            if (page == 1024) {
                compare(attention(backend, resident, 1, page, 128, -1, n_rows),
                        actual, "CLOCK sweep crosses cache end");
            }
        }

        compare(attention(backend, resident, 1, 0, 128, -1, n_rows),
                attention(backend, paged,    1, 0, 128, -1, n_rows), "CLOCK revisit after wraparound");
    }

    ggml_backend_free(backend);
}

}  // namespace

int main() {
    ggml_backend_load_all();
    int tested = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        using buffer_type_fn = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t, uint32_t);
        auto fn              = reinterpret_cast<buffer_type_fn>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_qsa_kv_buffer_type"));

        if (fn) {
            printf("Testing %s\n", ggml_backend_dev_name(dev));
            test(dev, fn(dev, 64));
            test_clock_wrap(dev, fn(dev, 4100));
            ++tested;
        }
    }

    if (!tested) {
        puts("SKIP: no CUDA/HIP paged KV backend");
    }

    return 0;
}
