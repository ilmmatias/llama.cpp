// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_ZNQ2 = 0.06f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst, float amplitude = 2.0f) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + amplitude*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits_cpu * qfns_cpu, ggml_type src0_type, size_t test_size,
                               const float * test_data1, const float * test_data2,
                               const float * test_data3, const float * test_data4,
                               const int nrc) {
    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
    const size_t pad  = 64;
    const size_t bx   = ggml_row_size(src0_type, test_size) + pad;
    const size_t by   = ggml_row_size(qfns_cpu->vec_dot_type, test_size) + pad;

    std::vector<uint8_t> tmp_q1(bx * nrc);
    std::vector<uint8_t> tmp_q2(by * nrc);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    if (nrc == 1) {
        float result = INFINITY;
        qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

        const float dot_ref = dot_product(test_data1, test_data2, test_size);
        return fabsf(result - dot_ref) / test_size;
    }

    // nrc == 2: kernel computes a 2x2 dot product matrix
    // Output layout: s[0]=dot(vx0,vy0), s[1]=dot(vx1,vy0), s[bs]=dot(vx0,vy1), s[bs+1]=dot(vx1,vy1)
    // row and output strides are padded, same as in the mul_mat path
    qfns_cpu->from_float(test_data3, tmp_q1.data() + bx, test_size);
    vdot->from_float(test_data4, tmp_q2.data() + by, test_size);

    const size_t bs = 16;
    std::vector<float> result(bs + 2, INFINITY);
    qfns_cpu->vec_dot(test_size, result.data(), bs, tmp_q1.data(), bx, tmp_q2.data(), by, 2);

    const float ref00 = dot_product(test_data1, test_data2, test_size);
    const float ref10 = dot_product(test_data3, test_data2, test_size);
    const float ref01 = dot_product(test_data1, test_data4, test_size);
    const float ref11 = dot_product(test_data3, test_data4, test_size);

    const auto err = [test_size](float val, float ref) {
        const float e = fabsf(val - ref) / test_size;
        return std::isfinite(e) ? e : INFINITY;
    };

    return std::max({err(result[0], ref00), err(result[1], ref10), err(result[bs], ref01), err(result[bs + 1], ref11)});
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    std::vector<float> test_data3(test_size);
    std::vector<float> test_data4(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_data(3.0, test_data3.size(), test_data3.data(), 1.0f);
    generate_data(4.0, test_data4.size(), test_data4.data(), 1.5f);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_ZNQ2    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_ZNQ3    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), nullptr, nullptr, 1);
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_ZNQ3
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : type == GGML_TYPE_ZNQ2
                ? MAX_DOT_PRODUCT_ERROR_ZNQ2
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }

            // Test nrc=2 path for types that support it
            if (qfns_cpu->nrows == 2) {
                const float vec_dot_error_nrc2 = dot_product_error(qfns_cpu, type, test_size, test_data.data(), test_data2.data(), test_data3.data(), test_data4.data(), 2);
                failed = !(vec_dot_error_nrc2 < max_allowed_error);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%5s dot product error (nrc=2):    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error_nrc2);
                }
            }
        }
    }

    return num_failed;
}

static int test_znq3_repack(bool verbose) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    assert(backend);
    auto * dev = ggml_backend_get_device(backend.get());
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto get_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack_buft = nullptr;
    if (get_bufts != nullptr) {
        for (auto * buft = get_bufts(dev); buft != nullptr && *buft != nullptr; ++buft) {
            if (std::string(ggml_backend_buft_name(*buft)) == "CPU_REPACK") {
                repack_buft = *buft;
                break;
            }
        }
    }
    if (repack_buft == nullptr) {
        return 0;
    }

    int num_failed = 0;
    // Short batches, GEMM tails, output tiles, and shared gate/up inputs.
    const int shapes[][4] = {
        {32, 56, 2, 2}, {256, 280, 3, 2}, {256, 56, 6, 2}, {4128, 56, 7, 2},
        {32, 56, 1, 4}, {256, 24, 2, 4}, {256, 56, 3, 4},
        {256, 280, 4, 8}, {256, 280, 6, 8}, {4128, 56, 7, 8},
    };
    for (const auto & shape : shapes) {
        const int k = shape[0], m = shape[1], n = shape[2];
        const int n_used = shape[3], n_expert = 2*n_used;
        ggml_init_params params = { 1024*1024, nullptr, true };
        ggml_context_ptr weight_ctx[2] = { ggml_context_ptr(ggml_init(params)), ggml_context_ptr(ggml_init(params)) };
        ggml_tensor * weights[2];
        ggml_backend_buffer_ptr weight_buf[2];
        for (int i = 0; i < 2; ++i) {
            weights[i] = ggml_new_tensor_3d(weight_ctx[i].get(), GGML_TYPE_ZNQ3, k, m, n_expert);
            weight_buf[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx[i].get(), i == 0 ? ggml_backend_cpu_buffer_type() : repack_buft));
            assert(weight_buf[i]);
            ggml_backend_buffer_set_usage(weight_buf[i].get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        for (bool broadcast : { false, true }) {
            ggml_context_ptr ctx(ggml_init(params));
            auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, broadcast ? 1 : n_used, n);
            auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n);
            ggml_tensor * out[2] = {
                ggml_mul_mat_id(ctx.get(), weights[0], input, ids),
                ggml_mul_mat_id(ctx.get(), weights[1], input, ids),
            };
            if (!ggml_backend_supports_op(backend.get(), out[1])) {
                return num_failed;
            }

            auto * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, out[0]);
            ggml_build_forward_expand(graph, out[1]);
            ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            assert(buf);

            std::vector<float> data(ggml_nelements(weights[0]));
            generate_data(0.0f, data.size(), data.data());
            std::vector<uint8_t> quantized(ggml_nbytes(weights[0]));
            ggml_quantize_chunk(GGML_TYPE_ZNQ3, data.data(), quantized.data(), 0, m*n_expert, k, nullptr);
            for (auto * weight : weights) {
                ggml_backend_tensor_set(weight, quantized.data(), 0, quantized.size());
            }
            data.resize(ggml_nelements(input));
            generate_data(1.0f, data.size(), data.data());
            ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
            std::vector<int32_t> routes(n*n_used);
            for (int t = 0; t < n; ++t) {
                for (int r = 0; r < n_used; ++r) {
                    routes[n_used*t + r] = n_used >= 4 && r == n_used - 1 && t % 3 == 0
                                          ? n_expert - 1 : 2*((r + t) % n_used);
                }
            }
            ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));

            std::vector<float> expected(ggml_nelements(out[0])), actual(expected.size());
            for (int threads : { 1, 3, 8 }) {
                ggml_backend_cpu_set_n_threads(backend.get(), threads);
                for (auto & route : routes) {
                    route = (route + 1) % n_expert;
                }
                ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
                std::fill(actual.begin(), actual.end(), NAN);
                ggml_backend_tensor_set(out[1], actual.data(), 0, ggml_nbytes(out[1]));
                assert(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_get(out[0], expected.data(), 0, ggml_nbytes(out[0]));
                ggml_backend_tensor_get(out[1], actual.data(), 0, ggml_nbytes(out[1]));
                double error = 0, norm = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    const double diff = actual[i] - expected[i];
                    error += diff * diff;
                    norm += (double) expected[i] * expected[i];
                }
                const bool failed = !(error <= 1e-6 * std::max(norm, 1e-20));
                num_failed += failed;
                if (failed || verbose) {
                    printf("znq3 repack k=%d m=%d n=%d used=%d broadcast=%d threads=%d: %s\n", k, m, n, n_used, broadcast, threads, RESULT_STR[failed]);
                }
            }
        }
    }
    return num_failed;
}

// In every group, all values with importance are equal, and the max (0) has none.
// The scale search in make_qkx3_quants then fits min == max and passes inf/nan to nearest_int (#29804).
static int test_quantize_imatrix_degenerate(bool verbose) {
    const int64_t n = 256;
    std::vector<float> x(n);
    std::vector<float> imatrix(n);
    std::vector<float> out(n);
    int num_failed = 0;

    printf("Testing degenerate imatrix:\n");
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
        const int64_t group = type == GGML_TYPE_Q2_K ? 16 : 32;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t g = i / group;
            const int64_t p = i % group;
            const bool important = p % 3 == 1;
            x[i] = important ? -0.02f*(g + 1) : (p % 2 ? -1.0f : 0.0f);
            imatrix[i] = important ? 1.0f : 0.0f;
        }
        printf("  - %s\n", ggml_type_name(type));

        std::vector<uint8_t> q(ggml_row_size(type, n));
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, x.data(), q.data(), 0, 1, n, imatrix.data());
        ggml_get_type_traits(type)->to_float(q.data(), out.data(), n);

        const bool failed = !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
        num_failed += failed;
        if (failed || verbose) {
            printf("%5s imatrix degenerate groups:      %s\n", ggml_type_name(type), RESULT_STR[failed]);
        }
    }

    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_znq3_repack(verbose);
    num_failed += test_quantize_imatrix_degenerate(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
