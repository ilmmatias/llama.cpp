// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
constexpr float MAX_DOT_PRODUCT_ERROR_ZNQ2_M = 0.06f;
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

// Positive finite E4M3 values in encoding order, from OCP FP8 rev. 1.1, section 5.1.
static const float F8_E4M3_VALUES[] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f,
};

static uint8_t fp32_to_f8_e4m3_ref(float value) {
    if (isnan(value)) {
        return 0x7F;
    }
    const uint8_t sign = signbit(value) ? 0x80 : 0;
    const double magnitude = fabs((double) value);
    if (magnitude >= F8_E4M3_VALUES[126]) {
        return sign | 126;
    }
    int nearest = 0;
    double distance = magnitude;
    for (int i = 1; i < 127; ++i) {
        const double candidate = fabs(magnitude - F8_E4M3_VALUES[i]);
        if (candidate < distance || (candidate == distance && (i & 1) == 0)) {
            nearest = i;
            distance = candidate;
        }
    }
    return sign | (uint8_t) nearest;
}

static int test_f8_e4m3(bool verbose) {
    static_assert(sizeof(ggml_fp8_e4m3_t) == 1);

    const auto * traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    const auto * traits_cpu = ggml_get_type_traits_cpu(GGML_TYPE_F8_E4M3);
    std::vector<uint8_t> encoded(256);
    std::vector<uint8_t> roundtrip(256);
    std::vector<float> decoded(256);
    int num_failed = 0;

    const bool traits_failed = traits->blck_size != 1 || traits->type_size != sizeof(ggml_fp8_e4m3_t) || traits->is_quantized;
    num_failed += traits_failed;
    if (verbose || traits_failed) {
        printf("f8_e4m3 scalar type traits:          %s\n", RESULT_STR[traits_failed]);
    }

    for (int i = 0; i < 256; ++i) {
        encoded[i] = (uint8_t) i;
    }
    traits->to_float(encoded.data(), decoded.data(), decoded.size());
    traits_cpu->from_float(decoded.data(), roundtrip.data(), decoded.size());

    for (int i = 0; i < 256; ++i) {
        const bool is_nan = (i & 0x7F) == 0x7F;
        const float expected = is_nan ? NAN : copysignf(F8_E4M3_VALUES[i & 0x7F], i & 0x80 ? -1.0f : 1.0f);
        const bool decode_failed = is_nan ? !isnan(decoded[i])
            : decoded[i] != expected || (expected == 0.0f && signbit(decoded[i]) != signbit(expected));
        const bool encode_failed = !is_nan && roundtrip[i] != encoded[i];
        if (decode_failed || encode_failed) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 code 0x%02x failed: decoded=%f expected=%f roundtrip=0x%02x\n",
                        i, decoded[i], expected, roundtrip[i]);
            }
        }
    }

    std::vector<float> inputs;
    for (float value : F8_E4M3_VALUES) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }
    for (int i = 0; i < 126; ++i) {
        const float midpoint = (F8_E4M3_VALUES[i] + F8_E4M3_VALUES[i + 1]) / 2.0f;
        for (float value : {nextafterf(midpoint, -INFINITY), midpoint, nextafterf(midpoint, INFINITY)}) {
            inputs.push_back(value);
            inputs.push_back(-value);
        }
    }
    for (float value : {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
                       nextafterf(448.0f, INFINITY), nextafterf(464.0f, -INFINITY), 464.0f,
                       nextafterf(464.0f, INFINITY), INFINITY, NAN}) {
        inputs.push_back(value);
        inputs.push_back(-value);
    }

    std::vector<uint8_t> quantized(inputs.size());
    std::vector<uint8_t> quantized_ref(inputs.size());
    traits_cpu->from_float(inputs.data(), quantized.data(), inputs.size());
    traits->from_float_ref(inputs.data(), quantized_ref.data(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const uint8_t expected = fp32_to_f8_e4m3_ref(inputs[i]);
        if (quantized[i] != expected || quantized_ref[i] != expected) {
            num_failed++;
            if (verbose) {
                printf("f8_e4m3 input %a failed: encoded=0x%02x reference=0x%02x expected=0x%02x\n",
                       inputs[i], quantized[i], quantized_ref[i], expected);
            }
        }
    }

    for (int n : {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 129, 257}) {
        std::vector<uint8_t> x(n + 1);
        std::vector<float> y(n + 1);
        double expected = 0.0;
        for (int i = 0; i < n; ++i) {
            const uint8_t code = (uint8_t) (i % 254 + (i % 254 >= 127));
            x[i + 1] = code;
            y[i + 1] = (i % 7 - 3) * 0.125f;
            expected += copysignf(F8_E4M3_VALUES[code & 0x7F], code & 0x80 ? -1.0f : 1.0f) * y[i + 1];
        }
        float actual;
        traits_cpu->vec_dot(n, &actual, 0, x.data() + 1, 0, y.data() + 1, 0, 1);
        const bool failed = fabs(actual - expected) > 1e-5 * std::max(1.0, fabs(expected));
        num_failed += failed;
        if (verbose || failed) {
            printf("f8_e4m3 dot n=%d:                  %s (ref=%f got=%f)\n", n, RESULT_STR[failed], expected, actual);
        }
    }

    for (uint8_t code : {0x7F, 0xFF}) {
        std::vector<uint8_t> x(65, 0);
        std::vector<float> y(65, 1.0f);
        x[0] = x.back() = code;
        float actual;
        traits_cpu->vec_dot(x.size(), &actual, 0, x.data(), 0, y.data(), 0, 1);
        num_failed += !isnan(actual);
    }

    if (verbose || num_failed) {
        printf("f8_e4m3 exhaustive conversion:      %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
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
                (type == GGML_TYPE_Q1_0 || type == GGML_TYPE_ZNQ2_S || type == GGML_TYPE_ZNQ2_SM) ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                (type == GGML_TYPE_ZNQ2_M || type == GGML_TYPE_ZNQ2_L || type == GGML_TYPE_ZNQ3_S || type == GGML_TYPE_ZNQ3_SM) ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                (type == GGML_TYPE_ZNQ3_M || type == GGML_TYPE_ZNQ3_L || type == GGML_TYPE_ZNQ4_S || type == GGML_TYPE_ZNQ4_SM) ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
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
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_ZNQ3_M || type == GGML_TYPE_ZNQ3_L || type == GGML_TYPE_ZNQ4_S || type == GGML_TYPE_ZNQ4_SM
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0 || type == GGML_TYPE_ZNQ2_S || type == GGML_TYPE_ZNQ2_SM
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : (type == GGML_TYPE_ZNQ2_M || type == GGML_TYPE_ZNQ2_L || type == GGML_TYPE_ZNQ3_S || type == GGML_TYPE_ZNQ3_SM)
                ? MAX_DOT_PRODUCT_ERROR_ZNQ2_M
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

// Fixed endpoints test the wire contract independently of quantization and dot kernels.
static int test_znq_new_layouts(bool verbose) {
    struct fixture {
        ggml_type type;
        int bits;
        int sbits;
        int8_t lo[16];
        int8_t hi[16];
    };
    const fixture cases[] = {
        {GGML_TYPE_ZNQ2_S, 1, 3,
            {-98, -61, -28, -16, -127, 31, -43, 0},
            {127, 63, 28, 18, -36, 127, 0, 57}},
        {GGML_TYPE_ZNQ2_SM, 1, 4,
            {-98, -61, -28, -16, -127, 31, -43, 0, -123, -28, -77, -104, 35, -85, -26, -127},
            {127, 63, 28, 18, -36, 127, 0, 57, 33, 127, -37, 82, 83, 5, 84, 0}},
        {GGML_TYPE_ZNQ3_SM, 2, 4,
            {-61, -104, -51, -80, -40, -93, -127, -62, -98, -123, -32, -91, -38, -69, -94, -73},
            {63, 82, 121, 49, 36, 82, 57, 92, 127, 33, 104, 76, 68, 83, 60, 119}},
        {GGML_TYPE_ZNQ4_SM, 3, 4,
            {-98, -109, -127, -76, -65, -127, -93, -89, -127, -127, -117, -100, -112, -75, -76, -125},
            {113, 84, 127, 70, 56, 101, 96, 127, 127, 97, 65, 127, 106, 115, 127, 80}},
        {GGML_TYPE_ZNQ5_SM, 4, 4,
            {-122, -127, -108, -108, -127, -121, -127, -115, -70, -127, -127, -111, -97, -122, -111, -118},
            {119, 117, 127, 108, 123, 127, 127, 112, 77, 126, 103, 118, 95, 122, 123, 127}},
        {GGML_TYPE_ZNQ6_SM, 5, 4,
            {-124, -116, -127, -108, -127, -125, -122, -118, -74, -102, -124, -124, -108, -125, -115, -124},
            {122, 123, 127, 126, 122, 117, 125, 118, 77, 100, 124, 109, 108, 122, 112, 126}},
        {GGML_TYPE_ZNQ7_SM, 6, 4,
            {-124, -125, -125, -116, -125, -122, -124, -118, -74, -102, -124, -108, -115, -108, -124, -125},
            {125, 123, 126, 123, 122, 125, 124, 118, 77, 100, 109, 125, 112, 108, 123, 117}},
    };
    int num_failed = 0;
    for (const auto & f : cases) {
        const int nbooks = 1 << f.sbits;
        const size_t size = 1 + f.sbits + 4 * f.bits;
        bool failed = ggml_blck_size(f.type) != 32 || ggml_type_size(f.type) != size;
        for (int base = 0; base < nbooks; ++base) {
            for (int scale : {0, 1, 7, 8, 120, 255}) {
                // Exactly one unaligned block: no padded payload or selector reads are valid.
                std::vector<uint8_t> bytes(size + 1, 0);
                auto * b = bytes.data() + 1;
                b[0] = scale;
                for (int g = 0; g < 8; ++g) {
                    const int book = (base + g) % nbooks;
                    for (int k = 0; k < f.sbits; ++k) {
                        const int bit = f.sbits * g + k;
                        b[1 + bit / 8] |= ((book >> k) & 1) << (bit % 8);
                    }
                }
                for (int j = 0; j < 32; ++j) {
                    if (j % 2) {
                        for (int k = 0; k < f.bits; ++k) {
                            const int bit = f.bits * j + k;
                            b[1 + f.sbits + bit / 8] |= 1 << (bit % 8);
                        }
                    }
                }
                float actual[32];
                ggml_get_type_traits(f.type)->to_float(b, actual, 32);
                const float d = scale < 8 ? std::ldexp(float(scale), -17) :
                    std::ldexp(1.0f + float(scale % 8) / 8.0f, scale / 8 - 15);
                for (int j = 0; j < 32; ++j) {
                    const int book = (base + j / 4) % nbooks;
                    failed |= actual[j] != d * (j % 2 ? f.hi[book] : f.lo[book]);
                }
            }
        }
        num_failed += failed;
        if (failed || verbose) {
            printf("%s wire selectors/endpoints/scales: %s\n", ggml_type_name(f.type), RESULT_STR[failed]);
        }
    }
    return num_failed;
}

static int test_znq_sm_error(bool verbose) {
    const ggml_type pairs[][2] = {
        {GGML_TYPE_ZNQ2_S, GGML_TYPE_ZNQ2_SM}, {GGML_TYPE_ZNQ3_S, GGML_TYPE_ZNQ3_SM},
        {GGML_TYPE_ZNQ4_S, GGML_TYPE_ZNQ4_SM}, {GGML_TYPE_ZNQ5_S, GGML_TYPE_ZNQ5_SM},
        {GGML_TYPE_ZNQ6_S, GGML_TYPE_ZNQ6_SM}, {GGML_TYPE_ZNQ7_S, GGML_TYPE_ZNQ7_SM},
    };
    int num_failed = 0;
    for (const auto & pair : pairs) {
        for (int n : {32, 96, 544}) {
            std::vector<float> x(n), importance(n), decoded[2];
            for (int pattern = 0; pattern < 4; ++pattern) {
                for (int j = 0; j < n; ++j) {
                    const float value = std::sin(float(17 * j + 31 * pattern));
                    x[j] = pattern == 0 ? value : pattern == 1 ? value * std::ldexp(1.0f, (j / 32) % 15 - 7) :
                           pattern == 2 ? (j % 9 == 0 ? 9.0f * value : 0.02f * value) : (j / 4 % 3 == 0 ? 0 : value + 2);
                }
                for (int mode = 0; mode < 4; ++mode) {
                    for (int j = 0; j < n; ++j) {
                        importance[j] = mode == 3 || (mode == 2 && j % 3 != 0) ? 0 : 0.25f * (1 + j % 13);
                    }
                    const float * imatrix = mode ? importance.data() : nullptr;
                    for (int v = 0; v < 2; ++v) {
                        std::vector<uint8_t> bytes(ggml_row_size(pair[v], n));
                        decoded[v].resize(n);
                        ggml_quantize_chunk(pair[v], x.data(), bytes.data(), 0, 1, n, imatrix);
                        ggml_get_type_traits(pair[v])->to_float(bytes.data(), decoded[v].data(), n);
                    }
                    bool failed = false;
                    for (int b = 0; b < n; b += 32) {
                        float sigma2 = 0;
                        for (int j = 0; j < 32; ++j) {
                            sigma2 += x[b + j] * x[b + j];
                        }
                        sigma2 *= 2.0f / 32;
                        double error[2] = {0, 0};
                        for (int j = 0; j < 32; ++j) {
                            const float w = imatrix ? importance[b + j] * std::sqrt(sigma2 + x[b + j] * x[b + j]) : 1;
                            for (int v = 0; v < 2; ++v) {
                                const double diff = double(x[b + j]) - decoded[v][b + j];
                                error[v] += w * diff * diff;
                            }
                        }
                        // SM retains every S book and explicitly evaluates the S scale.
                        failed |= !(error[1] <= error[0] * (1.0 + 1e-6) + 1e-20);
                    }
                    num_failed += failed;
                    if (failed || verbose) {
                        printf("%s retains S objective n=%d pattern=%d imatrix=%d: %s\n",
                               ggml_type_name(pair[1]), n, pattern, mode, RESULT_STR[failed]);
                    }
                }
            }
        }
    }
    return num_failed;
}

static int test_quantize_znq_rows(bool verbose) {
    int num_failed = 0;
    const int nrows = 3;

    for (ggml_type type : {GGML_TYPE_ZNQ2_S, GGML_TYPE_ZNQ2_SM, GGML_TYPE_ZNQ3_SM, GGML_TYPE_ZNQ4_SM, GGML_TYPE_ZNQ5_SM, GGML_TYPE_ZNQ6_SM, GGML_TYPE_ZNQ7_SM, GGML_TYPE_ZNQ2_M, GGML_TYPE_ZNQ2_L, GGML_TYPE_ZNQ3_S, GGML_TYPE_ZNQ3_M, GGML_TYPE_ZNQ3_L,
                          GGML_TYPE_ZNQ4_S, GGML_TYPE_ZNQ4_M, GGML_TYPE_ZNQ4_L, GGML_TYPE_ZNQ5_S, GGML_TYPE_ZNQ5_M,
                          GGML_TYPE_ZNQ5_L, GGML_TYPE_ZNQ6_S, GGML_TYPE_ZNQ6_M, GGML_TYPE_ZNQ6_L, GGML_TYPE_ZNQ7_S,
                          GGML_TYPE_ZNQ7_M, GGML_TYPE_ZNQ7_L}) {
        const auto * traits = ggml_get_type_traits(type);
        for (int n : {32, 96, 544}) {
            std::vector<float> x((nrows + 2) * n), importance(n), zero_importance(n, 0.0f);
            for (int row = 0; row < nrows + 2; ++row) {
                for (int j = 0; j < n; ++j) {
                    const float value = float((37*j + 19*row) % 513 - 256) * 0.5f;
                    const float nearby = j % 3 == 0 ? std::nextafter(value, -INFINITY) : j % 3 == 1 ? value : std::nextafter(value, INFINITY);
                    x[row*n + j] = row == 2 ? 0.0f : nearby * (row == 1 ? 0.000001f : 0.125f);
                }
            }
            for (int j = 0; j < n; ++j) {
                importance[j] = j % 5 == 0 ? 0.0f : 0.25f * (1 + j % 7);
            }

            const size_t row_size = ggml_row_size(type, n);
            for (int weighting : {0, 1, 2}) {
                // Leave guard rows around the chunk and use an unaligned output pointer.
                std::vector<uint8_t> actual((nrows + 2) * row_size + 2, 0xa5);
                std::vector<uint8_t> expected = actual;
                const float * imatrix = weighting == 1 ? importance.data() : weighting == 2 ? zero_importance.data() : nullptr;
                const size_t size = ggml_quantize_chunk(type, x.data(), actual.data() + 1, n, nrows, n, imatrix);
                for (int row = 1; row <= nrows; ++row) {
                    if (imatrix) {
                        ggml_quantize_chunk(type, x.data(), expected.data() + 1, row*n, 1, n, imatrix);
                    } else {
                        traits->from_float_ref(x.data() + row*n, expected.data() + 1 + row*row_size, n);
                    }
                }

                const bool failed = size != nrows * row_size || actual != expected;
                num_failed += failed;
                if (failed || verbose) {
                    printf("%s quantize rows=%d n=%d imatrix=%d: %s\n", ggml_type_name(type), nrows, n, weighting, RESULT_STR[failed]);
                }
            }
        }
    }
    return num_failed;
}

static int test_vec_dot_znq(bool verbose) {
    const auto * activations = ggml_get_type_traits(GGML_TYPE_Q8_0);
    int num_failed = 0;

    for (ggml_type type : {GGML_TYPE_ZNQ2_S, GGML_TYPE_ZNQ2_SM, GGML_TYPE_ZNQ3_SM, GGML_TYPE_ZNQ4_SM, GGML_TYPE_ZNQ5_SM, GGML_TYPE_ZNQ6_SM, GGML_TYPE_ZNQ7_SM, GGML_TYPE_ZNQ2_M, GGML_TYPE_ZNQ2_L, GGML_TYPE_ZNQ3_S, GGML_TYPE_ZNQ3_M, GGML_TYPE_ZNQ3_L, GGML_TYPE_ZNQ4_S, GGML_TYPE_ZNQ4_M,
                          GGML_TYPE_ZNQ4_L, GGML_TYPE_ZNQ5_S, GGML_TYPE_ZNQ5_M, GGML_TYPE_ZNQ5_L,
                          GGML_TYPE_ZNQ6_S, GGML_TYPE_ZNQ6_M, GGML_TYPE_ZNQ6_L, GGML_TYPE_ZNQ7_S,
                          GGML_TYPE_ZNQ7_M, GGML_TYPE_ZNQ7_L}) {
        const auto * weights = ggml_get_type_traits(type);
        const auto * cpu = ggml_get_type_traits_cpu(type);
        assert(cpu->vec_dot_type == GGML_TYPE_Q8_0);
        ggml_quantize_init(type);

        for (int n : {32, 64, 96, 160, 10240}) {
            std::vector<float> x(n), y(n), dx(n), dy(n);
            std::vector<uint8_t> qx(ggml_row_size(type, n) + 1);
            std::vector<uint8_t> qy(ggml_row_size(GGML_TYPE_Q8_0, n));

            const auto check = [&](const char * input, int seed) {
                weights->to_float(qx.data() + 1, dx.data(), n);
                activations->to_float(qy.data(), dy.data(), n);

                float result = INFINITY;
                cpu->vec_dot(n, &result, 0, qx.data() + 1, 0, qy.data(), 0, 1);
                const float reference = dot_product(dx.data(), dy.data(), n);
                const bool failed = !(fabsf(result - reference) / n < MAX_QUANTIZATION_REFERENCE_ERROR);
                num_failed += failed;
                if (failed || verbose) {
                    printf("%s native dot %s n=%d seed=%d: %s (ref=%f got=%f)\n",
                           ggml_type_name(type), input, n, seed, RESULT_STR[failed], reference, result);
                }
            };

            // Keep the expensive reference quantization bounded for long rows.
            for (int seed = 0; seed < (n == 10240 ? 2 : 8); ++seed) {
                for (int i = 0; i < n; ++i) {
                    x[i] = sinf(float(i * 17 + seed * 31)) * (1 + (i / 32) % 7);
                    y[i] = cosf(float(i * 13 + seed * 19));
                }
                weights->from_float_ref(x.data(), qx.data() + 1, n);
                activations->from_float_ref(y.data(), qy.data(), n);
                check("quantized", seed);
            }

            if (n == 10240) {
                continue;
            }

            const size_t bx = ggml_type_size(type);
            const size_t by = ggml_type_size(GGML_TYPE_Q8_0);
            // Sweep every selector/payload byte, then isolate zero and subnormal scales and the last block.
            for (int seed = 0; seed < 264; ++seed) {
                for (int ib = 0; ib < n / 32; ++ib) {
                    uint8_t * px = qx.data() + 1 + ib*bx;
                    uint8_t * py = qy.data() + ib*by;
                    for (size_t j = 1; j < bx; ++j) {
                        px[j] = seed < 256 ? uint8_t(seed + 37*j + 61*ib) :
                                seed == 256 ? 0 : seed == 257 ? 0x55 : seed == 258 ? 0xaa : 0xff;
                    }
                    px[0] = seed == 258 ? 0 : seed == 259 ? 1 : seed == 260 ? 7 : seed == 261 ? 8 :
                            seed == 262 ? 120 : seed == 263 && ib != n / 32 - 1 ? 0 : 104 + ib;

                    // Q8_0 is a half scale followed by 32 signed bytes; include -128, which quantization does not emit.
                    const ggml_fp16_t scale = seed == 256 ? 0 : seed == 257 ? 1 : ggml_fp32_to_fp16(1.0f / 128.0f);
                    std::memcpy(py, &scale, sizeof(scale));
                    for (int j = 0; j < 32; ++j) {
                        py[sizeof(scale) + j] = seed % 4 == 0 ? (j % 2 == 0 ? 0x80 : 0x7f) :
                                               seed % 4 == 1 ? 0x7f : seed % 4 == 2 ? 0x80 :
                                               uint8_t(13*j + 17*seed + 29*ib);
                    }
                }
                check("packed", seed);
            }
        }
    }
    return num_failed;
}

static int test_mul_mat_znq(bool verbose) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    assert(backend);
    int num_failed = 0;
    // Plane sizes deliberately do not align with four-column batches or 16-column chunks.
    const int shapes[][7] = {
        // k, m, n, weight planes 2/3, input planes 2/3
        {32,   1,  3, 1, 1, 2, 2},
        {32,   1,  4, 1, 1, 1, 1},
        {32,   3,  4, 2, 1, 4, 1},
        {96,   3,  5, 2, 1, 4, 1},
        {32,   9,  7, 1, 2, 1, 4},
        {96,   9, 17, 2, 2, 4, 2},
        {96, 324,  5, 1, 1, 1, 1},
        {96, 324, 17, 2, 1, 2, 1},
    };
    const auto * q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const auto quantize_input = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;

    for (ggml_type type : {GGML_TYPE_ZNQ2_S, GGML_TYPE_ZNQ2_SM, GGML_TYPE_ZNQ3_SM, GGML_TYPE_ZNQ4_SM, GGML_TYPE_ZNQ5_SM, GGML_TYPE_ZNQ6_SM, GGML_TYPE_ZNQ7_SM, GGML_TYPE_ZNQ2_M, GGML_TYPE_ZNQ2_L, GGML_TYPE_ZNQ3_S, GGML_TYPE_ZNQ3_M, GGML_TYPE_ZNQ3_L, GGML_TYPE_ZNQ4_S, GGML_TYPE_ZNQ4_M,
                          GGML_TYPE_ZNQ4_L, GGML_TYPE_ZNQ5_S, GGML_TYPE_ZNQ5_M, GGML_TYPE_ZNQ5_L,
                          GGML_TYPE_ZNQ6_S, GGML_TYPE_ZNQ6_M, GGML_TYPE_ZNQ6_L, GGML_TYPE_ZNQ7_S,
                          GGML_TYPE_ZNQ7_M, GGML_TYPE_ZNQ7_L}) {
        ggml_quantize_init(type);
        const auto * traits = ggml_get_type_traits(type);
        for (const auto & shape : shapes) {
            const int k = shape[0], m = shape[1], n = shape[2];
            const int w2 = shape[3], w3 = shape[4], a2 = shape[5], a3 = shape[6];
            const size_t weight_row_size = ggml_row_size(type, k);
            const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_0, k);
            std::vector<uint8_t> weight_data(weight_row_size*m*w2*w3);
            std::vector<float> weight_float(k*m*w2*w3);
            // Valid packed payloads avoid expensive weight quantization and vary every row and plane.
            for (int row = 0; row < m*w2*w3; ++row) {
                for (int block = 0; block < k/32; ++block) {
                    auto * packed = weight_data.data() + row*weight_row_size + block*ggml_type_size(type);
                    packed[0] = 100 + (7*row + block) % 20;
                    for (size_t j = 1; j < ggml_type_size(type); ++j) {
                        packed[j] = uint8_t(37*j + 61*block + 17*row + 11*(row/256));
                    }
                }
                traits->to_float(weight_data.data() + row*weight_row_size, weight_float.data() + row*k, k);
            }

            for (int mode = 0; mode < 3; ++mode) {
                const ggml_type input_type = mode == 0 ? GGML_TYPE_F32 : GGML_TYPE_Q8_0;
                const int stride = mode == 2 ? 2 : 1;
                ggml_init_params params = { 1024*1024, nullptr, true };
                ggml_context_ptr weight_ctx(ggml_init(params));
                ggml_context_ptr ctx(ggml_init(params));
                auto * weights = ggml_new_tensor_4d(weight_ctx.get(), type, k, m, w2, w3);
                ggml_backend_buffer_ptr weight_buf(ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx.get(), ggml_backend_cpu_buffer_type()));
                assert(weight_buf);
                ggml_backend_buffer_set_usage(weight_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                auto * storage = ggml_new_tensor_4d(ctx.get(), input_type, k, stride*n, a2, a3);
                const size_t input_row_size = ggml_row_size(input_type, k);
                const size_t offset = stride == 2 ? input_row_size : 0;
                auto * input = ggml_view_4d(ctx.get(), storage, k, n, a2, a3,
                                           stride*input_row_size, storage->nb[2], storage->nb[3], offset);
                auto * out = ggml_mul_mat(ctx.get(), weights, input);
                if (!ggml_backend_supports_op(backend.get(), out)) {
                    fprintf(stderr, "%s native matmul does not support %s input\n", ggml_type_name(type), ggml_type_name(input_type));
                    return num_failed + 1;
                }
                auto * graph = ggml_new_graph(ctx.get());
                ggml_build_forward_expand(graph, out);
                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
                assert(buf);
                ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size());

                std::vector<float> data(ggml_nelements(storage));
                std::vector<uint8_t> quantized(q8_row_size*stride*n*a2*a3);
                for (int row = 0; row < stride*n*a2*a3; ++row) {
                    for (int i = 0; i < k; ++i) {
                        data[row*k + i] = sinf(float(13*i + 19*row)) * (1 + row % 5) + 0.03f*(row % 7);
                    }
                    quantize_input(data.data() + row*k, quantized.data() + row*q8_row_size, k);
                }
                if (input_type == GGML_TYPE_F32) {
                    ggml_backend_tensor_set(storage, data.data(), 0, ggml_nbytes(storage));
                } else {
                    ggml_backend_tensor_set(storage, quantized.data(), 0, quantized.size());
                }

                std::vector<float> expected(ggml_nelements(out)), actual(expected.size()), input_float(k);
                for (int i3 = 0; i3 < a3; ++i3) {
                    for (int i2 = 0; i2 < a2; ++i2) {
                        const int weight_plane = i2/(a2/w2) + w2*(i3/(a3/w3));
                        for (int col = 0; col < n; ++col) {
                            const int input_row = stride*(col + n*(i2 + a2*i3)) + (stride == 2 ? 1 : 0);
                            q8->to_float(quantized.data() + input_row*q8_row_size, input_float.data(), k);
                            for (int row = 0; row < m; ++row) {
                                const int index = row + m*(col + n*(i2 + a2*i3));
                                expected[index] = dot_product(weight_float.data() + k*(row + m*weight_plane), input_float.data(), k);
                            }
                        }
                    }
                }

                for (int threads : { 1, 3 }) {
                    ggml_backend_cpu_set_n_threads(backend.get(), threads);
                    std::fill(actual.begin(), actual.end(), NAN);
                    ggml_backend_tensor_set(out, actual.data(), 0, ggml_nbytes(out));
                    assert(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
                    size_t mismatch = actual.size();
                    for (size_t i = 0; i < actual.size(); ++i) {
                        if (!(fabsf(actual[i] - expected[i])/k < MAX_QUANTIZATION_REFERENCE_ERROR)) {
                            mismatch = i;
                            break;
                        }
                    }
                    const bool failed = mismatch != actual.size();
                    num_failed += failed;
                    if (failed || verbose) {
                        printf("%s native matmul input=%s stride=%d k=%d m=%d n=%d weights=%dx%d inputs=%dx%d threads=%d: %s\n",
                               ggml_type_name(type), ggml_type_name(input_type), stride, k, m, n, w2, w3, a2, a3, threads, RESULT_STR[failed]);
                        if (failed) {
                            printf("  output[%zu]: ref=%f got=%f\n", mismatch, expected[mismatch], actual[mismatch]);
                        }
                    }
                }
            }
        }
    }
    return num_failed;
}

static int test_repack(bool verbose, ggml_type weight_type, ggml_type input_type, bool dense = false, int dense_planes = 1) {
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
        {32, 56, 2, 2}, {640, 24, 7, 2}, {256, 280, 3, 2}, {256, 56, 6, 2}, {4128, 56, 7, 2},
        {32, 56, 1, 4}, {256, 24, 2, 4}, {256, 56, 3, 4}, {96, 24, 5, 4},
        {256, 280, 4, 8}, {256, 280, 6, 8}, {4128, 56, 7, 8},
        {32, 8, 20, 2}, {640, 24, 25, 2}, {640, 40, 20, 2}, {4128, 56, 25, 2}, {256, 280, 64, 2},
    };
    for (const auto & shape : shapes) {
        const int k = shape[0], m = shape[1], n = shape[2];
        if (n >= 20 && (!dense || (weight_type != GGML_TYPE_ZNQ2_S && weight_type != GGML_TYPE_ZNQ2_SM && weight_type != GGML_TYPE_ZNQ3_SM && weight_type != GGML_TYPE_ZNQ4_SM && weight_type != GGML_TYPE_ZNQ5_SM && weight_type != GGML_TYPE_ZNQ6_SM && weight_type != GGML_TYPE_ZNQ7_SM && weight_type != GGML_TYPE_ZNQ2_M && weight_type != GGML_TYPE_ZNQ2_L &&
            weight_type != GGML_TYPE_ZNQ3_S && weight_type != GGML_TYPE_ZNQ3_M && weight_type != GGML_TYPE_ZNQ3_L && weight_type != GGML_TYPE_ZNQ4_S && weight_type != GGML_TYPE_ZNQ4_M && weight_type != GGML_TYPE_ZNQ4_L && weight_type != GGML_TYPE_ZNQ5_S && weight_type != GGML_TYPE_ZNQ5_M && weight_type != GGML_TYPE_ZNQ5_L && weight_type != GGML_TYPE_ZNQ6_S && weight_type != GGML_TYPE_ZNQ6_M && weight_type != GGML_TYPE_ZNQ6_L && weight_type != GGML_TYPE_ZNQ7_S && weight_type != GGML_TYPE_ZNQ7_M && weight_type != GGML_TYPE_ZNQ7_L))) {
            continue;
        }
        if (dense && std::any_of(shapes, &shape, [&](const auto & previous) {
                return previous[0] == k && previous[1] == m && previous[2] == n;
            })) {
            continue;
        }
        if (k % ggml_blck_size(weight_type) != 0 || k % ggml_blck_size(input_type) != 0) {
            continue;
        }
        const int n_used = shape[3], n_expert = dense ? 1 : 2*n_used;
        ggml_init_params params = { 1024*1024, nullptr, true };
        ggml_context_ptr weight_ctx[2] = { ggml_context_ptr(ggml_init(params)), ggml_context_ptr(ggml_init(params)) };
        ggml_tensor * weights[2];
        ggml_backend_buffer_ptr weight_buf[2];
        for (int i = 0; i < 2; ++i) {
            weights[i] = dense ? ggml_new_tensor_2d(weight_ctx[i].get(), weight_type, k, m)
                               : ggml_new_tensor_3d(weight_ctx[i].get(), weight_type, k, m, n_expert);
            weight_buf[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx[i].get(), i == 0 ? ggml_backend_cpu_buffer_type() : repack_buft));
            assert(weight_buf[i]);
            ggml_backend_buffer_set_usage(weight_buf[i].get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        for (bool broadcast : { false, true }) {
            if (dense && broadcast) {
                continue;
            }
            ggml_context_ptr ctx(ggml_init(params));
            const int slots = dense ? n : (broadcast ? 1 : n_used);
            const int planes = dense ? dense_planes : n;
            const size_t row_size = ggml_row_size(input_type, k);
            const int stride = input_type == GGML_TYPE_F32 && dense ? 1 : 2;
            auto * storage = ggml_new_tensor_3d(ctx.get(), input_type, k, stride*slots, planes);
            auto * input = ggml_view_3d(ctx.get(), storage, k, slots, planes, stride*row_size, stride*slots*row_size, 0);
            auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used, n);
            ggml_tensor * out[2] = {
                dense ? ggml_mul_mat(ctx.get(), weights[0], input) : ggml_mul_mat_id(ctx.get(), weights[0], input, ids),
                dense ? ggml_mul_mat(ctx.get(), weights[1], input) : ggml_mul_mat_id(ctx.get(), weights[1], input, ids),
            };
            auto * float_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, slots, planes);
            auto * float_out = dense ? ggml_mul_mat(ctx.get(), weights[1], float_input)
                                    : ggml_mul_mat_id(ctx.get(), weights[1], float_input, ids);
            if (!ggml_backend_supports_op(backend.get(), float_out)) {
                continue;
            }
            if (!ggml_backend_supports_op(backend.get(), out[1])) {
                fprintf(stderr, "%s repack does not support %s input\n", ggml_type_name(weight_type), ggml_type_name(input_type));
                return num_failed + 1;
            }

            auto * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, out[0]);
            ggml_build_forward_expand(graph, out[1]);
            ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            assert(buf);

            std::vector<float> data(ggml_nelements(weights[0]));
            generate_data(0.0f, data.size(), data.data());
            std::vector<uint8_t> quantized(ggml_nbytes(weights[0]));
            ggml_quantize_chunk(weight_type, data.data(), quantized.data(), 0, m*n_expert, k, nullptr);
            for (auto * weight : weights) {
                ggml_backend_tensor_set(weight, quantized.data(), 0, quantized.size());
            }
            data.resize(ggml_nelements(storage));
            generate_data(1.0f, data.size(), data.data());
            auto set_input = [&] {
                if (input_type == GGML_TYPE_F32) {
                    ggml_backend_tensor_set(storage, data.data(), 0, ggml_nbytes(storage));
                } else {
                    quantized.resize(ggml_nbytes(storage));
                    const auto from_float = ggml_get_type_traits_cpu(input_type)->from_float;
                    for (int row = 0; row < stride*slots*planes; ++row) {
                        from_float(data.data() + row*k, quantized.data() + row*row_size, k);
                    }
                    ggml_backend_tensor_set(storage, quantized.data(), 0, quantized.size());
                }
            };
            set_input();
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
                if (dense && dense_planes > 1) {
                    for (float & value : data) value = -0.9f*value + 0.001f;
                    set_input();
                }
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
                    printf("%s repack input=%s dense=%d k=%d m=%d n=%d planes=%d used=%d broadcast=%d threads=%d: %s\n", ggml_type_name(weight_type), ggml_type_name(input_type), dense, k, m, n, planes, n_used, broadcast, threads, RESULT_STR[failed]);
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
    for (ggml_type type : {GGML_TYPE_ZNQ2_S, GGML_TYPE_ZNQ2_SM, GGML_TYPE_ZNQ3_SM, GGML_TYPE_ZNQ4_SM, GGML_TYPE_ZNQ5_SM, GGML_TYPE_ZNQ6_SM, GGML_TYPE_ZNQ7_SM, GGML_TYPE_ZNQ2_L, GGML_TYPE_ZNQ3_S, GGML_TYPE_ZNQ3_L, GGML_TYPE_ZNQ4_S, GGML_TYPE_ZNQ4_L, GGML_TYPE_ZNQ5_S, GGML_TYPE_ZNQ5_M, GGML_TYPE_ZNQ5_L, GGML_TYPE_ZNQ6_S, GGML_TYPE_ZNQ6_M, GGML_TYPE_ZNQ6_L, GGML_TYPE_ZNQ7_S, GGML_TYPE_ZNQ7_M, GGML_TYPE_ZNQ7_L, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1}) {
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
    num_failed += test_f8_e4m3(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_znq_new_layouts(verbose);
    num_failed += test_znq_sm_error(verbose);
    num_failed += test_quantize_znq_rows(verbose);
    num_failed += test_vec_dot_znq(verbose);
    num_failed += test_mul_mat_znq(verbose);
    for (ggml_type type : { GGML_TYPE_ZNQ2_M, GGML_TYPE_ZNQ3_M, GGML_TYPE_ZNQ4_M }) {
        num_failed += test_repack(verbose, type, GGML_TYPE_F32);
    }
    for (ggml_type type : {GGML_TYPE_ZNQ2_S, GGML_TYPE_ZNQ2_SM, GGML_TYPE_ZNQ3_SM, GGML_TYPE_ZNQ4_SM, GGML_TYPE_ZNQ5_SM, GGML_TYPE_ZNQ6_SM, GGML_TYPE_ZNQ7_SM, GGML_TYPE_ZNQ2_L, GGML_TYPE_ZNQ3_S, GGML_TYPE_ZNQ3_L, GGML_TYPE_ZNQ4_S, GGML_TYPE_ZNQ4_L, GGML_TYPE_ZNQ5_S, GGML_TYPE_ZNQ5_M, GGML_TYPE_ZNQ5_L, GGML_TYPE_ZNQ6_S, GGML_TYPE_ZNQ6_M, GGML_TYPE_ZNQ6_L, GGML_TYPE_ZNQ7_S, GGML_TYPE_ZNQ7_M, GGML_TYPE_ZNQ7_L}) {
        num_failed += test_repack(verbose, type, GGML_TYPE_F32);
        num_failed += test_repack(verbose, type, GGML_TYPE_F32, true);
        // Shared sums must cover every plane, including Q8 tails and strided input rows.
        num_failed += test_repack(verbose, type, GGML_TYPE_F32, true, 3);
        num_failed += test_repack(verbose, type, GGML_TYPE_Q8_0, true, 3);
    }
    for (ggml_type type : {GGML_TYPE_ZNQ2_S, GGML_TYPE_ZNQ2_SM, GGML_TYPE_ZNQ3_SM, GGML_TYPE_ZNQ4_SM, GGML_TYPE_ZNQ5_SM, GGML_TYPE_ZNQ6_SM, GGML_TYPE_ZNQ7_SM, GGML_TYPE_ZNQ2_M, GGML_TYPE_ZNQ2_L, GGML_TYPE_ZNQ3_S, GGML_TYPE_ZNQ3_M, GGML_TYPE_ZNQ3_L, GGML_TYPE_ZNQ4_S, GGML_TYPE_ZNQ4_M, GGML_TYPE_ZNQ4_L, GGML_TYPE_ZNQ5_S, GGML_TYPE_ZNQ5_M, GGML_TYPE_ZNQ5_L, GGML_TYPE_ZNQ6_S, GGML_TYPE_ZNQ6_M, GGML_TYPE_ZNQ6_L, GGML_TYPE_ZNQ7_S, GGML_TYPE_ZNQ7_M, GGML_TYPE_ZNQ7_L, GGML_TYPE_Q4_0,
                           GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        const ggml_type input_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
        num_failed += test_repack(verbose, type, input_type);
        num_failed += test_repack(verbose, type, input_type, true);
    }
    for (ggml_type type : {GGML_TYPE_ZNQ2_M, GGML_TYPE_ZNQ3_M, GGML_TYPE_ZNQ4_M}) {
        num_failed += test_repack(verbose, type, GGML_TYPE_F32, true, 3);
        num_failed += test_repack(verbose, type, GGML_TYPE_Q8_0, true, 3);
    }
    num_failed += test_quantize_imatrix_degenerate(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
