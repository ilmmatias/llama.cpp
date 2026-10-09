#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "common.cuh"
#include "fusion-rtc.h"
#include "ggml-fusion.h"
#include "ggml-cpu.h"
#include <hip/hiprtc.h>

#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#ifdef __linux__
#include "hiprtc-test-interpose.h"
#include <dlfcn.h>
#endif

#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1); \
        } \
    } while (0)

#define RTC_CHECK(expression) \
    do { \
        const hiprtcResult status = (expression); \
        if (status != HIPRTC_SUCCESS) { \
            std::fprintf(stderr, "%s: %s\n", #expression, hiprtcGetErrorString(status)); \
            std::exit(1); \
        } \
    } while (0)

static void raw(bool capture) {
    hipDeviceProp_t props;
    CUDA_CHECK(hipGetDeviceProperties(&props, 0));

    int major, minor, runtime, driver;
    RTC_CHECK(hiprtcVersion(&major, &minor));
    CUDA_CHECK(hipRuntimeGetVersion(&runtime));
    CUDA_CHECK(hipDriverGetVersion(&driver));

    std::printf("target=%s wave=%d hiprtc=%d.%d header=%d runtime=%d driver=%d\n",
                props.gcnArchName, props.warpSize, major, minor, HIP_VERSION, runtime, driver);

    std::string source = R"(
extern "C" __global__ void probe(float * out, const float * in, unsigned long long n, float bias) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + bias;
}
)";
    if (capture) {
        source += R"(
extern "C" __global__ void partial(float * out, const float * in, unsigned long long n, float bias) {
    __shared__ float sums[256];
    unsigned long long base = (unsigned long long) blockIdx.x * 4096;
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < 4096 && base + i < n; i += 256) sum += in[base + i] * 2.0f + bias;
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = 128; offset; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (!threadIdx.x) out[blockIdx.x] = sums[0];
}
extern "C" __global__ void finalize(float * out, const float * in, unsigned long long n) {
    __shared__ float sums[256];
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < n; i += 256) sum += in[i];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = 128; offset; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (!threadIdx.x) out[0] = sums[0];
}
)";
    }

    const std::string target = "--gpu-architecture=" + std::string(props.gcnArchName);
    const char * options[] = {target.c_str(), "-std=c++17", "-O3", "-ffp-contract=off"};

    for (int iteration = 0; iteration < 100; ++iteration) {
        hipStream_t stream;
        CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

        hiprtcProgram program = nullptr;
        RTC_CHECK(hiprtcCreateProgram(&program, source.c_str(), "probe.cpp", 0, nullptr, nullptr));
        const auto compiled = hiprtcCompileProgram(program, 4, options);

        size_t size;
        RTC_CHECK(hiprtcGetProgramLogSize(program, &size));
        std::vector<char> log(size);
        if (size) {
            RTC_CHECK(hiprtcGetProgramLog(program, log.data()));
        }

        if (compiled != HIPRTC_SUCCESS) {
            std::fprintf(stderr, "%s\n", log.empty() ? "compiler log unavailable" : log.data());
        }
        RTC_CHECK(compiled);

        RTC_CHECK(hiprtcGetCodeSize(program, &size));
        std::vector<char> code(size);
        RTC_CHECK(hiprtcGetCode(program, code.data()));
        RTC_CHECK(hiprtcDestroyProgram(&program));
        program = nullptr;

        hipModule_t module;
        hipFunction_t pointwise, partial_function = nullptr, final_function = nullptr;
        CUDA_CHECK(hipModuleLoadData(&module, code.data()));
        CUDA_CHECK(hipModuleGetFunction(&pointwise, module, "probe"));
        if (capture) {
            CUDA_CHECK(hipModuleGetFunction(&partial_function, module, "partial"));
            CUDA_CHECK(hipModuleGetFunction(&final_function, module, "finalize"));
        }

        const size_t capacity = capture ? 8193 : 513;
        std::vector<float> input(capacity), output(capacity);
        float * in, * out, * alternate = nullptr, * scratch = nullptr;
        CUDA_CHECK(hipMalloc(&in, capacity * sizeof(float)));
        CUDA_CHECK(hipMalloc(&out, capacity * sizeof(float)));
        if (capture) {
            CUDA_CHECK(hipMalloc(&alternate, capacity * sizeof(float)));
            CUDA_CHECK(hipMalloc(&scratch, 3 * sizeof(float)));
        }

        for (int reduction = 0; reduction <= int(capture); ++reduction) {
            hipGraph_t graph = nullptr;
            hipGraphExec_t executable = nullptr;
            unsigned long long n = reduction ? 8193 : 513;
            float * current_input = in;
            float bias = 1.0f;

            for (int step = 0; step < (capture ? 3 : 1); ++step) {
                if (step == 2) {
                    CUDA_CHECK(hipGraphExecDestroy(executable));
                    CUDA_CHECK(hipGraphDestroy(graph));
                    n = reduction ? 4097 : 257;
                    current_input = alternate;
                    bias = 2.0f;
                }

                for (size_t i = 0; i < input.size(); ++i) {
                    input[i] = step ? float(step) : float(int(i % 17) - 8);
                }
                CUDA_CHECK(hipMemcpyAsync(current_input, input.data(), capacity * sizeof(float), hipMemcpyHostToDevice, stream));

                if (!capture || step != 1) {
                    if (capture) {
                        CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
                    }

                    if (reduction) {
                        unsigned long long partials = (n + 4095) / 4096;
                        void * args[] = {&scratch, &current_input, &n, &bias};
                        void * final_args[] = {&out, &scratch, &partials};
                        CUDA_CHECK(hipModuleLaunchKernel(partial_function, partials, 1, 1, 256, 1, 1, 0, stream, args, nullptr));
                        CUDA_CHECK(hipModuleLaunchKernel(final_function, 1, 1, 1, 256, 1, 1, 0, stream, final_args, nullptr));
                    } else {
                        void * args[] = {&out, &current_input, &n, &bias};
                        CUDA_CHECK(hipModuleLaunchKernel(pointwise, (n + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr));
                    }

                    if (capture) {
                        CUDA_CHECK(hipStreamEndCapture(stream, &graph));
                        CUDA_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                    }
                }

                if (capture) {
                    CUDA_CHECK(hipGraphLaunch(executable, stream));
                }

                const size_t count = reduction ? 1 : size_t(n);
                CUDA_CHECK(hipMemcpyAsync(output.data(), out, count * sizeof(float), hipMemcpyDeviceToHost, stream));
                CUDA_CHECK(hipStreamSynchronize(stream));

                if (reduction) {
                    double expected = 0;
                    for (size_t i = 0; i < n; ++i) {
                        expected += input[i] * 2.0f + bias;
                    }

                    REQUIRE(output[0] == float(expected));
                } else {
                    for (size_t i = 0; i < n; ++i) {
                        REQUIRE(output[i] == input[i] * 2.0f + bias);
                    }
                }
            }

            if (capture) {
                CUDA_CHECK(hipGraphExecDestroy(executable));
                CUDA_CHECK(hipGraphDestroy(graph));
            }
        }

        if (capture) {
            CUDA_CHECK(hipFree(scratch));
            CUDA_CHECK(hipFree(alternate));
        }

        CUDA_CHECK(hipFree(out));
        CUDA_CHECK(hipFree(in));
        CUDA_CHECK(hipModuleUnload(module));
        CUDA_CHECK(hipStreamDestroy(stream));
    }

    std::puts(capture ? "raw-capture: 100 pointwise/two-stage replay and rebinding lifecycles passed" :
                       "raw: 100 compile/load/launch/unload lifecycles passed");
}

static float decode_scalar(const unsigned char * data, ggml_type type) {
    if (type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    uint16_t value;
    std::memcpy(&value, data, sizeof(value));
    return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(value) : ggml_bf16_to_fp32({value});
}

static float round_scalar(float value, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
        default:
            return value;
    }
}

static void mixed_silu(ggml_backend_t backend) {
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        for (bool silu : {false, true}) {
            if (!silu && type == GGML_TYPE_F32) {
                continue;
            }

            auto * ctx = ggml_init({1 << 20, nullptr, true});
            REQUIRE(ctx);

            auto * input = ggml_new_tensor_1d(ctx, type, 257);
            auto * rhs = ggml_new_tensor_1d(ctx, silu ? type : GGML_TYPE_F32, 257);
            auto * value = ggml_add(ctx, input, rhs);
            auto * out = ggml_neg(ctx, ggml_sqr(ctx, silu ? ggml_silu(ctx, value) : value));

            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);

            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            REQUIRE(buffer);

            const float probes[] = {0.0f, -0.0f, 1.0009765625f, 1.0078125f, -1.0f, -4.0f, 4.0f, 0x1p-24f};
            std::array<float, 257> inputs, addends;
            for (size_t i = 0; i < inputs.size(); ++i) {
                inputs[i] = probes[i % 8];
                addends[i] = type == GGML_TYPE_F16 ? 0x1p-11f : 0x1p-8f;
            }

            auto set = [](ggml_tensor * tensor, const std::array<float, 257> & values) {
                if (tensor->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
                } else if (tensor->type == GGML_TYPE_F16) {
                    std::array<ggml_fp16_t, 257> storage;
                    ggml_fp32_to_fp16_row(values.data(), storage.data(), values.size());
                    ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size() * sizeof(ggml_fp16_t));
                } else {
                    std::array<ggml_bf16_t, 257> storage;
                    ggml_fp32_to_bf16_row_ref(values.data(), storage.data(), values.size());
                    ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size() * sizeof(ggml_bf16_t));
                }
            };

            set(input, inputs);
            set(rhs, addends);

            for (int i = 0; i < graph->n_nodes; ++i) {
                auto view = ggml_graph_view(graph, i, i + 1);
                REQUIRE(ggml_backend_graph_compute(backend, &view) == GGML_STATUS_SUCCESS);
            }

            std::array<unsigned char, 257 * sizeof(float)> ordinary, fused;
            ggml_backend_tensor_get(out, ordinary.data(), 0, ggml_nbytes(out));

            auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
            REQUIRE(ggml_cuda_rtc_fusion_try(cuda_ctx, graph, 0, true) == graph->n_nodes - 1);
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(out, fused.data(), 0, ggml_nbytes(out));

            const size_t size = ggml_type_size(out->type);
            double error = 0, magnitude = 0;
            bool counterexample = false;

            for (size_t i = 0; i < inputs.size(); ++i) {
                const float a = decode_scalar(ordinary.data() + i * size, out->type);
                const float b = decode_scalar(fused.data() + i * size, out->type);
                REQUIRE(std::isfinite(a) && std::isfinite(b));
                error += (double(a) - b) * (double(a) - b);
                magnitude += double(a) * a;

                if (!silu) {
                    REQUIRE(std::memcmp(ordinary.data() + i * size, fused.data() + i * size, size) == 0);
                    const float sum = round_scalar(inputs[i], input->type) + round_scalar(addends[i], rhs->type);
                    counterexample |= b != round_scalar(-(sum * sum), out->type);
                }
            }

            REQUIRE(error / magnitude <= 1e-7);
            REQUIRE(silu || counterexample);

            std::printf("typed %s %s: nmse=%g%s\n", ggml_type_name(type), silu ? "SiLU" : "F32 RHS",
                        error / magnitude, silu ? "" : " finite unrounded counterexample verified");

            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
}

static void test_scale_contraction(ggml_backend_t backend) {
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
        auto * ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * input = ggml_new_tensor_1d(ctx, type, 257);
        auto * affine = ggml_scale_bias(ctx, input, 0x1.000002p0f, -1.5f);
        auto * out = ggml_neg(ctx, affine);

        auto * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        std::vector<float> values(257, 1.5f);
        if (type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        } else {
            std::vector<ggml_bf16_t> storage(values.size());
            ggml_fp32_to_bf16_row_ref(values.data(), storage.data(), values.size());
            ggml_backend_tensor_set(input, storage.data(), 0, storage.size() * sizeof(ggml_bf16_t));
        }

        for (int i = 0; i < g->n_nodes; ++i) {
            auto view = ggml_graph_view(g, i, i + 1);
            REQUIRE(ggml_backend_graph_compute(backend, &view) == GGML_STATUS_SUCCESS);
        }

        const size_t bytes = ggml_nbytes(out);
        std::vector<unsigned char> ordinary(bytes), fused(bytes);
        ggml_backend_tensor_get(out, ordinary.data(), 0, bytes);

        REQUIRE(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, fused.data(), 0, bytes);

        std::printf("SCALE cancellation %s: AOT=", ggml_type_name(type));
        for (size_t i = 0; i < ggml_type_size(type); ++i) {
            std::printf("%02x", ordinary[i]);
        }
        std::printf(" RTC=");
        for (size_t i = 0; i < ggml_type_size(type); ++i) {
            std::printf("%02x", fused[i]);
        }
        std::puts("");
        REQUIRE(ordinary == fused);

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

static void test_typed_boundaries(ggml_backend_t backend) {
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        auto * ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * input = ggml_new_tensor_1d(ctx, type, 257);
        auto * rhs = ggml_new_tensor_1d(ctx, type, 257);
        auto * rounded = ggml_add(ctx, input, rhs);
        auto * out = ggml_neg(ctx, ggml_sqr(ctx, rounded));

        auto * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        const float probes[] = {0.0f, -0.0f, 1.0f, 1.0009765625f, 1.0078125f,
                                0x1p-24f, 0x1p-126f, 0x1p-133f, 65504.0f, 0x1p127f,
                                INFINITY, -INFINITY, NAN};
        std::vector<float> values(257);

        auto set = [&](ggml_tensor * tensor, bool addend) {
            for (size_t i = 0; i < values.size(); ++i) {
                const size_t probe = i % (sizeof(probes) / sizeof(probes[0]));
                if (addend && probe >= 2) {
                    values[i] = i % 2 ? 0x1p-11f : 0x1p-8f;
                } else {
                    values[i] = probes[probe];
                }
            }

            if (type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
            } else if (type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> storage(values.size());
                ggml_fp32_to_fp16_row(values.data(), storage.data(), values.size());
                ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size() * sizeof(ggml_fp16_t));
            } else {
                std::vector<ggml_bf16_t> storage(values.size());
                ggml_fp32_to_bf16_row_ref(values.data(), storage.data(), values.size());
                ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size() * sizeof(ggml_bf16_t));
            }
        };

        set(input, false);
        set(rhs, true);

        for (int i = 0; i < g->n_nodes; ++i) {
            auto view = ggml_graph_view(g, i, i + 1);
            REQUIRE(ggml_backend_graph_compute(backend, &view) == GGML_STATUS_SUCCESS);
        }

        const size_t bytes = ggml_nbytes(out);
        std::vector<unsigned char> ordinary(bytes), fused(bytes);
        ggml_backend_tensor_get(out, ordinary.data(), 0, bytes);

        REQUIRE(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, fused.data(), 0, bytes);

        auto decode = [&](const unsigned char * data) {
            return decode_scalar(data, type);
        };

        const size_t size = ggml_type_size(type);
        for (size_t i = 0; i < values.size(); ++i) {
            const float a = decode(ordinary.data() + i * size);
            const float b = decode(fused.data() + i * size);
            REQUIRE((std::isnan(a) && std::isnan(b)) ||
                    std::memcmp(ordinary.data() + i * size, fused.data() + i * size, size) == 0);
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);

        std::printf("typed boundaries %s: finite ties, underflow, overflow and nonfinite classification passed\n", ggml_type_name(type));
    }
}

static void types() {
    auto backend = ggml_backend_cuda_init(0);
    REQUIRE(backend);

    test_scale_contraction(backend);
    test_typed_boundaries(backend);
    mixed_silu(backend);

    ggml_backend_free(backend);

    std::puts("types: SCALE contraction matches ordinary per-node execution");
}

static void graph() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    REQUIRE(backend);

    for (int n : {513, 257, 513, 1, 255, 256}) {
        auto * ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        auto * u = ggml_add(ctx, a, b);
        auto * v = ggml_mul(ctx, u, c);
        auto * affine = ggml_scale_bias(ctx, v, 0.5f, -1.0f);
        auto * relu = ggml_relu(ctx, affine);
        auto * out = ggml_sqr(ctx, relu);

        auto * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        const float av[] = {-4, -1, 2, 4};
        const float bv[] = {2, 3, -2, 0};
        const float replacement_b[] = {4, 3, 0, 0};
        std::vector<float> data(n), result(n), sentinel(n, 777.0f);

        for (int pass = 0; pass < 3; ++pass) {
            for (int i = 0; i < n; ++i) {
                data[i] = av[i % 4];
            }
            ggml_backend_tensor_set(a, data.data(), 0, n * sizeof(float));

            for (int i = 0; i < n; ++i) {
                data[i] = (pass == 0 ? bv : replacement_b)[i % 4];
            }
            ggml_backend_tensor_set(b, data.data(), 0, n * sizeof(float));

            std::fill(data.begin(), data.end(), 2.0f);
            ggml_backend_tensor_set(c, data.data(), 0, n * sizeof(float));

            const float bias = pass == 2 ? 0.0f : -1.0f;
            std::memcpy(reinterpret_cast<char *>(affine->op_params) + sizeof(float), &bias, sizeof(bias));

            for (auto * t : {u, v, affine, relu}) {
                ggml_backend_tensor_set(t, sentinel.data(), 0, n * sizeof(float));
            }

            REQUIRE(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, result.data(), 0, n * sizeof(float));
            const float expected[][4] = {{0, 1, 0, 9}, {0, 1, 1, 9}, {0, 4, 4, 16}};
            for (int i = 0; i < n; ++i) {
                REQUIRE(result[i] == expected[pass][i % 4]);
            }

            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
            REQUIRE(cuda_ctx->rtc_fusion);
            for (auto * t : {u, v, affine, relu}) {
                ggml_backend_tensor_get(t, result.data(), 0, n * sizeof(float));
                for (float value : result) {
                    REQUIRE(value == 777.0f);
                }
            }
        }

        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    std::puts("graph: rebinding, typed stores, tails and untouched intermediates passed");
}

static void check_pattern(const ggml_tensor * tensor, const float * expected, int n) {
    std::vector<float> values(n);
    ggml_backend_tensor_get(tensor, values.data(), 0, n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        REQUIRE(values[i] == expected[i % 4]);
    }
}

struct schedule_observer {
    ggml_tensor * tensor;
    int calls = 0;
};

static bool observe_u(ggml_tensor * tensor, bool ask, void * data) {
    auto & observer = *static_cast<schedule_observer *>(data);
    if (ask) {
        return tensor == observer.tensor;
    }

    REQUIRE(tensor == observer.tensor);

    const float expected[] = {-2, 2, 0, 4};
    check_pattern(tensor, expected, 513);

    ++observer.calls;
    return true;
}

static void schedule() {
    auto backend = ggml_backend_cuda_init(0);
    auto cpu = ggml_backend_cpu_init();
    REQUIRE(backend && cpu);
    ggml_backend_t backends[] = {backend, cpu};

    for (int mode = 0; mode < 5; ++mode) {
        auto * ctx = ggml_init({4 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
        auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
        auto * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
        for (auto * input : {a, b, c}) {
            ggml_set_input(input);
        }

        auto * u = ggml_add(ctx, a, b);
        auto * v = ggml_mul(ctx, u, c);
        auto * t = ggml_sqr(ctx, ggml_neg(ctx, u));
        auto * out = mode == 0 ? ggml_sqr(ctx, ggml_relu(ctx, ggml_scale_bias(ctx, v, 0.5f, -1.0f))) : ggml_add(ctx, v, t);
        ggml_set_output(out);
        if (mode == 2) {
            ggml_set_output(v);
            ggml_set_output(t);
        }

        auto * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
        REQUIRE(sched);

        for (auto * input : {a, b, c}) {
            ggml_backend_sched_set_tensor_backend(sched, input, backend);
        }
        for (int i = 0; i < g->n_nodes; ++i) {
            ggml_backend_sched_set_tensor_backend(sched, g->nodes[i], mode == 4 && g->nodes[i] == u ? cpu : backend);
        }

        schedule_observer observer {u};
        if (mode == 3) {
            ggml_backend_sched_set_eval_callback(sched, observe_u, &observer);
        }

        REQUIRE(ggml_backend_sched_alloc_graph(sched, g));

        std::vector<float> data(513);
        const float av[] = {-4, -1, 2, 4};
        const float bv[] = {2, 3, -2, 0};
        for (int i = 0; i < 513; ++i) {
            data[i] = av[i % 4];
        }
        ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));

        for (int i = 0; i < 513; ++i) {
            data[i] = bv[i % 4];
        }
        ggml_backend_tensor_set(b, data.data(), 0, data.size() * sizeof(float));

        std::fill(data.begin(), data.end(), 2.0f);
        ggml_backend_tensor_set(c, data.data(), 0, data.size() * sizeof(float));

        REQUIRE(ggml_backend_sched_graph_compute(sched, g) == GGML_STATUS_SUCCESS);

        const float basic[] = {0, 1, 0, 9};
        const float diamond[] = {0, 8, 0, 24};
        check_pattern(out, mode == 0 ? basic : diamond, 513);

        if (mode == 2) {
            const float expected_v[] = {-4, 4, 0, 8};
            const float expected_t[] = {4, 4, 0, 16};
            check_pattern(v, expected_v, 513);
            check_pattern(t, expected_t, 513);
            REQUIRE(v->data != t->data && out->data != v->data && out->data != t->data);
        }

        if (mode == 3) {
            REQUIRE(observer.calls == 1);
        }

        std::printf("schedule mode=%d u=%p v=%p t=%p out=%p: passed\n", mode, u->data, v->data, t->data, out->data);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(backend);
}

static void test_output_input_aliases(ggml_backend_t backend) {
    for (int offset : {0, 1}) {
        auto * ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 515);
        auto * a = ggml_view_1d(ctx, backing, 513, 0);
        auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
        auto * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
        auto * u = ggml_add(ctx, a, b);
        auto * out = ggml_mul(ctx, u, c);

        auto * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        std::vector<float> original(515), values(513, 2.0f);
        for (int i = 0; i < 515; ++i) {
            original[i] = float(i % 17);
        }
        ggml_backend_tensor_set(backing, original.data(), 0, 515 * sizeof(float));
        ggml_backend_tensor_set(b, values.data(), 0, 513 * sizeof(float));
        ggml_backend_tensor_set(c, values.data(), 0, 513 * sizeof(float));

        out->buffer = backing->buffer;
        out->data = static_cast<float *>(backing->data) + offset;
        REQUIRE(out->view_src == nullptr);

        auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
        const int skipped = ggml_cuda_rtc_fusion_try(cuda_ctx, g, 1, true);
        REQUIRE(skipped == (offset == 0 ? 1 : 0));
        if (!skipped) {
            REQUIRE(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        }

        ggml_backend_synchronize(backend);

        std::vector<float> result(515);
        ggml_backend_tensor_get(backing, result.data(), 0, 515 * sizeof(float));
        for (int i = 0; i < 515; ++i) {
            const float expected = i >= offset && i < 513 + offset ? (original[i - offset] + 2.0f) * 2.0f : original[i];
            REQUIRE(result[i] == expected);
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

static void test_indexed_aliases(ggml_backend_t backend) {
    for (int mapping : {0, 1, 2}) {
        auto * ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1028);
        auto * a = ggml_view_1d(ctx, backing, 512, 0);
        auto * b = ggml_view_1d(ctx, backing, mapping == 2 ? 256 : 512, sizeof(float));
        if (mapping == 1) {
            b->nb[0] = 2 * sizeof(float);
        }

        auto * u = ggml_add(ctx, a, b);
        auto * out = ggml_sqr(ctx, u);

        auto * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        std::vector<float> original(1028);
        for (int i = 0; i < 1028; ++i) {
            original[i] = float(i % 17);
        }
        ggml_backend_tensor_set(backing, original.data(), 0, original.size() * sizeof(float));

        if (mapping) {
            out->buffer = backing->buffer;
            out->data = b->data;
        }

        int start = 0;
        while (g->nodes[start] != u) {
            ++start;
        }

        auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
        const int skipped = ggml_cuda_rtc_fusion_try(cuda_ctx, g, start, true);
        REQUIRE(skipped == (mapping == 0 ? 1 : 0));
        if (!skipped) {
            REQUIRE(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        }

        ggml_backend_synchronize(backend);

        std::vector<float> result(512);
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        for (int i = 0; i < 512; ++i) {
            int rhs = i;
            if (mapping == 1) {
                rhs = i * 2;
            } else if (mapping == 2) {
                rhs = i % 256;
            }

            const float sum = original[i] + original[1 + rhs];
            REQUIRE(result[i] == sum * sum);
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

static void test_output_output_aliases(ggml_backend_t backend) {
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    REQUIRE(ctx);

    auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
    auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
    auto * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
    auto * u = ggml_add(ctx, a, b);
    auto * v = ggml_mul(ctx, u, c);
    auto * t = ggml_neg(ctx, u);
    auto * out = ggml_add(ctx, v, t);
    ggml_set_output(v);
    ggml_set_output(t);

    auto * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, v);
    ggml_build_forward_expand(g, out);

    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer);

    t->buffer = v->buffer;
    t->data = v->data;
    REQUIRE(t->view_src == nullptr);

    std::vector<float> values(513, 2.0f);
    std::vector<float> ordinary(513);
    std::vector<float> fused(513);
    std::vector<float> shared(513);
    ggml_backend_tensor_set(a, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(b, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(c, values.data(), 0, values.size() * sizeof(float));

    for (int i = 0; i < g->n_nodes; ++i) {
        ggml_cgraph node = *g;
        node.nodes = g->nodes + i;
        node.n_nodes = 1;
        REQUIRE(ggml_backend_graph_compute(backend, &node) == GGML_STATUS_SUCCESS);
    }

    ggml_backend_tensor_get(out, ordinary.data(), 0, ordinary.size() * sizeof(float));
    for (float value : ordinary) {
        REQUIRE(value == -8.0f);
    }

    REQUIRE(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, fused.data(), 0, fused.size() * sizeof(float));
    ggml_backend_tensor_get(v, shared.data(), 0, shared.size() * sizeof(float));
    for (size_t i = 0; i < fused.size(); ++i) {
        REQUIRE(shared[i] == -4.0f);
        REQUIRE(fused[i] == ordinary[i]);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void alias() {
    auto backend = ggml_backend_cuda_init(0);
    REQUIRE(backend);

    test_output_input_aliases(backend);
    test_indexed_aliases(backend);
    test_output_output_aliases(backend);

    ggml_backend_free(backend);

    std::puts("alias: exact mapping and read-only overlaps admitted; shifted, indexed and repeated overlaps declined");
}

static void reductions() {
    {
        auto backend = ggml_backend_cuda_init(0);
        REQUIRE(backend);

        for (ggml_op op : {GGML_OP_SUM, GGML_OP_SUM_ROWS, GGML_OP_MEAN}) {
            auto * ctx = ggml_init({1 << 20, nullptr, true});
            REQUIRE(ctx);

            const int64_t columns = op == GGML_OP_SUM ? 513 : 257;
            const int64_t rows = op == GGML_OP_SUM ? 1 : 4;

            auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
            auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
            auto * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
            auto * producer = ggml_mul(ctx, ggml_add(ctx, a, b), c);
            ggml_tensor * out;
            switch (op) {
                case GGML_OP_SUM:
                    out = ggml_sum(ctx, producer);
                    break;
                case GGML_OP_MEAN:
                    out = ggml_mean(ctx, producer);
                    break;
                default:
                    out = ggml_sum_rows(ctx, producer);
                    break;
            }

            auto * g = ggml_new_graph(ctx);
            ggml_build_forward_expand(g, out);

            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            REQUIRE(buffer);

            const float av[] = {-4, -1, 2, 4}, bv[] = {2, 3, -2, 0};
            std::vector<float> values(size_t(columns * rows));
            for (auto * input : {a, b, c}) {
                for (size_t i = 0; i < values.size(); ++i) {
                    if (input == a) {
                        values[i] = av[i % 4];
                    } else if (input == b) {
                        values[i] = bv[i % 4];
                    } else {
                        values[i] = 2.0f;
                    }
                }
                ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
            }

            auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
            REQUIRE(ggml_cuda_rtc_fusion_try(cuda_ctx, g, 0, true) == 2);
            ggml_backend_synchronize(backend);

            float result[4] = {};
            ggml_backend_tensor_get(out, result, 0, size_t(rows) * sizeof(float));
            const float expected[] = {508, 516, 512, 520};
            for (int64_t row = 0; row < rows; ++row) {
                float reference = expected[row];
                if (op == GGML_OP_SUM) {
                    reference = 1020.0f;
                } else if (op == GGML_OP_MEAN) {
                    reference /= 257.0f;
                }

                REQUIRE(result[row] == reference);
            }

            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }

        ggml_backend_free(backend);
    }

    for (int iteration = 0; iteration < 100; ++iteration) {
        auto backend = ggml_backend_cuda_init(0);
        REQUIRE(backend);
        auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);

        for (ggml_op op : {GGML_OP_SUM_ROWS, GGML_OP_MEAN, GGML_OP_SUM}) {
            for (int64_t columns : {1, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097, 8193}) {
                if ((iteration && (op != GGML_OP_SUM || columns != 4097)) ||
                    (op != GGML_OP_SUM && columns > 4096)) {
                    continue;
                }

                auto * ctx = ggml_init({1 << 20, nullptr, true});
                REQUIRE(ctx);

                const int64_t rows = op == GGML_OP_SUM ? 1 : 3;
                auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
                auto * producer = ggml_neg(ctx, ggml_scale(ctx, input, 2.0f));
                ggml_tensor * reduced;
                switch (op) {
                    case GGML_OP_SUM:
                        reduced = ggml_sum(ctx, producer);
                        break;
                    case GGML_OP_MEAN:
                        reduced = ggml_mean(ctx, producer);
                        break;
                    default:
                        reduced = ggml_sum_rows(ctx, producer);
                        break;
                }

                ggml_set_output(reduced);
                auto * out = ggml_neg(ctx, reduced);
                auto * g = ggml_new_graph(ctx);
                ggml_build_forward_expand(g, out);

                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                REQUIRE(buffer);

                std::vector<float> values(size_t(columns * rows));
                for (size_t i = 0; i < values.size(); ++i) {
                    const float probes[] = {1e6f, 0.001f, -1e6f, 3.0f, -0.125f, 0x1p-20f, 0.5f};
                    values[i] = probes[i % 7];
                }
                ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));

                for (int node = 0; node < 2; ++node) {
                    auto view = ggml_graph_view(g, node, node + 1);
                    REQUIRE(ggml_backend_graph_compute(backend, &view) == GGML_STATUS_SUCCESS);
                }

                std::vector<float> ordinary_values(values.size());
                ggml_backend_tensor_get(producer, ordinary_values.data(), 0, ordinary_values.size() * sizeof(float));

                REQUIRE(ggml_cuda_rtc_fusion_try(cuda_ctx, g, 0, true) == g->n_nodes - 1);
                ggml_backend_synchronize(backend);

                std::vector<float> result(size_t(rows), 0.0f), intermediate(size_t(rows), 0.0f);
                ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
                ggml_backend_tensor_get(reduced, intermediate.data(), 0, intermediate.size() * sizeof(float));

                double largest_error = 0, largest_relative = 0;
                for (int64_t row = 0; row < rows; ++row) {
                    double reference = 0, magnitude = 0;
                    for (int64_t column = 0; column < columns; ++column) {
                        const float value = -ordinary_values[size_t(row * columns + column)];
                        reference += value;
                        magnitude += std::fabs(value);
                    }

                    const double divisor = op == GGML_OP_MEAN ? double(columns) : 1.0;
                    reference /= divisor;
                    magnitude /= divisor;

                    const int threads = columns <= 256 ? ggml_cuda_info().devices[0].warp_size : 256;
                    const int steps = columns > 4096 ? 16 + 8 + int((columns + 4095) / 4096) + 8 :
                        int((columns + threads - 1) / threads) + int(std::log2(threads));
                    const double u = 0x1p-24;
                    const double gamma = (steps + 2) * u / (1.0 - (steps + 2) * u);

                    const double error = std::fabs(double(result[size_t(row)]) - reference);
                    REQUIRE(error <= gamma * magnitude);
                    REQUIRE(result[size_t(row)] == -intermediate[size_t(row)]);

                    largest_error = std::max(largest_error, error);
                    largest_relative = std::max(largest_relative, error / std::max(1.0, std::fabs(reference)));
                }

                if (!iteration) {
                    std::printf("reduction op=%s columns=%lld rows=%lld abs=%g rel=%g\n",
                                ggml_op_name(op), (long long) columns, (long long) rows, largest_error, largest_relative);
                }

                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
        ggml_backend_free(backend);
    }

    std::puts("reductions: native wave tree, tails, exposed reduced output and 100 two-stage backend lifecycles passed");
}

#ifdef __linux__
static void observer_snapshot(uint64_t (&values)[HIPRTC_TEST_COUNTER_COUNT]) {
    static auto snapshot = reinterpret_cast<decltype(&hiprtc_test_snapshot)>(dlsym(RTLD_DEFAULT, "hiprtc_test_snapshot"));
    REQUIRE(snapshot);
    snapshot(values);
}

struct cache_fixture {
    ggml_backend_t backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * input;
    ggml_tensor * affine;
    ggml_tensor * out;
    ggml_tensor * squared;
    ggml_cgraph * graphs[2];
    int64_t n;
    float bias;

    cache_fixture(int index) : n(257 + index * 32), bias(float(index + 1)) {
        backend = ggml_backend_cuda_init(0);
        REQUIRE(backend);

        ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        affine = ggml_scale_bias(ctx, input, 2.0f, bias);
        out = ggml_neg(ctx, affine);
        squared = ggml_sqr(ctx, out);

        graphs[0] = ggml_new_graph(ctx);
        graphs[1] = ggml_new_graph(ctx);
        ggml_build_forward_expand(graphs[0], out);
        ggml_build_forward_expand(graphs[1], squared);

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);
    }

    void run(int variant, bool failure, bool whole_graph = false) {
        const float probes[] = {-4, -1, 2, 4};
        std::vector<float> values(size_t(n), 0.0f);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = probes[i % 4];
        }

        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));

        auto * graph = graphs[variant];
        if (whole_graph) {
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        } else {
            auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
            const int skipped = ggml_cuda_rtc_fusion_try(cuda_ctx, graph, 0, true);
            REQUIRE(skipped == (failure ? 0 : graph->n_nodes - 1));

            if (failure) {
                for (int node = 0; node < graph->n_nodes; ++node) {
                    auto view = ggml_graph_view(graph, node, node + 1);
                    REQUIRE(ggml_backend_graph_compute(backend, &view) == GGML_STATUS_SUCCESS);
                }
            }
        }

        ggml_backend_synchronize(backend);

        ggml_backend_tensor_get(variant ? squared : out, values.data(), 0, values.size() * sizeof(float));

        for (size_t i = 0; i < values.size(); ++i) {
            const float value = -(probes[i % 4] * 2.0f + bias);
            REQUIRE(values[i] == (variant ? value * value : value));
        }
    }

    ~cache_fixture() {
        ggml_backend_free(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

struct capture_fixture {
    ggml_backend_t backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * a;
    ggml_tensor * b;
    ggml_tensor * c;
    ggml_tensor * alternate;
    ggml_tensor * backing;
    ggml_tensor * u;
    ggml_tensor * affine;
    ggml_tensor * out;
    ggml_cgraph * graph;
    bool diamond;
    bool changed_inputs = false;
    bool changed_address = false;
    bool changed_edge = false;
    int n = 513;
    float scale;
    float bias;

    capture_fixture(bool diamond) : diamond(diamond), scale(diamond ? 1.0f : 0.5f), bias(diamond ? 0.0f : -1.0f) {
        backend = ggml_backend_cuda_init(0);
        REQUIRE(backend);

        ctx = ggml_init({4 << 20, nullptr, true});
        REQUIRE(ctx);

        a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * n);
        b = ggml_view_1d(ctx, backing, n, 0);
        c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        alternate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        for (auto * input : {a, backing, c}) {
            ggml_set_input(input);
        }

        u = ggml_add(ctx, a, b);
        auto * v = ggml_mul(ctx, u, c);
        affine = ggml_scale_bias(ctx, v, scale, bias);
        out = diamond ? ggml_add(ctx, affine, ggml_sqr(ctx, ggml_neg(ctx, u))) : ggml_sqr(ctx, ggml_relu(ctx, affine));
        ggml_set_output(out);
        if (diamond) {
            ggml_set_output(u);
        }

        graph = ggml_new_graph(ctx);
        rebuild();
        graph->uid = 1;

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        upload();
    }

    void rebuild() {
        ggml_graph_clear(graph);
        ggml_build_forward_expand(graph, b);
        ggml_build_forward_expand(graph, out);
    }

    float a_value(int i) const {
        static const float original[] = {-4, -1, 2, 4};
        static const float replacement[] = {-2, 1, 4, 6};
        return (changed_address ? replacement : original)[i % 4];

    }

    float b_value(int i) const {
        static const float original[] = {2, 3, -2, 0};
        static const float replacement[] = {4, 3, 0, 0};
        return (changed_inputs ? replacement : original)[i % 4];

    }

    void upload() {
        std::vector<float> values(2 * 513);
        for (int i = 0; i < 2 * 513; ++i) {
            values[i] = b_value(i);
        }
        ggml_backend_tensor_set(backing, values.data(), 0, values.size() * sizeof(float));

        for (int i = 0; i < n; ++i) {
            values[i] = a_value(i);
        }
        ggml_backend_tensor_set(a, values.data(), 0, n * sizeof(float));

        std::fill(values.begin(), values.end(), 2.0f);
        ggml_backend_tensor_set(c, values.data(), 0, n * sizeof(float));
    }

    void check() {
        std::vector<float> values(n);
        ggml_backend_tensor_get(out, values.data(), 0, n * sizeof(float));

        for (int i = 0; i < n; ++i) {
            float value = a_value(i) + (changed_edge ? 2.0f : b_value(i * int(b->nb[0] / sizeof(float))));
            const float transformed = (value * 2.0f) * scale + bias;
            const float positive = std::max(transformed, 0.0f);
            REQUIRE(values[i] == (diamond ? transformed + value * value : positive * positive));
        }

        if (u->flags & GGML_TENSOR_FLAG_OUTPUT) {
            ggml_backend_tensor_get(u, values.data(), 0, n * sizeof(float));
            for (int i = 0; i < n; ++i) {
                REQUIRE(values[i] == a_value(i) + (changed_edge ? 2.0f : b_value(i * int(b->nb[0] / sizeof(float)))));
            }
        }
    }

    void run() {
        REQUIRE(graph->uid == 1 && graph->nodes[0] == b);
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

        check();
    }

    void recapture(bool same_program) {
        upload();

        uint64_t before[HIPRTC_TEST_COUNTER_COUNT], direct[HIPRTC_TEST_COUNTER_COUNT];
        uint64_t captured[HIPRTC_TEST_COUNTER_COUNT], replayed[HIPRTC_TEST_COUNTER_COUNT];

        observer_snapshot(before);
        run();
        observer_snapshot(direct);

        REQUIRE(direct[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN]);
        REQUIRE(direct[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH]);
        REQUIRE(direct[HIPRTC_TEST_MODULE_LAUNCH] > before[HIPRTC_TEST_MODULE_LAUNCH]);

        if (same_program) {
            REQUIRE(direct[HIPRTC_TEST_COMPILE] == before[HIPRTC_TEST_COMPILE]);
        } else {
            REQUIRE(direct[HIPRTC_TEST_COMPILE] == before[HIPRTC_TEST_COMPILE] + 1);
        }

        run();
        observer_snapshot(captured);

        REQUIRE(captured[HIPRTC_TEST_CAPTURE_BEGIN] == direct[HIPRTC_TEST_CAPTURE_BEGIN] + 1);
        REQUIRE(captured[HIPRTC_TEST_CAPTURE_END] == direct[HIPRTC_TEST_CAPTURE_END] + 1);
        REQUIRE(captured[HIPRTC_TEST_GRAPH_LAUNCH] == direct[HIPRTC_TEST_GRAPH_LAUNCH] + 1);

        run();
        observer_snapshot(replayed);
        REQUIRE(replayed[HIPRTC_TEST_GRAPH_LAUNCH] == captured[HIPRTC_TEST_GRAPH_LAUNCH] + 1);

        for (auto counter : {HIPRTC_TEST_CREATE, HIPRTC_TEST_COMPILE, HIPRTC_TEST_LOAD, HIPRTC_TEST_UNLOAD,
                             HIPRTC_TEST_MODULE_LAUNCH, HIPRTC_TEST_CAPTURE_BEGIN, HIPRTC_TEST_CAPTURE_END,
                             HIPRTC_TEST_RESOURCE_QUERY, HIPRTC_TEST_ALLOCATE, HIPRTC_TEST_VERSION}) {
            REQUIRE(captured[counter] == direct[counter] + (counter == HIPRTC_TEST_MODULE_LAUNCH ||
                    counter == HIPRTC_TEST_CAPTURE_BEGIN || counter == HIPRTC_TEST_CAPTURE_END ? 1 : 0));
            REQUIRE(replayed[counter] == captured[counter]);
        }

        REQUIRE(replayed[HIPRTC_TEST_SETUP_DURING_CAPTURE] == 0);
    }

    void resize(int count) {
        n = count;
        for (auto * tensor : {a, c}) {
            tensor->ne[0] = n;
            tensor->nb[1] = tensor->nb[2] = tensor->nb[3] = n * sizeof(float);
        }

        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * tensor = graph->nodes[i];
            tensor->ne[0] = n;
            tensor->nb[1] = tensor->nb[2] = tensor->nb[3] = n * tensor->nb[0];
        }

        upload();
    }

    ~capture_fixture() {
        ggml_backend_free(backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

static void capture_reductions() {
    for (int iteration = 0; iteration < 100; ++iteration) {
        const bool full = iteration % 2 != 0;
        const int columns = full ? 8193 : 257;
        const int rows = full ? 1 : 4;
        const int stages = full ? 2 : 1;

        auto backend = ggml_backend_cuda_init(0);
        REQUIRE(backend);

        auto * ctx = ggml_init({2 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
        auto * alternate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
        ggml_set_input(input);
        auto * affine = ggml_scale_bias(ctx, input, 2.0f, 1.0f);
        auto * producer = ggml_neg(ctx, affine);
        auto * reduced = full ? ggml_sum(ctx, producer) : ggml_sum_rows(ctx, producer);
        auto * out = ggml_scale_bias(ctx, reduced, 0.5f, -1.0f);
        ggml_set_output(out);

        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        graph->uid = 1;

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        std::vector<float> values(columns * rows);
        float bias = 1.0f;

        auto upload = [&](int shift) {
            const float probes[] = {-4, 4, 0, 8};
            for (int i = 0; i < columns * rows; ++i) {
                values[i] = probes[(i + shift) % 4];
            }
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        };

        auto check = [&] {
            std::vector<float> result(rows);
            ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

            for (int row = 0; row < rows; ++row) {
                double sum = 0;
                for (int column = 0; column < columns; ++column) {
                    sum -= 2.0 * values[row * columns + column] + bias;
                }

                REQUIRE(result[row] == float(sum * 0.5 - 1.0));
            }
        };

        auto run = [&] {
            REQUIRE(graph->uid == 1);
            REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            check();
        };

        auto warm = [&] {
            uint64_t before[HIPRTC_TEST_COUNTER_COUNT], direct[HIPRTC_TEST_COUNTER_COUNT];
            uint64_t captured[HIPRTC_TEST_COUNTER_COUNT], replayed[HIPRTC_TEST_COUNTER_COUNT];

            observer_snapshot(before);
            run();
            observer_snapshot(direct);

            REQUIRE(direct[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN]);
            REQUIRE(direct[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH]);
            REQUIRE(direct[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH] + stages);

            run();
            observer_snapshot(captured);

            REQUIRE(captured[HIPRTC_TEST_CAPTURE_BEGIN] == direct[HIPRTC_TEST_CAPTURE_BEGIN] + 1);
            REQUIRE(captured[HIPRTC_TEST_CAPTURE_END] == direct[HIPRTC_TEST_CAPTURE_END] + 1);
            REQUIRE(captured[HIPRTC_TEST_GRAPH_LAUNCH] == direct[HIPRTC_TEST_GRAPH_LAUNCH] + 1);
            REQUIRE(captured[HIPRTC_TEST_MODULE_LAUNCH] == direct[HIPRTC_TEST_MODULE_LAUNCH] + stages);

            run();
            observer_snapshot(replayed);

            REQUIRE(replayed[HIPRTC_TEST_GRAPH_LAUNCH] == captured[HIPRTC_TEST_GRAPH_LAUNCH] + 1);
            REQUIRE(replayed[HIPRTC_TEST_MODULE_LAUNCH] == captured[HIPRTC_TEST_MODULE_LAUNCH]);
            REQUIRE(replayed[HIPRTC_TEST_CAPTURE_BEGIN] == captured[HIPRTC_TEST_CAPTURE_BEGIN]);
            REQUIRE(replayed[HIPRTC_TEST_CAPTURE_END] == captured[HIPRTC_TEST_CAPTURE_END]);

            for (auto counter : {HIPRTC_TEST_CREATE, HIPRTC_TEST_COMPILE, HIPRTC_TEST_LOAD, HIPRTC_TEST_UNLOAD,
                                 HIPRTC_TEST_RESOURCE_QUERY, HIPRTC_TEST_ALLOCATE, HIPRTC_TEST_VERSION}) {
                REQUIRE(captured[counter] == direct[counter]);
                REQUIRE(replayed[counter] == captured[counter]);
            }

            REQUIRE(replayed[HIPRTC_TEST_SETUP_DURING_CAPTURE] == 0);
        };

        upload(0);
        warm();

        upload(1);

        uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];
        observer_snapshot(before);
        run();
        observer_snapshot(after);

        REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 1);
        REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN]);
        REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH]);

        input->data = alternate->data;
        bias = 2.0f;
        const float parameters[] = {2.0f, bias};
        ggml_set_op_params(affine, parameters, sizeof(parameters));
        upload(2);

        observer_snapshot(before);
        warm();
        observer_snapshot(after);
        REQUIRE(after[HIPRTC_TEST_COMPILE] == before[HIPRTC_TEST_COMPILE]);

        observer_snapshot(before);
        for (int replay = 0; replay < 3; ++replay) {
            REQUIRE(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
        }

        observer_snapshot(after);
        REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 3);
        REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH]);
        REQUIRE(after[HIPRTC_TEST_UNLOAD] == before[HIPRTC_TEST_UNLOAD]);

        ggml_backend_free(backend);
        check();
        observer_snapshot(after);
        REQUIRE(after[HIPRTC_TEST_UNLOAD] == before[HIPRTC_TEST_UNLOAD] + 1);

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    std::puts("capture: row and two-stage reduction replay, rebinding and 100 queued teardown lifecycles passed");
}

struct capture_limit_graph {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_cgraph * graph;
    ggml_tensor * input;
    ggml_tensor * out;
    int nodes;
    float value;

    capture_limit_graph(ggml_backend_t backend, int nodes, int index) : nodes(nodes), value(float(index % 4)) {
        ctx = ggml_init({8 << 20, nullptr, true});
        REQUIRE(ctx);

        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_input(input);

        out = input;
        for (int i = 0; i < nodes; ++i) {
            out = ggml_scale_bias(ctx, out, 1.0f, 1.0f);
        }
        ggml_set_output(out);

        graph = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(graph, out);
        graph->uid = 1;
        REQUIRE(graph->n_nodes == nodes);

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);
    }

    void run(ggml_backend_t backend) {
        ggml_backend_tensor_set(input, &value, 0, sizeof(value));
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

        float result;
        ggml_backend_tensor_get(out, &result, 0, sizeof(result));
        REQUIRE(result == value + float(nodes));
    }

    ~capture_limit_graph() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

static void capture_limits() {
    for (int nodes : {4096, 6144}) {
        auto backend = ggml_backend_cuda_init(0);
        REQUIRE(backend);

        std::vector<std::unique_ptr<capture_limit_graph>> fixtures;
        uint64_t start[HIPRTC_TEST_COUNTER_COUNT], before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];

        observer_snapshot(start);
        size_t captured = 0;
        bool refused = false;

        for (int index = 0; index < 65; ++index) {
            fixtures.emplace_back(new capture_limit_graph(backend, nodes, index));
            auto & fixture = *fixtures.back();

            observer_snapshot(before);
            fixture.run(backend);
            fixture.run(backend);
            fixture.run(backend);
            observer_snapshot(after);

            REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] > before[HIPRTC_TEST_MODULE_LAUNCH]);
            REQUIRE(after[HIPRTC_TEST_UNLOAD] == start[HIPRTC_TEST_UNLOAD]);
            REQUIRE(after[HIPRTC_TEST_SETUP_DURING_CAPTURE] == 0);

            const bool admitted = after[HIPRTC_TEST_CAPTURE_BEGIN] != before[HIPRTC_TEST_CAPTURE_BEGIN];
            if (admitted) {
                REQUIRE(!refused);
                REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN] + 1);
                REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 2);
                ++captured;
            } else {
                refused = true;
                REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH]);
            }

            for (size_t retained = 0; retained < captured; ++retained) {
                observer_snapshot(before);
                fixtures[retained]->run(backend);
                observer_snapshot(after);
                REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 1);
                REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN]);
                REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH]);
                REQUIRE(after[HIPRTC_TEST_UNLOAD] == start[HIPRTC_TEST_UNLOAD]);
            }
        }

        REQUIRE(refused);
        REQUIRE(captured > 0 && captured < fixtures.size());
        std::printf("capture limits: nodes=%d admitted=%zu refused=%zu live modules retained\n", nodes, captured, fixtures.size() - captured);

        ggml_backend_free(backend);
        observer_snapshot(after);
        REQUIRE(after[HIPRTC_TEST_UNLOAD] == start[HIPRTC_TEST_UNLOAD] + 1);
    }
}

static void capture_mixed() {
    auto backend = ggml_backend_cuda_init(0);
    REQUIRE(backend);
    auto * ctx = ggml_init({2 << 20, nullptr, true});
    REQUIRE(ctx);

    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
    auto * alternate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
    ggml_set_input(input);
    ggml_set_input(weight);
    auto * producer = ggml_neg(ctx, ggml_scale_bias(ctx, input, 2.0f, 1.0f));
    auto * mm = ggml_mul_mat(ctx, weight, producer);
    auto * out = ggml_neg(ctx, ggml_scale_bias(ctx, mm, 0.5f, 2.0f));
    ggml_set_output(out);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    graph->uid = 1;
    auto * other_out = ggml_sqr(ctx, alternate);
    ggml_set_output(other_out);
    auto * other_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(other_graph, other_out);
    other_graph->uid = 1;
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer);

    std::vector<float> inputs(128, 1.0f);
    std::vector<float> weights(256, 0.25f);
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(float));
    std::fill(weights.begin(), weights.end(), 0.5f);
    ggml_backend_tensor_set(alternate, weights.data(), 0, weights.size() * sizeof(float));

    auto run = [&](float expected) {
        REQUIRE(ggml_backend_graph_compute(backend, other_graph) == GGML_STATUS_SUCCESS);
        float other_result[256];
        ggml_backend_tensor_get(other_out, other_result, 0, sizeof(other_result));
        for (float value : other_result) {
            REQUIRE(value == 0.25f);
        }
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        float result[32];
        ggml_backend_tensor_get(out, result, 0, sizeof(result));
        for (float value : result) {
            REQUIRE(value == expected);
        }
    };

    for (int phase = 0; phase < 3; ++phase) {
        if (phase == 1) {
            std::fill(inputs.begin(), inputs.end(), 2.0f);
            ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
        } else if (phase == 2) {
            weight->data = alternate->data;
        }
        const float expected = phase == 0 ? 10.0f : phase == 1 ? 18.0f : 38.0f;
        uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];
        observer_snapshot(before);
        run(expected);
        if (phase != 1) {
            run(expected);
        }
        observer_snapshot(after);
        REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + (phase == 2 ? 3 : 2));
        REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN] + (phase == 0 ? 2 : phase == 1 ? 0 : 1));
        REQUIRE(after[HIPRTC_TEST_SETUP_DURING_CAPTURE] == 0);

        observer_snapshot(before);
        run(expected);
        observer_snapshot(after);
        REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 2);
        REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH]);
        REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN]);
    }

    ggml_backend_free(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::puts("capture: interleaved mixed graphs, input updates and same-UID weight rebinding passed");
}

static void capture() {
    capture_mixed();

    for (bool diamond : {false, true}) {

        capture_fixture fixture(diamond);
        fixture.recapture(false);

        uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];

        fixture.changed_inputs = true;
        fixture.upload();

        observer_snapshot(before);
        fixture.run();
        observer_snapshot(after);

        REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 1);
        REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN]);
        REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH]);

        fixture.changed_address = true;
        fixture.a->data = fixture.alternate->data;
        fixture.upload();
        fixture.recapture(true);

        fixture.resize(257);
        fixture.recapture(true);
        fixture.resize(513);
        fixture.recapture(true);

        fixture.bias += 1.0f;
        const float parameters[] = {fixture.scale, fixture.bias};
        ggml_set_op_params(fixture.affine, parameters, sizeof(parameters));
        fixture.recapture(true);

        fixture.b->nb[0] = 2 * sizeof(float);
        fixture.b->nb[1] = fixture.b->nb[2] = fixture.b->nb[3] = fixture.n * fixture.b->nb[0];
        fixture.recapture(false);

        if (!diamond) {
            ggml_set_output(fixture.u);
            fixture.recapture(false);
        }

        fixture.changed_edge = true;
        fixture.u->src[1] = fixture.c;
        fixture.rebuild();
        fixture.recapture(false);
    }

    capture_reductions();
    std::puts("capture: same-UID contents, address, tail, scalar, stride, output and source mutations passed");
}

static void concurrency() {
    uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];

    observer_snapshot(before);
    {
        std::vector<std::unique_ptr<cache_fixture>> fixtures;
        for (int i = 0; i < 8; ++i) {
            fixtures.emplace_back(new cache_fixture(i));
        }

        for (int variant = 0; variant < 2; ++variant) {
            std::mutex mutex;
            std::condition_variable changed;
            int ready = 0;
            bool start = false;
            std::vector<std::thread> threads;

            for (auto & fixture : fixtures) {
                auto * instance = fixture.get();
                threads.emplace_back([&, instance] {
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        ++ready;
                        changed.notify_all();
                        changed.wait(lock, [&] { return start; });
                    }

                    instance->run(variant, false);
                });
            }

            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return ready == 8; });
                start = true;
                changed.notify_all();
            }

            for (auto & thread : threads) {
                thread.join();
            }

            observer_snapshot(after);
            REQUIRE(after[HIPRTC_TEST_COMPILE] - before[HIPRTC_TEST_COMPILE] == uint64_t(variant + 1));
            REQUIRE(after[HIPRTC_TEST_CREATE] - before[HIPRTC_TEST_CREATE] == uint64_t(variant + 1));
            REQUIRE(after[HIPRTC_TEST_LOAD] - before[HIPRTC_TEST_LOAD] == uint64_t((variant + 1) * 8));
        }
    }

    observer_snapshot(after);
    REQUIRE(after[HIPRTC_TEST_UNLOAD] - before[HIPRTC_TEST_UNLOAD] == 16);
    REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] - before[HIPRTC_TEST_MODULE_LAUNCH] == 16);
    REQUIRE(after[HIPRTC_TEST_SETUP_DURING_CAPTURE] == 0);
    std::puts("concurrency: eight contexts, two single-flight real compiles, 16 independent module lifecycles passed");
}

static void concurrency_capture() {
    concurrency();

    uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];

    {
        cache_fixture recapturer(9);
        recapturer.graphs[0]->uid = recapturer.graphs[1]->uid = 1;
        for (int variant = 0; variant < 2; ++variant) {
            for (int pass = 0; pass < 3; ++pass) {
                recapturer.run(variant, false, true);
            }
        }

        std::vector<std::unique_ptr<cache_fixture>> fixtures;
        for (int i = 0; i < 8; ++i) {
            fixtures.emplace_back(new cache_fixture(i));
        }

        std::mutex mutex;
        std::condition_variable changed;
        int ready = 0;
        bool start = false;
        std::vector<std::thread> threads;

        observer_snapshot(before);
        for (int index = 0; index < 9; ++index) {
            threads.emplace_back([&, index] {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ++ready;
                    changed.notify_all();
                    changed.wait(lock, [&] { return start; });
                }

                if (index < 8) {
                    for (int variant = 0; variant < 2; ++variant) {
                        fixtures[index]->run(variant, false);
                    }
                } else {
                    for (int iteration = 0; iteration < 32; ++iteration) {
                        recapturer.bias += 1.0f;
                        const float parameters[] = {2.0f, recapturer.bias};
                        ggml_set_op_params(recapturer.affine, parameters, sizeof(parameters));
                        for (int pass = 0; pass < 3; ++pass) {
                            recapturer.run(iteration % 2, false, true);
                        }
                    }
                }
            });
        }

        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&] { return ready == 9; });
            start = true;
            changed.notify_all();
        }

        for (auto & thread : threads) {
            thread.join();
        }

        observer_snapshot(after);
        REQUIRE(after[HIPRTC_TEST_COMPILE] == before[HIPRTC_TEST_COMPILE]);
        REQUIRE(after[HIPRTC_TEST_LOAD] == before[HIPRTC_TEST_LOAD] + 16);
        REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN] + 32);
        REQUIRE(after[HIPRTC_TEST_CAPTURE_END] == before[HIPRTC_TEST_CAPTURE_END] + 32);
        REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 64);
        REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH] + 80);
        REQUIRE(after[HIPRTC_TEST_SETUP_DURING_CAPTURE] == 0);
    }

    observer_snapshot(after);
    REQUIRE(after[HIPRTC_TEST_UNLOAD] == before[HIPRTC_TEST_UNLOAD] + 18);
    std::puts("concurrency: eight cold module contexts and independently recapturing context passed without setup during capture");
}

static void failure() {
    REQUIRE(getenv("GGML_TEST_HIPRTC_FAIL_CREATE") && std::atoi(getenv("GGML_TEST_HIPRTC_FAIL_CREATE")));

    uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];
    observer_snapshot(before);

    for (int context = 0; context < 2; ++context) {
        cache_fixture fixture(context);
        for (int variant = 0; variant < 2; ++variant) {
            for (int iteration = 0; iteration < 3; ++iteration) {
                fixture.run(variant, true);
                observer_snapshot(after);
                const uint64_t expected = context ? 2 : uint64_t(variant + 1);
                REQUIRE(after[HIPRTC_TEST_CREATE] - before[HIPRTC_TEST_CREATE] == expected);
                REQUIRE(after[HIPRTC_TEST_COMPILE] - before[HIPRTC_TEST_COMPILE] == expected);
                REQUIRE(after[HIPRTC_TEST_LOAD] == before[HIPRTC_TEST_LOAD]);
                REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH]);
            }
        }
    }
    std::puts("failure: two real syntax failures cached across repeated requests and two contexts; ordinary outputs passed");
}

static void limits() {
    auto backend = ggml_backend_cuda_init(0);
    REQUIRE(backend);

    uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];
    observer_snapshot(before);

    for (size_t stride = 2; stride < 259; ++stride) {
        auto * ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        auto * backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, stride + 2);
        auto * b = ggml_view_2d(ctx, backing, 2, 2, stride * sizeof(float), 0);
        auto * out = ggml_sqr(ctx, ggml_neg(ctx, ggml_mul(ctx, ggml_add(ctx, a, b), a)));

        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        const float av[] = {-4, -1, 2, 4};
        std::vector<float> bv(stride + 2, 2.0f);
        ggml_backend_tensor_set(a, av, 0, sizeof(av));
        ggml_backend_tensor_set(backing, bv.data(), 0, bv.size() * sizeof(float));

        auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
        const int skipped = ggml_cuda_rtc_fusion_try(cuda_ctx, graph, 1, true);
        REQUIRE(skipped == (stride < 258 ? 3 : 0));
        if (!skipped) {
            for (int node = 1; node < graph->n_nodes; ++node) {
                auto view = ggml_graph_view(graph, node, node + 1);
                REQUIRE(ggml_backend_graph_compute(backend, &view) == GGML_STATUS_SUCCESS);
            }
        }

        ggml_backend_synchronize(backend);

        float values[4];
        ggml_backend_tensor_get(out, values, 0, sizeof(values));
        for (int i = 0; i < 4; ++i) {
            const float value = (av[i] + 2.0f) * av[i];
            REQUIRE(values[i] == value * value);
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    for (bool wide : {false, true}) {
        auto * ctx = ggml_init({1 << 20, nullptr, true});
        REQUIRE(ctx);

        auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        auto * graph = ggml_new_graph(ctx);
        std::vector<ggml_tensor *> inputs, outputs;
        auto * chain = a;
        for (int i = 0; i < (wide ? 64 : 32); ++i) {
            if (wide) {
                auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
                inputs.push_back(b);
                auto * out = ggml_add(ctx, a, b);
                ggml_set_output(out);
                outputs.push_back(out);
                ggml_build_forward_expand(graph, out);
            } else {
                chain = ggml_scale_bias(ctx, chain, 0.5f, 1.0f);
            }
        }

        if (!wide) {
            outputs.push_back(chain);
            ggml_build_forward_expand(graph, chain);
        }

        ggml_fusion_region region;
        ggml_fusion_program program;
        REQUIRE(ggml_fusion_build(graph, 0, region, program));
        REQUIRE(region.count == (wide ? 8 : 16));
        REQUIRE(program.outputs == (wide ? 8 : 1));
        REQUIRE(program.inputs == (wide ? 9 : 1));

        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        REQUIRE(buffer);

        const float av[] = {-4, -1, 2, 4};
        ggml_backend_tensor_set(a, av, 0, sizeof(av));
        for (size_t i = 0; i < inputs.size(); ++i) {
            const float bv[] = {float(i + 1), float(i + 1), float(i + 1), float(i + 1)};
            ggml_backend_tensor_set(inputs[i], bv, 0, sizeof(bv));
        }

        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        for (size_t i = 0; i < outputs.size(); ++i) {
            float values[4];
            ggml_backend_tensor_get(outputs[i], values, 0, sizeof(values));
            for (int j = 0; j < 4; ++j) {
                float expected = av[j];
                if (wide) {
                    expected += float(i + 1);
                } else {
                    for (int op = 0; op < 32; ++op) {
                        expected = expected * 0.5f + 1.0f;
                    }
                }
                REQUIRE(values[j] == expected);
            }
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);

    observer_snapshot(after);
    REQUIRE(after[HIPRTC_TEST_CREATE] - before[HIPRTC_TEST_CREATE] == 256);
    REQUIRE(after[HIPRTC_TEST_COMPILE] - before[HIPRTC_TEST_COMPILE] == 256);
    REQUIRE(after[HIPRTC_TEST_LOAD] - before[HIPRTC_TEST_LOAD] == 256);
    REQUIRE(after[HIPRTC_TEST_UNLOAD] - before[HIPRTC_TEST_UNLOAD] == 256);
    std::puts("limits: 257 layout keys, 256 real compiles/modules, final ordinary fallback and teardown passed");
}

static void streams() {
    auto backend = ggml_backend_cuda_init(0);
    auto cpu = ggml_backend_cpu_init();
    REQUIRE(backend && cpu);
    ggml_backend_t backends[] = {backend, cpu};

    auto * ctx = ggml_init({4 << 20, nullptr, true});
    REQUIRE(ctx);

    auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513);
    ggml_set_input(a);
    auto * root = ggml_scale(ctx, a, 0.5f);
    ggml_set_name(root, "attn_norm");
    ggml_set_output(root); // keep the shared fork input out of branch storage

    auto * v = ggml_sqr(ctx, ggml_neg(ctx, root));
    auto * t = ggml_scale(ctx, ggml_sqr(ctx, root), 2.0f);
    auto * q = ggml_scale(ctx, ggml_relu(ctx, root), 3.0f);

    auto * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, v);
    ggml_build_forward_expand(g, t);
    ggml_build_forward_expand(g, q);
    auto * out = ggml_add(ctx, ggml_add(ctx, v, t), q);
    ggml_set_output(out);
    ggml_build_forward_expand(g, out);

    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 2048, false, true);
    REQUIRE(sched);
    ggml_backend_sched_set_tensor_backend(sched, a, backend);
    for (int i = 0; i < g->n_nodes; ++i) {
        ggml_backend_sched_set_tensor_backend(sched, g->nodes[i], backend);
    }
    REQUIRE(ggml_backend_sched_alloc_graph(sched, g));

    auto & cuda_ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    REQUIRE(cuda_ctx.stream_context().concurrent_events.size() == 1);
    const auto & event = cuda_ctx.stream_context().concurrent_events.at(root);
    REQUIRE(event.n_streams == 3);

    std::vector<float> input(513);
    for (int i = 0; i < 513; ++i) {
        input[i] = float(2 * (i % 4) - 4);
    }

    uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];
    observer_snapshot(before);
    for (int repeat = 0; repeat < 4; ++repeat) {
        ggml_backend_tensor_set(a, input.data(), 0, input.size() * sizeof(float));
        REQUIRE(ggml_backend_sched_graph_compute(sched, g) == GGML_STATUS_SUCCESS);

        const float expected[] = {12, 3, 0, 6};
        check_pattern(out, expected, 513);
        const float expected_root[] = {-2, -1, 0, 1};
        check_pattern(root, expected_root, 513);
    }

    observer_snapshot(after);
    REQUIRE(after[HIPRTC_TEST_COMPILE] == before[HIPRTC_TEST_COMPILE]);
    REQUIRE(after[HIPRTC_TEST_MODULE_LAUNCH] == before[HIPRTC_TEST_MODULE_LAUNCH]);

    for (float & value : input) {
        value = -value;
    }
    observer_snapshot(before);
    ggml_backend_tensor_set(a, input.data(), 0, input.size() * sizeof(float));
    REQUIRE(ggml_backend_sched_graph_compute(sched, g) == GGML_STATUS_SUCCESS);
    observer_snapshot(after);
    const float expected[] = {18, 6, 0, 3};
    check_pattern(out, expected, 513);
    const float expected_root[] = {2, 1, 0, -1};
    check_pattern(root, expected_root, 513);
    REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] == before[HIPRTC_TEST_GRAPH_LAUNCH] + 1);
    REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] == before[HIPRTC_TEST_CAPTURE_BEGIN]);
    REQUIRE(after[HIPRTC_TEST_EVENT_RECORD] == before[HIPRTC_TEST_EVENT_RECORD]);
    REQUIRE(after[HIPRTC_TEST_STREAM_WAIT] == before[HIPRTC_TEST_STREAM_WAIT]);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    ggml_free(ctx);

    std::puts("streams: three real side streams replay changed inputs without host fork/join calls");
}

#include "hiprtc-fusion-bench.h"
#endif

int main(int argc, char ** argv) {
    int count = 0;
    const auto status = hipGetDeviceCount(&count);
    if (status == hipErrorNoDevice || (status == hipSuccess && count == 0)) {
        return 77;
    }

    CUDA_CHECK(status);
    REQUIRE(argc == 2);

#ifndef USE_CUDA_GRAPH
    if (std::strcmp(argv[1], "capture") == 0 || std::strcmp(argv[1], "capture-limits") == 0 ||
        std::strcmp(argv[1], "concurrency-capture") == 0 || std::strcmp(argv[1], "streams") == 0) {
        std::fputs("This mode requires GGML_HIP_GRAPHS=ON\n", stderr);
        return 77;
    }
#endif

    if (std::strcmp(argv[1], "raw") == 0) {
        raw(false);
    } else if (std::strcmp(argv[1], "raw-capture") == 0) {
#ifdef __linux__
        const bool observed = dlsym(RTLD_DEFAULT, "hiprtc_test_snapshot") != nullptr;
        uint64_t before[HIPRTC_TEST_COUNTER_COUNT], after[HIPRTC_TEST_COUNTER_COUNT];
        if (observed) {
            observer_snapshot(before);
        }
#endif
        raw(true);
#ifdef __linux__
        if (observed) {
            observer_snapshot(after);
            REQUIRE(after[HIPRTC_TEST_CREATE] - before[HIPRTC_TEST_CREATE] == 100);
            REQUIRE(after[HIPRTC_TEST_COMPILE] - before[HIPRTC_TEST_COMPILE] == 100);
            REQUIRE(after[HIPRTC_TEST_LOAD] - before[HIPRTC_TEST_LOAD] == 100);
            REQUIRE(after[HIPRTC_TEST_UNLOAD] - before[HIPRTC_TEST_UNLOAD] == 100);
            REQUIRE(after[HIPRTC_TEST_CAPTURE_BEGIN] - before[HIPRTC_TEST_CAPTURE_BEGIN] == 400);
            REQUIRE(after[HIPRTC_TEST_CAPTURE_END] - before[HIPRTC_TEST_CAPTURE_END] == 400);
            REQUIRE(after[HIPRTC_TEST_GRAPH_LAUNCH] - before[HIPRTC_TEST_GRAPH_LAUNCH] == 600);
            REQUIRE(after[HIPRTC_TEST_SETUP_DURING_CAPTURE] == before[HIPRTC_TEST_SETUP_DURING_CAPTURE]);
            std::puts("raw-capture: 400 observed capture intervals, 600 graph replays, zero cold setup during capture");
        }
#endif
    } else if (std::strcmp(argv[1], "graph") == 0) {
        graph();
    } else if (std::strcmp(argv[1], "schedule") == 0) {
        schedule();
    } else if (std::strcmp(argv[1], "alias") == 0) {
        alias();
    } else if (std::strcmp(argv[1], "types") == 0) {
        types();
    } else if (std::strcmp(argv[1], "reductions") == 0) {
        reductions();
#ifdef __linux__
    } else if (std::strcmp(argv[1], "streams") == 0) {
        streams();
    } else if (std::strcmp(argv[1], "bench") == 0) {
        bench();
    } else if (std::strcmp(argv[1], "capture") == 0) {
        capture();
    } else if (std::strcmp(argv[1], "capture-limits") == 0) {
        capture_limits();
    } else if (std::strcmp(argv[1], "concurrency") == 0) {
        concurrency();
    } else if (std::strcmp(argv[1], "concurrency-capture") == 0) {
        concurrency_capture();
    } else if (std::strcmp(argv[1], "failure") == 0) {
        failure();
    } else if (std::strcmp(argv[1], "limits") == 0) {
        limits();
#endif
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
        return 1;
    }
}
