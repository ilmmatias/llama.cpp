#include "fusion-rtc.h"
#include "common.cuh"
#include <hip/hiprtc.h>

#include <cstdlib>
#include <string>
#include <vector>
#include <climits>
#include <cstdint>

static constexpr int GGML_CUDA_RTC_MAX_OPS = 8;
static constexpr int GGML_CUDA_RTC_MAX_INPUTS = 16;
static constexpr int GGML_CUDA_RTC_MAX_PARAMS = 16;

struct ggml_cuda_rtc_instruction {
    ggml_op op = GGML_OP_NONE;
    ggml_unary_op unary = GGML_UNARY_OP_NEG;
    int src[2] = {0, 0};
    int param = -1;
};

struct ggml_cuda_rtc_program {
    int count = 0;
    int inputs = 0;
    int params = 0;
    ggml_cuda_rtc_instruction instructions[GGML_CUDA_RTC_MAX_OPS];
};

struct ggml_cuda_rtc_binding {
    const ggml_tensor * tensors[GGML_CUDA_RTC_MAX_INPUTS] = {};
    const float * inputs[GGML_CUDA_RTC_MAX_INPUTS] = {};
    float params[GGML_CUDA_RTC_MAX_PARAMS] = {};
    float * output = nullptr;
    unsigned long long n = 0;
};

static int rtc_source_count(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL: return 2;
        case GGML_OP_SCALE:
        case GGML_OP_SQR: return 1;
        case GGML_OP_UNARY: {
            const auto unary = ggml_get_unary_op(node);
            return unary == GGML_UNARY_OP_NEG || unary == GGML_UNARY_OP_RELU || unary == GGML_UNARY_OP_SILU ? 1 : 0;
        }
        default: return 0;
    }
}

static bool rtc_element_count(const ggml_tensor * tensor, unsigned long long & n) {
    n = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] <= 0 || n > uint64_t(INT64_MAX) / uint64_t(tensor->ne[d])) {
            return false;
        }
        n *= uint64_t(tensor->ne[d]);
    }
    return n <= SIZE_MAX / sizeof(float);
}

static bool rtc_tensor_valid(const ggml_tensor * tensor, const ggml_tensor * shape, int device) {
    unsigned long long n;
    return tensor && tensor->type == GGML_TYPE_F32 && rtc_element_count(tensor, n) &&
        ggml_are_same_shape(tensor, shape) && ggml_is_contiguous(tensor) &&
        tensor->data && tensor->buffer &&
        ggml_backend_buffer_get_type(tensor->buffer) == ggml_backend_cuda_buffer_type(device);
}

static bool rtc_bind(const ggml_cgraph * graph, int start, int count, int device,
                     ggml_cuda_rtc_program & program, ggml_cuda_rtc_binding & binding) {
    program = {};
    binding = {};
    const ggml_tensor * output = graph->nodes[start + count - 1];
    if (!rtc_element_count(output, binding.n)) {
        return false;
    }
    program.count = count;
    binding.output = static_cast<float *>(output->data);
    for (int i = 0; i < count; ++i) {
        const auto * node = graph->nodes[start + i];
        auto & instruction = program.instructions[i];
        instruction.op = node->op;
        if (node->op == GGML_OP_UNARY) {
            instruction.unary = ggml_get_unary_op(node);
        }
        if (node->op == GGML_OP_SCALE) {
            if (program.params + 2 > GGML_CUDA_RTC_MAX_PARAMS) {
                return false;
            }
            instruction.param = program.params;
            binding.params[program.params++] = ggml_get_op_params_f32(node, 0);
            binding.params[program.params++] = ggml_get_op_params_f32(node, 1);
        }
        for (int s = 0; s < rtc_source_count(node); ++s) {
            const auto * source = node->src[s];
            int internal = -1;
            for (int j = 0; j < count; ++j) {
                if (source == graph->nodes[start + j]) {
                    internal = j;
                    break;
                }
            }
            if (internal >= 0) {
                if (internal >= i) {
                    return false;
                }
                instruction.src[s] = internal;
                continue;
            }
            for (const auto * ancestor = source->view_src; ancestor; ancestor = ancestor->view_src) {
                for (int j = 0; j < count; ++j) {
                    if (ancestor == graph->nodes[start + j]) {
                        return false;
                    }
                }
            }
            if (!rtc_tensor_valid(source, output, device)) {
                return false;
            }
            int slot = 0;
            while (slot < program.inputs && binding.tensors[slot] != source) {
                ++slot;
            }
            if (slot == program.inputs) {
                if (program.inputs == GGML_CUDA_RTC_MAX_INPUTS) {
                    return false;
                }
                binding.tensors[slot] = source;
                binding.inputs[slot] = static_cast<const float *>(source->data);
                ++program.inputs;
            }
            instruction.src[s] = -1 - slot;
        }
    }
    const size_t bytes = size_t(binding.n) * sizeof(float);
    const uintptr_t out = reinterpret_cast<uintptr_t>(output->data);
    if (out > UINTPTR_MAX - bytes) {
        return false;
    }
    for (int k = 0; k < program.inputs; ++k) {
        const auto * input = binding.tensors[k];
        const uintptr_t in = reinterpret_cast<uintptr_t>(input->data);
        if (in > UINTPTR_MAX - bytes) {
            return false;
        }
        if (out < in + bytes && in < out + bytes &&
            (out != in || !ggml_are_same_layout(output, input))) {
            return false;
        }
    }
    return true;
}

static bool rtc_discover(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int start,
                         ggml_cuda_rtc_program & program, ggml_cuda_rtc_binding & binding) {
    if (!ggml_cuda_rtc_fusion_enabled() || start < 0 || start >= graph->n_nodes ||
        ctx.curr_stream_no != 0 || !ctx.stream_context().concurrent_events.empty()) {
        return false;
    }
    ggml_op ops[GGML_CUDA_RTC_MAX_OPS];
    int count = 0;
    while (count < GGML_CUDA_RTC_MAX_OPS && count < graph->n_nodes - start) {
        const auto * node = graph->nodes[start + count];
        const int sources = rtc_source_count(node);
        if (!sources || !(node->flags & GGML_TENSOR_FLAG_COMPUTE) || node->view_src ||
            !rtc_tensor_valid(node, graph->nodes[start], ctx.device)) {
            break;
        }
        bool valid = true;
        bool connected = count == 0;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (s < sources) {
                valid = valid && rtc_tensor_valid(node->src[s], node, ctx.device);
                connected = connected || (count > 0 && node->src[s] == graph->nodes[start + count - 1]);
            } else {
                valid = valid && node->src[s] == nullptr;
            }
        }
        if (!valid || !connected) {
            break;
        }
        ops[count++] = node->op;
    }
    for (; count >= 2; --count) {
        const int last = start + count - 1;
        if (ggml_can_fuse_subgraph(graph, start, count, ops, &last, 1) &&
            rtc_bind(graph, start, count, ctx.device, program, binding)) {
            return true;
        }
    }
    return false;
}

static std::string rtc_value(int reference) {
    return reference < 0 ? "x" + std::to_string(-1 - reference) : "v" + std::to_string(reference);
}

static std::string rtc_emit(const ggml_cuda_rtc_program & program) {
    std::string source = "extern \"C\" __global__ void ggml_fused_elementwise(float * out";
    for (int k = 0; k < program.inputs; ++k) {
        source += ", const float * in" + std::to_string(k);
    }
    source += ", unsigned long long n";
    for (int k = 0; k < program.params; ++k) {
        source += ", float p" + std::to_string(k);
    }
    source += ") {\nunsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
              "unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;\nwhile (i < n) {\n";
    for (int k = 0; k < program.inputs; ++k) {
        source += "float x" + std::to_string(k) + " = in" + std::to_string(k) + "[i];\n";
    }
    for (int j = 0; j < program.count; ++j) {
        const auto & instruction = program.instructions[j];
        const auto a = rtc_value(instruction.src[0]);
        const auto b = rtc_value(instruction.src[1]);
        std::string expression;
        switch (instruction.op) {
            case GGML_OP_ADD: expression = a + " + " + b; break;
            case GGML_OP_MUL: expression = a + " * " + b; break;
            case GGML_OP_SQR: expression = a + " * " + a; break;
            case GGML_OP_SCALE:
                expression = "p" + std::to_string(instruction.param) + " * " + a + " + p" + std::to_string(instruction.param + 1);
                break;
            case GGML_OP_UNARY:
                switch (instruction.unary) {
                    case GGML_UNARY_OP_NEG: expression = "-" + a; break;
                    case GGML_UNARY_OP_RELU: expression = "fmaxf(" + a + ", 0.0f)"; break;
                    case GGML_UNARY_OP_SILU: expression = a + " / (1.0f + expf(-" + a + "))"; break;
                    default: GGML_ABORT("unsupported RTC unary");
                }
                break;
            default: GGML_ABORT("unsupported RTC operation");
        }
        source += "float v" + std::to_string(j) + " = " + expression + ";\n";
    }
    source += "out[i] = v" + std::to_string(program.count - 1) + ";\nif (n - i <= stride) break;\ni += stride;\n}\n}\n";
    return source;
}

bool ggml_cuda_rtc_fusion_enabled() {
    static const bool enabled = getenv("GGML_HIP_RTC_FUSION") && std::atoi(getenv("GGML_HIP_RTC_FUSION")) &&
        !(getenv("GGML_CUDA_DISABLE_FUSION") && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION")));
    return enabled;
}

struct rtc_program_hash {
    size_t operator()(const ggml_cuda_rtc_program & program) const {
        size_t hash = 1; // code-generator ABI
        auto add = [&hash](int value) { hash = hash * 31 + size_t(value); };
        add(program.count);
        add(program.inputs);
        add(program.params);
        for (int i = 0; i < program.count; ++i) {
            const auto & op = program.instructions[i];
            add(op.op);
            add(op.unary);
            add(op.src[0]);
            add(op.src[1]);
            add(op.param);
        }
        return hash;
    }
};

struct rtc_program_equal {
    bool operator()(const ggml_cuda_rtc_program & a, const ggml_cuda_rtc_program & b) const {
        if (a.count != b.count || a.inputs != b.inputs || a.params != b.params) {
            return false;
        }
        for (int i = 0; i < a.count; ++i) {
            const auto & x = a.instructions[i];
            const auto & y = b.instructions[i];
            if (x.op != y.op || x.unary != y.unary || x.src[0] != y.src[0] || x.src[1] != y.src[1] || x.param != y.param) {
                return false;
            }
        }
        return true;
    }
};

struct rtc_module {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    ~rtc_module() {
        if (module) {
            CUDA_CHECK(hipModuleUnload(module));
        }
    }
};

struct rtc_compiler_program {
    hiprtcProgram program = nullptr;
    ~rtc_compiler_program() {
        if (program) {
            const auto error = hiprtcDestroyProgram(&program);
            if (error != HIPRTC_SUCCESS) {
                GGML_LOG_WARN("HIPRTC fusion: destroy program: %s\n", hiprtcGetErrorString(error));
            }
        }
    }
};

struct ggml_cuda_rtc_fusion {
    std::string target;
    int version_major = 0;
    int version_minor = 0;
    bool ready = false;
    int physical_device = -1;
    std::unordered_map<ggml_cuda_rtc_program, std::unique_ptr<rtc_module>, rtc_program_hash, rtc_program_equal> cache;
};

static bool rtc_initialize(ggml_backend_cuda_context & ctx) {
    ctx.rtc_fusion = new ggml_cuda_rtc_fusion;
    auto & state = *ctx.rtc_fusion;
    ggml_cuda_set_device(ctx.device);
    state.physical_device = ggml_cuda_info().devices[ctx.device].physical_device;
    hipDeviceProp_t properties;
    const auto device_error = hipGetDeviceProperties(&properties, state.physical_device);
    const auto compiler_error = hiprtcVersion(&state.version_major, &state.version_minor);
    if (device_error != hipSuccess || compiler_error != HIPRTC_SUCCESS) {
        GGML_LOG_WARN("HIPRTC fusion: initialization failed: device %s, compiler %s\n",
                      hipGetErrorString(device_error), hiprtcGetErrorString(compiler_error));
        return false;
    }
    state.target = properties.gcnArchName;
    state.ready = true;
    return true;
}

static std::unique_ptr<rtc_module> rtc_compile(const ggml_cuda_rtc_fusion & state, const ggml_cuda_rtc_program & program) {
    const std::string source = rtc_emit(program);
    const std::string target = "--gpu-architecture=" + state.target;
    const char * options[] = {target.c_str(), "-std=c++17", "-O3", "-ffp-contract=off"};
    rtc_compiler_program compiler;
    std::vector<char> log;
    auto check = [&](hiprtcResult error, const char * stage) {
        if (error == HIPRTC_SUCCESS) {
            return true;
        }
        GGML_LOG_WARN("HIPRTC fusion: %s %s: %s\n%s\n", state.target.c_str(), stage,
                      hiprtcGetErrorString(error), log.empty() ? "" : log.data());
        return false;
    };
    if (!check(hiprtcCreateProgram(&compiler.program, source.c_str(), "fusion.cpp", 0, nullptr, nullptr), "create")) {
        return nullptr;
    }
    const auto result = hiprtcCompileProgram(compiler.program, 4, options);
    size_t size = 0;
    if (!check(hiprtcGetProgramLogSize(compiler.program, &size), "log size")) {
        return nullptr;
    }
    if (size) {
        log.resize(size);
        if (!check(hiprtcGetProgramLog(compiler.program, log.data()), "log")) {
            return nullptr;
        }
    }
    if (!check(result, "compile") || !check(hiprtcGetCodeSize(compiler.program, &size), "code size")) {
        return nullptr;
    }
    std::vector<char> code(size);
    if (!check(hiprtcGetCode(compiler.program, code.data()), "code") ||
        !check(hiprtcDestroyProgram(&compiler.program), "destroy program")) {
        return nullptr;
    }
    // HIPRTC does not clear the destroyed handle.
    compiler.program = nullptr;
    auto module = std::make_unique<rtc_module>();
    auto error = hipModuleLoadData(&module->module, code.data());
    const char * stage = "load";
    if (error == hipSuccess) {
        stage = "function";
        error = hipModuleGetFunction(&module->function, module->module, "ggml_fused_elementwise");
    }
    if (error != hipSuccess) {
        GGML_LOG_WARN("HIPRTC fusion: %s %s: %s\n%s\n", state.target.c_str(), stage,
                      hipGetErrorString(error), log.empty() ? "" : log.data());
        return nullptr;
    }
    GGML_LOG_DEBUG("HIPRTC fusion: compiled %d ops for %s\n", program.count, state.target.c_str());
    return module;
}

int ggml_cuda_rtc_fusion_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int node_idx) {
    ggml_cuda_rtc_program program;
    ggml_cuda_rtc_binding binding;
    if (!rtc_discover(ctx, graph, node_idx, program, binding)) {
        return 0;
    }
    if (!ctx.rtc_fusion && !rtc_initialize(ctx)) {
        return 0;
    }
    auto & state = *ctx.rtc_fusion;
    if (!state.ready) {
        return 0;
    }
    ggml_cuda_set_device(ctx.device);
    auto entry = state.cache.find(program);
    if (entry == state.cache.end()) {
        if (state.cache.size() == 256) {
            return 0;
        }
        entry = state.cache.emplace(program, rtc_compile(state, program)).first;
    }
    if (!entry->second) {
        return 0;
    }
    void * args[2 + GGML_CUDA_RTC_MAX_INPUTS + GGML_CUDA_RTC_MAX_PARAMS];
    int arg = 0;
    args[arg++] = &binding.output;
    for (int k = 0; k < program.inputs; ++k) {
        args[arg++] = &binding.inputs[k];
    }
    args[arg++] = &binding.n;
    for (int k = 0; k < program.params; ++k) {
        args[arg++] = &binding.params[k];
    }
    const unsigned int blocks = unsigned(std::min((binding.n - 1) / 256 + 1, 65535ULL));
    CUDA_CHECK(hipModuleLaunchKernel(entry->second->function, blocks, 1, 1, 256, 1, 1, 0, ctx.stream(), args, nullptr));
    return program.count - 1;
}

void ggml_cuda_rtc_fusion_free(ggml_backend_cuda_context & ctx) {
    delete ctx.rtc_fusion;
    ctx.rtc_fusion = nullptr;
}
