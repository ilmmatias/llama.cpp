#include "fusion-rtc.h"
#include "common.cuh"
#include "ggml-fusion.h"
#include "ggml-backend-impl.h"
#include <hip/hiprtc.h>

#include <algorithm>
#include <climits>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ggml_cuda_rtc_binding {
    const unsigned char * inputs[GGML_FUSION_MAX_INPUTS] = {};
    float params[GGML_FUSION_MAX_PARAMS] = {};
    unsigned char * outputs[GGML_FUSION_MAX_OUTPUTS] = {};
    unsigned long long n = 0;
};

static bool rtc_tensor_valid(const ggml_tensor * tensor, int device) {
    const auto * original = tensor;
    const uintptr_t address = reinterpret_cast<uintptr_t>(tensor ? tensor->data : nullptr);
    size_t original_span = 0;
    int depth = 0;
    for (; tensor; tensor = tensor->view_src) {
        if (++depth > 64 || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) ||
            !tensor->data || !tensor->buffer ||
            ggml_backend_buffer_get_type(tensor->buffer) != ggml_backend_cuda_buffer_type(device)) {
            return false;
        }

        const size_t element = ggml_type_size(tensor->type);
        size_t span = element;
        uint64_t n = 1;
        for (int d = 0; d < 4; ++d) {
            if (tensor->ne[d] <= 0 || n > uint64_t(INT64_MAX) / uint64_t(tensor->ne[d]) ||
                (tensor->nb[d] % element) != 0 || (tensor->ne[d] > 1 && !tensor->nb[d]) ||
                uint64_t(tensor->ne[d] - 1) > (SIZE_MAX - span) / (tensor->nb[d] ? tensor->nb[d] : 1)) {
                return false;
            }
            n *= uint64_t(tensor->ne[d]);
            span += size_t(tensor->ne[d] - 1) * tensor->nb[d];
        }
        if (n > SIZE_MAX / element) {
            return false;
        }

        const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
        const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
        const size_t size = ggml_backend_buffer_get_size(tensor->buffer);
        if (data % element || base > UINTPTR_MAX - size || data < base ||
            data > UINTPTR_MAX - span || data + span > base + size) {
            return false;
        }
        if (tensor == original) {
            original_span = span;
        } else if (address < data || address > data + span || original_span > data + span - address) {
            return false;
        }
    }
    return depth != 0;
}

static bool rtc_bind(const ggml_cgraph * graph, int device, const ggml_fusion_region & region,
                     const ggml_fusion_program & program, ggml_cuda_rtc_binding & binding) {
    binding = {};

    for (int i = 0; i < region.count; ++i) {
        if (!rtc_tensor_valid(graph->nodes[region.members[i]], device)) {
            return false;
        }
    }

    binding.n = region.n;
    for (int o = 0; o < program.outputs; ++o) {
        const auto * output = region.outputs[o];
        const size_t output_bytes = size_t(ggml_nelements(output)) * ggml_type_size(output->type);
        const uintptr_t output_address = reinterpret_cast<uintptr_t>(output->data);

        for (int j = 0; j < o; ++j) {
            const uintptr_t other_address = reinterpret_cast<uintptr_t>(region.outputs[j]->data);
            const size_t other_bytes = size_t(ggml_nelements(region.outputs[j])) * ggml_type_size(region.outputs[j]->type);
            if (output_address < other_address + other_bytes && other_address < output_address + output_bytes) {
                return false;
            }
        }

        for (int k = 0; k < program.inputs; ++k) {
            const auto * input = region.inputs[k];
            if (!rtc_tensor_valid(input, device)) {
                return false;
            }

            const uintptr_t input_address = reinterpret_cast<uintptr_t>(input->data);
            size_t span = ggml_type_size(input->type);
            for (int d = 0; d < 4; ++d) {
                span += size_t(input->ne[d] - 1) * input->nb[d];
            }

            const bool overlaps = output_address < input_address + span && input_address < output_address + output_bytes;
            const bool scalar_reduction = program.reduction.value >= 0 &&
                program.values[program.reduction.value].op != GGML_OP_RMS_NORM;
            if (overlaps && (scalar_reduction || output_address != input_address ||
                             !ggml_are_same_layout(output, input) || program.accesses[k].repeat)) {
                return false;
            }

            if (output_address == input_address) {
                // Ordinary execution stores this output before later nodes read the aliased input.
                for (int j = program.roots[o] + 1; j < program.count; ++j) {
                    if (program.values[j].src[0] == -1 - k || program.values[j].src[1] == -1 - k) {
                        return false;
                    }
                }
            }

            binding.inputs[k] = static_cast<const unsigned char *>(input->data);
        }

        binding.outputs[o] = static_cast<unsigned char *>(output->data);
    }

    for (int p = 0; p < program.params; ++p) {
        binding.params[p] = region.params[p];
    }

    return true;
}

static std::string rtc_value(int reference) {
    return reference < 0 ? "x" + std::to_string(-1 - reference) : "v" + std::to_string(reference);
}

static bool rtc_normalizes(const ggml_fusion_program & program) {
    return program.reduction.value >= 0 && program.values[program.reduction.value].op == GGML_OP_RMS_NORM;
}

static bool rtc_reduction_preferred(const ggml_cgraph * graph, const ggml_fusion_region & region, const ggml_fusion_program & program) {
    if (!rtc_normalizes(program) || program.count != 3 || program.outputs != 1 || program.reduction.value != 0) {
        return false;
    }

    const auto prefix = program.values[1].op;
    const int last = region.members[region.count - 1];
    const auto * output = graph->nodes[last];
    if ((prefix != GGML_OP_MUL && prefix != GGML_OP_SCALE) || output->op != GGML_OP_MUL || last + 1 >= graph->n_nodes) {
        return false;
    }

    // Keep the multiply available for the next row reduction.
    const auto * next = graph->nodes[last + 1];
    return (next->op == GGML_OP_SUM_ROWS || next->op == GGML_OP_MEAN) && next->src[0] == output && !(output->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_node_get_use_count(graph, last) == 1;
}

static bool rtc_index32(const ggml_fusion_program & program) {
    uint64_t n = 1;

    for (const int64_t extent : program.ne) {
        if (extent <= 0 || uint64_t(extent) > UINT32_MAX / n) {
            return false;
        }
        n *= uint64_t(extent);
    }

    return true;
}

static bool rtc_flat_access(const ggml_fusion_access & access) {
    if (access.repeat) {
        return false;
    }

    uint64_t stride = ggml_type_size(access.type);
    for (int d = 0; d < 4; ++d) {
        if (access.ne[d] > 1 && access.nb[d] != stride) {
            return false;
        }
        stride *= uint64_t(access.ne[d]);
    }

    return true;
}

static void rtc_row_coordinates(std::string & source, const ggml_fusion_program & program) {
    if (!program.indexed || program.reduction.axes != 1) {
        return;
    }

    int last = 3;
    while (last > 0 && program.ne[last] == 1) {
        --last;
    }

    if (last == 0) {
        return;
    }

    const bool narrow = rtc_index32(program);
    const char * type = narrow ? "unsigned int" : "unsigned long long";
    const char * suffix = narrow ? "U" : "ULL";
    source += "    " + std::string(type) + " row_index = " + (narrow ? "(unsigned int)row" : "row") + ";\n";

    for (int d = 1; d <= last; ++d) {
        if (program.ne[d] > 1) {
            source += "    " + std::string(type) + " r" + std::to_string(d) + " = row_index";
            if (d < last) {
                source += " % " + std::to_string(program.ne[d]) + suffix;
            }
            source += ";\n";

            if (d < last) {
                source += "    row_index /= " + std::to_string(program.ne[d]) + suffix + ";\n";
            }
        }
    }
}

static const char * rtc_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return "float";
        case GGML_TYPE_F16:
            return "_Float16";
        case GGML_TYPE_BF16:
            return "__bf16";
        default:
            GGML_ABORT("unsupported RTC type");
    }
}

static void rtc_instructions(std::string & source, const ggml_fusion_program & program, uint8_t domain) {
    for (int j = 0; j < program.count; ++j) {
        const auto & instruction = program.values[j];
        if (instruction.domain != domain || j == program.reduction.value) {
            continue;
        }

        const auto a = rtc_value(instruction.src[0]);
        const auto b = rtc_value(instruction.src[1]);
        std::string expression;
        switch (instruction.op) {
            case GGML_OP_ADD:
                expression = a + " + " + b;
                break;
            case GGML_OP_SUB:
                expression = a + " - " + b;
                break;
            case GGML_OP_MUL:
                expression = a + " * " + b;
                break;
            case GGML_OP_DIV:
                expression = a + " / " + b;
                break;
            case GGML_OP_SQR:
                expression = a + " * " + a;
                break;
            case GGML_OP_SQRT:
                expression = "sqrtf(" + a + ")";
                break;
            case GGML_OP_LOG:
                expression = "logf(" + a + ")";
                break;
            case GGML_OP_CLAMP: {
                const std::string lower = "p" + std::to_string(instruction.param);
                const std::string upper = "p" + std::to_string(instruction.param + 1);
                const std::string cast = instruction.type == GGML_TYPE_F16 ? "(float)(_Float16)" : "";
                expression = "fminf(fmaxf(" + a + ", " + cast + lower + "), " + cast + upper + ")";
                break;
            }
            case GGML_OP_SCALE:
                expression = "fmaf(p" + std::to_string(instruction.param) + ", " + a + ", p" + std::to_string(instruction.param + 1) + ")";
                break;
            case GGML_OP_UNARY:
                switch (instruction.unary) {
                    case GGML_UNARY_OP_NEG:
                        expression = "-" + a;
                        break;
                    case GGML_UNARY_OP_RELU:
                        expression = "fmaxf(" + a + ", 0.0f)";
                        break;
                    case GGML_UNARY_OP_SILU:
                        expression = a + " / (1.0f + expf(-" + a + "))";
                        break;
                    case GGML_UNARY_OP_SIGMOID:
                        expression = "1.0f / (1.0f + expf(-" + a + "))";
                        break;
                    case GGML_UNARY_OP_SOFTPLUS:
                        expression = "(" + a + " > 20.0f ? " + a + " : logf(1.0f + expf(" + a + ")))";
                        break;
                    case GGML_UNARY_OP_ABS:
                        expression = "fabsf(" + a + ")";
                        break;
                    case GGML_UNARY_OP_SGN:
                        expression = "(" + a + " > 0.0f ? 1.0f : (" + a + " < 0.0f ? -1.0f : 0.0f))";
                        break;
                    case GGML_UNARY_OP_EXP:
                        expression = "expf(" + a + ")";
                        break;
                    case GGML_UNARY_OP_TANH:
                        expression = "tanhf(" + a + ")";
                        break;
                    default:
                        GGML_ABORT("unsupported RTC unary");
                }
                break;
            default:
                GGML_ABORT("unsupported RTC operation");
        }

        if (instruction.type != GGML_TYPE_F32) {
            expression = "(float)(" + std::string(rtc_type(instruction.type)) + ")(" + expression + ")";
        }

        source += "        float v" + std::to_string(j) + " = " + expression + ";\n";
    }
}

static bool rtc_parameter_domain(const ggml_fusion_program & program, int slot, uint8_t domain) {
    for (int i = 0; i < program.count; ++i) {
        const auto & value = program.values[i];
        const int params = value.op == GGML_OP_RMS_NORM ? 1 : 2;
        if (value.domain == domain && value.param >= 0 && slot >= value.param && slot < value.param + params) {
            return true;
        }
    }

    return false;
}

static std::string rtc_reduce_signature(const ggml_fusion_program & program, const char * name, int stage) {
    std::string source = "extern \"C\" __global__ void " + std::string(name) + "(";

    if (stage == 1) {
        source += "float * partial";
    } else {
        for (int k = 0; k < program.outputs; ++k) {
            source += (k ? ", " : "") + std::string("float * out") + std::to_string(k);
        }
    }

    for (int k = 0; k < program.inputs; ++k) {
        if (!stage || program.accesses[k].domain == stage) {
            source += ", const unsigned char * in" + std::to_string(k);
        }
    }

    if (stage == 2) {
        source += ", const float * partial, unsigned long long columns";
    } else if (stage == 1) {
        source += ", unsigned long long n";
    } else {
        source += ", unsigned long long rows, unsigned long long columns";
    }

    for (int k = 0; k < program.params; ++k) {
        if (!stage || rtc_parameter_domain(program, k, stage)) {
            source += ", float p" + std::to_string(k);
        }
    }

    return source + ") {\n";
}

static void rtc_reduce_loads(std::string & source, const ggml_fusion_program & program, uint8_t domain) {
    const bool narrow = rtc_index32(program);
    const char * suffix = narrow ? "U" : "ULL";
    const char * index = narrow ? "(unsigned int)i" : "i";

    for (int k = 0; k < program.inputs; ++k) {
        const auto & access = program.accesses[k];
        if (access.domain != domain) {
            continue;
        }

        std::string offset = "i * 4ULL";
        if (program.indexed && !rtc_flat_access(access)) {
            offset = "0ULL";
            uint64_t divisor = 1;
            for (int d = 0; d < 4; ++d) {
                const uint64_t extent = domain == 2 && !rtc_normalizes(program) && (program.reduction.axes & (1 << d)) ? 1 : program.ne[d];
                if (access.ne[d] > 1) {
                    std::string coordinate;
                    if (program.reduction.axes == 1) {
                        coordinate = d == 0 ? "column" : "r" + std::to_string(d);
                    } else {
                        coordinate = "((" + std::string(index) + " / " + std::to_string(divisor) + suffix + ") % " + std::to_string(extent) + suffix + ")";
                    }

                    if (access.repeat & (1 << d)) {
                        coordinate = "(" + coordinate + " % " + std::to_string(access.ne[d]) + suffix + ")";
                    }

                    offset += " + " + coordinate + " * " + std::to_string(access.nb[d]) + "ULL";
                }

                divisor *= extent;
            }
        }

        source += "        float x" + std::to_string(k) + " = *(const float *)(in" + std::to_string(k) +
            " + " + offset + ");\n";
    }
}

static std::string rtc_emit_reduction(const ggml_fusion_program & program, const ggml_fusion_schedule & schedule) {
    const auto & reduction = program.values[program.reduction.value];
    const std::string wave = std::to_string(schedule.wave);
    std::string source =
        "__device__ float rtc_wave_sum(float value) {\n"
        "    union { float f; unsigned int u; } bits;\n"
        "    unsigned int lane = threadIdx.x % " + wave + ";\n"
        "    for (unsigned int offset = " + wave + " / 2; offset; offset /= 2) {\n"
        "        bits.f = value;\n"
        "        unsigned int other = __builtin_amdgcn_ds_bpermute((lane + offset) * 4, bits.u);\n"
        "        bits.u = other;\n"
        "        if (lane + offset < " + wave + ") value += bits.f;\n"
        "    }\n"
        "    return value;\n"
        "}\n"
        "__device__ float rtc_block_sum(float value, float * shared) {\n"
        "    value = rtc_wave_sum(value);\n"
        "    unsigned int lane = threadIdx.x % " + wave + ";\n"
        "    unsigned int wave = threadIdx.x / " + wave + ";\n"
        "    if (lane == 0) shared[wave] = value;\n"
        "    __syncthreads();\n"
        "    value = threadIdx.x < blockDim.x / " + wave + " ? shared[lane] : 0.0f;\n"
        "    if (wave == 0) value = rtc_wave_sum(value);\n"
        "    return value;\n"
        "}\n";
    const bool two_stage = schedule.stages == 2;
    const char * kernel_name = rtc_normalizes(program) ? "ggml_fused_normalize_rows" : two_stage ? "ggml_fused_reduce_partial" : "ggml_fused_reduce_rows";

    source += rtc_reduce_signature(program, kernel_name, two_stage ? 1 : 0);
    source += "    __shared__ float shared[8];\n";
    if (two_stage) {
        source += "    unsigned long long begin = (unsigned long long)blockIdx.x * 4096ULL;\n"
                  "    unsigned long long columns = n - begin < 4096ULL ? n - begin : 4096ULL;\n";
    } else {
        source += "    for (unsigned long long row = blockIdx.x; row < rows; row += gridDim.x) {\n"
                  "    unsigned long long begin = row * columns;\n";
        rtc_row_coordinates(source, program);
    }

    source += "    float sum = 0.0f;\n"
              "    for (unsigned long long column = threadIdx.x; column < columns; column += blockDim.x) {\n"
              "        unsigned long long i = begin + column;\n";
    rtc_reduce_loads(source, program, 1);
    rtc_instructions(source, program, 1);

    const std::string input = rtc_value(reduction.src[0]);
    source += rtc_normalizes(program) ? "        sum = fmaf(" + input + ", " + input + ", sum);\n" : "        sum += " + input + ";\n";
    source += "    }\n    sum = rtc_block_sum(sum, shared);\n";

    if (rtc_normalizes(program)) {
        source += "    if (threadIdx.x == 0) shared[0] = rsqrtf(sum / (float)columns + p" + std::to_string(reduction.param) + ");\n"
                  "    __syncthreads();\n"
                  "    float inv_rms = shared[0];\n"
                  "    for (unsigned long long column = threadIdx.x; column < columns; column += blockDim.x) {\n"
                  "        unsigned long long i = begin + column;\n";
        rtc_reduce_loads(source, program, 1);
        rtc_instructions(source, program, 1);

        source += "        float v" + std::to_string(program.reduction.value) + " = " + input + " * inv_rms;\n";
        rtc_reduce_loads(source, program, 2);
        rtc_instructions(source, program, 2);

        for (int k = 0; k < program.outputs; ++k) {
            source += "        out" + std::to_string(k) + "[i] = v" + std::to_string(program.roots[k]) + ";\n";
        }

        return source + "    }\n    __syncthreads();\n    }\n}\n";
    }

    source += "    if (threadIdx.x == 0) {\n";
    if (two_stage) {
        source += "        partial[blockIdx.x] = sum;\n    }\n}\n";
        source += rtc_reduce_signature(program, "ggml_fused_reduce_finalize", 2);
        source += "    __shared__ float shared[8];\n"
                  "    float sum = 0.0f;\n"
                  "    for (unsigned long long i = threadIdx.x; i < columns; i += blockDim.x) sum += partial[i];\n"
                  "    sum = rtc_block_sum(sum, shared);\n"
                  "    if (threadIdx.x == 0) {\n"
                  "        unsigned long long i = 0;\n";
    } else {
        source += "        unsigned long long i = row;\n";
    }
    source += "        float v" + std::to_string(program.reduction.value) +
        (reduction.op == GGML_OP_MEAN ? " = sum / (float)columns;\n" : " = sum;\n");
    rtc_reduce_loads(source, program, 2);
    rtc_instructions(source, program, 2);
    for (int k = 0; k < program.outputs; ++k) {
        source += "        out" + std::to_string(k) + "[i] = v" + std::to_string(program.roots[k]) + ";\n";
    }
    source += "    }\n";
    if (!two_stage) {
        source += "    __syncthreads();\n    }\n";
    }
    return source + "}\n";
}

static std::string rtc_emit_pointwise(const ggml_fusion_program & program) {
    std::string source = "extern \"C\" __global__ void ggml_fused_elementwise(";

    for (int k = 0; k < program.outputs; ++k) {
        source += (k ? ", " : "") + std::string("unsigned char * out") + std::to_string(k);
    }

    for (int k = 0; k < program.inputs; ++k) {
        source += ", const unsigned char * in" + std::to_string(k);
    }

    source += ", unsigned long long n";
    for (int k = 0; k < program.params; ++k) {
        source += ", float p" + std::to_string(k);
    }

    source += ") {\n"
              "    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
              "    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;\n\n"
              "    while (i < n) {\n";

    const bool narrow = rtc_index32(program);
    const char * index_type = narrow ? "unsigned int" : "unsigned long long";
    const char * suffix = narrow ? "U" : "ULL";

    if (program.indexed) {
        int last = 3;
        while (last > 0 && program.ne[last] == 1) {
            --last;
        }

        source += "        " + std::string(index_type) + " remaining = " + (narrow ? "(unsigned int)i" : "i") + ";\n";

        for (int d = 0; d < 4; ++d) {
            if (program.ne[d] > 1) {
                source += "        " + std::string(index_type) + " q" + std::to_string(d) + " = remaining";
                if (d < last) {
                    source += " % " + std::to_string(program.ne[d]) + suffix;
                }
                source += ";\n";

                if (d < last) {
                    source += "        remaining /= " + std::to_string(program.ne[d]) + suffix + ";\n";
                }
            }
        }

        for (int k = 0; k < program.inputs; ++k) {
            for (int d = 0; d < 4; ++d) {
                const auto & access = program.accesses[k];
                if (!(access.repeat & (1 << d)) || access.ne[d] == 1) {
                    continue;
                }

                bool emitted = false;
                for (int j = 0; j < k; ++j) {
                    emitted |= (program.accesses[j].repeat & (1 << d)) &&
                        program.accesses[j].ne[d] == access.ne[d];
                }

                if (!emitted) {
                    source += "        " + std::string(index_type) + " r" + std::to_string(d) + "_" + std::to_string(access.ne[d]) + " = q" + std::to_string(d) + " % " + std::to_string(access.ne[d]) + suffix + ";\n";
                }
            }
        }
        source += "\n";
    }

    for (int k = 0; k < program.inputs; ++k) {
        std::string offset = "i * " + std::to_string(ggml_type_size(program.accesses[k].type)) + "ULL";
        if (program.indexed && !rtc_flat_access(program.accesses[k])) {
            const auto & access = program.accesses[k];
            offset = "0ULL";
            for (int d = 0; d < 4; ++d) {
                if (access.ne[d] == 1) {
                    continue;
                }

                const std::string coordinate = access.repeat & (1 << d) ?
                    "r" + std::to_string(d) + "_" + std::to_string(access.ne[d]) : "q" + std::to_string(d);
                offset += " + " + coordinate + " * " + std::to_string(access.nb[d]) + "ULL";
            }
        }

        source += "        float x" + std::to_string(k) + " = (float)*(const " + rtc_type(program.accesses[k].type) +
            " *)(in" + std::to_string(k) + " + " + offset + ");\n";
    }

    source += "\n";

    rtc_instructions(source, program, 0);
    source += "\n";

    for (int k = 0; k < program.outputs; ++k) {
        const std::string type = rtc_type(program.stores[k].type);
        source += "        ((" + type + " *)out" + std::to_string(k) + ")[i] = (" + type + ")v" +
            std::to_string(program.roots[k]) + ";\n";
    }

    source += "\n        if (n - i <= stride) break;\n"
              "        i += stride;\n"
              "    }\n"
              "}\n";

    return source;
}

bool ggml_cuda_rtc_fusion_enabled() {
    static const bool enabled = getenv("GGML_HIP_RTC_FUSION") && std::atoi(getenv("GGML_HIP_RTC_FUSION")) &&
        !(getenv("GGML_CUDA_DISABLE_FUSION") && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION")));
    return enabled;
}

struct rtc_key_hash {
    size_t operator()(const ggml_fusion_cache_key & key) const {
        uint64_t hash = 14695981039346656037ULL;
        for (size_t i = 0; i < key.size; ++i) {
            hash = (hash ^ key.bytes[i]) * 1099511628211ULL;
        }
        return size_t(hash);
    }
};

struct rtc_key_equal {
    bool operator()(const ggml_fusion_cache_key & a, const ggml_fusion_cache_key & b) const {
        return a.size == b.size && std::memcmp(a.bytes, b.bytes, a.size) == 0;
    }
};

static constexpr size_t RTC_CACHE_BYTES = 64 * 1024 * 1024;

enum rtc_compile_state {
    GGML_CUDA_RTC_COMPILE_COMPILING,
    GGML_CUDA_RTC_COMPILE_READY,
    GGML_CUDA_RTC_COMPILE_FAILED,
};

struct rtc_artifact {
    std::atomic<rtc_compile_state> state{GGML_CUDA_RTC_COMPILE_COMPILING};
    std::condition_variable changed;
    std::vector<char> code;
};

static std::mutex rtc_artifact_mutex;
static size_t rtc_artifact_bytes = 0;
static std::unordered_map<ggml_fusion_cache_key, std::shared_ptr<rtc_artifact>, rtc_key_hash, rtc_key_equal> rtc_artifacts;

static bool rtc_key_integer(ggml_fusion_cache_key & key, uint32_t value) {
    if (sizeof(key.bytes) - key.size < 4) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        key.bytes[key.size++] = uint8_t(value >> (8 * i));
    }
    return true;
}

static bool rtc_key_string(ggml_fusion_cache_key & key, const char * value) {
    const size_t size = std::strlen(value);
    if (size > UINT32_MAX || !rtc_key_integer(key, uint32_t(size)) || size > sizeof(key.bytes) - key.size) {
        return false;
    }
    std::memcpy(key.bytes + key.size, value, size);
    key.size += size;
    return true;
}

static bool rtc_make_key(ggml_fusion_program & program, const ggml_fusion_schedule & schedule, ggml_fusion_cache_key & key) {
    if (!program.indexed) {
        return ggml_fusion_make_key(program, schedule, key);
    }

    const bool narrow = rtc_index32(program);

    // Normalize only the key. Binding and code generation need the original layouts.
    int64_t original_ne[4];
    unsigned char original_accesses[sizeof(program.accesses)];
    const size_t access_bytes = program.inputs * sizeof(program.accesses[0]);
    std::memcpy(original_ne, program.ne, sizeof(original_ne));
    std::memcpy(original_accesses, program.accesses, access_bytes);

    const bool row_coordinates = program.reduction.value >= 0 && program.reduction.axes == 1;
    const int first = row_coordinates ? 1 : 0;
    int required = first - 1;
    for (int k = 0; k < program.inputs; ++k) {
        auto & access = program.accesses[k];
        const bool flat = rtc_flat_access(access);
        for (int d = 0; d < 4; ++d) {
            if (flat) {
                access.ne[d] = 0;
                access.nb[d] = 0;
            } else if (access.ne[d] == 1) {
                access.nb[d] = 0;
                access.repeat &= ~(1 << d);
            } else {
                if (d >= first) {
                    required = std::max(required, d);
                }
                if (!(access.repeat & (1 << d))) {
                    access.ne[d] = 2;
                }
            }
        }
    }

    if (program.reduction.value < 0 || row_coordinates) {
        int last = 3;
        while (last > first && original_ne[last] == 1) {
            --last;
        }

        for (int d = 0; d < 4; ++d) {
            if (d < first || d > required) {
                program.ne[d] = 1;
            } else if (d == required && d == last) {
                program.ne[d] = 2;
            }
        }
        if (required >= first && required < last) {
            program.ne[required + 1] = 2;
        }
    }

    const bool result = ggml_fusion_make_key(program, schedule, key);
    std::memcpy(program.ne, original_ne, sizeof(original_ne));
    std::memcpy(program.accesses, original_accesses, access_bytes);
    return result && rtc_key_integer(key, narrow);
}

struct rtc_module {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    hipFunction_t finalize = nullptr;

    ~rtc_module() {
        if (module) {
            CUDA_CHECK(hipModuleUnload(module));
        }
    }
};

struct rtc_node_reference {
    int node = -1;
    int slot = -1;
};

struct rtc_action {
    int start = 0;
    int members[GGML_FUSION_MAX_VALUES] = {};
    int outputs[GGML_FUSION_MAX_OUTPUTS] = {};
    rtc_node_reference inputs[GGML_FUSION_MAX_INPUTS];
    rtc_node_reference parameters[GGML_FUSION_MAX_PARAMS];
    ggml_fusion_program program;
    ggml_fusion_schedule schedule;
    ggml_fusion_cache_key key;
    std::shared_ptr<rtc_module> module;
};

struct ggml_cuda_rtc_plan {
    std::vector<uint64_t> signature;
    std::vector<rtc_action> actions;
    std::vector<std::shared_ptr<const rtc_artifact>> pending;
    bool retry = false;
    const float * scratch = nullptr;

    size_t bytes() const {
        return sizeof(*this) + signature.capacity() * sizeof(uint64_t) + actions.capacity() * sizeof(rtc_action) + pending.capacity() * sizeof(pending[0]);
    }
};

static constexpr size_t RTC_PLAN_BYTES = 32 * 1024 * 1024;

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
    ggml_fusion_cache_key identity;
    bool ready = false;
    bool initialization_attempted = false;

    int physical_device = -1;
    std::unordered_map<ggml_fusion_cache_key, std::shared_ptr<rtc_module>, rtc_key_hash, rtc_key_equal> cache;
    size_t code_bytes = 0;

    int max_threads = 0;
    size_t max_shared = 0;
    float * partial = nullptr;
    bool scratch_attempted = false;

    bool recording = false;
    // signature_count is the active prefix; keep the workspace size between graphs.
    std::vector<uint64_t> signature;
    size_t signature_count = 0;
    std::vector<ggml_bitset_t> seen;
    std::vector<rtc_action> actions;
    std::vector<std::shared_ptr<const rtc_artifact>> pending;
    bool retry = false;
    int pending_until = 0;
    std::vector<ggml_cuda_rtc_binding> bindings;

    std::shared_ptr<const ggml_cuda_rtc_plan> direct_plan;
    std::shared_ptr<const ggml_cuda_rtc_plan> capture_plan;
    size_t capture_action = 0;

    ~ggml_cuda_rtc_fusion() {
        capture_plan.reset();
        direct_plan.reset();
        actions.clear();
        cache.clear();

        if (partial) {
            CUDA_CHECK(hipFree(partial));
        }
    }
};

static bool rtc_signature_matches(const ggml_cuda_rtc_fusion & state, const ggml_cuda_rtc_plan & plan) {
    return plan.signature.size() == state.signature_count && std::memcmp(plan.signature.data(), state.signature.data(), state.signature_count * sizeof(uint64_t)) == 0;
}

static size_t rtc_metadata_bytes(const ggml_backend_cuda_context & ctx) {
    const auto & state = *ctx.rtc_fusion;
    size_t bytes = state.signature.capacity() * sizeof(uint64_t) +
        state.seen.capacity() * sizeof(ggml_bitset_t) +
        state.actions.capacity() * sizeof(rtc_action) + state.bindings.capacity() * sizeof(ggml_cuda_rtc_binding) +
        state.pending.capacity() * sizeof(state.pending[0]);
    bool direct_retained = false;

#ifdef USE_CUDA_GRAPH
    for (const auto & entry : ctx.cuda_graphs) {
        const auto & plan = entry.second->rtc_plan;
        if (plan) {
            bytes += plan->bytes();
            direct_retained |= plan == state.direct_plan;
        }
    }
#endif

    if (state.direct_plan && !direct_retained) {
        bytes += state.direct_plan->bytes();
    }

    return bytes;
}

template <typename T>
static bool rtc_reserve(ggml_backend_cuda_context & ctx, std::vector<T> & values, size_t count) {
    if (count <= values.capacity()) {
        return true;
    }

    const size_t capacity = std::max(count, std::max(size_t(8), values.capacity() * 2));
    const size_t bytes = rtc_metadata_bytes(ctx);
    if (capacity > RTC_PLAN_BYTES / sizeof(T) || bytes > RTC_PLAN_BYTES ||
        (capacity - values.capacity()) * sizeof(T) > RTC_PLAN_BYTES - bytes) {
        return false;
    }

    values.reserve(capacity);
    return true;
}

static bool rtc_signature_add(ggml_backend_cuda_context & ctx, uint64_t value) {
    auto & state = *ctx.rtc_fusion;
    auto & signature = state.signature;
    if (!rtc_reserve(ctx, signature, state.signature_count + 1)) {
        return false;
    }

    if (state.signature_count == signature.size()) {
        signature.push_back(value);
    } else {
        signature[state.signature_count] = value;
    }

    ++state.signature_count;
    return true;
}

static bool rtc_signature_tensor(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, const ggml_tensor * tensor) {
    auto & state = *ctx.rtc_fusion;
    auto & signature = state.signature;
    int depth = 0;

    do {
        const size_t count = tensor ? 11 + 2 * GGML_MAX_DIMS + GGML_MAX_OP_PARAMS / sizeof(int32_t) + GGML_MAX_SRC : 1;
        const size_t start = state.signature_count;
        if (!rtc_reserve(ctx, signature, start + count)) {
            return false;
        }

        if (signature.size() < start + count) {
            signature.resize(start + count);
        }

        state.signature_count = start + count;
        uint64_t * value = signature.data() + start;
        *value++ = reinterpret_cast<uintptr_t>(tensor);

        if (!tensor) {
            return true;
        }
        if (++depth > 64) {
            return false;
        }

        const size_t hash = ggml_hash_find(&graph->visited_hash_set, tensor);
        *value++ = ggml_bitset_get(graph->visited_hash_set.used, hash) ? graph->use_counts[hash] : 0;
        *value++ = tensor->op;
        *value++ = tensor->type;
        *value++ = tensor->flags & (GGML_TENSOR_FLAG_COMPUTE | GGML_TENSOR_FLAG_OUTPUT);
        *value++ = reinterpret_cast<uintptr_t>(tensor->data);
        *value++ = reinterpret_cast<uintptr_t>(tensor->buffer);
        *value++ = reinterpret_cast<uintptr_t>(tensor->buffer ? ggml_backend_buffer_get_type(tensor->buffer) : nullptr);
        *value++ = reinterpret_cast<uintptr_t>(tensor->buffer ? ggml_backend_buffer_get_base(tensor->buffer) : nullptr);
        *value++ = tensor->buffer ? ggml_backend_buffer_get_size(tensor->buffer) : 0;
        *value++ = tensor->view_offs;

        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            *value++ = uint64_t(tensor->ne[d]);
            *value++ = tensor->nb[d];
        }
        for (int p = 0; p < GGML_MAX_OP_PARAMS / int(sizeof(int32_t)); ++p) {
            *value++ = uint32_t(tensor->op_params[p]);
        }
        for (const auto * source : tensor->src) {
            *value++ = reinterpret_cast<uintptr_t>(source);
        }

        tensor = tensor->view_src;
    } while (true);
}

static bool rtc_signature_streams(ggml_backend_cuda_context & ctx) {
    const auto & events = ctx.concurrent_stream_context.concurrent_events;
    if (!rtc_signature_add(ctx, events.size())) {
        return false;
    }

    for (const auto & entry : events) {
        const auto & event = entry.second;
        if (!rtc_signature_add(ctx, reinterpret_cast<uintptr_t>(entry.first)) ||
            !rtc_signature_add(ctx, reinterpret_cast<uintptr_t>(event.join_node)) ||
            !rtc_signature_add(ctx, reinterpret_cast<uintptr_t>(event.fork_event)) ||
            !rtc_signature_add(ctx, event.n_streams) ||
            !rtc_signature_add(ctx, event.join_events.size())) {
            return false;
        }

        for (const auto join : event.join_events) {
            if (!rtc_signature_add(ctx, reinterpret_cast<uintptr_t>(join))) {
                return false;
            }
        }

        if (!rtc_signature_add(ctx, event.stream_mapping.size())) {
            return false;
        }
        for (const auto & mapping : event.stream_mapping) {
            if (!rtc_signature_add(ctx, reinterpret_cast<uintptr_t>(mapping.first)) ||
                !rtc_signature_add(ctx, mapping.second)) {
                return false;
            }
        }

        if (!rtc_signature_add(ctx, event.original_order.size())) {
            return false;
        }
        for (const auto * node : event.original_order) {
            if (!rtc_signature_add(ctx, reinterpret_cast<uintptr_t>(node))) {
                return false;
            }
        }
    }

    return true;
}

static bool rtc_action_bind(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph,
                            const rtc_action & action, ggml_cuda_rtc_binding & binding) {
    ggml_fusion_region region;
    const auto & program = action.program;
    region.count = program.count;
    region.n = 1;
    for (const int64_t extent : program.ne) {
        region.n *= uint64_t(extent);
    }

    for (int i = 0; i < program.count; ++i) {
        region.members[i] = action.members[i];
    }

    for (int o = 0; o < program.outputs; ++o) {
        region.outputs[o] = graph->nodes[action.outputs[o]];
    }

    for (int i = 0; i < program.inputs; ++i) {
        region.inputs[i] = graph->nodes[action.inputs[i].node]->src[action.inputs[i].slot];
    }

    for (int p = 0; p < program.params; ++p) {
        region.params[p] = ggml_get_op_params_f32(graph->nodes[action.parameters[p].node], action.parameters[p].slot);
    }

    return rtc_bind(graph, ctx.device, region, program, binding);
}

bool ggml_cuda_rtc_fusion_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph,
                                  ggml_cuda_graph * capture_graph, bool graph_compatible) {
    if (ctx.rtc_fusion) {
        ctx.rtc_fusion->recording = false;
        ctx.rtc_fusion->capture_plan.reset();
        ctx.rtc_fusion->capture_action = 0;
        ctx.rtc_fusion->actions.clear();
        ctx.rtc_fusion->bindings.clear();
        ctx.rtc_fusion->pending.clear();
        ctx.rtc_fusion->retry = false;
        ctx.rtc_fusion->pending_until = 0;
    }

#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph permission;
    permission.disable_due_to_gpu_arch = ggml_cuda_info().devices[ctx.device].cc < GGML_CUDA_CC_VOLTA;

    if (!ggml_cuda_rtc_fusion_enabled() || !permission.is_enabled() || graph->n_nodes == 0) {
        return false;
    }

    if (!graph_compatible || ctx.curr_stream_no != 0) {
        static const bool logged = [] {
            GGML_LOG_INFO("HIPRTC fusion: capture unsupported for this graph; using direct execution\n");
            return true;
        }();
        GGML_UNUSED(logged);

        if (ctx.rtc_fusion) {
            ctx.rtc_fusion->direct_plan.reset();
        }

        return false;
    }

    if (!ctx.rtc_fusion) {
        ctx.rtc_fusion = new ggml_cuda_rtc_fusion;
    }

    auto & state = *ctx.rtc_fusion;
    state.signature_count = 0;
    if (!rtc_signature_add(ctx, uint64_t(graph->n_nodes)) || !rtc_signature_streams(ctx)) {
        state.direct_plan.reset();
        return false;
    }

    const size_t words = ggml_bitset_size(graph->visited_hash_set.size);
    if (!rtc_reserve(ctx, state.seen, words)) {
        state.direct_plan.reset();
        return false;
    }
    state.seen.assign(words, 0);
    for (int i = 0; i < graph->n_nodes; ++i) {
        const size_t hash = ggml_hash_find(&graph->visited_hash_set, graph->nodes[i]);
        if (ggml_bitset_get(graph->visited_hash_set.used, hash)) {
            ggml_bitset_set(state.seen.data(), hash);
        }
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        if (!rtc_signature_tensor(ctx, graph, node)) {
            state.direct_plan.reset();
            return false;
        }

        for (const auto * source : node->src) {
            if (!source) {
                continue;
            }
            const size_t hash = ggml_hash_find(&graph->visited_hash_set, source);
            const bool visited = ggml_bitset_get(graph->visited_hash_set.used, hash);
            if (visited && ggml_bitset_get(state.seen.data(), hash)) {
                continue;
            }
            if (!rtc_signature_tensor(ctx, graph, source)) {
                state.direct_plan.reset();
                return false;
            }
            if (visited) {
                ggml_bitset_set(state.seen.data(), hash);
            }
        }
    }

    state.recording = true;
    const auto plan = capture_graph && capture_graph->rtc_plan ? capture_graph->rtc_plan : state.direct_plan;
    if (!plan || !rtc_signature_matches(state, *plan) ||
        (plan->scratch && plan->scratch != state.partial)) {
        state.direct_plan.reset();
        return false;
    }

    if (plan->retry || std::any_of(plan->pending.begin(), plan->pending.end(), [](const std::shared_ptr<const rtc_artifact> & artifact) {
            return artifact->state.load(std::memory_order_acquire) == GGML_CUDA_RTC_COMPILE_READY;
        })) {
        state.direct_plan.reset();
        return false;
    }

    if (!capture_graph || !capture_graph->instance) {
        if (!rtc_reserve(ctx, state.bindings, plan->actions.size())) {
            state.direct_plan.reset();
            return false;
        }
        state.bindings.resize(plan->actions.size());
        for (size_t i = 0; i < plan->actions.size(); ++i) {
            const auto & action = plan->actions[i];
            const auto entry = state.cache.find(action.key);
            if (entry == state.cache.end() || entry->second != action.module ||
                !rtc_action_bind(ctx, graph, action, state.bindings[i])) {
                state.direct_plan.reset();
                return false;
            }
        }
    }

    state.direct_plan = plan;
    state.capture_plan = plan;

    if (capture_graph) {
        capture_graph->rtc_plan = plan;
    }

    return true;
#else
    GGML_UNUSED(graph);
    GGML_UNUSED(capture_graph);
    GGML_UNUSED(graph_compatible);
    return false;
#endif
}

void ggml_cuda_rtc_fusion_record(ggml_backend_cuda_context & ctx, ggml_cuda_graph * capture_graph) {
    if (!ctx.rtc_fusion || !ctx.rtc_fusion->recording) {
        return;
    }

    auto & state = *ctx.rtc_fusion;
    if (capture_graph && !capture_graph->rtc_plan && state.capture_plan && state.actions.empty()) {
        capture_graph->rtc_plan = state.capture_plan;
        return;
    }

    state.capture_plan.reset();
    const auto previous = capture_graph && capture_graph->rtc_plan ? capture_graph->rtc_plan : state.direct_plan;
    bool unchanged = previous && rtc_signature_matches(state, *previous) && previous->actions.size() == state.actions.size() &&
        previous->pending == state.pending && previous->retry == state.retry;
    for (size_t i = 0; unchanged && i < state.actions.size(); ++i) {
        const auto & before = previous->actions[i];
        const auto & after = state.actions[i];
        unchanged = before.start == after.start && before.module == after.module &&
            before.key.size == after.key.size && std::memcmp(before.key.bytes, after.key.bytes, after.key.size) == 0;
    }

    if (unchanged) {
        state.direct_plan = previous;
    } else {
        state.direct_plan.reset();

        const size_t bytes = sizeof(ggml_cuda_rtc_plan) + state.signature_count * sizeof(uint64_t) + state.actions.size() * sizeof(rtc_action) + state.pending.size() * sizeof(state.pending[0]);
        const size_t retained = rtc_metadata_bytes(ctx);
        if (retained > RTC_PLAN_BYTES || bytes > RTC_PLAN_BYTES - retained) {
            state.recording = false;
            state.actions.clear();
            return;
        }

        auto plan = std::make_shared<ggml_cuda_rtc_plan>();
        plan->signature.assign(state.signature.begin(), state.signature.begin() + state.signature_count);
        plan->actions = state.actions;
        plan->pending = state.pending;
        plan->retry = state.retry;
        for (const auto & action : plan->actions) {
            if (action.schedule.scratch) {
                plan->scratch = state.partial;
                break;
            }
        }

        state.direct_plan = std::move(plan);
    }

    if (capture_graph) {
        capture_graph->rtc_plan = state.direct_plan;
    }

    state.actions.clear();
}

static void rtc_record_action(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph,
                              const ggml_fusion_region & region, const ggml_fusion_program & program,
                              const ggml_fusion_schedule & schedule, const ggml_fusion_cache_key & key,
                              const std::shared_ptr<rtc_module> & module) {
    auto & state = *ctx.rtc_fusion;
    if (!state.recording) {
        return;
    }

    if (!rtc_reserve(ctx, state.actions, state.actions.size() + 1)) {
        state.recording = false;
        state.actions.clear();
        return;
    }

    state.actions.emplace_back();
    auto & action = state.actions.back();
    action.start = region.members[0];
    action.program = program;
    action.schedule = schedule;
    action.key = key;
    action.module = module;

    for (int i = 0; i < region.count; ++i) {
        action.members[i] = region.members[i];
    }

    for (int o = 0; o < program.outputs; ++o) {
        action.outputs[o] = region.output_indices[o];
    }

    for (int input = 0; input < program.inputs; ++input) {
        for (int i = 0; i < region.count && action.inputs[input].node < 0; ++i) {
            const int member = region.members[i];
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                if (graph->nodes[member]->src[source] == region.inputs[input]) {
                    action.inputs[input] = {member, source};
                    break;
                }
            }
        }

        GGML_ASSERT(action.inputs[input].node >= 0);
    }

    for (int parameter = 0; parameter < program.params; ++parameter) {
        for (int i = 0; i < region.count; ++i) {
            if (graph->nodes[region.members[i]] == region.parameters[parameter]) {
                action.parameters[parameter] = {region.members[i], region.parameter_slots[parameter]};
                break;
            }
        }
    }
}

static bool rtc_initialize(ggml_backend_cuda_context & ctx) {
    if (!ctx.rtc_fusion) {
        ctx.rtc_fusion = new ggml_cuda_rtc_fusion;
    }

    auto & state = *ctx.rtc_fusion;
    state.initialization_attempted = true;

    ggml_cuda_rtc_setup_guard setup;
    ggml_cuda_set_device(ctx.device);
    state.physical_device = ggml_cuda_info().devices[ctx.device].physical_device;

    hipDeviceProp_t properties;
    int major = 0;
    int minor = 0;
    int runtime = 0;
    int driver = 0;

    const auto device_error = hipGetDeviceProperties(&properties, state.physical_device);
    const auto compiler_error = hiprtcVersion(&major, &minor);
    const auto runtime_error = hipRuntimeGetVersion(&runtime);
    const auto driver_error = hipDriverGetVersion(&driver);
    if (device_error != hipSuccess || compiler_error != HIPRTC_SUCCESS ||
        runtime_error != hipSuccess || driver_error != hipSuccess) {
        GGML_LOG_WARN("HIPRTC fusion: initialization failed: device %s, compiler %s, runtime %s, driver %s\n",
                      hipGetErrorString(device_error), hiprtcGetErrorString(compiler_error),
                      hipGetErrorString(runtime_error), hipGetErrorString(driver_error));
        return false;
    }

    state.target = properties.gcnArchName;
    state.max_threads = properties.maxThreadsPerBlock;
    state.max_shared = properties.sharedMemPerBlock;

    auto & identity = state.identity;
    if (!rtc_key_string(identity, "hiprtc") || !rtc_key_integer(identity, major) ||
        !rtc_key_integer(identity, minor) || !rtc_key_integer(identity, HIP_VERSION) ||
        !rtc_key_integer(identity, runtime) || !rtc_key_integer(identity, driver) ||
        !rtc_key_string(identity, properties.gcnArchName) || !rtc_key_integer(identity, properties.warpSize) ||
        !rtc_key_string(identity, "--gpu-architecture=") || !rtc_key_string(identity, properties.gcnArchName) ||
        !rtc_key_string(identity, "-std=c++17") || !rtc_key_string(identity, "-O3") ||
        !rtc_key_string(identity, "-ffp-contract=off")) {
        return false;
    }

    GGML_LOG_DEBUG("HIPRTC fusion: target=%s wave=%d hiprtc=%d.%d header=%d runtime=%d driver=%d\n",
                   state.target.c_str(), properties.warpSize, major, minor, HIP_VERSION, runtime, driver);
    state.ready = true;
    return true;
}

static std::vector<char> rtc_compile(const std::string & architecture, const std::string & source, int count, int64_t source_us) {
    const int64_t start = ggml_time_us();
    if (source.size() > 64 * 1024) {
        GGML_LOG_WARN("HIPRTC fusion: source exceeds 64 KiB\n");
        return {};
    }

    const std::string target = "--gpu-architecture=" + architecture;
    const char * options[] = {target.c_str(), "-std=c++17", "-O3", "-ffp-contract=off"};
    rtc_compiler_program compiler;
    std::vector<char> log;
    auto check = [&](hiprtcResult error, const char * stage) {
        if (error == HIPRTC_SUCCESS) {
            return true;
        }

        GGML_LOG_WARN("HIPRTC fusion: %s %s: %s\n%s\n", architecture.c_str(), stage,
                      hiprtcGetErrorString(error), log.empty() ? "" : log.data());
        return false;
    };

    if (!check(hiprtcCreateProgram(&compiler.program, source.c_str(), "fusion.cpp", 0, nullptr, nullptr), "create")) {
        return {};
    }

    const auto result = hiprtcCompileProgram(compiler.program, 4, options);
    size_t size = 0;
    if (!check(hiprtcGetProgramLogSize(compiler.program, &size), "log size")) {
        return {};
    }

    if (size > 1024 * 1024) {
        GGML_LOG_WARN("HIPRTC fusion: compiler log unavailable: exceeds 1 MiB\n");
    } else if (size) {
        log.resize(size);
        if (!check(hiprtcGetProgramLog(compiler.program, log.data()), "log")) {
            return {};
        }
        log.back() = '\0';
    }

    if (!check(result, "compile") || !check(hiprtcGetCodeSize(compiler.program, &size), "code size")) {
        return {};
    }

    if (!size || size > 4 * 1024 * 1024) {
        GGML_LOG_WARN("HIPRTC fusion: code object exceeds size limit or is empty\n");
        return {};
    }

    std::vector<char> code(size);
    if (!check(hiprtcGetCode(compiler.program, code.data()), "code") ||
        !check(hiprtcDestroyProgram(&compiler.program), "destroy program")) {
        return {};
    }

    // HIPRTC does not clear the destroyed handle.
    compiler.program = nullptr;

    GGML_LOG_DEBUG("HIPRTC fusion: compiled %d ops for %s source_us=%lld compile_us=%lld code_bytes=%zu\n",
                   count, architecture.c_str(), (long long) source_us,
                   (long long) (ggml_time_us() - start), code.size());

    return code;
}

struct rtc_compile_job {
    std::shared_ptr<rtc_artifact> artifact;
    std::string architecture;
    std::string source;
    int count;
    int64_t source_us;
};

struct rtc_compiler {
    static constexpr size_t max_pending = 32;
    std::deque<rtc_compile_job> jobs;
    std::condition_variable changed;
    bool stopping = false;
    std::thread worker;

    rtc_compiler() : worker([this] { run(); }) {}

    ~rtc_compiler() {
        {
            std::lock_guard<std::mutex> lock(rtc_artifact_mutex);
            stopping = true;
            for (auto & job : jobs) {
                job.artifact->state.store(GGML_CUDA_RTC_COMPILE_FAILED, std::memory_order_release);
                job.artifact->changed.notify_all();
            }
            jobs.clear();
        }
        changed.notify_one();
        worker.join();
    }

    void run() {
        for (;;) {
            std::unique_lock<std::mutex> lock(rtc_artifact_mutex);
            changed.wait(lock, [&] { return stopping || !jobs.empty(); });
            if (stopping) {
                return;
            }

            auto job = std::move(jobs.front());
            jobs.pop_front();
            lock.unlock();

            std::vector<char> code;
            try {
                code = rtc_compile(job.architecture, job.source, job.count, job.source_us);
            } catch (const std::exception & error) {
                GGML_LOG_WARN("HIPRTC fusion: compiler setup: %s\n", error.what());
            } catch (...) {
                GGML_LOG_WARN("HIPRTC fusion: compiler setup: unknown exception\n");
            }

            lock.lock();
            if (!code.empty() && code.size() <= RTC_CACHE_BYTES - rtc_artifact_bytes) {
                rtc_artifact_bytes += code.size();
                job.artifact->code = std::move(code);
                job.artifact->state.store(GGML_CUDA_RTC_COMPILE_READY, std::memory_order_release);
            } else {
                job.artifact->state.store(GGML_CUDA_RTC_COMPILE_FAILED, std::memory_order_release);
            }
            lock.unlock();
            job.artifact->changed.notify_all();
        }
    }
};

static bool rtc_async() {
    static const bool enabled = !getenv("GGML_HIP_RTC_FUSION_ASYNC") || std::atoi(getenv("GGML_HIP_RTC_FUSION_ASYNC"));
    return enabled;
}

static std::shared_ptr<const rtc_artifact> rtc_artifact_get(const ggml_cuda_rtc_fusion & state,
                                                          const ggml_fusion_cache_key & key,
                                                          const ggml_fusion_program & program,
                                                          const ggml_fusion_schedule & schedule,
                                                          bool & retry) {
    std::unique_lock<std::mutex> lock(rtc_artifact_mutex);
    auto found = rtc_artifacts.find(key);
    std::shared_ptr<rtc_artifact> artifact;
    if (found != rtc_artifacts.end()) {
        artifact = found->second;
    } else {
        if (rtc_artifacts.size() >= 256 || rtc_artifact_bytes >= RTC_CACHE_BYTES) {
            return nullptr;
        }

        static rtc_compiler compiler;
        if (compiler.jobs.size() >= rtc_compiler::max_pending) {
            retry = true;
            return nullptr;
        }

        const int64_t start = ggml_time_us();
        auto source = program.reduction.value >= 0 ? rtc_emit_reduction(program, schedule) : rtc_emit_pointwise(program);
        const int64_t source_us = ggml_time_us() - start;
        artifact = std::make_shared<rtc_artifact>();
        rtc_artifacts.emplace(key, artifact);
        try {
            compiler.jobs.push_back({artifact, state.target, std::move(source), program.count, source_us});
        } catch (...) {
            rtc_artifacts.erase(key);
            throw;
        }
        compiler.changed.notify_one();
    }

    if (!rtc_async()) {
        artifact->changed.wait(lock, [&] {
            return artifact->state.load(std::memory_order_acquire) != GGML_CUDA_RTC_COMPILE_COMPILING;
        });
    }
    return artifact;
}

static bool rtc_force() {
    static const bool force = getenv("GGML_HIP_RTC_FUSION_FORCE") && std::atoi(getenv("GGML_HIP_RTC_FUSION_FORCE"));
    return force;
}

static std::shared_ptr<rtc_module> rtc_load(const ggml_cuda_rtc_fusion & state, const rtc_artifact & artifact,
                                           const ggml_fusion_program & program, const ggml_fusion_schedule & schedule) {
    ggml_cuda_rtc_setup_guard setup;
    const int64_t start = ggml_time_us();

    auto module = std::make_shared<rtc_module>();
    auto error = hipModuleLoadData(&module->module, artifact.code.data());
    const char * stage = "load";
    if (error == hipSuccess) {
        stage = "function";
        const char * name;
        if (program.reduction.value < 0) {
            name = "ggml_fused_elementwise";
        } else if (rtc_normalizes(program)) {
            name = "ggml_fused_normalize_rows";
        } else if (schedule.stages == 2) {
            name = "ggml_fused_reduce_partial";
        } else {
            name = "ggml_fused_reduce_rows";
        }

        error = hipModuleGetFunction(&module->function, module->module, name);
        if (error == hipSuccess && schedule.stages == 2) {
            error = hipModuleGetFunction(&module->finalize, module->module, "ggml_fused_reduce_finalize");
        }
    }

    if (error != hipSuccess) {
        GGML_LOG_WARN("HIPRTC fusion: %s %s: %s\n", state.target.c_str(), stage, hipGetErrorString(error));
        return nullptr;
    }

    for (auto function : {module->function, module->finalize}) {
        if (!function) {
            continue;
        }

        int registers = -1;
        int local = -1;
        int shared = -1;
        int threads = -1;
        int blocks = -1;

        const auto reg_error = hipFuncGetAttribute(&registers, HIP_FUNC_ATTRIBUTE_NUM_REGS, function);
        const auto local_error = hipFuncGetAttribute(&local, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function);
        const auto shared_error = hipFuncGetAttribute(&shared, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function);
        const auto thread_error = hipFuncGetAttribute(&threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
        const auto occupancy_error = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, function, schedule.threads, 0);
        if (reg_error != hipSuccess || local_error != hipSuccess || shared_error != hipSuccess ||
            thread_error != hipSuccess || occupancy_error != hipSuccess) {
            GGML_LOG_WARN("HIPRTC fusion: resource query unavailable; using static limits\n");
        }

        if (schedule.threads > uint32_t(state.max_threads) ||
            (thread_error == hipSuccess && threads < int(schedule.threads)) ||
            (shared_error == hipSuccess && (shared < 0 || size_t(shared) > state.max_shared)) ||
            (occupancy_error == hipSuccess && blocks <= 0)) {
            GGML_LOG_WARN("HIPRTC fusion: unsafe kernel resources\n");
            return nullptr;
        }

        GGML_LOG_DEBUG("HIPRTC fusion: loaded %s load_us=%lld registers=%d local=%d shared=%d max_threads=%d blocks=%d\n",
                       state.target.c_str(), (long long) (ggml_time_us() - start), registers, local, shared, threads, blocks);

        if (!rtc_force() && ((local_error == hipSuccess && local > 0) ||
            (program.reduction.value < 0 && occupancy_error == hipSuccess && blocks < 2))) {
            return nullptr;
        }
    }

    return module;
}


static bool rtc_profitable(const ggml_fusion_program & program, uint64_t n, uint64_t rows, int device) {
    if (rtc_force()) {
        return true;
    }

    if (program.reduction.value >= 0) {
        if (rtc_normalizes(program) && program.count < 3) {
            return false;
        }
        int producer = 0;
        bool silu = false;

        for (int i = 0; i < program.count; ++i) {
            if (program.values[i].domain == 1) {
                ++producer;
                silu |= program.values[i].op == GGML_OP_UNARY && program.values[i].unary == GGML_UNARY_OP_SILU;
            }
        }

        return program.reduction.axes != 1 || rows >= uint64_t(ggml_cuda_info().devices[device].nsm) ||
            (!silu && producer <= 4);
    }

    if (program.count < 2) {
        return false;
    }

    if (!program.indexed || program.count >= 3) {
        return true;
    }

    uint64_t bytes = 0;
    for (int i = 0; i < program.count; ++i) {
        bool stored = false;
        for (int o = 0; o < program.outputs; ++o) {
            stored |= program.roots[o] == i;
        }

        if (!stored) {
            const uint64_t size = ggml_type_size(program.values[i].type);
            if (n > (UINT64_MAX - bytes) / size) {
                return true;
            }

            bytes += n * size;
        }
    }

    return bytes >= 65536;
}

static bool rtc_select_schedule(const ggml_fusion_program & program, uint64_t n, int device, ggml_fusion_schedule & schedule) {
    schedule.kind = program.indexed ? GGML_FUSION_SCHEDULE_POINTWISE_INDEXED : GGML_FUSION_SCHEDULE_POINTWISE_FLAT;
    uint64_t columns = n;
    uint64_t rows = 1;

    if (program.reduction.value >= 0) {
        schedule.wave = ggml_cuda_info().devices[device].warp_size;
        if (schedule.wave != 32 && schedule.wave != 64) {
            return false;
        }

        if (program.reduction.axes == 1) {
            columns = program.ne[0];
            rows = n / columns;
        }

        schedule.kind = rtc_normalizes(program) ? GGML_FUSION_SCHEDULE_NORMALIZE_ROWS : GGML_FUSION_SCHEDULE_REDUCE_ROWS;
        schedule.threads = columns <= 256 ? schedule.wave : 256;

        if (program.reduction.axes == 15 && n > 4096) {
            schedule.kind = GGML_FUSION_SCHEDULE_REDUCE_ALL_TWO_STAGE;
            schedule.tile = 4096;
            schedule.stages = 2;
            schedule.scratch = 16384;
        }
    }

    return rtc_profitable(program, n, rows, device);
}

static int rtc_launch(ggml_backend_cuda_context & ctx, const ggml_fusion_program & program,
                      const ggml_fusion_schedule & schedule, ggml_cuda_rtc_binding & binding, const rtc_module & module) {
    auto & state = *ctx.rtc_fusion;

    if (program.reduction.value >= 0) {
        unsigned long long columns = program.reduction.axes == 1 ? uint64_t(program.ne[0]) : binding.n;
        unsigned long long rows = binding.n / columns;
        unsigned long long partials = (binding.n - 1) / 4096 + 1;

        void * args[3 + GGML_FUSION_MAX_OUTPUTS + GGML_FUSION_MAX_INPUTS + GGML_FUSION_MAX_PARAMS];
        const int first_stage = schedule.stages == 2 ? 1 : 0;
        const int last_stage = schedule.stages == 2 ? 2 : 0;

        for (int stage = first_stage; stage <= last_stage; ++stage) {
            int arg = 0;
            if (stage == 1) {
                args[arg++] = &state.partial;
            } else {
                for (int k = 0; k < program.outputs; ++k) {
                    args[arg++] = &binding.outputs[k];
                }
            }

            for (int k = 0; k < program.inputs; ++k) {
                if (!stage || program.accesses[k].domain == stage) {
                    args[arg++] = &binding.inputs[k];
                }
            }

            if (stage == 2) {
                args[arg++] = &state.partial;
                args[arg++] = &partials;
            } else if (stage == 1) {
                args[arg++] = &binding.n;
            } else {
                args[arg++] = &rows;
                args[arg++] = &columns;
            }

            for (int k = 0; k < program.params; ++k) {
                if (!stage || rtc_parameter_domain(program, k, stage)) {
                    args[arg++] = &binding.params[k];
                }
            }

            unsigned int blocks;
            if (stage == 1) {
                blocks = unsigned(partials);
            } else if (stage == 2) {
                blocks = 1;
            } else {
                blocks = unsigned(std::min(rows, 65535ULL));
            }

            const hipFunction_t function = stage == 2 ? module.finalize : module.function;
            CUDA_CHECK(hipModuleLaunchKernel(function, blocks, 1, 1, schedule.threads, 1, 1, 0, ctx.stream(), args, nullptr));
        }

        return program.count - 1;
    }

    void * args[1 + GGML_FUSION_MAX_OUTPUTS + GGML_FUSION_MAX_INPUTS + GGML_FUSION_MAX_PARAMS];
    int arg = 0;
    for (int k = 0; k < program.outputs; ++k) {
        args[arg++] = &binding.outputs[k];
    }

    for (int k = 0; k < program.inputs; ++k) {
        args[arg++] = &binding.inputs[k];
    }

    args[arg++] = &binding.n;
    for (int k = 0; k < program.params; ++k) {
        args[arg++] = &binding.params[k];
    }

    const unsigned int blocks = unsigned(std::min((binding.n - 1) / 256 + 1, 65535ULL));
    CUDA_CHECK(hipModuleLaunchKernel(module.function, blocks, 1, 1, 256, 1, 1, 0, ctx.stream(), args, nullptr));

    return program.count - 1;
}

int ggml_cuda_rtc_fusion_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int node_idx, bool allow_compile, int min_count) {
    if (!ggml_cuda_rtc_fusion_enabled() || ctx.curr_stream_no != 0 ||
        !ctx.stream_context().concurrent_events.empty()) {
        return 0;
    }

    if (!allow_compile) {
        if (!ctx.rtc_fusion || !ctx.rtc_fusion->capture_plan) {
            return 0;
        }

        auto & state = *ctx.rtc_fusion;
        const auto & actions = state.capture_plan->actions;
        while (state.capture_action < actions.size() && actions[state.capture_action].start < node_idx) {
            ++state.capture_action;
        }

        if (state.capture_action == actions.size() || actions[state.capture_action].start != node_idx) {
            return 0;
        }

        const size_t index = state.capture_action;
        const auto & action = actions[index];
        if (action.program.count < min_count) {
            return 0;
        }
        ++state.capture_action;
        return rtc_launch(ctx, action.program, action.schedule, state.bindings[index], *action.module);
    }

    if (ctx.rtc_fusion && node_idx < ctx.rtc_fusion->pending_until) {
        return 0;
    }

    ggml_fusion_region region;
    ggml_fusion_program program;
    ggml_cuda_rtc_binding binding;
    ggml_cgraph slice = *graph;

    for (;;) {
        if (!ggml_fusion_build(&slice, node_idx, region, program) || program.count < min_count) {
            return 0;
        }

        if (rtc_reduction_preferred(graph, region, program)) {
            return 0;
        }

        if (rtc_bind(graph, ctx.device, region, program, binding)) {
            break;
        }

        slice.n_nodes = node_idx + region.count - 1;
    }

    ggml_fusion_schedule schedule;
    if (!rtc_select_schedule(program, binding.n, ctx.device, schedule)) {
        return 0;
    }

    ggml_fusion_cache_key key;
    if (!rtc_make_key(program, schedule, key)) {
        return 0;
    }

    try {
        if ((!ctx.rtc_fusion || !ctx.rtc_fusion->initialization_attempted) && !rtc_initialize(ctx)) {
            return 0;
        }
    } catch (const std::exception & error) {
        GGML_LOG_WARN("HIPRTC fusion: initialization: %s\n", error.what());
        return 0;
    }

    auto & state = *ctx.rtc_fusion;
    if (!state.ready) {
        return 0;
    }

    if (state.identity.size > sizeof(key.bytes) - key.size) {
        return 0;
    }

    std::memcpy(key.bytes + key.size, state.identity.bytes, state.identity.size);
    key.size += state.identity.size;

    ggml_cuda_set_device(ctx.device);
    auto entry = state.cache.find(key);
    if (entry == state.cache.end()) {
        if (state.cache.size() >= 256 || state.code_bytes >= RTC_CACHE_BYTES) {
            return 0;
        }

        try {
            auto artifact = rtc_artifact_get(state, key, program, schedule, state.retry);
            if (!artifact) {
                state.pending_until = region.members[region.count - 1] + 1;
                return 0;
            }
            const auto status = artifact->state.load(std::memory_order_acquire);
            if (status == GGML_CUDA_RTC_COMPILE_COMPILING) {
                state.pending_until = region.members[region.count - 1] + 1;
                if (state.recording && std::find(state.pending.begin(), state.pending.end(), artifact) == state.pending.end()) {
                    if (!rtc_reserve(ctx, state.pending, state.pending.size() + 1)) {
                        state.recording = false;
                    } else {
                        state.pending.push_back(artifact);
                    }
                }
                return 0;
            }

            entry = state.cache.emplace(key, nullptr).first;
            if (status == GGML_CUDA_RTC_COMPILE_READY && artifact->code.size() <= RTC_CACHE_BYTES - state.code_bytes) {
                entry->second = rtc_load(state, *artifact, program, schedule);
                if (entry->second) {
                    state.code_bytes += artifact->code.size();
                }
            }
        } catch (const std::exception & error) {
            GGML_LOG_WARN("HIPRTC fusion: module setup: %s\n", error.what());
            return 0;
        }
    }

    if (!entry->second) {
        state.pending_until = region.members[region.count - 1] + 1;
        return 0;
    }

    if (schedule.stages == 2 && !state.scratch_attempted) {
        ggml_cuda_rtc_setup_guard setup;
        state.scratch_attempted = true;
        const auto error = hipMalloc(&state.partial, size_t(schedule.scratch));
        if (error != hipSuccess) {
            GGML_LOG_WARN("HIPRTC fusion: reduction scratch: %s\n", hipGetErrorString(error));
        }
    }

    if (schedule.stages == 2 && !state.partial) {
        return 0;
    }

    const int skipped = rtc_launch(ctx, program, schedule, binding, *entry->second);

    try {
        rtc_record_action(ctx, graph, region, program, schedule, key, entry->second);
    } catch (const std::exception & error) {
        GGML_LOG_WARN("HIPRTC fusion: capture plan: %s\n", error.what());
        state.recording = false;
        state.actions.clear();
    }

    return skipped;
}

void ggml_cuda_rtc_fusion_alloc_deps(const ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    if (!ggml_cuda_rtc_fusion_enabled()) {
        return;
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_fusion_region region;
        ggml_fusion_program program;
        if (!ggml_fusion_build(graph, i, region, program) || program.count < 3 || !rtc_normalizes(program) || rtc_reduction_preferred(graph, region, program)) {
            continue;
        }

        auto * until = graph->nodes[region.members[region.count - 1]];
        for (int k = 0; k < program.inputs; ++k) {
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(region.inputs[k]), until);
        }

        i += region.count - 1;
    }
}

void ggml_cuda_rtc_fusion_free(ggml_backend_cuda_context & ctx) {
    delete ctx.rtc_fusion;
    ctx.rtc_fusion = nullptr;
}
