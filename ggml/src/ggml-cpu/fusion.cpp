#include "fusion.h"
#include "ggml-fusion.h"
#include "ggml-impl.h"

#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

// Emit vector IR without relying on C++ auto-vectorization.
constexpr unsigned fusion_width          = 8;
constexpr size_t   fusion_cache_capacity = 128;

using fusion_kernel = void (*)(const float * const *, float * const *, const float *, uint64_t, uint64_t);

bool fusion_supported_op(ggml_op op, ggml_unary_op unary) {
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
            return true;
        case GGML_OP_UNARY:
            return unary == GGML_UNARY_OP_NEG || unary == GGML_UNARY_OP_ABS || unary == GGML_UNARY_OP_RELU;
        default:
            return false;
    }
}

bool fusion_supported(const ggml_fusion_program & program) {
    if (program.indexed || program.reduction.value >= 0) {
        return false;
    }

    for (int i = 0; i < program.inputs; ++i) {
        if (program.accesses[i].type != GGML_TYPE_F32) {
            return false;
        }
    }

    for (int i = 0; i < program.count; ++i) {
        const auto & value = program.values[i];

        if (value.type != GGML_TYPE_F32 || !fusion_supported_op(value.op, value.unary)) {
            return false;
        }
    }

    return true;
}

// Each iteration loads all inputs before storing, so exact in-place bindings are safe.
// Shifted overlaps can create dependencies between iterations.
bool fusion_bindings_valid(const ggml_fusion_region & region, const ggml_fusion_program & program) {
    if (!region.n || region.n > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }

    const size_t bytes = size_t(region.n) * sizeof(float);

    auto valid = [&](const ggml_tensor * tensor) {
        return tensor->type == GGML_TYPE_F32 &&
               ggml_is_contiguous(tensor) &&
               tensor->data &&
               uintptr_t(tensor->data) % alignof(float) == 0 &&
               uintptr_t(tensor->data) <= std::numeric_limits<uintptr_t>::max() - bytes &&
               uint64_t(ggml_nelements(tensor)) == region.n;
    };

    auto overlaps = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t x = uintptr_t(a->data);
        const uintptr_t y = uintptr_t(b->data);

        return x < y + bytes && y < x + bytes;
    };

    for (int i = 0; i < program.inputs; ++i) {
        if (!valid(region.inputs[i])) {
            return false;
        }
    }

    for (int i = 0; i < program.outputs; ++i) {
        if (!valid(region.outputs[i])) {
            return false;
        }

        for (int j = 0; j < program.inputs; ++j) {
            if (overlaps(region.outputs[i], region.inputs[j]) && region.outputs[i]->data != region.inputs[j]->data) {
                return false;
            }
        }

        for (int j = 0; j < i; ++j) {
            if (overlaps(region.outputs[i], region.outputs[j])) {
                return false;
            }
        }
    }

    return true;
}

llvm::PointerType * fusion_pointer_type(llvm::Type * element) {
#if LLVM_VERSION_MAJOR >= 15
    return llvm::PointerType::getUnqual(element->getContext());
#else
    return llvm::PointerType::getUnqual(element);
#endif
}

std::unique_ptr<llvm::Module> fusion_module(const ggml_fusion_program & program, llvm::LLVMContext & context, const llvm::DataLayout & layout, const std::string & name) {
    auto module = std::make_unique<llvm::Module>(name, context);
    module->setDataLayout(layout);

    llvm::IRBuilder<> b(context);

    auto * f32      = b.getFloatTy();
    auto * fp       = fusion_pointer_type(f32);
    auto * fpp      = fusion_pointer_type(fp);
    auto * type     = llvm::FunctionType::get(b.getVoidTy(), {fpp, fpp, fp, b.getInt64Ty(), b.getInt64Ty()}, false);
    auto * function = llvm::Function::Create(type, llvm::Function::ExternalLinkage, name, *module);

    auto * entry       = llvm::BasicBlock::Create(context, "entry",       function);
    auto * vector_test = llvm::BasicBlock::Create(context, "vector.test", function);
    auto * vector_body = llvm::BasicBlock::Create(context, "vector.body", function);
    auto * scalar_test = llvm::BasicBlock::Create(context, "scalar.test", function);
    auto * scalar_body = llvm::BasicBlock::Create(context, "scalar.body", function);
    auto * done        = llvm::BasicBlock::Create(context, "done",        function);

    b.SetInsertPoint(entry);

    std::vector<llvm::Value *> inputs, outputs, params;

    for (int i = 0; i < program.inputs; ++i) {
        inputs.push_back(b.CreateLoad(fp, b.CreateInBoundsGEP(fp, function->getArg(0), b.getInt64(i))));
    }

    for (int i = 0; i < program.outputs; ++i) {
        outputs.push_back(b.CreateLoad(fp, b.CreateInBoundsGEP(fp, function->getArg(1), b.getInt64(i))));
    }

    for (int i = 0; i < program.params; ++i) {
        params.push_back(b.CreateLoad(f32, b.CreateInBoundsGEP(f32, function->getArg(2), b.getInt64(i))));
    }

    auto emit = [&](llvm::Value * index, unsigned width) {
        llvm::Type * value_type = width == 1 ? f32 : llvm::FixedVectorType::get(f32, width);

        auto splat = [&](llvm::Value * value) -> llvm::Value * {
            return width == 1 ? value : b.CreateVectorSplat(width, value);
        };

        std::vector<llvm::Value *> sources;

        for (auto * input : inputs) {
            auto * pointer = b.CreateInBoundsGEP(f32, input, index);
            pointer = b.CreateBitCast(pointer, fusion_pointer_type(value_type));
            sources.push_back(b.CreateAlignedLoad(value_type, pointer, llvm::Align(alignof(float))));
        }

        std::vector<llvm::Value *> values;

        auto source = [&](int ref) { return ref < 0 ? sources[-1 - ref] : values[ref]; };

        for (int i = 0; i < program.count; ++i) {
            const auto & op = program.values[i];

            auto * x = source(op.src[0]);
            llvm::Value * value = nullptr;

            switch (op.op) {
                case GGML_OP_ADD: value = b.CreateFAdd(x, source(op.src[1])); break;
                case GGML_OP_SUB: value = b.CreateFSub(x, source(op.src[1])); break;
                case GGML_OP_MUL: value = b.CreateFMul(x, source(op.src[1])); break;
                case GGML_OP_DIV: value = b.CreateFDiv(x, source(op.src[1])); break;
                case GGML_OP_SCALE: {
                    auto * product = b.CreateFMul(x, splat(params[op.param]));
                    auto * bias    = params[op.param + 1];

                    // The CPU kernel omits the addition for either signed zero.
                    value = b.CreateSelect(b.CreateFCmpOEQ(bias, llvm::ConstantFP::get(f32, 0)), product, b.CreateFAdd(product, splat(bias)));
                } break;
                case GGML_OP_CLAMP: {
                    auto * lo    = splat(params[op.param]);
                    auto * hi    = splat(params[op.param + 1]);
                    auto * upper = b.CreateSelect(b.CreateFCmpOLT(x, hi), x, hi);

                    value = b.CreateSelect(b.CreateFCmpOGT(upper, lo), upper, lo);
                } break;
                case GGML_OP_SQR:  value = b.CreateFMul(x, x);                               break;
                case GGML_OP_SQRT: value = b.CreateUnaryIntrinsic(llvm::Intrinsic::sqrt, x); break;
                case GGML_OP_CONT: value = x;                                                break;
                case GGML_OP_UNARY:
                    switch (op.unary) {
                        case GGML_UNARY_OP_NEG: value = b.CreateFNeg(x);                                  break;
                        case GGML_UNARY_OP_ABS: value = b.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, x); break;
                        case GGML_UNARY_OP_RELU: {
                            auto * zero = llvm::Constant::getNullValue(value_type);

                            value = b.CreateSelect(b.CreateFCmpOGT(x, zero), x, zero);
                        } break;
                        default: GGML_ABORT("unsupported CPU fusion unary op");
                    }
                    break;
                default: GGML_ABORT("unsupported CPU fusion op");
            }

            values.push_back(value);
        }

        for (int i = 0; i < program.outputs; ++i) {
            auto * pointer = b.CreateInBoundsGEP(f32, outputs[i], index);
            pointer = b.CreateBitCast(pointer, fusion_pointer_type(value_type));
            b.CreateAlignedStore(values[program.roots[i]], pointer, llvm::Align(alignof(float)));
        }
    };

    b.CreateBr(vector_test);

    b.SetInsertPoint(vector_test);
    auto * vi = b.CreatePHI(b.getInt64Ty(), 2);
    vi->addIncoming(function->getArg(3), entry);
    b.CreateCondBr(b.CreateICmpUGE(b.CreateSub(function->getArg(4), vi), b.getInt64(fusion_width)), vector_body, scalar_test);

    b.SetInsertPoint(vector_body);
    emit(vi, fusion_width);
    auto * next_vector = b.CreateAdd(vi, b.getInt64(fusion_width));
    b.CreateBr(vector_test);
    vi->addIncoming(next_vector, vector_body);

    b.SetInsertPoint(scalar_test);
    auto * si = b.CreatePHI(b.getInt64Ty(), 2);
    si->addIncoming(vi, vector_test);
    b.CreateCondBr(b.CreateICmpULT(si, function->getArg(4)), scalar_body, done);

    b.SetInsertPoint(scalar_body);
    emit(si, 1);
    auto * next_scalar = b.CreateAdd(si, b.getInt64(1));
    b.CreateBr(scalar_test);
    si->addIncoming(next_scalar, scalar_body);

    b.SetInsertPoint(done);
    b.CreateRetVoid();

    return module;
}

class fusion_cache {
    std::mutex                           mutex;
    std::unique_ptr<llvm::orc::LLJIT>    jit;
    std::map<std::string, fusion_kernel> kernels;

public:
    fusion_cache() {
        if (llvm::InitializeNativeTarget() || llvm::InitializeNativeTargetAsmPrinter()) {
            return;
        }

        auto result = llvm::orc::LLJITBuilder().create();
        if (result) {
            jit = std::move(*result);
        } else {
            GGML_LOG_WARN("CPU fusion: %s\n", llvm::toString(result.takeError()).c_str());
        }
    }

    fusion_kernel get(const ggml_fusion_program & program) {
        ggml_fusion_schedule schedule;
        schedule.threads = 1;
        schedule.tile    = fusion_width;

        ggml_fusion_cache_key key;
        if (!ggml_fusion_make_key(program, schedule, key)) {
            return nullptr;
        }

        std::string bytes(reinterpret_cast<const char *>(key.bytes), key.size);
        std::lock_guard<std::mutex> lock(mutex);

        const auto found = kernels.find(bytes);
        if (found != kernels.end()) {
            return found->second;
        }

        // Bound executable memory; existing kernels remain valid for active plans.
        if (!jit || kernels.size() == fusion_cache_capacity) {
            return nullptr;
        }

        const std::string name = "ggml_cpu_fusion_" + std::to_string(kernels.size());
        auto inserted = kernels.emplace(std::move(bytes), nullptr).first;
        auto context  = std::make_unique<llvm::LLVMContext>();
        auto module   = fusion_module(program, *context, jit->getDataLayout(), name);

        if (llvm::verifyModule(*module, &llvm::errs())) {
            return nullptr;
        }

        llvm::LoopAnalysisManager     loops;
        llvm::FunctionAnalysisManager functions;
        llvm::CGSCCAnalysisManager    cgscc;
        llvm::ModuleAnalysisManager   modules;
        llvm::PassBuilder             passes;

        passes.registerModuleAnalyses(modules);
        passes.registerCGSCCAnalyses(cgscc);
        passes.registerFunctionAnalyses(functions);
        passes.registerLoopAnalyses(loops);
        passes.crossRegisterProxies(loops, functions, cgscc, modules);

        auto pipeline = passes.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
        pipeline.run(*module, modules);

        if (auto error = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(module), std::move(context)))) {
            GGML_LOG_WARN("CPU fusion: %s\n", llvm::toString(std::move(error)).c_str());
            return nullptr;
        }

        auto symbol = jit->lookup(name);
        if (!symbol) {
            GGML_LOG_WARN("CPU fusion: %s\n", llvm::toString(symbol.takeError()).c_str());
            return nullptr;
        }

#if LLVM_VERSION_MAJOR >= 15
        inserted->second = symbol->toPtr<fusion_kernel>();
#else
        inserted->second = reinterpret_cast<fusion_kernel>(uintptr_t(symbol->getAddress()));
#endif

        return inserted->second;
    }
};

struct fusion_binding {
    fusion_kernel kernel = nullptr;

    std::array<const float *, GGML_FUSION_MAX_INPUTS> inputs  = {};
    std::array<float *, GGML_FUSION_MAX_OUTPUTS>      outputs = {};
    std::array<float, GGML_FUSION_MAX_PARAMS>         params  = {};

    uint64_t n    = 0;
    int      skip = 0;
};

} // namespace

struct ggml_cpu_fusion_plan {
    std::vector<int>            nodes;
    std::vector<fusion_binding> bindings;
};

ggml_cpu_fusion_plan * ggml_cpu_fusion_plan_create(const ggml_cgraph * graph) {
    if (!graph || graph->n_nodes < 2) {
        return nullptr;
    }

    try {
        auto plan = std::make_unique<ggml_cpu_fusion_plan>();
        plan->nodes.resize(graph->n_nodes, -1);

        for (int i = 0; i < graph->n_nodes; ++i) {
            int end = i;
            while (end < graph->n_nodes && end - i < GGML_FUSION_MAX_VALUES) {
                const auto * node = graph->nodes[end];

                if (node->type != GGML_TYPE_F32 || !fusion_supported_op(node->op, node->op == GGML_OP_UNARY ? ggml_get_unary_op(node) : GGML_UNARY_OP_NEG)) {
                    break;
                }

                ++end;
            }

            if (end - i < 2) {
                continue;
            }

            // Keep use counts for consumers outside the candidate window.
            auto window = *graph;
            window.n_nodes = end;

            ggml_fusion_region region;
            ggml_fusion_program program;

            if (!ggml_fusion_build(&window, i, region, program) || !fusion_supported(program) || !fusion_bindings_valid(region, program)) {
                continue;
            }

            static fusion_cache cache;
            auto kernel = cache.get(program);

            if (!kernel) {
                continue;
            }

            fusion_binding binding;
            binding.kernel = kernel;
            binding.n      = region.n;
            binding.skip   = region.count - 1;

            for (int j = 0; j < program.inputs; ++j) {
                binding.inputs[j] = static_cast<const float *>(region.inputs[j]->data);
            }

            for (int j = 0; j < program.outputs; ++j) {
                binding.outputs[j] = static_cast<float *>(region.outputs[j]->data);
            }

            std::copy_n(region.params, program.params, binding.params.begin());
            plan->nodes[i] = int(plan->bindings.size());
            plan->bindings.push_back(binding);

            i += binding.skip;
        }

        return plan->bindings.empty() ? nullptr : plan.release();
    } catch (const std::exception & error) {
        GGML_LOG_WARN("CPU fusion: %s\n", error.what());
        return nullptr;
    }
}

void ggml_cpu_fusion_plan_free(ggml_cpu_fusion_plan * plan) {
    delete plan;
}

int ggml_cpu_fusion_compute(const ggml_cpu_fusion_plan * plan, int node, int ith, int nth) {
    if (!plan || node < 0 || size_t(node) >= plan->nodes.size() || plan->nodes[node] < 0) {
        return 0;
    }

    const auto & binding = plan->bindings[plan->nodes[node]];
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);

    const uint64_t chunk = binding.n / uint64_t(nth);
    const uint64_t extra = binding.n % uint64_t(nth);
    const uint64_t begin = chunk * uint64_t(ith) + std::min(uint64_t(ith), extra);
    const uint64_t end   = begin + chunk + (uint64_t(ith) < extra);

    if (begin != end) {
        binding.kernel(binding.inputs.data(), binding.outputs.data(), binding.params.data(), begin, end);
    }

    return binding.skip;
}
