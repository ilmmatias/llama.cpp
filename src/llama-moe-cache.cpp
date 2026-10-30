#include "llama-moe-cache.h"

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<typename T>
T llama_moe_cache_proc(ggml_backend_reg_t reg, const char * name) {
    auto fn = reinterpret_cast<T>(ggml_backend_reg_get_proc_address(reg, name));
    if (fn == nullptr) {
        throw std::runtime_error(std::string("CPU backend does not provide ") + name);
    }
    return fn;
}

ggml_backend_t llama_moe_cache_cpu_backend(const std::vector<ggml_backend_t> & backends) {
    for (auto backend : backends) {
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return backend;
        }
    }
    return nullptr;
}

}

struct llama_moe_cache::impl {
    struct device {
        ggml_backend_dev_t dev;
        ggml_backend_buffer_type_t buft;
        std::vector<const ggml_tensor *> weights;
        size_t host_bytes = 0;
        size_t reservation = 0;
        uint32_t slots = 0;
    };

    std::vector<device> devices;
    ggml_backend_cpu_set_expert_cache_t set_cache = nullptr;
    ggml_backend_cpu_expert_cache_device_t cache_device = nullptr;
    ggml_backend_cpu_expert_cache_device_t cache_offload_device = nullptr;
    ggml_backend_cpu_expert_cache_prepare_t cache_prepare = nullptr;
    ggml_backend_cpu_expert_cache_copy_t cache_copy = nullptr;
    ggml_backend_cpu_expert_cache_memory_t cache_memory = nullptr;
    std::unique_ptr<ggml_backend_cpu_expert_cache, ggml_backend_cpu_expert_cache_free_t> root{nullptr, nullptr};

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, ggml_backend_t backend_cpu,
            uint32_t slots, uint32_t admit_window, uint32_t workers, size_t size) {
        // Fit estimation reserves the cache separately and must not start workers or allocate GPU buffers.
        if (no_alloc) {
            return;
        }
        if (slots != 0 && size != 0) {
            throw std::invalid_argument("expert cache slots and byte size are mutually exclusive");
        }
        if (workers == 0) {
            throw std::invalid_argument("expert cache requires at least one conversion worker");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("expert cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || model.hparams.n_expert_used_max() == 0) {
            throw std::runtime_error("expert cache requires a MoE model");
        }
        for (auto backend : backends) {
            auto dev = ggml_backend_get_device(backend);
            const auto type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                devices.push_back({dev, ggml_backend_dev_buffer_type(dev), {}});
            }
        }
        if (backend_cpu == nullptr || devices.empty()) {
            throw std::runtime_error("expert cache requires CPU and GPU backends");
        }

        size_t host_bytes = 0;
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto it = std::find_if(devices.begin(), devices.end(), [&](const device & d) {
                return d.dev == model.dev_layer(il);
            });
            if (it == devices.end()) {
                continue;
            }
            const auto & layer = model.layers[il];
            for (const auto * tensor : {layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
                if (llama_model_expert_cache_is_host_weight(tensor)) {
                    it->weights.push_back(tensor);
                    const size_t bytes = ggml_nbytes(tensor);
                    it->host_bytes += bytes;
                    host_bytes += bytes;
                }
            }
        }
        if (host_bytes == 0) {
            LLAMA_LOG_WARN("%s: no host-resident experts are assigned to a GPU, expert cache is disabled\n", __func__);
            return;
        }

        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        const auto create = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_create_t>(reg, "ggml_backend_cpu_expert_cache_create");
        const auto free_cache = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_free_t>(reg, "ggml_backend_cpu_expert_cache_free");
        const auto add_device = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_add_device_t>(reg, "ggml_backend_cpu_expert_cache_add_device");
        const auto register_weight = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_register_t>(reg, "ggml_backend_cpu_expert_cache_register");
        set_cache = llama_moe_cache_proc<ggml_backend_cpu_set_expert_cache_t>(reg, "ggml_backend_cpu_set_expert_cache");
        cache_device = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_device_t>(reg, "ggml_backend_cpu_expert_cache_device");
        cache_offload_device = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_device_t>(reg, "ggml_backend_cpu_expert_cache_offload_device");
        cache_prepare = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_prepare_t>(reg, "ggml_backend_cpu_expert_cache_prepare");
        cache_copy = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_copy_t>(reg, "ggml_backend_cpu_expert_cache_copy");
        cache_memory = llama_moe_cache_proc<ggml_backend_cpu_expert_cache_memory_t>(reg, "ggml_backend_cpu_expert_cache_memory");

        bool enabled = false;
        for (auto & d : devices) {
            if (d.weights.empty()) {
                continue;
            }
            d.slots = slots;
            if (size != 0) {
                // Split by the model's host expert placement, independent of transient free memory.
                const size_t budget = (size_t) ((long double) size * d.host_bytes / host_bytes);
                uint32_t lo = 0;
                uint32_t hi = model.hparams.n_expert;
                while (lo < hi) {
                    const uint32_t mid = lo + (hi - lo) / 2 + 1;
                    if (llama_model_expert_cache_get_cache_size(&model, d.dev, mid) <= budget) {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                d.slots = lo;
            }
            if (d.slots == 0) {
                continue;
            }
            d.reservation = llama_model_expert_cache_get_cache_size(&model, d.dev, d.slots) +
                llama_model_expert_cache_get_buffer_size(&model, d.dev);
            enabled = true;
            LLAMA_LOG_INFO("%s: %s: %u expert cache slots, reserving %.2f MiB including workspace\n",
                    __func__, ggml_backend_dev_name(d.dev), d.slots, d.reservation / (1024.0 * 1024.0));
        }
        if (!enabled) {
            throw std::runtime_error("expert cache byte size is too small to hold one expert slot");
        }

        root = decltype(root)(create(), free_cache);
        if (!root) {
            throw std::runtime_error("failed to create expert cache");
        }
        for (const auto & d : devices) {
            if (d.slots == 0) {
                continue;
            }
            if (!add_device(root.get(), d.slots, admit_window, workers, d.dev)) {
                throw std::runtime_error(std::string("failed to initialize expert cache on ") + ggml_backend_dev_name(d.dev));
            }
            for (const auto * tensor : d.weights) {
                register_weight(root.get(), d.dev, tensor);
            }
        }
    }
};

llama_moe_cache::llama_moe_cache(
        const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends,
        uint32_t slots, uint32_t admit_window, uint32_t workers, size_t size) :
    backend_cpu(llama_moe_cache_cpu_backend(backends)),
    pimpl(std::make_shared<impl>(model, no_alloc, backends, backend_cpu, slots, admit_window, workers, size)) {
    if (pimpl->root) {
        pimpl->set_cache(backend_cpu, pimpl->root.get());
    }
}

llama_moe_cache::llama_moe_cache(const llama_moe_cache & cache, ggml_backend_t backend_cpu) :
    backend_cpu(backend_cpu),
    owns_memory(false),
    pimpl(cache.pimpl) {
    if (pimpl->root) {
        pimpl->set_cache(backend_cpu, pimpl->root.get());
    }
}

llama_moe_cache::~llama_moe_cache() {
    if (pimpl->root) {
        pimpl->set_cache(backend_cpu, nullptr);
    }
}

ggml_backend_dev_t llama_moe_cache::device(const ggml_tensor * op) const {
    return pimpl->root ? pimpl->cache_device(pimpl->root.get(), op) : nullptr;
}

ggml_backend_dev_t llama_moe_cache::offload_device(const ggml_tensor * op) const {
    return pimpl->root ? pimpl->cache_offload_device(pimpl->root.get(), op) : nullptr;
}

void llama_moe_cache::prepare(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    if (pimpl->root) {
        pimpl->cache_prepare(pimpl->root.get(), sched, graph);
    }
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    return pimpl->root && pimpl->cache_copy(pimpl->root.get(), backend, src, dst);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    if (owns_memory && pimpl->root) {
        for (const auto & d : pimpl->devices) {
            if (d.slots != 0) {
                // The cache allocates lazily; count the planned total or the actual total, never both.
                result[d.buft] += std::max(d.reservation, pimpl->cache_memory(pimpl->root.get(), d.dev));
            }
        }
    }
    return result;
}
