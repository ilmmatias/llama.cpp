#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// Context-owned CPU-miss/GPU-hit cache, explicitly shared with same-model MTP contexts.
class llama_moe_cache {
public:
    llama_moe_cache(
            const llama_model & model,
            const std::vector<ggml_backend_t> & backends,
            uint32_t slots, uint32_t admit_window, uint32_t workers, size_t size);
    llama_moe_cache(const llama_moe_cache & cache, ggml_backend_t backend_cpu);
    llama_moe_cache(const llama_moe_cache &) = delete;
    llama_moe_cache & operator=(const llama_moe_cache &) = delete;
    ~llama_moe_cache();

    ggml_backend_dev_t device(const ggml_tensor * op) const;
    ggml_backend_dev_t offload_device(const ggml_tensor * op) const;
    void prepare(ggml_backend_sched_t sched, ggml_cgraph * graph);
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    ggml_backend_t backend_cpu = nullptr;
    bool owns_memory = true;
    std::shared_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
