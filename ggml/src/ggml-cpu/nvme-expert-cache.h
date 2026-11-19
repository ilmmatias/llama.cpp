#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

class nvme_expert_cache {
public:
    nvme_expert_cache(size_t ram_budget_bytes, uint32_t admit_window);
    ~nvme_expert_cache();

    nvme_expert_cache(const nvme_expert_cache &) = delete;
    nvme_expert_cache & operator=(const nvme_expert_cache &) = delete;

    bool is_valid() const;
    size_t memory_bytes() const;

    void register_tensor(const ggml_tensor * tensor, const char * file_path, uint64_t file_offset);
    void register_router(int layer_id, const ggml_tensor * gate_inp, const ggml_tensor * exp_probs_b);
    bool has_tensor(const ggml_tensor * tensor) const;

    // Decode / small-batch lifecycle (called by thread 0):
    // begin_op submits io_uring O_DIRECT reads for all L3-missing experts (across all parts of the layer)
    // and pins L2-resident experts. Returns true if this tensor is NVMe-backed.
    bool begin_op(const ggml_tensor * op, const uint64_t * l1_ready_masks);
    void end_op(const ggml_tensor * op);

    // Get pointer to expert data in RAM for op->src[0].
    // If wait == false: returns pointer if already ready in RAM (L2 hit or completed IO), else nullptr.
    // If wait == true:  blocks (thread 0 only!) until io_uring CQE completes, then returns pointer.
    const void * get_expert(const ggml_tensor * op, int32_t expert, bool wait);
    // Returns 1 if already repacked, 2 if in an L2 slot eligible for in-place parallel repack during PP, 0 otherwise.
    int is_expert_repacked(const ggml_tensor * op, int32_t expert);

    // Multi-token PP sliding-window sequential streaming (called by thread 0):
    // Submits io_uring reads for up to PP_WINDOW active experts ahead of active_idx and waits for active_experts[active_idx].
    const void * pp_prepare_expert(const ggml_tensor * op, const int32_t * active_experts, int active_idx, int n_active);
    void pp_release_expert(const ggml_tensor * op, int32_t expert);

    // Thread-safe copy of an expert slice for L1 VRAM background promotion workers.
    bool copy_expert_to_staging(const ggml_tensor * tensor, int32_t expert, void * dst, size_t bytes);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
