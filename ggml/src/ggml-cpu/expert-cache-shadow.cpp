#include "expert-cache-shadow.h"

#include "ggml-alloc.h"
#include "quants.h"
#include "repack.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

enum class expert_cache_mode : uint32_t {
    shadow = 1,
    hybrid = 2,
};

enum class expert_cache_part {
    none,
    gate,
    up,
    gate_up,
    down,
};

static expert_cache_part expert_cache_parse_tensor(const char * name, int & layer) {
    int consumed = 0;
    layer = -1;

#define MATCH_EXPERT_CACHE_PART(suffix, value) \
    do { \
        consumed = 0; \
        if (sscanf(name, "blk.%d." suffix "%n", &layer, &consumed) == 1 && name[consumed] == '\0') { \
            return value; \
        } \
    } while (0)

    MATCH_EXPERT_CACHE_PART("ffn_gate_exps.weight",    expert_cache_part::gate);
    MATCH_EXPERT_CACHE_PART("ffn_up_exps.weight",      expert_cache_part::up);
    MATCH_EXPERT_CACHE_PART("ffn_gate_up_exps.weight", expert_cache_part::gate_up);
    MATCH_EXPERT_CACHE_PART("ffn_down_exps.weight",    expert_cache_part::down);

#undef MATCH_EXPERT_CACHE_PART

    layer = -1;
    return expert_cache_part::none;
}

struct expert_cache_hybrid_template {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;
    bool input_q8 = false;
};

struct expert_cache_layer {
    const ggml_tensor * gate    = nullptr;
    const ggml_tensor * up      = nullptr;
    const ggml_tensor * gate_up = nullptr;
    const ggml_tensor * down    = nullptr;

    std::vector<const ggml_tensor *> parts;
    std::vector<ggml_tensor *> cache_parts;
    std::vector<size_t> part_bytes;

    ggml_tensor * shadow_cache_tensor = nullptr;

    ggml_tensor * cache_gate = nullptr;
    ggml_tensor * cache_up   = nullptr;
    ggml_tensor * cache_gate_up = nullptr;
    ggml_tensor * cache_down = nullptr;

    int64_t n_expert = 0;
    size_t bundle_bytes = 0;

    ggml_context * cache_ctx = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;

    std::vector<int32_t> expert_to_slot;
    std::vector<int32_t> slot_to_expert;
    std::vector<uint64_t> slot_stamp;
    std::vector<uint64_t> slot_generation;
    std::vector<uint8_t> slot_ready;
    std::vector<int64_t> last_seen;

    uint64_t stamp = 0;
    bool disabled = false;

    // Hybrid execution is deliberately structural rather than architecture
    // specific. v1 accepts a direct pair of gate/up MUL_MAT_ID nodes feeding a
    // GLU node which then feeds the down MUL_MAT_ID.
    bool hybrid_checked = false;
    bool hybrid_compatible = false;
    int64_t n_expert_used = 0;
    int64_t input_dim = 0;
    int64_t ffn_dim = 0;
    int64_t output_dim = 0;
    enum ggml_glu_op glu_op = GGML_GLU_OP_COUNT;
    std::array<uint8_t, GGML_MAX_OP_PARAMS> glu_params = {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> gate_params = {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> up_params = {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> down_params = {};

    std::vector<std::unique_ptr<expert_cache_hybrid_template>> hybrid_templates;
    std::vector<uint8_t> hybrid_template_failed;

    // One decode token is active from the first gate/up projection until down
    // finishes. The ready set is frozen here so an admission that completes in
    // the middle of the token only becomes visible on the next token.
    bool active = false;
    const ggml_tensor * active_ids = nullptr;
    uint64_t active_mapped_mask = 0;
    uint64_t active_policy_ready_mask = 0;
    uint64_t active_ready_mask = 0;
    std::vector<int32_t> active_slots;
    std::vector<int32_t> active_route_positions;
    std::vector<int32_t> active_route_slots;
    int active_hits = 0;
    bool active_gpu_launched = false;
    std::chrono::steady_clock::time_point active_launch_time;
};

struct expert_cache_model {
    std::unordered_map<int, std::unique_ptr<expert_cache_layer>> layers;
    int last_layer = -1;
    int64_t token = -1;
};

struct expert_cache_request {
    const void * model_key;
    int layer;
    int32_t expert;
    int32_t slot;
    uint64_t generation;
};


struct expert_cache_converted_request {
    expert_cache_request req;
    uint32_t staging_index;
};

struct expert_cache_staging {
    ggml_backend_buffer_t buffer = nullptr;
    void * ptr = nullptr;
    size_t size = 0;
    std::vector<uint8_t> fallback;

    std::mutex mutex;
    std::condition_variable cv;
    bool available = true;
};

static bool is_cpu_repack_tensor(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->buffer != nullptr &&
        ggml_backend_buffer_get_type(tensor->buffer) == ggml_backend_cpu_repack_buffer_type();
}

static constexpr uint16_t znq3_spread4(uint16_t x) {
    return (uint16_t) (((x & 0x1u) << 0) |
                       ((x & 0x2u) << 2) |
                       ((x & 0x4u) << 4) |
                       ((x & 0x8u) << 6));
}

static constexpr uint16_t znq3_spread4_table[16] = {
    znq3_spread4(0x0), znq3_spread4(0x1), znq3_spread4(0x2), znq3_spread4(0x3),
    znq3_spread4(0x4), znq3_spread4(0x5), znq3_spread4(0x6), znq3_spread4(0x7),
    znq3_spread4(0x8), znq3_spread4(0x9), znq3_spread4(0xa), znq3_spread4(0xb),
    znq3_spread4(0xc), znq3_spread4(0xd), znq3_spread4(0xe), znq3_spread4(0xf),
};

static inline uint16_t znq3_pack_four_codes(const block_znq3x8 & in, int group, int row) {
    const int shift = 4*row;
    const uint32_t p0 = (in.planes[group][0] >> shift) & 0xfu;
    const uint32_t p1 = (in.planes[group][1] >> shift) & 0xfu;
    const uint32_t p2 = (in.planes[group][2] >> shift) & 0xfu;
    return (uint16_t) (znq3_spread4_table[p0] |
                       (znq3_spread4_table[p1] << 1) |
                       (znq3_spread4_table[p2] << 2));
}

static bool unpack_repacked_znq_expert(const ggml_tensor * tensor, int32_t expert, void * dst, size_t dst_size) {
    if (tensor == nullptr || tensor->data == nullptr || expert < 0 || expert >= tensor->ne[2] ||
        tensor->ne[0] % QK_ZNQ != 0 || tensor->ne[1] % 8 != 0) {
        return false;
    }

    const int64_t nblocks = tensor->ne[0] / QK_ZNQ;
    const int64_t row_groups = tensor->ne[1] / 8;
    const uint8_t * src_base = (const uint8_t *) tensor->data + (size_t) expert * tensor->nb[2];

    switch (tensor->type) {
        case GGML_TYPE_ZNQ2: {
            if (dst_size != (size_t) tensor->ne[1] * (size_t) nblocks * sizeof(block_znq2)) {
                return false;
            }
            const auto * src = (const block_znq2x8 *) src_base;
            auto * out = (block_znq2 *) dst;
            for (int64_t rg = 0; rg < row_groups; ++rg) {
                for (int64_t ib = 0; ib < nblocks; ++ib) {
                    const auto & in = src[rg*nblocks + ib];
                    for (int r = 0; r < 8; ++r) {
                        auto & b = out[(rg*8 + r)*nblocks + ib];
                        b.d = in.d[r];
                        b.books = in.books[r];
                        for (int g = 0; g < QK_ZNQ/4; ++g) {
                            b.qs[g] = in.qs[g][r];
                        }
                    }
                }
            }
            return true;
        }
        case GGML_TYPE_ZNQ3: {
            if (dst_size != (size_t) tensor->ne[1] * (size_t) nblocks * sizeof(block_znq3)) {
                return false;
            }
            const auto * src = (const block_znq3x8 *) src_base;
            auto * out = (block_znq3 *) dst;
            for (int64_t rg = 0; rg < row_groups; ++rg) {
                for (int64_t ib = 0; ib < nblocks; ++ib) {
                    const auto & in = src[rg*nblocks + ib];
                    for (int r = 0; r < 8; ++r) {
                        auto & b = out[(rg*8 + r)*nblocks + ib];
                        b.d = in.d[r];
                        b.books = in.books[r];

                        // CPU_REPACK transposes eight native ZNQ3 rows into
                        // three 32-bit bit-planes for each four-weight group.
                        // Rebuild two 12-bit groups at a time, yielding exactly
                        // three native packed bytes. This is ~4x less scalar
                        // bit work than reconstructing all 32 codes one by one.
                        for (int pair = 0; pair < QK_ZNQ/8; ++pair) {
                            const uint32_t packed =
                                (uint32_t) znq3_pack_four_codes(in, 2*pair + 0, r) |
                                ((uint32_t) znq3_pack_four_codes(in, 2*pair + 1, r) << 12);
                            b.qs[3*pair + 0] = (uint8_t) (packed >> 0);
                            b.qs[3*pair + 1] = (uint8_t) (packed >> 8);
                            b.qs[3*pair + 2] = (uint8_t) (packed >> 16);
                        }
                    }
                }
            }
            return true;
        }
        case GGML_TYPE_ZNQ4: {
            if (dst_size != (size_t) tensor->ne[1] * (size_t) nblocks * sizeof(block_znq4)) {
                return false;
            }
            const auto * src = (const block_znq4x8 *) src_base;
            auto * out = (block_znq4 *) dst;
            for (int64_t rg = 0; rg < row_groups; ++rg) {
                for (int64_t ib = 0; ib < nblocks; ++ib) {
                    const auto & in = src[rg*nblocks + ib];
                    for (int r = 0; r < 8; ++r) {
                        auto & b = out[(rg*8 + r)*nblocks + ib];
                        b.d = in.d[r];
                        b.books = in.books[r];
                        for (int g = 0; g < QK_ZNQ/4; ++g) {
                            memcpy(b.qs + 2*g, in.qs + 16*g + 2*r, 2);
                        }
                    }
                }
            }
            return true;
        }
        default:
            return false;
    }
}

static bool copy_native_expert(const ggml_tensor * tensor, int32_t expert, void * dst, size_t bytes) {
    if (tensor == nullptr || tensor->data == nullptr || expert < 0 || expert >= tensor->ne[2]) {
        return false;
    }
    if (is_cpu_repack_tensor(tensor)) {
        return unpack_repacked_znq_expert(tensor, expert, dst, bytes);
    }
    memcpy(dst, (const uint8_t *) tensor->data + (size_t) expert * tensor->nb[2], bytes);
    return true;
}

class expert_cache {
public:
    expert_cache(expert_cache_mode mode, uint32_t slots, uint32_t admit_window, uint32_t convert_workers, bool print_stats, ggml_backend_dev_t device) :
        mode(mode), slots(slots), admit_window(admit_window), convert_workers(std::max<uint32_t>(1, convert_workers)),
        print_stats(print_stats), device(device) {
        upload_backend = ggml_backend_dev_init(device, nullptr);
        if (mode == expert_cache_mode::hybrid) {
            compute_backend = ggml_backend_dev_init(device, nullptr);
        }
        device_buft = ggml_backend_dev_buffer_type(device);
        host_buft = ggml_backend_dev_host_buffer_type(device);

        if (upload_backend == nullptr || device_buft == nullptr ||
            (mode == expert_cache_mode::hybrid && compute_backend == nullptr)) {
            fprintf(stderr, "%s: failed to initialize %s expert cache on %s\n", __func__, mode_name(), ggml_backend_dev_name(device));
            valid = false;
            return;
        }

        valid = true;
        if (mode == expert_cache_mode::shadow) {
            worker = std::thread([this]() { worker_main_shadow(); });
        } else {
            hybrid_staging.reserve(this->convert_workers);
            convert_threads.reserve(this->convert_workers);
            for (uint32_t i = 0; i < this->convert_workers; ++i) {
                hybrid_staging.push_back(std::make_unique<expert_cache_staging>());
                convert_threads.emplace_back([this, i]() { worker_main_hybrid_convert(i); });
            }
            upload_thread = std::thread([this]() { worker_main_hybrid_upload(); });
        }

        if (print_stats) {
            fprintf(stderr, "expert_cache: enabled on %s: mode=%s slots/layer=%u admit_window=%u converters=%u\n",
                    ggml_backend_dev_name(device), mode_name(), slots, admit_window,
                    mode == expert_cache_mode::hybrid ? this->convert_workers : 1);
        }
    }

    ~expert_cache() {
        if (valid) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                stop = true;
            }
            queue_cv.notify_all();

            if (mode == expert_cache_mode::shadow) {
                if (worker.joinable()) {
                    worker.join();
                }
            } else {
                for (auto & t : convert_threads) {
                    if (t.joinable()) {
                        t.join();
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(converted_mutex);
                    converters_done = true;
                }
                converted_cv.notify_all();
                if (upload_thread.joinable()) {
                    upload_thread.join();
                }
            }
        }

        if (compute_backend != nullptr) {
            ggml_backend_synchronize(compute_backend);
        }

        const uint64_t tokens = total_tokens_locked();
        const double hit_pct = selections ? 100.0 * (double) simulated_hits / (double) selections : 0.0;
        const double ready_pct = selections ? 100.0 * (double) ready_hits / (double) selections : 0.0;
        const double pending_pct = selections ? 100.0 * (double) pending_hits / (double) selections : 0.0;
        const double cache_mib = (double) allocated_bytes / (1024.0 * 1024.0);
        const double upload_mib = (double) upload_bytes / (1024.0 * 1024.0);
        const double upload_mib_per_token = tokens ? upload_mib / (double) tokens : 0.0;
        const double worker_gib_s = upload_us ? ((double) upload_bytes / (1024.0 * 1024.0 * 1024.0)) / ((double) upload_us / 1.0e6) : 0.0;
        const uint64_t converted_bytes = conversion_bytes.load(std::memory_order_relaxed);
        const uint64_t converted_us = conversion_us.load(std::memory_order_relaxed);
        const double convert_gib_s = converted_us ? ((double) converted_bytes / (1024.0 * 1024.0 * 1024.0)) / ((double) converted_us / 1.0e6) : 0.0;
        const double wall_s = route_last_us > route_first_us ? (double) (route_last_us - route_first_us) / 1.0e6 : 0.0;
        const double wall_gib_s = wall_s > 0 ? ((double) upload_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s : 0.0;

        if (print_stats && (selections || allocated_bytes)) {
            fprintf(stderr,
                    "~expert_cache: mode=%s cache=%.1f MiB tokens=%" PRIu64 " selections=%" PRIu64
                    " simulated_hit=%.2f%% ready_hit=%.2f%% pending_hit=%.2f%% admissions=%" PRIu64 "\n",
                    mode_name(), cache_mib, tokens, selections, hit_pct, ready_pct, pending_pct, admissions);
            if (mode == expert_cache_mode::shadow) {
                fprintf(stderr,
                        "~expert_cache: uploads=%" PRIu64 " copies=%" PRIu64 " stale=%" PRIu64 " upload=%.1f MiB (%.2f MiB/token) "
                        "worker=%.2f GiB/s wall=%.2f GiB/s max_queue=%zu\n",
                        uploads, upload_ops, stale_requests.load(std::memory_order_relaxed), upload_mib, upload_mib_per_token,
                        worker_gib_s, wall_gib_s, max_queue);
            } else {
                fprintf(stderr,
                        "~expert_cache: uploads=%" PRIu64 " copies=%" PRIu64 " stale=%" PRIu64 " upload=%.1f MiB (%.2f MiB/token) "
                        "h2d=%.2f GiB/s convert=%.2f GiB/s/thread-eq converters=%u wall=%.2f GiB/s max_queue=%zu max_converted=%zu\n",
                        uploads, upload_ops, stale_requests.load(std::memory_order_relaxed), upload_mib, upload_mib_per_token,
                        worker_gib_s, convert_gib_s, convert_workers, wall_gib_s, max_queue, max_converted_queue);
            }
        }
        if (print_stats && mode == expert_cache_mode::hybrid && (hybrid_launches || hybrid_compatible_layers)) {
            const double avg_routes = hybrid_launches ? (double) hybrid_routes / (double) hybrid_launches : 0.0;
            const double wait_ms_per_launch = hybrid_launches ? (double) hybrid_wait_us / 1000.0 / (double) hybrid_launches : 0.0;
            const double overlap_ms_per_launch = hybrid_launches ? (double) hybrid_elapsed_us / 1000.0 / (double) hybrid_launches : 0.0;
            const double preq_us_per_input = hybrid_preq_inputs ? (double) hybrid_preq_us / (double) hybrid_preq_inputs : 0.0;
            fprintf(stderr,
                    "~expert_cache: hybrid_layers=%" PRIu64 " launches=%" PRIu64 " routes=%" PRIu64
                    " routes/launch=%.2f sync_wait=%.3f ms/launch launch_to_join=%.3f ms/launch fallbacks=%" PRIu64
                    " preq8=%" PRIu64 " q8_cpu=%.2f us/input\n",
                    hybrid_compatible_layers, hybrid_launches, hybrid_routes, avg_routes,
                    wait_ms_per_launch, overlap_ms_per_launch, hybrid_fallbacks, hybrid_preq_inputs, preq_us_per_input);
        }

        if (staging_buffer != nullptr) {
            ggml_backend_buffer_free(staging_buffer);
        }
        for (auto & staging : hybrid_staging) {
            if (staging && staging->buffer != nullptr) {
                ggml_backend_buffer_free(staging->buffer);
                staging->buffer = nullptr;
            }
        }
        if (hybrid_input_q8_buffer != nullptr) {
            ggml_backend_buffer_free(hybrid_input_q8_buffer);
        }
        if (hybrid_output_buffer != nullptr) {
            ggml_backend_buffer_free(hybrid_output_buffer);
        }
        for (auto & model_it : models) {
            for (auto & layer_it : model_it.second.layers) {
                free_layer(*layer_it.second);
            }
        }
        if (compute_backend != nullptr) {
            ggml_backend_free(compute_backend);
        }
        if (upload_backend != nullptr) {
            ggml_backend_free(upload_backend);
        }
    }

    void route(const ggml_tensor * weights, const ggml_tensor * ids) {
        if (!valid || weights == nullptr || ids == nullptr || ids->type != GGML_TYPE_I32) {
            return;
        }

        int layer_id = -1;
        const auto part = expert_cache_parse_tensor(weights->name, layer_id);
        if (part == expert_cache_part::none || layer_id < 0) {
            return;
        }

        const void * model_key = weights->buffer != nullptr ? (const void *) weights->buffer : weights->data;
        if (model_key == nullptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(state_mutex);
        auto & model = models[model_key];
        auto & layer = get_layer(model, layer_id);
        register_part(layer, part, weights);

        // Gate/up execute before down. Down is the single policy observation
        // point, avoiding three counts for the same route set.
        if (part != expert_cache_part::down) {
            return;
        }
        if (mode == expert_cache_mode::hybrid) {
            // Hybrid structure is validated from the live decode graph before
            // any cache VRAM is allocated. Policy/admission is then performed
            // only after the down projection joins.
            return;
        }
        if (!prepare_layer(layer, layer_id)) {
            return;
        }

        if (!is_decode_ids(ids)) {
            return;
        }
        observe_shadow_locked(model_key, model, layer, layer_id, ids);
    }

    uint64_t hybrid_begin(ggml_tensor * op) {
        if (!valid || mode != expert_cache_mode::hybrid || op == nullptr || op->op != GGML_OP_MUL_MAT_ID ||
            op->src[0] == nullptr || op->src[2] == nullptr || !is_cpu_repack_tensor(op->src[0])) {
            return 0;
        }

        int layer_id = -1;
        const auto part = expert_cache_parse_tensor(op->src[0]->name, layer_id);
        if (part == expert_cache_part::none || layer_id < 0 || !is_decode_ids(op->src[2]) || op->src[2]->ne[0] > 64) {
            return 0;
        }

        const void * model_key = op->src[0]->buffer != nullptr ? (const void *) op->src[0]->buffer : op->src[0]->data;
        if (model_key == nullptr) {
            return 0;
        }

        std::lock_guard<std::mutex> lock(state_mutex);
        auto mit = models.find(model_key);
        if (mit == models.end()) {
            return 0;
        }
        auto lit = mit->second.layers.find(layer_id);
        if (lit == mit->second.layers.end()) {
            return 0;
        }
        auto & layer = *lit->second;
        register_part(layer, part, op->src[0]);

        if (part == expert_cache_part::down && !layer.hybrid_checked) {
            detect_hybrid_locked(layer, layer_id, op);
            if (!layer.hybrid_compatible || !prepare_layer(layer, layer_id)) {
                layer.hybrid_compatible = false;
                return 0;
            }
            // Gate/up for this token have already executed, so the first token
            // after structural discovery stays fully on CPU.
            start_active_locked(layer, op->src[2], /*allow_gpu=*/ false, op);
            return 0;
        }
        if (!layer.hybrid_compatible || layer.cache_buffer == nullptr) {
            return 0;
        }

        if (!layer.active) {
            if (part == expert_cache_part::down) {
                // This should only happen after an unexpected graph/order change.
                ++hybrid_fallbacks;
                start_active_locked(layer, op->src[2], /*allow_gpu=*/ false, op);
                return 0;
            }
            start_active_locked(layer, op->src[2], /*allow_gpu=*/ true, op);
        } else if (layer.active_ids != op->src[2]) {
            ++hybrid_fallbacks;
            return 0;
        }

        return layer.active_ready_mask;
    }

    void hybrid_end(ggml_tensor * op) {
        if (!valid || mode != expert_cache_mode::hybrid || op == nullptr || op->op != GGML_OP_MUL_MAT_ID ||
            op->src[0] == nullptr || op->src[2] == nullptr) {
            return;
        }

        int layer_id = -1;
        const auto part = expert_cache_parse_tensor(op->src[0]->name, layer_id);
        if (part != expert_cache_part::down || layer_id < 0 || !is_decode_ids(op->src[2])) {
            return;
        }
        const void * model_key = op->src[0]->buffer != nullptr ? (const void *) op->src[0]->buffer : op->src[0]->data;
        if (model_key == nullptr) {
            return;
        }

        expert_cache_layer * layer_ptr = nullptr;
        expert_cache_model * model_ptr = nullptr;
        bool launched = false;
        int hit_count = 0;
        std::vector<int32_t> route_positions;
        size_t output_row_bytes = 0;
        std::chrono::steady_clock::time_point launch_time;

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            auto mit = models.find(model_key);
            if (mit == models.end()) {
                return;
            }
            auto lit = mit->second.layers.find(layer_id);
            if (lit == mit->second.layers.end()) {
                return;
            }
            layer_ptr = lit->second.get();
            model_ptr = &mit->second;
            auto & layer = *layer_ptr;
            if (!layer.hybrid_compatible || !layer.active || layer.active_ids != op->src[2]) {
                return;
            }
            launched = layer.active_gpu_launched;
            hit_count = layer.active_hits;
            route_positions = layer.active_route_positions;
            output_row_bytes = (size_t) layer.output_dim * sizeof(float);
            launch_time = layer.active_launch_time;
        }

        if (launched) {
            const auto wait0 = std::chrono::steady_clock::now();
            ggml_backend_synchronize(compute_backend);
            const auto wait1 = std::chrono::steady_clock::now();
            hybrid_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(wait1 - wait0).count();
            hybrid_elapsed_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(wait1 - launch_time).count();

            const uint8_t * src = (const uint8_t *) hybrid_output_ptr;
            for (int i = 0; i < hit_count; ++i) {
                const int32_t route = route_positions[(size_t) i];
                memcpy((uint8_t *) op->data + (size_t) route * op->nb[1],
                       src + (size_t) i * output_row_bytes, output_row_bytes);
            }
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            // model/layer storage cannot be erased while configured; use the
            // pointers obtained above after the GPU wait.
            observe_hybrid_locked(model_key, *model_ptr, *layer_ptr, layer_id, op->src[2]);
            clear_active_locked(*layer_ptr);
        }
    }

private:
    const char * mode_name() const {
        return mode == expert_cache_mode::hybrid ? "hybrid" : "shadow";
    }

    static bool is_decode_ids(const ggml_tensor * ids) {
        return ids != nullptr && ids->type == GGML_TYPE_I32 && ids->ne[1] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1;
    }

    static expert_cache_layer & get_layer(expert_cache_model & model, int layer_id) {
        auto & ptr = model.layers[layer_id];
        if (!ptr) {
            ptr = std::make_unique<expert_cache_layer>();
        }
        return *ptr;
    }

    static void register_part(expert_cache_layer & layer, expert_cache_part part, const ggml_tensor * weights) {
        switch (part) {
            case expert_cache_part::gate:    layer.gate    = weights; break;
            case expert_cache_part::up:      layer.up      = weights; break;
            case expert_cache_part::gate_up: layer.gate_up = weights; break;
            case expert_cache_part::down:    layer.down    = weights; break;
            case expert_cache_part::none:                         break;
        }
    }

    void free_template(expert_cache_hybrid_template & t) {
        if (t.buffer != nullptr) {
            ggml_backend_buffer_free(t.buffer);
            t.buffer = nullptr;
        }
        if (t.ctx != nullptr) {
            ggml_free(t.ctx);
            t.ctx = nullptr;
        }
    }

    void free_layer(expert_cache_layer & layer) {
        for (auto & t : layer.hybrid_templates) {
            if (t) {
                free_template(*t);
            }
        }
        if (layer.cache_buffer != nullptr) {
            ggml_backend_buffer_free(layer.cache_buffer);
            layer.cache_buffer = nullptr;
        }
        if (layer.cache_ctx != nullptr) {
            ggml_free(layer.cache_ctx);
            layer.cache_ctx = nullptr;
        }
    }

    bool prepare_layer(expert_cache_layer & layer, int layer_id) {
        if (layer.disabled) {
            return false;
        }
        if (layer.cache_buffer != nullptr) {
            return true;
        }

        layer.parts.clear();
        if (layer.gate_up != nullptr) {
            if (layer.down == nullptr) {
                return false;
            }
            layer.parts = { layer.gate_up, layer.down };
        } else {
            if (layer.gate == nullptr || layer.up == nullptr || layer.down == nullptr) {
                return false;
            }
            layer.parts = { layer.gate, layer.up, layer.down };
        }

        layer.n_expert = layer.down->ne[2];
        if (layer.n_expert <= 0) {
            layer.disabled = true;
            return false;
        }

        layer.bundle_bytes = 0;
        layer.part_bytes.clear();
        for (const auto * tensor : layer.parts) {
            if (tensor == nullptr || tensor->data == nullptr || tensor->ne[2] != layer.n_expert || tensor->ne[3] != 1) {
                layer.disabled = true;
                return false;
            }
            const size_t bytes = ggml_nbytes(tensor) / (size_t) layer.n_expert;
            if (bytes == 0 || tensor->nb[2] != bytes) {
                fprintf(stderr, "expert_cache: layer %d has non-contiguous expert slices in %s; disabling cache for layer\n",
                        layer_id, tensor->name);
                layer.disabled = true;
                return false;
            }
            if (is_cpu_repack_tensor(tensor) &&
                tensor->type != GGML_TYPE_ZNQ2 && tensor->type != GGML_TYPE_ZNQ3 && tensor->type != GGML_TYPE_ZNQ4) {
                fprintf(stderr, "expert_cache: layer %d CPU_REPACK type %s is not convertible to native GPU layout\n",
                        layer_id, ggml_type_name(tensor->type));
                layer.disabled = true;
                return false;
            }
            layer.part_bytes.push_back(bytes);
            layer.bundle_bytes += bytes;
        }

        if (mode == expert_cache_mode::shadow) {
            const size_t cache_bytes = layer.bundle_bytes * (size_t) slots;
            layer.cache_buffer = ggml_backend_buft_alloc_buffer(device_buft, cache_bytes);
            if (layer.cache_buffer == nullptr) {
                fprintf(stderr, "expert_cache: failed to allocate %.1f MiB for layer %d cache\n",
                        (double) cache_bytes / (1024.0 * 1024.0), layer_id);
                layer.disabled = true;
                return false;
            }
            ggml_backend_buffer_set_usage(layer.cache_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            ggml_init_params shadow_ctx_params = {
                /*.mem_size   =*/ 4 * ggml_tensor_overhead() + 1024,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            layer.cache_ctx = ggml_init(shadow_ctx_params);
            if (layer.cache_ctx == nullptr) {
                ggml_backend_buffer_free(layer.cache_buffer);
                layer.cache_buffer = nullptr;
                layer.disabled = true;
                return false;
            }

            layer.shadow_cache_tensor = ggml_new_tensor_1d(layer.cache_ctx, GGML_TYPE_I8, (int64_t) cache_bytes);
            if (ggml_backend_tensor_alloc(layer.cache_buffer, layer.shadow_cache_tensor,
                        ggml_backend_buffer_get_base(layer.cache_buffer)) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(layer.cache_buffer);
                layer.cache_buffer = nullptr;
                ggml_free(layer.cache_ctx);
                layer.cache_ctx = nullptr;
                layer.shadow_cache_tensor = nullptr;
                layer.disabled = true;
                return false;
            }

            layer.expert_to_slot.assign((size_t) layer.n_expert, -1);
            layer.slot_to_expert.assign(slots, -1);
            layer.slot_stamp.assign(slots, 0);
            layer.slot_generation.assign(slots, 0);
            layer.slot_ready.assign(slots, 0);
            layer.last_seen.assign((size_t) layer.n_expert, -1);

            allocated_bytes += cache_bytes;
            ++allocated_layers;
            return true;
        }

        ggml_init_params ctx_params = {
            /*.mem_size   =*/ (layer.parts.size() + 4) * ggml_tensor_overhead() + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        layer.cache_ctx = ggml_init(ctx_params);
        if (layer.cache_ctx == nullptr) {
            layer.disabled = true;
            return false;
        }

        layer.cache_parts.clear();
        for (const auto * src : layer.parts) {
            auto * cached = ggml_new_tensor_3d(layer.cache_ctx, src->type, src->ne[0], src->ne[1], (int64_t) slots);
            layer.cache_parts.push_back(cached);
        }

        const size_t alignment = ggml_backend_buft_get_alignment(device_buft);
        std::vector<size_t> offsets(layer.cache_parts.size());
        size_t cache_bytes = 0;
        for (size_t i = 0; i < layer.cache_parts.size(); ++i) {
            cache_bytes = GGML_PAD(cache_bytes, alignment);
            offsets[i] = cache_bytes;
            cache_bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(device_buft, layer.cache_parts[i]), alignment);
        }

        layer.cache_buffer = ggml_backend_buft_alloc_buffer(device_buft, cache_bytes);
        if (layer.cache_buffer == nullptr) {
            fprintf(stderr, "expert_cache: failed to allocate %.1f MiB for layer %d cache\n",
                    (double) cache_bytes / (1024.0 * 1024.0), layer_id);
            ggml_free(layer.cache_ctx);
            layer.cache_ctx = nullptr;
            layer.disabled = true;
            return false;
        }
        ggml_backend_buffer_set_usage(layer.cache_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        auto * base = (uint8_t *) ggml_backend_buffer_get_base(layer.cache_buffer);
        for (size_t i = 0; i < layer.cache_parts.size(); ++i) {
            if (ggml_backend_tensor_alloc(layer.cache_buffer, layer.cache_parts[i], base + offsets[i]) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "expert_cache: failed to bind layer %d cache tensor\n", layer_id);
                ggml_backend_buffer_free(layer.cache_buffer);
                layer.cache_buffer = nullptr;
                ggml_free(layer.cache_ctx);
                layer.cache_ctx = nullptr;
                layer.cache_parts.clear();
                layer.disabled = true;
                return false;
            }
        }

        if (layer.gate_up != nullptr) {
            layer.cache_gate_up = layer.cache_parts[0];
            layer.cache_down = layer.cache_parts[1];
        } else {
            layer.cache_gate = layer.cache_parts[0];
            layer.cache_up = layer.cache_parts[1];
            layer.cache_down = layer.cache_parts[2];
        }

        layer.expert_to_slot.assign((size_t) layer.n_expert, -1);
        layer.slot_to_expert.assign(slots, -1);
        layer.slot_stamp.assign(slots, 0);
        layer.slot_generation.assign(slots, 0);
        layer.slot_ready.assign(slots, 0);
        layer.last_seen.assign((size_t) layer.n_expert, -1);

        allocated_bytes += cache_bytes;
        ++allocated_layers;
        return true;
    }

    void detect_hybrid_locked(expert_cache_layer & layer, int layer_id, ggml_tensor * down_op) {
        layer.hybrid_checked = true;
        layer.hybrid_compatible = false;

        if (layer.gate_up != nullptr || layer.gate == nullptr || layer.up == nullptr || layer.down == nullptr) {
            return;
        }
        if (down_op->src[0] != layer.down || down_op->src[1] == nullptr || down_op->src[2] == nullptr) {
            return;
        }

        const ggml_tensor * act = down_op->src[1];
        if (act->op != GGML_OP_GLU || act->src[0] == nullptr || act->src[1] == nullptr) {
            return;
        }
        const ggml_tensor * gate_op = act->src[0];
        const ggml_tensor * up_op = act->src[1];
        if (gate_op->op != GGML_OP_MUL_MAT_ID || up_op->op != GGML_OP_MUL_MAT_ID ||
            gate_op->src[0] != layer.gate || up_op->src[0] != layer.up ||
            gate_op->src[1] == nullptr || gate_op->src[1] != up_op->src[1] ||
            gate_op->src[2] != down_op->src[2] || up_op->src[2] != down_op->src[2] ||
            gate_op->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate_op->src[1])) {
            return;
        }
        const ggml_tensor * input = gate_op->src[1];
        if (input->ne[1] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
            gate_op->ne[2] != 1 || up_op->ne[2] != 1 || down_op->ne[2] != 1) {
            return;
        }
        if (gate_op->ne[0] != up_op->ne[0] || gate_op->ne[1] != up_op->ne[1] ||
            act->ne[0] != gate_op->ne[0] || act->ne[1] != gate_op->ne[1] ||
            down_op->ne[1] != down_op->src[2]->ne[0]) {
            return;
        }

        layer.n_expert_used = down_op->src[2]->ne[0];
        if (layer.n_expert_used <= 0 || layer.n_expert_used > 64) {
            return;
        }
        layer.input_dim = input->ne[0];
        layer.ffn_dim = act->ne[0];
        layer.output_dim = down_op->ne[0];
        layer.glu_op = ggml_get_glu_op(act);
        memcpy(layer.glu_params.data(), act->op_params, GGML_MAX_OP_PARAMS);
        memcpy(layer.gate_params.data(), gate_op->op_params, GGML_MAX_OP_PARAMS);
        memcpy(layer.up_params.data(), up_op->op_params, GGML_MAX_OP_PARAMS);
        memcpy(layer.down_params.data(), down_op->op_params, GGML_MAX_OP_PARAMS);
        layer.hybrid_templates.resize((size_t) layer.n_expert_used + 1);
        layer.hybrid_template_failed.assign((size_t) layer.n_expert_used + 1, 0);
        layer.hybrid_compatible = true;
        ++hybrid_compatible_layers;
        (void) layer_id;
    }

    expert_cache_hybrid_template * get_hybrid_template_locked(expert_cache_layer & layer, int hit_count) {
        if (hit_count <= 0 || hit_count > layer.n_expert_used ||
            (size_t) hit_count >= layer.hybrid_templates.size() || layer.hybrid_template_failed[(size_t) hit_count]) {
            return nullptr;
        }
        if (layer.hybrid_templates[(size_t) hit_count]) {
            return layer.hybrid_templates[(size_t) hit_count].get();
        }

        auto t = std::make_unique<expert_cache_hybrid_template>();
        const size_t graph_size = 24;
        ggml_init_params params = {
            /*.mem_size   =*/ 32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false) + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        t->ctx = ggml_init(params);
        if (t->ctx == nullptr) {
            layer.hybrid_template_failed[(size_t) hit_count] = 1;
            return nullptr;
        }

        // Hybrid templates are single-token decode graphs. Quantize that one
        // activation on CPU and upload native Q8_1 so the GPU can enter MMVQ
        // directly instead of launching its own F32 -> Q8_1 conversion kernel.
        // This is independent of the number of routed cache hits: MUL_MAT_ID's
        // MMVQ batch limit applies to tokens (ne[2]), not expert routes.
        t->input_q8 = layer.input_dim % ggml_blck_size(GGML_TYPE_Q8_1) == 0;
        t->input = ggml_new_tensor_3d(t->ctx, t->input_q8 ? GGML_TYPE_Q8_1 : GGML_TYPE_F32, layer.input_dim, 1, 1);
        t->ids = ggml_new_tensor_2d(t->ctx, GGML_TYPE_I32, hit_count, 1);
        auto * gate = ggml_mul_mat_id(t->ctx, layer.cache_gate, t->input, t->ids);
        auto * up   = ggml_mul_mat_id(t->ctx, layer.cache_up,   t->input, t->ids);
        memcpy(gate->op_params, layer.gate_params.data(), GGML_MAX_OP_PARAMS);
        memcpy(up->op_params,   layer.up_params.data(),   GGML_MAX_OP_PARAMS);

        auto * act = ggml_glu_split(t->ctx, gate, up, layer.glu_op);
        memcpy(act->op_params, layer.glu_params.data(), GGML_MAX_OP_PARAMS);

        t->output = ggml_mul_mat_id(t->ctx, layer.cache_down, act, t->ids);
        memcpy(t->output->op_params, layer.down_params.data(), GGML_MAX_OP_PARAMS);
        t->graph = ggml_new_graph_custom(t->ctx, graph_size, false);
        ggml_build_forward_expand(t->graph, t->output);

        t->buffer = ggml_backend_alloc_ctx_tensors_from_buft(t->ctx, device_buft);
        if (t->buffer == nullptr) {
            free_template(*t);
            layer.hybrid_template_failed[(size_t) hit_count] = 1;
            ++hybrid_fallbacks;
            return nullptr;
        }
        ggml_backend_buffer_set_usage(t->buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        for (int i = 0; i < ggml_graph_n_nodes(t->graph); ++i) {
            if (!ggml_backend_supports_op(compute_backend, ggml_graph_node(t->graph, i))) {
                free_template(*t);
                layer.hybrid_template_failed[(size_t) hit_count] = 1;
                ++hybrid_fallbacks;
                return nullptr;
            }
        }

        auto * ret = t.get();
        layer.hybrid_templates[(size_t) hit_count] = std::move(t);
        return ret;
    }

    bool ensure_hybrid_q8_input(size_t size) {
        if (hybrid_input_q8_size >= size) {
            return true;
        }
        if (hybrid_input_q8_buffer != nullptr) {
            ggml_backend_buffer_free(hybrid_input_q8_buffer);
            hybrid_input_q8_buffer = nullptr;
            hybrid_input_q8_ptr = nullptr;
        }
        hybrid_input_q8_fallback.clear();

        if (host_buft != nullptr) {
            hybrid_input_q8_buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (hybrid_input_q8_buffer != nullptr) {
                hybrid_input_q8_ptr = ggml_backend_buffer_get_base(hybrid_input_q8_buffer);
                hybrid_input_q8_size = size;
                return true;
            }
        }
        hybrid_input_q8_fallback.resize(size);
        hybrid_input_q8_ptr = hybrid_input_q8_fallback.data();
        hybrid_input_q8_size = size;
        return hybrid_input_q8_ptr != nullptr;
    }

    bool ensure_hybrid_output(size_t size) {
        if (hybrid_output_size >= size) {
            return true;
        }
        if (hybrid_output_buffer != nullptr) {
            ggml_backend_buffer_free(hybrid_output_buffer);
            hybrid_output_buffer = nullptr;
            hybrid_output_ptr = nullptr;
        }
        hybrid_output_fallback.clear();

        if (host_buft != nullptr) {
            hybrid_output_buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (hybrid_output_buffer != nullptr) {
                hybrid_output_ptr = ggml_backend_buffer_get_base(hybrid_output_buffer);
                hybrid_output_size = size;
                return true;
            }
        }
        hybrid_output_fallback.resize(size);
        hybrid_output_ptr = hybrid_output_fallback.data();
        hybrid_output_size = size;
        return hybrid_output_ptr != nullptr;
    }

    void start_active_locked(expert_cache_layer & layer, const ggml_tensor * ids, bool allow_gpu, ggml_tensor * op) {
        layer.active = true;
        layer.active_ids = ids;
        layer.active_mapped_mask = 0;
        layer.active_policy_ready_mask = 0;
        layer.active_ready_mask = 0;
        layer.active_slots.clear();
        layer.active_route_positions.clear();
        layer.active_route_slots.assign((size_t) ids->ne[0], -1);
        layer.active_hits = 0;
        layer.active_gpu_launched = false;

        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0]);
            if (expert < 0 || expert >= layer.n_expert) {
                continue;
            }
            const int32_t slot = layer.expert_to_slot[(size_t) expert];
            if (slot >= 0) {
                layer.active_route_slots[(size_t) i] = slot;
                layer.active_mapped_mask |= UINT64_C(1) << i;
                if (layer.slot_ready[(size_t) slot]) {
                    layer.active_policy_ready_mask |= UINT64_C(1) << i;
                    layer.active_ready_mask |= UINT64_C(1) << i;
                    layer.active_slots.push_back(slot);
                    layer.active_route_positions.push_back((int32_t) i);
                }
            }
        }
        layer.active_hits = (int) layer.active_slots.size();

        if (!allow_gpu || layer.active_hits == 0 || op == nullptr || op->src[1] == nullptr ||
            op->src[1]->type != GGML_TYPE_F32 || op->src[1]->ne[1] != 1 || op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1) {
            layer.active_ready_mask = 0;
            layer.active_slots.clear();
            layer.active_route_positions.clear();
            layer.active_hits = 0;
            return;
        }

        auto * t = get_hybrid_template_locked(layer, layer.active_hits);
        if (t == nullptr) {
            layer.active_ready_mask = 0;
            layer.active_slots.clear();
            layer.active_route_positions.clear();
            layer.active_hits = 0;
            return;
        }

        const size_t out_bytes = (size_t) layer.output_dim * (size_t) layer.active_hits * sizeof(float);
        if (!ensure_hybrid_output(out_bytes)) {
            ++hybrid_fallbacks;
            layer.active_ready_mask = 0;
            layer.active_slots.clear();
            layer.active_route_positions.clear();
            layer.active_hits = 0;
            return;
        }

        const void * input_data = op->src[1]->data;
        if (t->input_q8) {
            const size_t q8_bytes = ggml_nbytes(t->input);
            if (!ensure_hybrid_q8_input(q8_bytes)) {
                ++hybrid_fallbacks;
                layer.active_ready_mask = 0;
                layer.active_slots.clear();
                layer.active_route_positions.clear();
                layer.active_hits = 0;
                return;
            }
            const auto q0 = std::chrono::steady_clock::now();
            quantize_row_q8_1((const float *) op->src[1]->data, hybrid_input_q8_ptr, layer.input_dim);
            const auto q1 = std::chrono::steady_clock::now();
            hybrid_preq_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(q1 - q0).count();
            ++hybrid_preq_inputs;
            input_data = hybrid_input_q8_ptr;
        }

        ggml_backend_tensor_set_async(compute_backend, t->input, input_data, 0, ggml_nbytes(t->input));
        ggml_backend_tensor_set_async(compute_backend, t->ids, layer.active_slots.data(), 0,
                (size_t) layer.active_hits * sizeof(int32_t));
        const enum ggml_status status = ggml_backend_graph_compute_async(compute_backend, t->graph);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_synchronize(compute_backend);
            ++hybrid_fallbacks;
            layer.active_ready_mask = 0;
            layer.active_slots.clear();
            layer.active_route_positions.clear();
            layer.active_hits = 0;
            return;
        }
        ggml_backend_tensor_get_async(compute_backend, t->output, hybrid_output_ptr, 0, out_bytes);
        layer.active_gpu_launched = true;
        layer.active_launch_time = std::chrono::steady_clock::now();
        ++hybrid_launches;
        hybrid_routes += (uint64_t) layer.active_hits;
    }

    static void clear_active_locked(expert_cache_layer & layer) {
        layer.active = false;
        layer.active_ids = nullptr;
        layer.active_mapped_mask = 0;
        layer.active_policy_ready_mask = 0;
        layer.active_ready_mask = 0;
        layer.active_slots.clear();
        layer.active_route_positions.clear();
        layer.active_route_slots.clear();
        layer.active_hits = 0;
        layer.active_gpu_launched = false;
    }

    int64_t begin_token_locked(expert_cache_model & model, int layer_id) {
        if (model.last_layer < 0 || layer_id <= model.last_layer) {
            ++model.token;
        }
        model.last_layer = layer_id;
        const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (route_first_us == 0) {
            route_first_us = now_us;
        }
        route_last_us = now_us;
        return model.token;
    }

    void observe_shadow_locked(const void * model_key, expert_cache_model & model, expert_cache_layer & layer,
                               int layer_id, const ggml_tensor * ids) {
        const int64_t token = begin_token_locked(model, layer_id);
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0]);
            if (expert < 0 || expert >= layer.n_expert) {
                continue;
            }
            ++selections;
            const int32_t slot = layer.expert_to_slot[(size_t) expert];
            if (slot >= 0) {
                ++simulated_hits;
                if (layer.slot_ready[(size_t) slot]) {
                    ++ready_hits;
                } else {
                    ++pending_hits;
                }
                layer.slot_stamp[(size_t) slot] = ++layer.stamp;
            } else {
                ++misses;
                maybe_admit_locked(model_key, layer, layer_id, expert, token);
            }
            layer.last_seen[(size_t) expert] = token;
        }
    }

    void observe_hybrid_locked(const void * model_key, expert_cache_model & model, expert_cache_layer & layer,
                               int layer_id, const ggml_tensor * ids) {
        const int64_t token = begin_token_locked(model, layer_id);
        std::vector<int32_t> admit_after_observation;
        admit_after_observation.reserve((size_t) ids->ne[0]);

        // Classify the whole token against the frozen begin snapshot before
        // admitting anything. This keeps a miss early in the top-k list from
        // evicting a ready expert that was already used by the GPU later in
        // the same token.
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0]);
            if (expert < 0 || expert >= layer.n_expert) {
                continue;
            }

            ++selections;
            const bool mapped_at_begin = (layer.active_mapped_mask & (UINT64_C(1) << i)) != 0;
            const bool ready_at_begin  = (layer.active_policy_ready_mask & (UINT64_C(1) << i)) != 0;
            const int32_t slot_at_begin = (size_t) i < layer.active_route_slots.size()
                ? layer.active_route_slots[(size_t) i] : -1;

            if (mapped_at_begin && slot_at_begin >= 0) {
                ++simulated_hits;
                if (ready_at_begin) {
                    ++ready_hits;
                } else {
                    ++pending_hits;
                }
                if ((size_t) slot_at_begin < layer.slot_stamp.size() &&
                    layer.slot_to_expert[(size_t) slot_at_begin] == expert) {
                    layer.slot_stamp[(size_t) slot_at_begin] = ++layer.stamp;
                }
            } else {
                ++misses;
                const int64_t prev = layer.last_seen[(size_t) expert];
                const bool admit = admit_window == 0 ||
                    (prev >= 0 && token >= prev && (uint64_t) (token - prev) <= admit_window);
                if (admit) {
                    admit_after_observation.push_back(expert);
                }
            }

            layer.last_seen[(size_t) expert] = token;
        }

        for (const int32_t expert : admit_after_observation) {
            if (layer.expert_to_slot[(size_t) expert] < 0) {
                admit_expert_locked(model_key, layer, layer_id, expert);
            }
        }
    }

    void maybe_admit_locked(const void * model_key, expert_cache_layer & layer, int layer_id, int32_t expert, int64_t token) {
        const int64_t prev = layer.last_seen[(size_t) expert];
        const bool admit = admit_window == 0 ||
            (prev >= 0 && token >= prev && (uint64_t) (token - prev) <= admit_window);
        if (admit) {
            admit_expert_locked(model_key, layer, layer_id, expert);
        }
    }

    void admit_expert_locked(const void * model_key, expert_cache_layer & layer, int layer_id, int32_t expert) {
        int32_t slot = -1;
        for (uint32_t i = 0; i < slots; ++i) {
            if (layer.slot_to_expert[i] < 0) {
                slot = (int32_t) i;
                break;
            }
        }
        if (slot < 0) {
            slot = (int32_t) std::distance(layer.slot_stamp.begin(),
                std::min_element(layer.slot_stamp.begin(), layer.slot_stamp.end()));
        }

        const int32_t old_expert = layer.slot_to_expert[(size_t) slot];
        if (old_expert >= 0) {
            layer.expert_to_slot[(size_t) old_expert] = -1;
        }

        layer.slot_to_expert[(size_t) slot] = expert;
        layer.expert_to_slot[(size_t) expert] = slot;
        layer.slot_stamp[(size_t) slot] = ++layer.stamp;
        layer.slot_ready[(size_t) slot] = 0;
        const uint64_t generation = ++layer.slot_generation[(size_t) slot];
        ++admissions;

        {
            std::lock_guard<std::mutex> qlock(queue_mutex);
            queue.push_back({ model_key, layer_id, expert, slot, generation });
            max_queue = std::max(max_queue, queue.size());
        }
        queue_cv.notify_one();
    }

    void ensure_staging(size_t size) {
        if (staging_size >= size) {
            return;
        }
        if (staging_buffer != nullptr) {
            ggml_backend_buffer_free(staging_buffer);
            staging_buffer = nullptr;
            staging_ptr = nullptr;
        }
        staging_fallback.clear();

        if (host_buft != nullptr) {
            staging_buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (staging_buffer != nullptr) {
                staging_ptr = ggml_backend_buffer_get_base(staging_buffer);
                staging_size = size;
                return;
            }
        }
        staging_fallback.resize(size);
        staging_ptr = staging_fallback.data();
        staging_size = size;
    }

    void worker_main_shadow() {
        for (;;) {
            expert_cache_request req;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this]() { return stop || !queue.empty(); });
                if (queue.empty()) {
                    if (stop) {
                        break;
                    }
                    continue;
                }
                req = queue.front();
                queue.pop_front();
            }

            size_t bundle_bytes = 0;
            ggml_tensor * shadow_cache_tensor = nullptr;
            std::vector<const ggml_tensor *> parts;
            std::vector<ggml_tensor *> cache_parts;
            std::vector<size_t> part_bytes;

            {
                std::lock_guard<std::mutex> lock(state_mutex);
                auto mit = models.find(req.model_key);
                if (mit == models.end()) {
                    ++stale_requests;
                    continue;
                }
                auto lit = mit->second.layers.find(req.layer);
                if (lit == mit->second.layers.end()) {
                    ++stale_requests;
                    continue;
                }
                auto & layer = *lit->second;
                if (req.slot < 0 || (size_t) req.slot >= layer.slot_generation.size() ||
                    layer.slot_generation[(size_t) req.slot] != req.generation ||
                    layer.slot_to_expert[(size_t) req.slot] != req.expert) {
                    ++stale_requests;
                    continue;
                }

                bundle_bytes = layer.bundle_bytes;
                shadow_cache_tensor = layer.shadow_cache_tensor;
                parts = layer.parts;
                cache_parts = layer.cache_parts;
                part_bytes = layer.part_bytes;
            }

            ensure_staging(bundle_bytes);
            if (staging_ptr == nullptr) {
                ++stale_requests;
                continue;
            }

            const auto t0 = std::chrono::steady_clock::now();
            uint8_t * dst = (uint8_t *) staging_ptr;
            bool converted = true;
            if (mode == expert_cache_mode::shadow) {
                // Preserve the original shadow experiment exactly: cache the
                // CPU-resident byte representation without converting it, since
                // shadow mode never consumes the cache for computation.
                for (size_t i = 0; i < parts.size(); ++i) {
                    const auto * tensor = parts[i];
                    const size_t bytes = part_bytes[i];
                    memcpy(dst, (const uint8_t *) tensor->data + (size_t) req.expert * tensor->nb[2], bytes);
                    dst += bytes;
                }
            } else {
                // CPU_REPACK rearranges ZNQ rows for x86 kernels. The GPU
                // kernels consume native GGML ZNQ layout, so hybrid mode must
                // invert that layout before admission.
                for (size_t i = 0; i < parts.size(); ++i) {
                    if (!copy_native_expert(parts[i], req.expert, dst, part_bytes[i])) {
                        converted = false;
                        break;
                    }
                    dst += part_bytes[i];
                }
            }
            if (!converted) {
                ++stale_requests;
                continue;
            }

            size_t staging_offset = 0;
            if (mode == expert_cache_mode::shadow) {
                GGML_ASSERT(shadow_cache_tensor != nullptr);
                size_t cache_offset = (size_t) req.slot * bundle_bytes;
                for (const size_t bytes : part_bytes) {
                    ggml_backend_tensor_set_async(upload_backend, shadow_cache_tensor,
                            (const uint8_t *) staging_ptr + staging_offset, cache_offset, bytes);
                    staging_offset += bytes;
                    cache_offset += bytes;
                    ++upload_ops;
                }
            } else {
                GGML_ASSERT(cache_parts.size() == part_bytes.size());
                for (size_t i = 0; i < part_bytes.size(); ++i) {
                    const size_t bytes = part_bytes[i];
                    const size_t slot_offset = (size_t) req.slot * cache_parts[i]->nb[2];
                    ggml_backend_tensor_set_async(upload_backend, cache_parts[i],
                            (const uint8_t *) staging_ptr + staging_offset, slot_offset, bytes);
                    staging_offset += bytes;
                    ++upload_ops;
                }
            }
            ggml_backend_synchronize(upload_backend);
            const auto t1 = std::chrono::steady_clock::now();

            upload_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            upload_bytes += bundle_bytes;
            ++uploads;

            {
                std::lock_guard<std::mutex> lock(state_mutex);
                auto mit = models.find(req.model_key);
                if (mit == models.end()) {
                    continue;
                }
                auto lit = mit->second.layers.find(req.layer);
                if (lit == mit->second.layers.end()) {
                    continue;
                }
                auto & layer = *lit->second;
                if ((size_t) req.slot < layer.slot_generation.size() &&
                    layer.slot_generation[(size_t) req.slot] == req.generation &&
                    layer.slot_to_expert[(size_t) req.slot] == req.expert) {
                    layer.slot_ready[(size_t) req.slot] = 1;
                } else {
                    ++stale_requests;
                }
            }
        }
    }

    bool ensure_hybrid_staging(uint32_t index, size_t size) {
        GGML_ASSERT(index < hybrid_staging.size() && hybrid_staging[index]);
        auto & staging = *hybrid_staging[index];
        if (staging.size >= size) {
            return staging.ptr != nullptr;
        }
        if (staging.buffer != nullptr) {
            ggml_backend_buffer_free(staging.buffer);
            staging.buffer = nullptr;
            staging.ptr = nullptr;
        }
        staging.fallback.clear();

        if (host_buft != nullptr) {
            staging.buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (staging.buffer != nullptr) {
                staging.ptr = ggml_backend_buffer_get_base(staging.buffer);
                staging.size = size;
                return staging.ptr != nullptr;
            }
        }

        staging.fallback.resize(size);
        staging.ptr = staging.fallback.data();
        staging.size = size;
        return staging.ptr != nullptr;
    }

    void release_hybrid_staging(uint32_t index) {
        GGML_ASSERT(index < hybrid_staging.size() && hybrid_staging[index]);
        auto & staging = *hybrid_staging[index];
        {
            std::lock_guard<std::mutex> lock(staging.mutex);
            staging.available = true;
        }
        staging.cv.notify_one();
    }

    void worker_main_hybrid_convert(uint32_t staging_index) {
        GGML_ASSERT(staging_index < hybrid_staging.size() && hybrid_staging[staging_index]);
        auto & staging = *hybrid_staging[staging_index];

        for (;;) {
            // One pinned buffer belongs to each converter. It cannot be reused
            // until the single uploader has finished the preceding H2D copy.
            {
                std::unique_lock<std::mutex> lock(staging.mutex);
                staging.cv.wait(lock, [&staging]() { return staging.available; });
            }

            expert_cache_request req;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this]() { return stop || !queue.empty(); });
                if (queue.empty()) {
                    if (stop) {
                        break;
                    }
                    continue;
                }
                req = queue.front();
                queue.pop_front();
            }

            size_t bundle_bytes = 0;
            std::vector<const ggml_tensor *> parts;
            std::vector<size_t> part_bytes;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                auto mit = models.find(req.model_key);
                if (mit == models.end()) {
                    ++stale_requests;
                    continue;
                }
                auto lit = mit->second.layers.find(req.layer);
                if (lit == mit->second.layers.end()) {
                    ++stale_requests;
                    continue;
                }
                auto & layer = *lit->second;
                if (req.slot < 0 || (size_t) req.slot >= layer.slot_generation.size() ||
                    layer.slot_generation[(size_t) req.slot] != req.generation ||
                    layer.slot_to_expert[(size_t) req.slot] != req.expert) {
                    ++stale_requests;
                    continue;
                }
                bundle_bytes = layer.bundle_bytes;
                parts = layer.parts;
                part_bytes = layer.part_bytes;
            }

            if (!ensure_hybrid_staging(staging_index, bundle_bytes)) {
                ++stale_requests;
                continue;
            }

            const auto t0 = std::chrono::steady_clock::now();
            uint8_t * dst = (uint8_t *) staging.ptr;
            bool converted = true;
            for (size_t i = 0; i < parts.size(); ++i) {
                if (!copy_native_expert(parts[i], req.expert, dst, part_bytes[i])) {
                    converted = false;
                    break;
                }
                dst += part_bytes[i];
            }
            const auto t1 = std::chrono::steady_clock::now();
            conversion_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
                                    std::memory_order_relaxed);
            if (!converted) {
                ++stale_requests;
                continue;
            }
            conversion_bytes.fetch_add(bundle_bytes, std::memory_order_relaxed);

            {
                std::lock_guard<std::mutex> lock(staging.mutex);
                GGML_ASSERT(staging.available);
                staging.available = false;
            }
            {
                std::lock_guard<std::mutex> lock(converted_mutex);
                converted_queue.push_back({ req, staging_index });
                max_converted_queue = std::max(max_converted_queue, converted_queue.size());
            }
            converted_cv.notify_one();
        }
    }

    void worker_main_hybrid_upload() {
        for (;;) {
            expert_cache_converted_request job;
            {
                std::unique_lock<std::mutex> lock(converted_mutex);
                converted_cv.wait(lock, [this]() { return converters_done || !converted_queue.empty(); });
                if (converted_queue.empty()) {
                    if (converters_done) {
                        break;
                    }
                    continue;
                }
                job = converted_queue.front();
                converted_queue.pop_front();
            }

            const auto & req = job.req;
            GGML_ASSERT(job.staging_index < hybrid_staging.size() && hybrid_staging[job.staging_index]);
            auto & staging = *hybrid_staging[job.staging_index];

            size_t bundle_bytes = 0;
            std::vector<ggml_tensor *> cache_parts;
            std::vector<size_t> part_bytes;
            bool current = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                auto mit = models.find(req.model_key);
                if (mit != models.end()) {
                    auto lit = mit->second.layers.find(req.layer);
                    if (lit != mit->second.layers.end()) {
                        auto & layer = *lit->second;
                        if (req.slot >= 0 && (size_t) req.slot < layer.slot_generation.size() &&
                            layer.slot_generation[(size_t) req.slot] == req.generation &&
                            layer.slot_to_expert[(size_t) req.slot] == req.expert) {
                            bundle_bytes = layer.bundle_bytes;
                            cache_parts = layer.cache_parts;
                            part_bytes = layer.part_bytes;
                            current = true;
                        }
                    }
                }
            }

            if (!current || cache_parts.size() != part_bytes.size()) {
                ++stale_requests;
                release_hybrid_staging(job.staging_index);
                continue;
            }

            const auto t0 = std::chrono::steady_clock::now();
            size_t staging_offset = 0;
            for (size_t i = 0; i < part_bytes.size(); ++i) {
                const size_t bytes = part_bytes[i];
                const size_t slot_offset = (size_t) req.slot * cache_parts[i]->nb[2];
                ggml_backend_tensor_set_async(upload_backend, cache_parts[i],
                        (const uint8_t *) staging.ptr + staging_offset, slot_offset, bytes);
                staging_offset += bytes;
                ++upload_ops;
            }
            ggml_backend_synchronize(upload_backend);
            const auto t1 = std::chrono::steady_clock::now();

            upload_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            upload_bytes += bundle_bytes;
            ++uploads;

            {
                std::lock_guard<std::mutex> lock(state_mutex);
                auto mit = models.find(req.model_key);
                if (mit != models.end()) {
                    auto lit = mit->second.layers.find(req.layer);
                    if (lit != mit->second.layers.end()) {
                        auto & layer = *lit->second;
                        if ((size_t) req.slot < layer.slot_generation.size() &&
                            layer.slot_generation[(size_t) req.slot] == req.generation &&
                            layer.slot_to_expert[(size_t) req.slot] == req.expert) {
                            layer.slot_ready[(size_t) req.slot] = 1;
                        } else {
                            ++stale_requests;
                        }
                    }
                }
            }

            release_hybrid_staging(job.staging_index);
        }
    }

    uint64_t total_tokens_locked() const {
        uint64_t total = 0;
        for (const auto & model_it : models) {
            if (model_it.second.token >= 0) {
                total += (uint64_t) model_it.second.token + 1;
            }
        }
        return total;
    }

private:
    const expert_cache_mode mode;
    const uint32_t slots;
    const uint32_t admit_window;
    const uint32_t convert_workers;
    const bool print_stats;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t upload_backend = nullptr;
    ggml_backend_t compute_backend = nullptr;
    ggml_backend_buffer_type_t device_buft = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
    bool valid = false;

    std::mutex state_mutex;
    std::unordered_map<const void *, expert_cache_model> models;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<expert_cache_request> queue;
    bool stop = false;
    std::thread worker;

    std::vector<std::unique_ptr<expert_cache_staging>> hybrid_staging;
    std::vector<std::thread> convert_threads;
    std::mutex converted_mutex;
    std::condition_variable converted_cv;
    std::deque<expert_cache_converted_request> converted_queue;
    bool converters_done = false;
    std::thread upload_thread;
    size_t max_converted_queue = 0;

    ggml_backend_buffer_t staging_buffer = nullptr;
    void * staging_ptr = nullptr;
    size_t staging_size = 0;
    std::vector<uint8_t> staging_fallback;

    ggml_backend_buffer_t hybrid_input_q8_buffer = nullptr;
    void * hybrid_input_q8_ptr = nullptr;
    size_t hybrid_input_q8_size = 0;
    std::vector<uint8_t> hybrid_input_q8_fallback;

    ggml_backend_buffer_t hybrid_output_buffer = nullptr;
    void * hybrid_output_ptr = nullptr;
    size_t hybrid_output_size = 0;
    std::vector<uint8_t> hybrid_output_fallback;

    uint64_t selections = 0;
    uint64_t simulated_hits = 0;
    uint64_t ready_hits = 0;
    uint64_t pending_hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t uploads = 0;
    uint64_t upload_ops = 0;
    uint64_t upload_bytes = 0;
    uint64_t upload_us = 0;
    std::atomic<uint64_t> stale_requests { 0 };
    std::atomic<uint64_t> conversion_bytes { 0 };
    std::atomic<uint64_t> conversion_us { 0 };
    uint64_t allocated_bytes = 0;
    uint64_t allocated_layers = 0;
    size_t max_queue = 0;
    int64_t route_first_us = 0;
    int64_t route_last_us = 0;

    uint64_t hybrid_compatible_layers = 0;
    uint64_t hybrid_launches = 0;
    uint64_t hybrid_routes = 0;
    uint64_t hybrid_fallbacks = 0;
    uint64_t hybrid_wait_us = 0;
    uint64_t hybrid_elapsed_us = 0;
    uint64_t hybrid_preq_inputs = 0;
    uint64_t hybrid_preq_us = 0;
};

std::mutex g_expert_cache_mutex;
std::unique_ptr<expert_cache> g_expert_cache;

static void configure_expert_cache(expert_cache_mode mode, uint32_t slots, uint32_t admit_window, uint32_t convert_workers, bool print_stats, ggml_backend_dev_t device) {
    std::lock_guard<std::mutex> lock(g_expert_cache_mutex);
    g_expert_cache.reset();
    if (slots == 0 || device == nullptr) {
        return;
    }
    g_expert_cache = std::make_unique<expert_cache>(mode, slots, admit_window, convert_workers, print_stats, device);
}

} // namespace

extern "C" void ggml_backend_cpu_expert_cache_shadow_configure(
        uint32_t slots,
        uint32_t admit_window,
        ggml_backend_dev_t device) {
    configure_expert_cache(expert_cache_mode::shadow, slots, admit_window, 1, false, device);
}

extern "C" void ggml_backend_cpu_expert_cache_hybrid_configure(
        uint32_t slots,
        uint32_t admit_window,
        uint32_t convert_workers,
        bool print_stats,
        ggml_backend_dev_t device) {
    configure_expert_cache(expert_cache_mode::hybrid, slots, admit_window, convert_workers, print_stats, device);
}

extern "C" void ggml_backend_cpu_expert_cache_shadow_route(
        const ggml_tensor * weights,
        const ggml_tensor * ids) {
    std::lock_guard<std::mutex> lock(g_expert_cache_mutex);
    if (g_expert_cache) {
        g_expert_cache->route(weights, ids);
    }
}

extern "C" uint64_t ggml_backend_cpu_expert_cache_hybrid_begin(ggml_tensor * op) {
    std::lock_guard<std::mutex> lock(g_expert_cache_mutex);
    return g_expert_cache ? g_expert_cache->hybrid_begin(op) : 0;
}

extern "C" void ggml_backend_cpu_expert_cache_hybrid_end(ggml_tensor * op) {
    std::lock_guard<std::mutex> lock(g_expert_cache_mutex);
    if (g_expert_cache) {
        g_expert_cache->hybrid_end(op);
    }
}
