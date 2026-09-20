#include "expert-cache-shadow.h"

#include <algorithm>
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

struct expert_cache_shadow_layer {
    const ggml_tensor * gate    = nullptr;
    const ggml_tensor * up      = nullptr;
    const ggml_tensor * gate_up = nullptr;
    const ggml_tensor * down    = nullptr;

    std::vector<const ggml_tensor *> parts;
    std::vector<size_t> part_bytes;

    int64_t n_expert = 0;
    size_t bundle_bytes = 0;

    ggml_context * cache_ctx = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;
    ggml_tensor * cache_tensor = nullptr;

    std::vector<int32_t> expert_to_slot;
    std::vector<int32_t> slot_to_expert;
    std::vector<uint64_t> slot_stamp;
    std::vector<uint64_t> slot_generation;
    std::vector<uint8_t> slot_ready;
    std::vector<int64_t> last_seen;

    uint64_t stamp = 0;
    bool disabled = false;
};

struct expert_cache_shadow_model {
    std::unordered_map<int, std::unique_ptr<expert_cache_shadow_layer>> layers;
    int last_layer = -1;
    int64_t token = -1;
};

struct expert_cache_shadow_request {
    const void * model_key;
    int layer;
    int32_t expert;
    int32_t slot;
    uint64_t generation;
};

class expert_cache_shadow {
public:
    expert_cache_shadow(uint32_t slots, uint32_t admit_window, ggml_backend_dev_t device) :
        slots(slots), admit_window(admit_window), device(device) {
        upload_backend = ggml_backend_dev_init(device, nullptr);
        device_buft = ggml_backend_dev_buffer_type(device);
        host_buft = ggml_backend_dev_host_buffer_type(device);

        if (upload_backend == nullptr || device_buft == nullptr) {
            fprintf(stderr, "%s: failed to initialize upload backend for %s\n", __func__, ggml_backend_dev_name(device));
            valid = false;
            return;
        }

        valid = true;
        worker = std::thread([this]() { worker_main(); });

        fprintf(stderr, "%s: enabled on %s: slots/layer=%u admit_window=%u (shadow mode)\n",
                __func__, ggml_backend_dev_name(device), slots, admit_window);
    }

    ~expert_cache_shadow() {
        if (valid) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                stop = true;
            }
            queue_cv.notify_all();
            if (worker.joinable()) {
                worker.join();
            }
        }

        const uint64_t tokens = total_tokens_locked();
        const double hit_pct = selections ? 100.0 * (double) simulated_hits / (double) selections : 0.0;
        const double ready_pct = selections ? 100.0 * (double) ready_hits / (double) selections : 0.0;
        const double pending_pct = selections ? 100.0 * (double) pending_hits / (double) selections : 0.0;
        const double cache_mib = (double) allocated_bytes / (1024.0 * 1024.0);
        const double upload_mib = (double) upload_bytes / (1024.0 * 1024.0);
        const double upload_mib_per_token = tokens ? upload_mib / (double) tokens : 0.0;
        const double worker_gib_s = upload_us ? ((double) upload_bytes / (1024.0 * 1024.0 * 1024.0)) / ((double) upload_us / 1.0e6) : 0.0;
        const double wall_s = route_last_us > route_first_us ? (double) (route_last_us - route_first_us) / 1.0e6 : 0.0;
        const double wall_gib_s = wall_s > 0 ? ((double) upload_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s : 0.0;

        if (selections || allocated_bytes) {
            fprintf(stderr,
                    "%s: cache=%.1f MiB tokens=%" PRIu64 " selections=%" PRIu64
                    " simulated_hit=%.2f%% ready_hit=%.2f%% pending_hit=%.2f%% admissions=%" PRIu64 "\n",
                    __func__, cache_mib, tokens, selections, hit_pct, ready_pct, pending_pct, admissions);
            fprintf(stderr,
                    "%s: uploads=%" PRIu64 " copies=%" PRIu64 " stale=%" PRIu64 " upload=%.1f MiB (%.2f MiB/token) "
                    "worker=%.2f GiB/s wall=%.2f GiB/s max_queue=%zu\n",
                    __func__, uploads, upload_ops, stale_requests, upload_mib, upload_mib_per_token,
                    worker_gib_s, wall_gib_s, max_queue);
        }

        if (staging_buffer != nullptr) {
            ggml_backend_buffer_free(staging_buffer);
        }
        for (auto & model_it : models) {
            for (auto & layer_it : model_it.second.layers) {
                auto & layer = *layer_it.second;
                if (layer.cache_buffer != nullptr) {
                    ggml_backend_buffer_free(layer.cache_buffer);
                }
                if (layer.cache_ctx != nullptr) {
                    ggml_free(layer.cache_ctx);
                }
            }
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
        auto & layer_ptr = model.layers[layer_id];
        if (!layer_ptr) {
            layer_ptr = std::make_unique<expert_cache_shadow_layer>();
        }
        auto & layer = *layer_ptr;

        switch (part) {
            case expert_cache_part::gate:    layer.gate    = weights; break;
            case expert_cache_part::up:      layer.up      = weights; break;
            case expert_cache_part::gate_up: layer.gate_up = weights; break;
            case expert_cache_part::down:    layer.down    = weights; break;
            case expert_cache_part::none:                         break;
        }

        // Gate/up execute before down in llama's MoE graph.  Use the down op as
        // the one route observation for this layer/token so the selected IDs are
        // not counted three times.
        if (part != expert_cache_part::down) {
            return;
        }

        if (!prepare_layer(layer, layer_id)) {
            return;
        }

        // Shadow policy is for autoregressive decode.  Prefill still discovers
        // tensor layout and allocates the cache, but it does not train/admit it.
        if (ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) {
            return;
        }

        if (model.last_layer < 0 || layer_id <= model.last_layer) {
            ++model.token;
        }
        model.last_layer = layer_id;

        const int64_t token = model.token;
        const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (route_first_us == 0) {
            route_first_us = now_us;
        }
        route_last_us = now_us;

        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0]);
            if (expert < 0 || expert >= layer.n_expert) {
                continue;
            }

            ++selections;
            const int32_t slot = layer.expert_to_slot[expert];
            if (slot >= 0) {
                ++simulated_hits;
                if (layer.slot_ready[slot]) {
                    ++ready_hits;
                } else {
                    ++pending_hits;
                }
                layer.slot_stamp[slot] = ++layer.stamp;
            } else {
                ++misses;
                const int64_t prev = layer.last_seen[expert];
                const bool admit = admit_window == 0 ||
                    (prev >= 0 && token >= prev && (uint64_t) (token - prev) <= admit_window);
                if (admit) {
                    admit_expert(model_key, layer, layer_id, expert);
                }
            }

            // Match the offline simulator: every observation, including a hit,
            // refreshes the reuse window used if this expert is later evicted.
            layer.last_seen[expert] = token;
        }
    }

private:
    bool prepare_layer(expert_cache_shadow_layer & layer, int layer_id) {
        if (layer.disabled) {
            return false;
        }
        if (layer.cache_tensor != nullptr) {
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
            // The shadow copy needs one contiguous source slice per expert.
            // This is true for llama's regular and CPU_REPACK MoE tensors.
            if (bytes == 0 || tensor->nb[2] != bytes) {
                fprintf(stderr, "%s: layer %d has non-contiguous expert slices in %s; disabling shadow cache for layer\n",
                        __func__, layer_id, tensor->name);
                layer.disabled = true;
                return false;
            }
            layer.part_bytes.push_back(bytes);
            layer.bundle_bytes += bytes;
        }

        const size_t cache_bytes = layer.bundle_bytes * (size_t) slots;
        layer.cache_buffer = ggml_backend_buft_alloc_buffer(device_buft, cache_bytes);
        if (layer.cache_buffer == nullptr) {
            fprintf(stderr, "%s: failed to allocate %.1f MiB for layer %d shadow cache\n",
                    __func__, (double) cache_bytes / (1024.0 * 1024.0), layer_id);
            layer.disabled = true;
            return false;
        }
        ggml_backend_buffer_set_usage(layer.cache_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_init_params ctx_params = {
            /*.mem_size   =*/ 4 * ggml_tensor_overhead() + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        layer.cache_ctx = ggml_init(ctx_params);
        if (layer.cache_ctx == nullptr) {
            ggml_backend_buffer_free(layer.cache_buffer);
            layer.cache_buffer = nullptr;
            layer.disabled = true;
            return false;
        }

        layer.cache_tensor = ggml_new_tensor_1d(layer.cache_ctx, GGML_TYPE_I8, (int64_t) cache_bytes);
        if (ggml_backend_tensor_alloc(layer.cache_buffer, layer.cache_tensor,
                    ggml_backend_buffer_get_base(layer.cache_buffer)) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(layer.cache_buffer);
            layer.cache_buffer = nullptr;
            ggml_free(layer.cache_ctx);
            layer.cache_ctx = nullptr;
            layer.cache_tensor = nullptr;
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

    void admit_expert(const void * model_key, expert_cache_shadow_layer & layer, int layer_id, int32_t expert) {
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

        const int32_t old_expert = layer.slot_to_expert[slot];
        if (old_expert >= 0) {
            layer.expert_to_slot[old_expert] = -1;
        }

        layer.slot_to_expert[slot] = expert;
        layer.expert_to_slot[expert] = slot;
        layer.slot_stamp[slot] = ++layer.stamp;
        layer.slot_ready[slot] = 0;
        const uint64_t generation = ++layer.slot_generation[slot];
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

    void worker_main() {
        for (;;) {
            expert_cache_shadow_request req;
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

            ggml_tensor * cache_tensor = nullptr;
            size_t cache_offset = 0;
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
                    layer.slot_generation[req.slot] != req.generation ||
                    layer.slot_to_expert[req.slot] != req.expert) {
                    ++stale_requests;
                    continue;
                }

                cache_tensor = layer.cache_tensor;
                cache_offset = (size_t) req.slot * layer.bundle_bytes;
                bundle_bytes = layer.bundle_bytes;
                parts = layer.parts;
                part_bytes = layer.part_bytes;
            }

            ensure_staging(bundle_bytes);
            if (staging_ptr == nullptr) {
                ++stale_requests;
                continue;
            }

            const auto t0 = std::chrono::steady_clock::now();
            uint8_t * dst = (uint8_t *) staging_ptr;
            for (size_t i = 0; i < parts.size(); ++i) {
                const auto * tensor = parts[i];
                const size_t bytes = part_bytes[i];
                const uint8_t * src = (const uint8_t *) tensor->data + (size_t) req.expert * tensor->nb[2];
                memcpy(dst, src, bytes);
                dst += bytes;
            }

            // Use one H2D operation per expert tensor, matching the future
            // gate/up/down (or gate_up/down) cache admission more closely than
            // one artificially coalesced bundle copy would.
            size_t staging_offset = 0;
            size_t cache_part_offset = cache_offset;
            for (const size_t bytes : part_bytes) {
                ggml_backend_tensor_set_async(upload_backend, cache_tensor,
                        (const uint8_t *) staging_ptr + staging_offset,
                        cache_part_offset, bytes);
                staging_offset += bytes;
                cache_part_offset += bytes;
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
                if (mit == models.end()) {
                    continue;
                }
                auto lit = mit->second.layers.find(req.layer);
                if (lit == mit->second.layers.end()) {
                    continue;
                }
                auto & layer = *lit->second;
                if ((size_t) req.slot < layer.slot_generation.size() &&
                    layer.slot_generation[req.slot] == req.generation &&
                    layer.slot_to_expert[req.slot] == req.expert) {
                    layer.slot_ready[req.slot] = 1;
                } else {
                    ++stale_requests;
                }
            }
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
    const uint32_t slots;
    const uint32_t admit_window;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t upload_backend = nullptr;
    ggml_backend_buffer_type_t device_buft = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
    bool valid = false;

    std::mutex state_mutex;
    std::unordered_map<const void *, expert_cache_shadow_model> models;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<expert_cache_shadow_request> queue;
    bool stop = false;
    std::thread worker;

    ggml_backend_buffer_t staging_buffer = nullptr;
    void * staging_ptr = nullptr;
    size_t staging_size = 0;
    std::vector<uint8_t> staging_fallback;

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
    uint64_t stale_requests = 0;
    uint64_t allocated_bytes = 0;
    uint64_t allocated_layers = 0;
    size_t max_queue = 0;
    int64_t route_first_us = 0;
    int64_t route_last_us = 0;
};

std::mutex g_expert_cache_shadow_mutex;
std::unique_ptr<expert_cache_shadow> g_expert_cache_shadow;

} // namespace

extern "C" void ggml_backend_cpu_expert_cache_shadow_configure(
        uint32_t slots,
        uint32_t admit_window,
        ggml_backend_dev_t device) {
    std::lock_guard<std::mutex> lock(g_expert_cache_shadow_mutex);
    g_expert_cache_shadow.reset();

    if (slots == 0 || device == nullptr) {
        return;
    }

    g_expert_cache_shadow = std::make_unique<expert_cache_shadow>(slots, admit_window, device);
}

extern "C" void ggml_backend_cpu_expert_cache_shadow_route(
        const ggml_tensor * weights,
        const ggml_tensor * ids) {
    std::lock_guard<std::mutex> lock(g_expert_cache_shadow_mutex);
    if (g_expert_cache_shadow) {
        g_expert_cache_shadow->route(weights, ids);
    }
}
