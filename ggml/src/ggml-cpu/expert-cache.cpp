#include "expert-cache.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "quants.h"
#include "repack.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
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
#include <unordered_set>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {
static bool supports_input(const ggml_tensor * op) {
    const ggml_type type = op->src[1]->type;
    return type == GGML_TYPE_F32 ||
           (type == ggml_get_type_traits_cpu(op->src[0]->type)->vec_dot_type &&
            ggml_get_type_traits(type)->to_float != nullptr);
}

static void pack_input_q8_1(const void * input, ggml_type type, void * output, int64_t n) {
    if (type == GGML_TYPE_F32) {
        quantize_row_q8_1(static_cast<const float *>(input), output, n);
    } else if (type == GGML_TYPE_Q8_0) {
        const auto * in = static_cast<const block_q8_0 *>(input);
        auto * out = static_cast<block_q8_1 *>(output);
        for (int64_t b = 0; b < n / QK8_0; ++b) {
            out[b].d = in[b].d;
            memcpy(out[b].qs, in[b].qs, QK8_0);
            int sum = 0;
            for (int j = 0; j < QK8_0; ++j) {
                sum += in[b].qs[j];
            }
            out[b].s = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(in[b].d) * sum);
        }
    } else {
        GGML_ASSERT(type == GGML_TYPE_Q8_K);
        const auto * in = static_cast<const block_q8_K *>(input);
        auto * out = static_cast<block_q8_1 *>(output);
        float values[QK_K];
        for (int64_t b = 0; b < n / QK_K; ++b) {
            dequantize_row_q8_K(in + b, values, QK_K);
            quantize_row_q8_1(values, out + b * (QK_K / QK8_1), QK_K);
        }
    }
}

static void pack_input_f32(const void * input, ggml_type type, void * output, int64_t n) {
    if (type == GGML_TYPE_F32) {
        memcpy(output, input, n * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(type);
        GGML_ASSERT(traits->to_float != nullptr);
        traits->to_float(input, static_cast<float *>(output), n);
    }
}

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

struct expert_cache_template {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;
    bool input_q8 = false;
    uint64_t workspace_generation = 0;
};

struct expert_cache_merge {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_event_t copied = nullptr;
    int64_t output_capacity = 0;
    int64_t expert_capacity = 0;
    int64_t token_capacity = 1;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * packed = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * misses = nullptr;
    ggml_tensor * output = nullptr;
    bool all_cached = false;

    void build(int64_t output_dim, int64_t n_expert_used, int64_t n_tokens) {
        ggml_reset(ctx);
        const int64_t rows = n_expert_used * n_tokens;
        packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output_dim, 2 * rows);
        ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
        misses = ggml_view_3d(ctx, packed, output_dim, n_expert_used, n_tokens,
                packed->nb[1], packed->nb[1] * n_expert_used, rows * packed->nb[1]);
        output = ggml_reshape_3d(ctx, ggml_get_rows(ctx, packed, ids), output_dim, n_expert_used, n_tokens);
        graph = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(graph, output);
    }

    ~expert_cache_merge() {
        ggml_backend_event_free(copied);
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
    }
};

static constexpr int EXPERT_CACHE_BATCH_SIZES = 4;

struct expert_cache_batch {
    expert_cache_template * t;
    size_t route_offset;
    size_t input_offset;
};

struct expert_cache_layer {
    const ggml_tensor * gate    = nullptr;
    const ggml_tensor * up      = nullptr;
    const ggml_tensor * gate_up = nullptr;
    const ggml_tensor * down    = nullptr;

    std::vector<const ggml_tensor *> parts;
    std::vector<ggml_tensor *> cache_parts;
    std::vector<size_t> part_bytes;

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
    std::vector<int32_t> admissions;
    std::vector<uint8_t> admission_pending;

    uint64_t stamp = 0;
    int64_t observed_tokens = 0;
    bool disabled = false;

    bool checked = false;
    bool compatible = false;
    bool projection_only = false;
    bool force_projection = false;
    int64_t n_expert_used = 0;
    int64_t input_dim = 0;
    int64_t ffn_dim = 0;
    int64_t output_dim = 0;
    enum ggml_glu_op glu_op = GGML_GLU_OP_COUNT;
    std::array<uint8_t, GGML_MAX_OP_PARAMS> glu_params = {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> gate_params = {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> gate_up_params = {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> up_params = {};
    std::array<uint8_t, GGML_MAX_OP_PARAMS> down_params = {};

    std::vector<std::unique_ptr<expert_cache_template>> templates;
    std::vector<uint8_t> template_failed;
    std::array<std::vector<std::unique_ptr<expert_cache_template>>, 5> projection_templates;
    std::array<std::vector<uint8_t>, 5> projection_failed;

    // Freeze all token routes until down finishes. New admissions become
    // visible only to the next batch.
    bool active = false;
    expert_cache_part active_part = expert_cache_part::none;
    const ggml_tensor * active_ids = nullptr;
    std::vector<uint64_t> active_ready_masks;
    std::vector<int32_t> active_slots;
    std::vector<int32_t> active_route_positions;
    std::vector<int32_t> active_route_slots;
    std::vector<expert_cache_batch> active_batches;
    int active_hits = 0;
    bool active_gpu_launched = false;
    bool active_deferred = false;
    std::vector<int32_t> merge_routes;
    ggml_backend_event_t reads_done = nullptr;
    bool reads_recorded = false;
};

struct expert_cache_request {
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
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX2__)
            const __m512i table = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) znq3_spread4_table));
            const __m512i shifts = _mm512_set_epi32(28, 24, 20, 16, 12, 8, 4, 0, 28, 24, 20, 16, 12, 8, 4, 0);
#endif
            for (int64_t rg = 0; rg < row_groups; ++rg) {
                for (int64_t ib = 0; ib < nblocks; ++ib) {
                    const auto & in = src[rg*nblocks + ib];
                    for (int r = 0; r < 8; ++r) {
                        auto & b = out[(rg*8 + r)*nblocks + ib];
                        b.d = in.d[r];
                        b.books = in.books[r];

#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX2__)
                        // Rebuild two four-code groups into three native bytes.
                        for (int pair = 0; pair < QK_ZNQ/8; ++pair) {
                            const uint32_t packed =
                                (uint32_t) znq3_pack_four_codes(in, 2*pair + 0, r) |
                                ((uint32_t) znq3_pack_four_codes(in, 2*pair + 1, r) << 12);
                            b.qs[3*pair + 0] = (uint8_t) (packed >> 0);
                            b.qs[3*pair + 1] = (uint8_t) (packed >> 8);
                            b.qs[3*pair + 2] = (uint8_t) (packed >> 16);
                        }
#endif
                    }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX2__)
                    // Expand both groups for all eight rows.
                    for (int pair = 0; pair < QK_ZNQ/8; ++pair) {
                        const auto spread = [&](int plane) {
                            const __m512i words = _mm512_inserti64x4(
                                    _mm512_castsi256_si512(_mm256_set1_epi32(in.planes[2*pair][plane])),
                                    _mm256_set1_epi32(in.planes[2*pair + 1][plane]), 1);
                            const __m512i indices = _mm512_and_si512(_mm512_srlv_epi32(words, shifts), _mm512_set1_epi32(15));
                            return _mm512_permutexvar_epi32(indices, table);
                        };
                        const __m512i codes = _mm512_ternarylogic_epi32(spread(0), _mm512_slli_epi32(spread(1), 1), _mm512_slli_epi32(spread(2), 2), 0xfe);
                        const __m256i packed = _mm256_or_si256(_mm512_castsi512_si256(codes), _mm256_slli_epi32(_mm512_extracti64x4_epi64(codes, 1), 12));
                        alignas(32) uint32_t packed_rows[8];
                        _mm256_store_si256((__m256i *) packed_rows, packed);
                        for (int r = 0; r < 8; ++r) {
                            memcpy(out[(rg*8 + r)*nblocks + ib].qs + 3*pair, packed_rows + r, 3);
                        }
                    }
#endif
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
    expert_cache(uint32_t slots, uint32_t admit_window, uint32_t convert_workers, ggml_backend_dev_t device) :
        slots(slots), admit_window(admit_window), convert_workers(std::max<uint32_t>(1, convert_workers)) {
        upload_backend  = ggml_backend_dev_init(device, nullptr);
        compute_backend = ggml_backend_dev_init(device, nullptr);
        device_buft = ggml_backend_dev_buffer_type(device);
        host_buft = ggml_backend_dev_host_buffer_type(device);

        if (upload_backend == nullptr || compute_backend == nullptr || device_buft == nullptr) {
            fprintf(stderr, "%s: failed to initialize expert cache on %s\n", __func__, ggml_backend_dev_name(device));
            valid = false;
            return;
        }

        valid = true;
        staging.reserve(this->convert_workers);
        convert_threads.reserve(this->convert_workers);
        for (uint32_t i = 0; i < this->convert_workers; ++i) {
            staging.push_back(std::make_unique<expert_cache_staging>());
            convert_threads.emplace_back([this, i]() { worker_main_convert(i); });
        }
        upload_thread = std::thread([this]() { worker_main_upload(); });
    }


    ~expert_cache() {
        if (valid) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                stop = true;
            }
            queue_cv.notify_all();

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

        if (compute_backend != nullptr) {
            ggml_backend_synchronize(compute_backend);
        }
        for (auto & s : staging) {
            if (s && s->buffer != nullptr) {
                ggml_backend_buffer_free(s->buffer);
                s->buffer = nullptr;
            }
        }
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
        }
        if (output_buffer != nullptr) {
            ggml_backend_buffer_free(output_buffer);
        }
        for (auto & layer : layers) {
            free_layer(*layer.second);
        }
        if (workspace != nullptr) {
            ggml_gallocr_free(workspace);
            workspace = nullptr;
        }
        merge.reset();
        if (compute_backend != nullptr) {
            ggml_backend_free(compute_backend);
        }
        if (upload_backend != nullptr) {
            ggml_backend_free(upload_backend);
        }
    }

    bool is_valid() const { return valid; }
    ggml_backend_dev_t get_device() const { return ggml_backend_get_device(compute_backend); }
    uint64_t hits() const { return gpu_hits.load(std::memory_order_relaxed); }

    void register_weights(const ggml_tensor * weights) {
        int layer_id = -1;
        const auto part = expert_cache_parse_tensor(weights->name, layer_id);
        if (part != expert_cache_part::none && layer_id >= 0) {
            registered.insert(weights);
            register_part(get_layer(layer_id), part, weights);
        }
    }

    size_t memory() {
        std::lock_guard<std::mutex> lock(state_mutex);
        size_t bytes = workspace ? ggml_gallocr_get_buffer_size(workspace, 0) : 0;
        if (merge) {
            bytes += ggml_gallocr_get_buffer_size(merge->allocator, 0);
        }
        for (const auto & layer : layers) {
            if (layer.second->cache_buffer) {
                bytes += ggml_backend_buffer_get_size(layer.second->cache_buffer);
            }
        }
        return bytes;
    }

    ggml_backend_dev_t device(const ggml_tensor * op) const {
        if (!valid || op == nullptr || op->op != GGML_OP_MUL_MAT_ID ||
            registered.count(op->src[0]) == 0) {
            return nullptr;
        }
        return ggml_backend_get_device(compute_backend);
    }

    ggml_backend_dev_t offload_device(const ggml_tensor * op) const {
        if (op == nullptr || !ggml_cpu_expert_cache_supports_ids(op->src[2]) ||
            op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
            return nullptr;
        }
        return device(op);
    }

    void prepare(ggml_backend_sched_t sched, ggml_cgraph * graph) {
        if (!valid) {
            return;
        }

        ggml_backend_synchronize(compute_backend);
        std::lock_guard<std::mutex> lock(state_mutex);
        pending_output = nullptr;
        host_intermediates = graph == nullptr;

        if (graph == nullptr) {
            deferred_starts.clear();
            return;
        }

        const auto root = [](const ggml_tensor * t) {
            while (t != nullptr && t->view_src != nullptr) {
                t = t->view_src;
            }

            return t;
        };

        const auto is_view = [](const ggml_tensor * t) {
            return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
                   t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
        };

        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * down = ggml_graph_node(graph, i);
            const auto dev = offload_device(down);
            if (dev == nullptr) {
                continue;
            }
            int discovered_id = -1;
            if (expert_cache_parse_tensor(down->src[0]->name, discovered_id) != expert_cache_part::down) {
                continue;
            }
            for (auto it = deferred_starts.begin(); it != deferred_starts.end();) {
                it = it->second == discovered_id ? deferred_starts.erase(it) : std::next(it);
            }
            auto & layer = get_layer(discovered_id);
            layer.force_projection = false;
            const auto * act = down->src[1];
            if (act && act->op == GGML_OP_GLU) {
                std::array<const ggml_tensor *, 5> intermediates = { act, act->src[0], act->src[1] };
                for (int j = 0; j < 2; ++j) {
                    const auto * input = act->src[j];
                    if (input && input->op == GGML_OP_VIEW) {
                        intermediates[3 + j] = input->src[0];
                    }
                }
                const auto internal = [&](const ggml_tensor * t) {
                    return t && std::find(intermediates.begin(), intermediates.end(), t) != intermediates.end();
                };
                for (int j = 0; j < ggml_graph_n_nodes(graph); ++j) {
                    const auto * node = ggml_graph_node(graph, j);
                    if (internal(node)) {
                        layer.force_projection |= (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
                    } else if (node != down) {
                        for (const auto * src : node->src) {
                            layer.force_projection |= internal(src);
                        }
                    }
                }
            }
            discover_locked(layer, down);
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            if (!props.caps.events || (down->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                continue;
            }

            bool eligible = true;
            bool consumer = false;
            bool next = true;

            for (int j = 0; j < ggml_graph_n_nodes(graph); ++j) {
                auto * node = ggml_graph_node(graph, j);
                if (node != down && node->op == GGML_OP_MUL_MAT_ID && node->src[0] == down->src[0]) {
                    eligible = false;
                }

                if (root(node) == down && (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    eligible = false;
                }

                if (is_view(node)) {
                    continue;
                }

                const auto backend = ggml_backend_sched_get_tensor_backend(sched, node);
                if (j > i && next) {
                    bool consumes_down = false;
                    for (const auto * src : node->src) {
                        consumes_down |= root(src) == down;
                    }
                    eligible &= consumes_down && backend != nullptr && ggml_backend_get_device(backend) == dev;
                    next = false;
                }

                for (const auto * src : node->src) {
                    if (root(src) == down) {
                        consumer = true;
                        eligible &= backend != nullptr && ggml_backend_get_device(backend) == dev;
                    }
                }
            }

            if (!eligible || !consumer) {
                continue;
            }

            int layer_id = -1;
            expert_cache_parse_tensor(down->src[0]->name, layer_id);

            for (int j = 0; j < i; ++j) {
                const auto * node = ggml_graph_node(graph, j);
                if (node->op != GGML_OP_MUL_MAT_ID || registered.count(node->src[0]) == 0) {
                    continue;
                }

                int id = -1;
                const auto part = expert_cache_parse_tensor(node->src[0]->name, id);
                if (id == layer_id && part != expert_cache_part::none && part != expert_cache_part::down) {
                    deferred_starts[node] = layer_id;
                }
            }
        }
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
        std::lock_guard<std::mutex> lock(state_mutex);

        if (src != pending_output ||
            ggml_backend_get_device(backend) != ggml_backend_get_device(compute_backend)) {
            return false;
        }

        GGML_ASSERT(merge != nullptr);
        auto & m = *merge;
        if (!m.all_cached) {
            ggml_backend_tensor_set_async(compute_backend, m.misses, src->data, 0, ggml_nbytes(src));
        }

        const auto status = ggml_backend_graph_compute_async(compute_backend, m.graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_copy_async(compute_backend, compute_backend, m.output, dst);

        // Copy on the compute stream before reusing the shared output.
        ggml_backend_event_record(m.copied, compute_backend);
        ggml_backend_event_wait(backend, m.copied);
        pending_output = nullptr;
        return true;
    }

    const uint64_t * begin(ggml_tensor * op) {
        if (!valid || op == nullptr || op->op != GGML_OP_MUL_MAT_ID ||
            op->src[0] == nullptr || op->src[2] == nullptr || registered.count(op->src[0]) == 0) {
            return nullptr;
        }

        int layer_id = -1;
        const auto part = expert_cache_parse_tensor(op->src[0]->name, layer_id);
        if (part == expert_cache_part::none || layer_id < 0 || !ggml_cpu_expert_cache_supports_ids(op->src[2])) {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(state_mutex);
        auto & layer = get_layer(layer_id);
        register_part(layer, part, op->src[0]);
        if (host_intermediates) {
            layer.projection_only = true;
        }

        if (part == expert_cache_part::down && (!layer.checked || layer.cache_buffer == nullptr)) {
            discover_locked(layer, op);
            if (!layer.compatible || !prepare_layer(layer, layer_id)) {
                layer.compatible = false;
                return nullptr;
            }
            // Gate/up have already executed, so the discovery batch stays on CPU.
            start_active_locked(layer, op->src[2], /*allow_gpu=*/ false, op);
            return nullptr;
        }
        if (!layer.compatible || layer.cache_buffer == nullptr) {
            return nullptr;
        }

        if (!layer.active) {
            if (layer.projection_only) {
                start_projection_locked(layer, part, op);
            } else if (part == expert_cache_part::down) {
                start_active_locked(layer, op->src[2], /*allow_gpu=*/ false, op);
                return nullptr;
            } else {
                start_active_locked(layer, op->src[2], /*allow_gpu=*/ true, op);
            }
        } else if (layer.active_ids != op->src[2]) {
            return nullptr;
        }

        return layer.active_gpu_launched ? layer.active_ready_masks.data() : nullptr;
    }

    void end(ggml_tensor * op) {
        if (!valid || op == nullptr || op->op != GGML_OP_MUL_MAT_ID ||
            op->src[0] == nullptr || op->src[2] == nullptr) {
            return;
        }

        int layer_id = -1;
        const auto part = expert_cache_parse_tensor(op->src[0]->name, layer_id);
        if (part == expert_cache_part::none || layer_id < 0 || !ggml_cpu_expert_cache_supports_ids(op->src[2])) {
            return;
        }

        expert_cache_layer * layer_ptr = nullptr;
        bool launched = false;
        bool deferred = false;
        int output_rows = 0;
        const std::vector<int32_t> * route_positions = nullptr;
        size_t output_row_bytes = 0;

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            auto lit = layers.find(layer_id);
            if (lit == layers.end()) {
                return;
            }
            layer_ptr = lit->second.get();
            auto & layer = *layer_ptr;
            if (part != expert_cache_part::down && !layer.projection_only) {
                return;
            }
            if (!layer.compatible || !layer.active || layer.active_ids != op->src[2]) {
                return;
            }
            launched = layer.active_gpu_launched;
            deferred = launched && layer.active_deferred;
            route_positions = &layer.active_route_positions;
            output_rows = (int) route_positions->size();
            output_row_bytes = (size_t) op->ne[0] * sizeof(float);
        }

        if (deferred) {
            std::lock_guard<std::mutex> lock(state_mutex);
            ggml_backend_event_record(layer_ptr->reads_done, compute_backend);
            layer_ptr->reads_recorded = true;
            pending_output = op;
            observe_locked(*layer_ptr, layer_id, op->src[2]);
            clear_active_locked(*layer_ptr);
            return;
        }

        if (launched) {
            ggml_backend_synchronize(compute_backend);

            const uint8_t * src = (const uint8_t *) output_ptr;
            for (int i = 0; i < output_rows; ++i) {
                const int32_t route = (*route_positions)[(size_t) i];
                if (route < 0) {
                    continue;
                }
                const size_t token = route / op->ne[1];
                const size_t id = route % op->ne[1];
                memcpy((uint8_t *) op->data + id * op->nb[1] + token * op->nb[2],
                       src + (size_t) i * output_row_bytes, output_row_bytes);
            }
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex);
            // Layer storage stays valid until the workers stop.
            if (part == expert_cache_part::down) {
                observe_locked(*layer_ptr, layer_id, op->src[2]);
            }
            clear_active_locked(*layer_ptr);
        }
    }

private:

    expert_cache_layer & get_layer(int layer_id) {
        auto & ptr = layers[layer_id];
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

    void free_template(expert_cache_template & t) {
        if (t.ctx != nullptr) {
            ggml_free(t.ctx);
            t.ctx = nullptr;
        }
    }

    void free_layer(expert_cache_layer & layer) {
        if (layer.reads_done != nullptr) {
            ggml_backend_event_free(layer.reads_done);
            layer.reads_done = nullptr;
        }
        for (auto & t : layer.templates) {
            if (t) {
                free_template(*t);
            }
        }
        for (auto & templates : layer.projection_templates) {
            for (auto & t : templates) {
                if (t) {
                    free_template(*t);
                }
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
            if (layer.up == nullptr || layer.down == nullptr) {
                return false;
            }
            layer.parts = { layer.up, layer.down };
            if (layer.gate != nullptr) {
                layer.parts.insert(layer.parts.begin(), layer.gate);
            }
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

        for (size_t i = 0; i < layer.parts.size(); ++i) {
            const auto * src = layer.parts[i];
            auto * cached = layer.cache_parts[i];
            if (src == layer.gate)    { layer.cache_gate = cached; }
            if (src == layer.up)      { layer.cache_up = cached; }
            if (src == layer.gate_up) { layer.cache_gate_up = cached; }
            if (src == layer.down)    { layer.cache_down = cached; }
        }

        layer.expert_to_slot.assign((size_t) layer.n_expert, -1);
        layer.slot_to_expert.assign(slots, -1);
        layer.slot_stamp.assign(slots, 0);
        layer.slot_generation.assign(slots, 0);
        layer.slot_ready.assign(slots, 0);
        layer.last_seen.assign((size_t) layer.n_expert, -1);
        layer.admissions.reserve((size_t) layer.n_expert);
        layer.admission_pending.assign((size_t) layer.n_expert, 0);

        return true;
    }

    void discover_locked(expert_cache_layer & layer, const ggml_tensor * down_op) {
        const bool checked = layer.checked;
        const bool projection = layer.projection_only;
        const auto dimensions = std::array<int64_t, 4>{ layer.n_expert_used, layer.input_dim, layer.ffn_dim, layer.output_dim };
        const auto glu_params = layer.glu_params;
        const auto gate_params = layer.gate_params;
        const auto up_params = layer.up_params;
        const auto gate_up_params = layer.gate_up_params;
        const auto down_params = layer.down_params;
        detect_locked(layer, down_op);
        layer.projection_only = host_intermediates || layer.force_projection || !layer.compatible;
        if (layer.projection_only && layer.down && (layer.up || layer.gate_up)) {
            layer.n_expert_used = down_op->src[2]->ne[0];
            layer.input_dim = layer.gate_up ? layer.gate_up->ne[0] : layer.up->ne[0];
            layer.ffn_dim = layer.down->ne[0];
            layer.output_dim = layer.down->ne[1];
            layer.compatible = true;
        }
        const auto new_dimensions = std::array<int64_t, 4>{ layer.n_expert_used, layer.input_dim, layer.ffn_dim, layer.output_dim };
        if (!checked || projection != layer.projection_only || dimensions != new_dimensions ||
            glu_params != layer.glu_params || gate_params != layer.gate_params || up_params != layer.up_params ||
            gate_up_params != layer.gate_up_params || down_params != layer.down_params) {
            for (auto & t : layer.templates) {
                if (t) {
                    free_template(*t);
                }
            }
            layer.templates.clear();
            for (auto & templates : layer.projection_templates) {
                for (auto & t : templates) {
                    if (t) {
                        free_template(*t);
                    }
                }
                templates.clear();
            }
            const size_t count = ((size_t) layer.n_expert_used + 1) * EXPERT_CACHE_BATCH_SIZES;
            layer.templates.resize(count);
            layer.template_failed.assign(count, 0);
            for (size_t i = 0; i < layer.projection_templates.size(); ++i) {
                layer.projection_templates[i].resize(count);
                layer.projection_failed[i].assign(count, 0);
            }
        }
    }

    void detect_locked(expert_cache_layer & layer, const ggml_tensor * down_op) {
        layer.checked = true;
        layer.compatible = false;

        if (layer.down == nullptr) {
            return;
        }
        if (down_op->src[0] != layer.down || down_op->src[1] == nullptr || down_op->src[2] == nullptr) {
            return;
        }

        const ggml_tensor * act = down_op->src[1];
        if (act->op != GGML_OP_GLU || act->src[0] == nullptr || act->src[1] == nullptr) {
            return;
        }

        const ggml_tensor * input = nullptr;

        if (layer.gate_up != nullptr) {
            const ggml_tensor * gate_view = act->src[0];
            const ggml_tensor * up_view   = act->src[1];

            if (gate_view->op != GGML_OP_VIEW || up_view->op != GGML_OP_VIEW ||
                gate_view->src[0] == nullptr || gate_view->src[0] != up_view->src[0]) {
                return;
            }

            const ggml_tensor * gate_up_op = gate_view->src[0];
            if (gate_up_op->op != GGML_OP_MUL_MAT_ID ||
                gate_up_op->src[0] != layer.gate_up ||
                gate_up_op->src[1] == nullptr ||
                gate_up_op->src[2] != down_op->src[2] ||
                !supports_input(gate_up_op) ||
                !ggml_is_contiguous(gate_up_op->src[1])) {
                return;
            }

            input = gate_up_op->src[1];
            if (input->ne[1] != 1 || input->ne[2] != down_op->src[2]->ne[1] || input->ne[3] != 1 ||
                gate_up_op->ne[2] != input->ne[2] || gate_view->ne[2] != input->ne[2] ||
                up_view->ne[2] != input->ne[2] || down_op->ne[2] != input->ne[2]) {
                return;
            }
            if (gate_view->ne[0] != up_view->ne[0] ||
                gate_view->ne[1] != up_view->ne[1] ||
                gate_up_op->ne[0] != 2 * gate_view->ne[0] ||
                gate_up_op->ne[1] != gate_view->ne[1] ||
                act->ne[0] != gate_view->ne[0] ||
                act->ne[1] != gate_view->ne[1] ||
                down_op->ne[1] != down_op->src[2]->ne[0]) {
                return;
            }

            memcpy(layer.gate_up_params.data(), gate_up_op->op_params, GGML_MAX_OP_PARAMS);
        } else {
            if (layer.gate == nullptr || layer.up == nullptr) {
                return;
            }

            const ggml_tensor * gate_op = act->src[0];
            const ggml_tensor * up_op   = act->src[1];
            if (gate_op->op != GGML_OP_MUL_MAT_ID || up_op->op != GGML_OP_MUL_MAT_ID ||
                gate_op->src[0] != layer.gate || up_op->src[0] != layer.up ||
                gate_op->src[1] == nullptr || gate_op->src[1] != up_op->src[1] ||
                gate_op->src[2] != down_op->src[2] || up_op->src[2] != down_op->src[2] ||
                !supports_input(gate_op) || !supports_input(up_op) || !ggml_is_contiguous(gate_op->src[1])) {
                return;
            }

            input = gate_op->src[1];
            if (input->ne[1] != 1 || input->ne[2] != down_op->src[2]->ne[1] || input->ne[3] != 1 ||
                gate_op->ne[2] != input->ne[2] || up_op->ne[2] != input->ne[2] || down_op->ne[2] != input->ne[2]) {
                return;
            }
            if (gate_op->ne[0] != up_op->ne[0] || gate_op->ne[1] != up_op->ne[1] ||
                act->ne[0] != gate_op->ne[0] || act->ne[1] != gate_op->ne[1] ||
                down_op->ne[1] != down_op->src[2]->ne[0]) {
                return;
            }

            memcpy(layer.gate_params.data(), gate_op->op_params, GGML_MAX_OP_PARAMS);
            memcpy(layer.up_params.data(), up_op->op_params, GGML_MAX_OP_PARAMS);
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
        memcpy(layer.down_params.data(), down_op->op_params, GGML_MAX_OP_PARAMS);
        layer.compatible = true;
    }

    bool allocate_template(expert_cache_template & t) {
        if (t.workspace_generation != 0 && t.workspace_generation == workspace_generation) {
            return true;
        }
        ggml_backend_synchronize(compute_backend);
        if (t.workspace_generation != 0) {
            const auto begin = reinterpret_cast<uintptr_t>(ggml_get_mem_buffer(t.ctx));
            const auto end = begin + ggml_get_mem_size(t.ctx);
            for (auto * tensor = ggml_get_first_tensor(t.ctx); tensor; tensor = ggml_get_next_tensor(t.ctx, tensor)) {
                const auto * root = tensor;
                while (root->view_src) {
                    root = root->view_src;
                }
                const auto address = reinterpret_cast<uintptr_t>(root);
                if (address < begin || address >= end) {
                    continue;
                }
                tensor->data = nullptr;
                tensor->buffer = nullptr;
            }
        }
        if (workspace == nullptr) {
            workspace = ggml_gallocr_new(device_buft);
        }
        const size_t before = ggml_gallocr_get_buffer_size(workspace, 0);
        if (!ggml_gallocr_alloc_graph(workspace, t.graph)) {
            return false;
        }
        if (before != ggml_gallocr_get_buffer_size(workspace, 0)) {
            ++workspace_generation;
        }
        t.workspace_generation = workspace_generation;
        return true;
    }

    expert_cache_template * get_template_locked(expert_cache_layer & layer, int hit_count, int batch_size) {
        int batch_idx = 0;
        while ((1 << batch_idx) < batch_size) {
            ++batch_idx;
        }
        GGML_ASSERT(batch_idx < EXPERT_CACHE_BATCH_SIZES && batch_size == (1 << batch_idx));
        const size_t key = (size_t) hit_count * EXPERT_CACHE_BATCH_SIZES + batch_idx;
        if (hit_count <= 0 || hit_count > layer.n_expert_used ||
            key >= layer.templates.size() || layer.template_failed[key]) {
            return nullptr;
        }
        if (layer.templates[key]) {
            return allocate_template(*layer.templates[key]) ? layer.templates[key].get() : nullptr;
        }

        auto t = std::make_unique<expert_cache_template>();
        const size_t graph_size = 24;
        ggml_init_params params = {
            /*.mem_size   =*/ 32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false) + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        t->ctx = ggml_init(params);
        if (t->ctx == nullptr) {
            layer.template_failed[key] = 1;
            return nullptr;
        }

        // Native Q8_1 skips the input conversion.
        const auto input_type = layer.gate_up ? layer.gate_up->type : layer.gate->type;
        t->input_q8 = (input_type == GGML_TYPE_ZNQ2 || input_type == GGML_TYPE_ZNQ3 || input_type == GGML_TYPE_ZNQ4) &&
            layer.input_dim % ggml_blck_size(GGML_TYPE_Q8_1) == 0;
        t->input = ggml_new_tensor_3d(t->ctx, t->input_q8 ? GGML_TYPE_Q8_1 : GGML_TYPE_F32, layer.input_dim, 1, batch_size);
        t->ids = ggml_new_tensor_2d(t->ctx, GGML_TYPE_I32, hit_count, batch_size);

        t->graph = ggml_new_graph_custom(t->ctx, graph_size, false);
        ggml_tensor * act = nullptr;

        if (layer.cache_gate_up != nullptr && t->input_q8) {
            auto * weights = layer.cache_gate_up;
            const int64_t n_ff = weights->ne[1] / 2;
            auto * gate_weights = ggml_view_3d(t->ctx, weights,
                    weights->ne[0], n_ff, weights->ne[2], weights->nb[1], weights->nb[2], 0);
            auto * up_weights = ggml_view_3d(t->ctx, weights,
                    weights->ne[0], n_ff, weights->ne[2], weights->nb[1], weights->nb[2], n_ff * weights->nb[1]);

            // Keep weight views before both matmuls so the native Q8_1 GLU/down fusion can match.
            ggml_build_forward_expand(t->graph, gate_weights);
            ggml_build_forward_expand(t->graph, up_weights);
            auto * gate = ggml_mul_mat_id(t->ctx, gate_weights, t->input, t->ids);
            auto * up   = ggml_mul_mat_id(t->ctx, up_weights,   t->input, t->ids);
            memcpy(gate->op_params, layer.gate_up_params.data(), GGML_MAX_OP_PARAMS);
            memcpy(up->op_params,   layer.gate_up_params.data(), GGML_MAX_OP_PARAMS);
            act = ggml_glu_split(t->ctx, gate, up, layer.glu_op);
        } else if (layer.cache_gate_up != nullptr) {
            auto * gate_up = ggml_mul_mat_id(t->ctx, layer.cache_gate_up, t->input, t->ids);
            memcpy(gate_up->op_params, layer.gate_up_params.data(), GGML_MAX_OP_PARAMS);

            GGML_ASSERT(gate_up->ne[0] % 2 == 0);
            const int64_t n_ff = gate_up->ne[0] / 2;

            auto * gate = ggml_view_3d(
                    t->ctx, gate_up,
                    n_ff, gate_up->ne[1], gate_up->ne[2],
                    gate_up->nb[1], gate_up->nb[2], 0);

            auto * up = ggml_view_3d(
                    t->ctx, gate_up,
                    n_ff, gate_up->ne[1], gate_up->ne[2],
                    gate_up->nb[1], gate_up->nb[2],
                    n_ff * gate_up->nb[0]);

            act = ggml_glu_split(t->ctx, gate, up, layer.glu_op);
        } else {
            auto * gate = ggml_mul_mat_id(t->ctx, layer.cache_gate, t->input, t->ids);
            auto * up   = ggml_mul_mat_id(t->ctx, layer.cache_up,   t->input, t->ids);
            memcpy(gate->op_params, layer.gate_params.data(), GGML_MAX_OP_PARAMS);
            memcpy(up->op_params,   layer.up_params.data(),   GGML_MAX_OP_PARAMS);

            act = ggml_glu_split(t->ctx, gate, up, layer.glu_op);
        }

        memcpy(act->op_params, layer.glu_params.data(), GGML_MAX_OP_PARAMS);

        t->output = ggml_mul_mat_id(t->ctx, layer.cache_down, act, t->ids);
        memcpy(t->output->op_params, layer.down_params.data(), GGML_MAX_OP_PARAMS);
        ggml_build_forward_expand(t->graph, t->output);

        if (!allocate_template(*t)) {
            free_template(*t);
            layer.template_failed[key] = 1;
            return nullptr;
        }

        for (int i = 0; i < ggml_graph_n_nodes(t->graph); ++i) {
            if (!ggml_backend_supports_op(compute_backend, ggml_graph_node(t->graph, i))) {
                free_template(*t);
                layer.template_failed[key] = 1;
                return nullptr;
            }
        }

        auto * ret = t.get();
        layer.templates[key] = std::move(t);
        return ret;
    }

    expert_cache_template * get_projection_template_locked(
            expert_cache_layer & layer, expert_cache_part part, const ggml_tensor * op, int hit_count, int batch_size) {
        size_t batch_idx = 0;
        while ((1 << batch_idx) < batch_size) {
            ++batch_idx;
        }
        const size_t key = hit_count * EXPERT_CACHE_BATCH_SIZES + batch_idx;
        auto & templates = layer.projection_templates[(size_t) part];
        auto & failed = layer.projection_failed[(size_t) part];
        if (key >= templates.size() || failed[key]) {
            return nullptr;
        }
        if (templates[key] && memcmp(templates[key]->output->op_params, op->op_params, GGML_MAX_OP_PARAMS) != 0) {
            free_template(*templates[key]);
            templates[key].reset();
        }
        if (templates[key]) {
            return allocate_template(*templates[key]) ? templates[key].get() : nullptr;
        }
        ggml_tensor * weights = nullptr;
        switch (part) {
            case expert_cache_part::gate:    weights = layer.cache_gate; break;
            case expert_cache_part::up:      weights = layer.cache_up; break;
            case expert_cache_part::gate_up: weights = layer.cache_gate_up; break;
            case expert_cache_part::down:    weights = layer.cache_down; break;
            case expert_cache_part::none: return nullptr;
        }
        if (!weights) {
            return nullptr;
        }
        auto t = std::make_unique<expert_cache_template>();
        t->ctx = ggml_init({ 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true });
        if (!t->ctx) {
            return nullptr;
        }
        t->input = ggml_new_tensor_3d(t->ctx, GGML_TYPE_F32, weights->ne[0], hit_count, batch_size);
        t->ids = ggml_new_tensor_2d(t->ctx, GGML_TYPE_I32, hit_count, batch_size);
        t->output = ggml_mul_mat_id(t->ctx, weights, t->input, t->ids);
        memcpy(t->output->op_params, op->op_params, GGML_MAX_OP_PARAMS);
        t->graph = ggml_new_graph_custom(t->ctx, 8, false);
        ggml_build_forward_expand(t->graph, t->output);
        if (!ggml_backend_supports_op(compute_backend, t->output) || !allocate_template(*t)) {
            free_template(*t);
            failed[key] = 1;
            return nullptr;
        }
        auto * result = t.get();
        templates[key] = std::move(t);
        return result;
    }

    bool prepare_merge_locked(expert_cache_layer & layer, const ggml_tensor * ids) {
        GGML_ASSERT(pending_output == nullptr);
        if (!merge) {
            auto m = std::make_unique<expert_cache_merge>();
            ggml_init_params params = {
                /*.mem_size   =*/ 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            m->ctx = ggml_init(params);
            m->allocator = ggml_gallocr_new(device_buft);
            m->copied = ggml_backend_event_new(ggml_backend_get_device(compute_backend));
            if (m->ctx == nullptr || m->allocator == nullptr || m->copied == nullptr) {
                return false;
            }
            merge = std::move(m);
        }

        auto & m = *merge;
        if (layer.output_dim > m.output_capacity || layer.n_expert_used > m.expert_capacity || ids->ne[1] > m.token_capacity) {
            ggml_backend_synchronize(compute_backend);
            const int64_t output_capacity = std::max(m.output_capacity, layer.output_dim);
            const int64_t expert_capacity = std::max(m.expert_capacity, layer.n_expert_used);
            int64_t token_capacity = m.token_capacity;
            while (token_capacity < ids->ne[1]) {
                token_capacity *= 2;
            }
            m.build(output_capacity, expert_capacity, token_capacity);
            if (!ggml_gallocr_reserve(m.allocator, m.graph)) {
                return false;
            }
            m.output_capacity = output_capacity;
            m.expert_capacity = expert_capacity;
            m.token_capacity = token_capacity;
        }

        if (m.output->ne[0] != layer.output_dim || m.output->ne[1] != layer.n_expert_used || m.output->ne[2] != ids->ne[1]) {
            m.build(layer.output_dim, layer.n_expert_used, ids->ne[1]);
        }
        if (m.output->buffer == nullptr) {
            for (int i = 0; i < ggml_graph_n_nodes(m.graph); ++i) {
                if (!ggml_backend_supports_op(compute_backend, ggml_graph_node(m.graph, i))) {
                    return false;
                }
            }
            if (!ggml_gallocr_alloc_graph(m.allocator, m.graph) ||
                ggml_backend_view_init(m.misses) != GGML_STATUS_SUCCESS) {
                return false;
            }
        }

        if (layer.reads_done == nullptr) {
            layer.reads_done = ggml_backend_event_new(ggml_backend_get_device(compute_backend));
            if (layer.reads_done == nullptr) {
                return false;
            }
        }

        const size_t rows = (size_t) layer.n_expert_used * ids->ne[1];
        layer.merge_routes.resize(rows);
        GGML_ASSERT(layer.active_route_positions.size() <= rows);
        for (size_t i = 0; i < rows; ++i) {
            layer.merge_routes[i] = rows + i;
        }
        for (size_t i = 0; i < layer.active_route_positions.size(); ++i) {
            const int32_t route = layer.active_route_positions[i];
            if (route >= 0) {
                layer.merge_routes[route] = i;
            }
        }
        m.all_cached = (size_t) layer.active_hits == rows;

        return true;
    }

    bool ensure_input(size_t size) {
        if (input_size >= size) {
            return true;
        }
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
            input_buffer = nullptr;
            input_ptr = nullptr;
        }
        input_fallback.clear();

        if (host_buft != nullptr) {
            input_buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (input_buffer != nullptr) {
                input_ptr = ggml_backend_buffer_get_base(input_buffer);
                input_size = size;
                return true;
            }
        }
        input_fallback.resize(size);
        input_ptr = input_fallback.data();
        input_size = size;
        return input_ptr != nullptr;
    }

    bool ensure_output(size_t size) {
        if (output_size >= size) {
            return true;
        }
        if (output_buffer != nullptr) {
            ggml_backend_buffer_free(output_buffer);
            output_buffer = nullptr;
            output_ptr = nullptr;
        }
        output_fallback.clear();

        if (host_buft != nullptr) {
            output_buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (output_buffer != nullptr) {
                output_ptr = ggml_backend_buffer_get_base(output_buffer);
                output_size = size;
                return true;
            }
        }
        output_fallback.resize(size);
        output_ptr = output_fallback.data();
        output_size = size;
        return output_ptr != nullptr;
    }

    void start_projection_locked(expert_cache_layer & layer, expert_cache_part part, ggml_tensor * op) {
        layer.active_part = part;
        start_active_locked(layer, op->src[2], true, op);
    }

    void start_active_locked(expert_cache_layer & layer, const ggml_tensor * ids, bool allow_gpu, ggml_tensor * op) {
        layer.active = true;

        layer.active_deferred = !layer.projection_only && deferred_starts.count(op) != 0;
        layer.active_ids = ids;
        layer.active_ready_masks.assign((size_t) ids->ne[1], 0);
        layer.active_slots.clear();
        layer.active_route_positions.clear();
        layer.active_route_slots.assign((size_t) ids->ne[0] * ids->ne[1], -1);
        layer.active_batches.clear();
        layer.active_hits = 0;
        layer.active_gpu_launched = false;

        for (int64_t token = 0; token < ids->ne[1]; ++token) {
            for (int64_t i = 0; i < ids->ne[0]; ++i) {
                const size_t route = (size_t) token * ids->ne[0] + i;
                const int32_t expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + i * ids->nb[0]);
                if (expert < 0 || expert >= layer.n_expert) {
                    continue;
                }
                const int32_t slot = layer.expert_to_slot[(size_t) expert];
                if (slot >= 0) {
                    layer.active_route_slots[route] = slot;
                    if (layer.slot_ready[(size_t) slot]) {
                        layer.active_ready_masks[(size_t) token] |= UINT64_C(1) << i;
                        layer.active_slots.push_back(slot);
                        layer.active_route_positions.push_back((int32_t) route);
                    }
                }
            }
        }
        layer.active_hits = (int) layer.active_slots.size();

        const bool projection = layer.projection_only;
        if (!allow_gpu || layer.active_hits == 0 || op == nullptr || op->src[1] == nullptr ||
            !supports_input(op) || (op->src[1]->ne[1] != 1 && (!projection || op->src[1]->ne[1] != ids->ne[0])) ||
            op->src[1]->ne[2] != ids->ne[1] || op->src[1]->ne[3] != 1) {
            return;
        }

        const int64_t input_dim = projection ? op->src[0]->ne[0] : layer.input_dim;
        const int64_t output_dim = projection ? op->ne[0] : layer.output_dim;
        const size_t output_row_bytes = (size_t) output_dim * sizeof(float);
        const size_t input_row_bytes = (size_t) input_dim * sizeof(float);
        if (!ensure_input(GGML_PAD(input_row_bytes * (projection ? ids->ne[0] : 1), 64) * ids->ne[1])) {
            return;
        }

        const auto get_template = [&](int hits, int batch) {
            return projection ? get_projection_template_locked(layer, layer.active_part, op, hits, batch) :
                get_template_locked(layer, hits, batch);
        };
        const int max_batch = 1 << (EXPERT_CACHE_BATCH_SIZES - 1);
        if (get_template(layer.n_expert_used, max_batch) == nullptr) {
            return;
        }

        std::vector<int> token_hits((size_t) ids->ne[1], 0);
        std::vector<int64_t> tokens;
        tokens.reserve((size_t) ids->ne[1]);
        for (int64_t token = 0; token < ids->ne[1]; ++token) {
            int hits = 0;
            for (int id = 0; id < ids->ne[0]; ++id) {
                hits += (layer.active_ready_masks[(size_t) token] >> id) & 1;
            }
            token_hits[(size_t) token] = hits;
            if (hits != 0) {
                tokens.push_back(token);
            }
        }

        layer.active_slots.clear();
        layer.active_route_positions.clear();
        size_t input_offset = 0;
        for (size_t first = 0; first < tokens.size();) {
            int batch_size = 1;
            while (batch_size < (1 << (EXPERT_CACHE_BATCH_SIZES - 1)) &&
                    first + 2 * batch_size <= tokens.size()) {
                batch_size *= 2;
            }

            int padded_hits = 0;
            expert_cache_template * t = nullptr;
            while (true) {
                padded_hits = 0;
                for (int i = 0; i < batch_size; ++i) {
                    padded_hits = std::max(padded_hits, token_hits[(size_t) tokens[first + i]]);
                }

                t = get_template(padded_hits, batch_size);
                if (t != nullptr || batch_size == 1) {
                    break;
                }
                batch_size /= 2;
            }
            if (t == nullptr) {
                return;
            }

            const size_t route_offset = layer.active_slots.size();
            for (int i = 0; i < batch_size; ++i) {
                const int64_t token = tokens[first + i];
                const auto * input_data = (const uint8_t *) op->src[1]->data + token * op->src[1]->nb[2];
                auto * packed = (uint8_t *) input_ptr + input_offset + i * t->input->nb[2];
                if (!projection) {
                    if (t->input_q8) {
                        pack_input_q8_1(input_data, op->src[1]->type, packed, input_dim);
                    } else {
                        pack_input_f32(input_data, op->src[1]->type, packed, input_dim);
                    }
                }

                int hits = 0;
                int32_t pad_slot = -1;
                for (int id = 0; id < ids->ne[0]; ++id) {
                    if (layer.active_ready_masks[(size_t) token] & (UINT64_C(1) << id)) {
                        const size_t route = (size_t) token * ids->ne[0] + id;
                        const int32_t slot = layer.active_route_slots[route];
                        layer.active_slots.push_back(slot);
                        layer.active_route_positions.push_back((int32_t) route);
                        if (pad_slot < 0) {
                            pad_slot = slot;
                        }
                        if (projection) {
                            const auto * row = input_data + (id % op->src[1]->ne[1]) * op->src[1]->nb[1];
                            pack_input_f32(row, op->src[1]->type, packed + hits * t->input->nb[1], input_dim);
                        }
                        ++hits;
                    }
                }

                GGML_ASSERT(hits == token_hits[(size_t) token] && pad_slot >= 0);
                for (; hits < padded_hits; ++hits) {
                    layer.active_slots.push_back(pad_slot);
                    layer.active_route_positions.push_back(-1);
                    if (projection) {
                        memcpy(packed + hits * t->input->nb[1], packed, input_row_bytes);
                    }
                }
            }

            layer.active_batches.push_back({ t, route_offset, input_offset });
            input_offset += GGML_PAD(ggml_nbytes(t->input), 64);
            first += batch_size;
        }

        GGML_ASSERT(layer.active_slots.size() == layer.active_route_positions.size());
        if (layer.active_deferred && !prepare_merge_locked(layer, ids)) {
            layer.active_deferred = false;
        }

        if (!layer.active_deferred && !ensure_output(output_row_bytes * layer.active_route_positions.size())) {
            return;
        }

        // Templates and host buffers are complete before any asynchronous use.
        if (layer.active_deferred) {
            ggml_backend_tensor_set_async(compute_backend, merge->ids,
                    layer.merge_routes.data(), 0, ggml_nbytes(merge->ids));
        }

        for (const auto & batch : layer.active_batches) {
            auto * t = batch.t;
            ggml_backend_tensor_set_async(compute_backend, t->input,
                    (const uint8_t *) input_ptr + batch.input_offset, 0, ggml_nbytes(t->input));
            ggml_backend_tensor_set_async(compute_backend, t->ids,
                    layer.active_slots.data() + batch.route_offset, 0, ggml_nbytes(t->ids));
            const enum ggml_status status = ggml_backend_graph_compute_async(compute_backend, t->graph);
            if (status != GGML_STATUS_SUCCESS) {
                ggml_backend_synchronize(compute_backend);
                return;
            }

            if (layer.active_deferred) {
                auto slice = *t->output;
                slice.buffer = merge->packed->buffer;
                slice.data = (uint8_t *) merge->packed->data + batch.route_offset * output_row_bytes;
                ggml_backend_tensor_copy_async(compute_backend, compute_backend, t->output, &slice);
            } else {
                ggml_backend_tensor_get_async(compute_backend, t->output,
                        (uint8_t *) output_ptr + batch.route_offset * output_row_bytes, 0, ggml_nbytes(t->output));
            }
        }
        layer.active_gpu_launched = true;
        gpu_hits.fetch_add(layer.active_hits, std::memory_order_relaxed);
    }

    static void clear_active_locked(expert_cache_layer & layer) {
        layer.active = false;
        layer.active_ids = nullptr;
        if (!layer.active_deferred) {
            layer.active_ready_masks.clear();
            layer.active_slots.clear();
            layer.active_route_positions.clear();
            layer.active_route_slots.clear();
            layer.active_batches.clear();
        }
        layer.active_hits = 0;
        layer.active_gpu_launched = false;
    }

    void observe_locked(expert_cache_layer & layer, int layer_id, const ggml_tensor * ids) {
        const int64_t first_token = layer.observed_tokens;
        layer.observed_tokens += ids->ne[1];
        layer.admissions.clear();

        // Classify all routes before admissions can replace any frozen slot.
        for (int64_t t = 0; t < ids->ne[1]; ++t) {
            const int64_t token = first_token + t;
            for (int64_t i = 0; i < ids->ne[0]; ++i) {
                const int32_t expert = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + i * ids->nb[0]);
                if (expert < 0 || expert >= layer.n_expert) {
                    continue;
                }

                const int32_t slot_at_begin = layer.active_route_slots[(size_t) t * ids->ne[0] + i];

                if (slot_at_begin >= 0) {
                    if ((size_t) slot_at_begin < layer.slot_stamp.size() &&
                        layer.slot_to_expert[(size_t) slot_at_begin] == expert) {
                        layer.slot_stamp[(size_t) slot_at_begin] = ++layer.stamp;
                    }
                } else {
                    const int64_t prev = layer.last_seen[(size_t) expert];
                    const bool admit = admit_window == 0 ||
                        (prev >= 0 && token >= prev && (uint64_t) (token - prev) <= admit_window);
                    if (admit && !layer.admission_pending[(size_t) expert]) {
                        layer.admission_pending[(size_t) expert] = 1;
                        layer.admissions.push_back(expert);
                    }
                }

                layer.last_seen[(size_t) expert] = token;
            }
        }

        for (const int32_t expert : layer.admissions) {
            layer.admission_pending[(size_t) expert] = 0;
            if (layer.expert_to_slot[(size_t) expert] < 0) {
                admit_expert_locked(layer, layer_id, expert);
            }
        }
    }


    void admit_expert_locked(expert_cache_layer & layer, int layer_id, int32_t expert) {
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

        {
            std::lock_guard<std::mutex> qlock(queue_mutex);
            queue.push_back({ layer_id, expert, slot, generation });
        }
        queue_cv.notify_one();
    }



    bool ensure_staging(uint32_t index, size_t size) {
        GGML_ASSERT(index < staging.size() && staging[index]);
        auto & s = *staging[index];
        if (s.size >= size) {
            return s.ptr != nullptr;
        }
        if (s.buffer != nullptr) {
            ggml_backend_buffer_free(s.buffer);
            s.buffer = nullptr;
            s.ptr = nullptr;
        }
        s.fallback.clear();

        if (host_buft != nullptr) {
            s.buffer = ggml_backend_buft_alloc_buffer(host_buft, size);
            if (s.buffer != nullptr) {
                s.ptr = ggml_backend_buffer_get_base(s.buffer);
                s.size = size;
                return s.ptr != nullptr;
            }
        }

        s.fallback.resize(size);
        s.ptr = s.fallback.data();
        s.size = size;
        return s.ptr != nullptr;
    }

    void release_staging(uint32_t index) {
        GGML_ASSERT(index < staging.size() && staging[index]);
        auto & s = *staging[index];
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.available = true;
        }
        s.cv.notify_one();
    }

    void worker_main_convert(uint32_t staging_index) {
        GGML_ASSERT(staging_index < staging.size() && staging[staging_index]);
        auto & s = *staging[staging_index];

        for (;;) {
            // One pinned buffer belongs to each converter. It cannot be reused
            // until the single uploader has finished the preceding H2D copy.
            {
                std::unique_lock<std::mutex> lock(s.mutex);
                s.cv.wait(lock, [&s]() { return s.available; });
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
                auto lit = layers.find(req.layer);
                if (lit == layers.end()) {
                    continue;
                }
                auto & layer = *lit->second;
                if (req.slot < 0 || (size_t) req.slot >= layer.slot_generation.size() ||
                    layer.slot_generation[(size_t) req.slot] != req.generation ||
                    layer.slot_to_expert[(size_t) req.slot] != req.expert) {
                    continue;
                }
                bundle_bytes = layer.bundle_bytes;
                parts = layer.parts;
                part_bytes = layer.part_bytes;
            }

            if (!ensure_staging(staging_index, bundle_bytes)) {
                continue;
            }

            uint8_t * dst = (uint8_t *) s.ptr;
            bool converted = true;
            for (size_t i = 0; i < parts.size(); ++i) {
                if (!copy_native_expert(parts[i], req.expert, dst, part_bytes[i])) {
                    converted = false;
                    break;
                }
                dst += part_bytes[i];
            }
            if (!converted) {
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(s.mutex);
                GGML_ASSERT(s.available);
                s.available = false;
            }
            {
                std::lock_guard<std::mutex> lock(converted_mutex);
                converted_queue.push_back({ req, staging_index });
            }
            converted_cv.notify_one();
        }
    }

    void worker_main_upload() {
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
            GGML_ASSERT(job.staging_index < staging.size() && staging[job.staging_index]);
            auto & s = *staging[job.staging_index];

            std::vector<ggml_tensor *> cache_parts;
            std::vector<size_t> part_bytes;
            bool current = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                auto lit = layers.find(req.layer);
                if (lit != layers.end()) {
                    auto & layer = *lit->second;
                    if (req.slot >= 0 && (size_t) req.slot < layer.slot_generation.size() &&
                        layer.slot_generation[(size_t) req.slot] == req.generation &&
                        layer.slot_to_expert[(size_t) req.slot] == req.expert) {
                        cache_parts = layer.cache_parts;
                        part_bytes = layer.part_bytes;
                        current = true;
                        if (layer.reads_recorded) {
                            ggml_backend_event_wait(upload_backend, layer.reads_done);
                        }
                    }
                }
            }

            if (!current || cache_parts.size() != part_bytes.size()) {
                release_staging(job.staging_index);
                continue;
            }

            size_t staging_offset = 0;
            for (size_t i = 0; i < part_bytes.size(); ++i) {
                const size_t bytes = part_bytes[i];
                const size_t slot_offset = (size_t) req.slot * cache_parts[i]->nb[2];
                ggml_backend_tensor_set_async(upload_backend, cache_parts[i],
                        (const uint8_t *) s.ptr + staging_offset, slot_offset, bytes);
                staging_offset += bytes;
            }
            ggml_backend_synchronize(upload_backend);

            {
                std::lock_guard<std::mutex> lock(state_mutex);
                auto lit = layers.find(req.layer);
                if (lit != layers.end()) {
                    auto & layer = *lit->second;
                    if ((size_t) req.slot < layer.slot_generation.size() &&
                        layer.slot_generation[(size_t) req.slot] == req.generation &&
                        layer.slot_to_expert[(size_t) req.slot] == req.expert) {
                        layer.slot_ready[(size_t) req.slot] = 1;
                    }
                }
            }

            release_staging(job.staging_index);
        }
    }


private:
    const uint32_t slots;
    const uint32_t admit_window;
    const uint32_t convert_workers;
    ggml_backend_t upload_backend = nullptr;
    ggml_backend_t compute_backend = nullptr;
    ggml_backend_buffer_type_t device_buft = nullptr;
    ggml_backend_buffer_type_t host_buft = nullptr;
    ggml_gallocr_t workspace = nullptr;
    uint64_t workspace_generation = 1;
    std::unique_ptr<expert_cache_merge> merge;
    bool valid = false;
    bool host_intermediates = false;
    std::atomic<uint64_t> gpu_hits{0};
    std::unordered_set<const ggml_tensor *> registered;

    std::mutex state_mutex;
    std::unordered_map<int, std::unique_ptr<expert_cache_layer>> layers;
    std::unordered_map<const ggml_tensor *, int> deferred_starts;
    const ggml_tensor * pending_output = nullptr;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<expert_cache_request> queue;
    bool stop = false;

    std::vector<std::unique_ptr<expert_cache_staging>> staging;
    std::vector<std::thread> convert_threads;
    std::mutex converted_mutex;
    std::condition_variable converted_cv;
    std::deque<expert_cache_converted_request> converted_queue;
    bool converters_done = false;
    std::thread upload_thread;


    ggml_backend_buffer_t input_buffer = nullptr;
    void * input_ptr = nullptr;
    size_t input_size = 0;
    std::vector<uint8_t> input_fallback;

    ggml_backend_buffer_t output_buffer = nullptr;
    void * output_ptr = nullptr;
    size_t output_size = 0;
    std::vector<uint8_t> output_fallback;


};


} // namespace


struct ggml_backend_cpu_expert_cache {
    std::vector<std::unique_ptr<expert_cache>> devices;
    std::unordered_map<const ggml_tensor *, expert_cache *> weights;

    expert_cache * find(const ggml_tensor * op) const {
        if (!op || op->op != GGML_OP_MUL_MAT_ID) {
            return nullptr;
        }
        const auto it = weights.find(op->src[0]);
        return it == weights.end() ? nullptr : it->second;
    }
};

extern "C" ggml_backend_cpu_expert_cache_t ggml_backend_cpu_expert_cache_create() {
    return new ggml_backend_cpu_expert_cache;
}

extern "C" void ggml_backend_cpu_expert_cache_free(ggml_backend_cpu_expert_cache_t cache) {
    delete cache;
}

extern "C" bool ggml_backend_cpu_expert_cache_add_device(
        ggml_backend_cpu_expert_cache_t cache, uint32_t slots, uint32_t admit_window,
        uint32_t convert_workers, ggml_backend_dev_t device) {
    if (!cache || !device || !slots || !convert_workers) {
        return false;
    }
    for (const auto & entry : cache->devices) {
        if (entry->get_device() == device) {
            return false;
        }
    }
    auto entry = std::make_unique<expert_cache>(slots, admit_window, convert_workers, device);
    if (!entry->is_valid()) {
        return false;
    }
    cache->devices.push_back(std::move(entry));
    return true;
}

extern "C" void ggml_backend_cpu_expert_cache_register(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_dev_t device, const ggml_tensor * weights) {
    if (!cache || !weights) {
        return;
    }
    for (const auto & entry : cache->devices) {
        if (entry->get_device() == device) {
            entry->register_weights(weights);
            cache->weights[weights] = entry.get();
            return;
        }
    }
}

extern "C" size_t ggml_backend_cpu_expert_cache_memory(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_dev_t device) {
    if (cache) {
        for (const auto & entry : cache->devices) {
            if (entry->get_device() == device) {
                return entry->memory();
            }
        }
    }
    return 0;
}

extern "C" uint64_t ggml_backend_cpu_expert_cache_hits(ggml_backend_cpu_expert_cache_t cache) {
    uint64_t hits = 0;
    if (cache) {
        for (const auto & entry : cache->devices) {
            hits += entry->hits();
        }
    }
    return hits;
}

extern "C" size_t ggml_backend_cpu_expert_cache_get_buffer_size(
        ggml_backend_dev_t device, int64_t input_dim, int64_t ffn_dim, int64_t output_dim, int64_t n_expert_used) {
    if (input_dim <= 0 || ffn_dim <= 0 || output_dim <= 0 || n_expert_used <= 0 || n_expert_used > 64) {
        return 0;
    }

    expert_cache_merge m;
    m.ctx = ggml_init({ 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true });
    GGML_ASSERT(m.ctx != nullptr);
    const auto buft = ggml_backend_dev_buffer_type(device);
    m.allocator = ggml_gallocr_new(buft);
    m.build(output_dim, n_expert_used, GGML_CPU_EXPERT_CACHE_MAX_TOKENS);
    size_t merge_bytes = 0;
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);
    if (props.caps.events) {
        ggml_gallocr_reserve_n_size(m.allocator, m.graph, nullptr, nullptr, &merge_bytes);
    }

    // Bound the largest template, including separate gate/up and F32 input.
    ggml_reset(m.ctx);
    const int batch_size = 1 << (EXPERT_CACHE_BATCH_SIZES - 1);
    ggml_new_tensor_3d(m.ctx, GGML_TYPE_F32, input_dim, 1, batch_size);
    ggml_new_tensor_2d(m.ctx, GGML_TYPE_I32, n_expert_used, batch_size);
    ggml_new_tensor_3d(m.ctx, GGML_TYPE_F32, 3 * ffn_dim, n_expert_used, batch_size);
    ggml_new_tensor_3d(m.ctx, GGML_TYPE_F32, output_dim, n_expert_used, batch_size);
    size_t workspace_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(m.ctx, buft);
    ggml_reset(m.ctx);
    ggml_new_tensor_3d(m.ctx, GGML_TYPE_F32, std::max(input_dim + 2 * ffn_dim, ffn_dim + output_dim), n_expert_used, batch_size);
    ggml_new_tensor_2d(m.ctx, GGML_TYPE_I32, n_expert_used, batch_size);
    workspace_bytes = std::max(workspace_bytes, ggml_backend_alloc_ctx_tensors_from_buft_size(m.ctx, buft));
    return merge_bytes + workspace_bytes;
}


extern "C" const uint64_t * ggml_backend_cpu_expert_cache_begin(ggml_backend_cpu_expert_cache_t cache, ggml_tensor * op) {
    auto * entry = cache ? cache->find(op) : nullptr;
    return entry ? entry->begin(op) : nullptr;
}

extern "C" void ggml_backend_cpu_expert_cache_end(ggml_backend_cpu_expert_cache_t cache, ggml_tensor * op) {
    if (auto * entry = cache ? cache->find(op) : nullptr) {
        entry->end(op);
    }
}

extern "C" ggml_backend_dev_t ggml_backend_cpu_expert_cache_device(
        ggml_backend_cpu_expert_cache_t cache, const ggml_tensor * op) {
    auto * entry = cache ? cache->find(op) : nullptr;
    return entry ? entry->device(op) : nullptr;
}

extern "C" ggml_backend_dev_t ggml_backend_cpu_expert_cache_offload_device(
        ggml_backend_cpu_expert_cache_t cache, const ggml_tensor * op) {
    auto * entry = cache ? cache->find(op) : nullptr;
    return entry ? entry->offload_device(op) : nullptr;
}

extern "C" void ggml_backend_cpu_expert_cache_prepare(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_sched_t sched, ggml_cgraph * graph) {
    if (cache) {
        for (const auto & entry : cache->devices) {
            entry->prepare(sched, graph);
        }
    }
}

extern "C" bool ggml_backend_cpu_expert_cache_copy(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    if (cache) {
        for (const auto & entry : cache->devices) {
            if (entry->copy(backend, src, dst)) {
                return true;
            }
        }
    }
    return false;
}

