#include "nvme-expert-cache.h"
#include "ggml-backend.h"
#include "traits.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(GGML_USE_IO_URING)
#include <liburing.h>
#endif

namespace {

enum class nvme_part : uint8_t {
    none    = 0,
    gate    = 1,
    up      = 2,
    gate_up = 3,
    down    = 4,
};

static nvme_part parse_nvme_tensor(const char * name, int & layer) {
    int consumed = 0;
    layer = -1;
#define MATCH_NVME_PART(suffix, val) \
    do { \
        consumed = 0; \
        if (sscanf(name, "blk.%d." suffix "%n", &layer, &consumed) == 1 && name[consumed] == '\0') { \
            return val; \
        } \
    } while (0)

    MATCH_NVME_PART("ffn_gate_exps.weight",    nvme_part::gate);
    MATCH_NVME_PART("ffn_up_exps.weight",      nvme_part::up);
    MATCH_NVME_PART("ffn_gate_up_exps.weight", nvme_part::gate_up);
    MATCH_NVME_PART("ffn_down_exps.weight",    nvme_part::down);
#undef MATCH_NVME_PART
    layer = -1;
    return nvme_part::none;
}

static inline size_t align_up(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

static inline uint64_t align_down_u64(uint64_t v, size_t a) {
    return v & ~(uint64_t)(a - 1);
}

static inline float dot_f32_fast(const float * a, const float * b, int n) {
#if defined(__AVX512F__)
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), sum1);
    }
    sum0 = _mm512_add_ps(sum0, sum1);
    float acc = _mm512_reduce_add_ps(sum0);
    for (; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#endif
}

struct prefetch_entry {
    int32_t expert = -1;
    int32_t scratch_slot = -1;
    bool adopted = false;
};

static constexpr size_t NVME_ALIGN = 4096;
static constexpr int    PP_WINDOW  = 32;

struct nvme_file_handle {
    int fd = -1;
    bool direct_io = false;
    size_t file_size = 0;
};

struct nvme_tensor_info {
    const ggml_tensor * tensor = nullptr;
    int layer_id = -1;
    int part_idx = -1; // index inside nvme_layer::parts
    nvme_part part = nvme_part::none;
    int fd = -1;
    bool direct_io = false;
    bool can_repack = false;
    uint64_t file_offset = 0;
    size_t expert_bytes = 0;
    int64_t n_expert = 0;
};

struct l2_slot_meta {
    int32_t layer_id = -1;
    int32_t expert   = -1;
    uint64_t generation = 0;
    uint32_t ref_count = 0;
    uint8_t second_chance = 0;
    bool valid = false;
    bool repacked = false;
    bool repack_queued = false;
};

struct repack_job {
    int32_t slot = -1;
    int32_t layer_id = -1;
    int32_t expert = -1;
    uint64_t generation = 0;
};

struct scratch_slot {
    uint8_t * base = nullptr;
    size_t capacity = 0;
    bool in_use = false;
};

struct nvme_layer {
    int layer_id = -1;
    int64_t n_expert = 0;
    bool can_repack_all = false;

    std::vector<nvme_tensor_info *> parts; // up to 3 (gate, up, down) or 2 (gate_up, down)
    std::vector<size_t> part_slot_offsets; // 4096-aligned offset of each part within a bundle slot
    std::vector<size_t> part_alloc_bytes;  // align_up(expert_bytes + 2*NVME_ALIGN, NVME_ALIGN)
    size_t bundle_alloc_bytes = 0;

    // L2 mapping (protected by pool_mutex)
    std::vector<int32_t> expert_to_slot; // -1 if not in L2
    std::vector<int64_t> last_seen_step;
    int64_t step = 0;

    // Active layer state (thread 0 writes in begin/wait/end; workers read after barrier)
    bool active = false;
    bool pp_mode = false;
    const ggml_tensor * active_ids = nullptr;

    // Per-part, per-expert RAM pointer for current op/layer (valid when ready)
    // Indexed as [part_idx][expert]
    std::vector<std::vector<const void *>> active_ptrs;
    std::vector<std::vector<int8_t>>       active_part_pending; // 1 if io_uring read for [part_idx][expert] is in flight
    std::vector<std::vector<int8_t>>       active_part_repacked; // 1 if [part_idx][expert] was repacked in-place during PP
    std::vector<int32_t> active_l2_slots;      // pinned L2 slot per expert (-1 if none)
    std::vector<int32_t> active_scratch_slots; // transient scratch slot per expert (-1 if none)
    std::vector<int8_t>  active_admit_l2;      // 1 if active_l2_slots[e] is a newly admitted slot to publish on down end
    std::vector<int8_t>  active_admit_from_scratch; // 1 if adopted prefetch in scratch should be copied into L2 at down end
    std::vector<int8_t>  active_is_repacked;   // 1 if active_ptrs[*][e] is in CPU_REPACK layout
    std::vector<int8_t>  active_can_repack_in_place; // 1 if L2 slot is exclusively pinned for in-place PP repack
    std::vector<int32_t> active_experts;       // list of experts touched in this layer (for fast cleanup)

    // Host copy of router weights for cross-layer speculative prefetching
    int router_n_embd = 0;
    std::vector<float> router_w; // [n_expert, router_n_embd]
    std::vector<float> router_b; // [n_expert] (optional)
};

} // namespace

struct nvme_expert_cache::impl {
    size_t ram_budget_bytes = 0;
    uint32_t admit_window = 16;
    uint32_t repack_workers = 2;
    bool prefetch_enabled = false;
    int  prefetch_max_k = 2;
    bool valid = false;
    bool pool_initialized = false;
    bool print_stats = false;

    // Cross-layer speculative prefetch state (thread 0 only)
    std::vector<float> last_hidden;
    int last_n_ids = 0;
    int prefetch_layer_id = -1;
    std::vector<prefetch_entry> prefetched;
    std::vector<int32_t> unused_prefetch_scratch;

    uint64_t stat_prefetch_issued = 0;
    uint64_t stat_prefetch_hits = 0;

    std::unordered_map<std::string, nvme_file_handle> files;
    std::unordered_map<const ggml_tensor *, std::unique_ptr<nvme_tensor_info>> tensors;
    std::unordered_map<int, std::unique_ptr<nvme_layer>> layers;

    // Global L2 RAM slab pool
    std::mutex pool_mutex;
    uint8_t * l2_pool_base = nullptr;
    size_t l2_pool_bytes = 0;
    size_t l2_slot_stride = 0;
    uint32_t l2_num_slots = 0;
    uint32_t l2_next_free = 0;
    uint32_t l2_clock_hand = 0;
    std::vector<l2_slot_meta> l2_slots;

    // Background CPU_REPACK workers for newly admitted L2 slots
    std::mutex repack_mutex;
    std::condition_variable repack_cv;
    std::deque<repack_job> repack_queue;
    std::vector<std::thread> repack_threads;
    bool repack_stop = false;

    // Transient aligned scratch ring (for cold misses + PP streaming)
    uint8_t * scratch_pool_base = nullptr;
    size_t scratch_pool_bytes = 0;
    size_t scratch_slot_stride = 0;
    std::vector<scratch_slot> scratch_slots;

    // io_uring (owned exclusively by thread 0 during compute)
#if defined(GGML_USE_IO_URING)
    struct io_uring ring = {};
    bool ring_ready = false;
    int ring_in_flight = 0;
#endif

    // Telemetry
    uint64_t stat_l2_hits = 0;
    uint64_t stat_l2_repacked_hits = 0;
    uint64_t stat_l3_misses = 0;
    uint64_t stat_l2_admits = 0;
    std::atomic<uint64_t> stat_repacked_done{0};
    uint64_t stat_bytes_read = 0;
    double   stat_wait_ms = 0.0;
    uint64_t stat_ops = 0;

    uint64_t stat_pp_hits = 0;
    uint64_t stat_pp_repacked_hits = 0;
    uint64_t stat_pp_misses = 0;
    uint64_t stat_pp_bytes_read = 0;
    double   stat_pp_wait_ms = 0.0;

    uint64_t stat_dec_hits = 0;
    uint64_t stat_dec_repacked_hits = 0;
    uint64_t stat_dec_misses = 0;
    uint64_t stat_dec_bytes_read = 0;
    double   stat_dec_wait_ms = 0.0;

    impl(size_t budget, uint32_t admit_win)
        : ram_budget_bytes(budget), admit_window(admit_win) {
        const char * env_stats = getenv("LLAMA_NVME_CACHE_STATS");
        print_stats = env_stats != nullptr && atoi(env_stats) != 0;

        const char * env_win = getenv("LLAMA_NVME_ADMIT_WINDOW");
        if (env_win != nullptr) {
            admit_window = (uint32_t) std::max(0, atoi(env_win));
        }

        const char * env_rw = getenv("LLAMA_NVME_REPACK_WORKERS");
        if (env_rw != nullptr) {
            repack_workers = (uint32_t) std::max(0, atoi(env_rw));
        }

        const char * env_pf = getenv("LLAMA_NVME_PREFETCH");
        if (env_pf != nullptr) {
            prefetch_enabled = atoi(env_pf) != 0;
        }
        const char * env_pfk = getenv("LLAMA_NVME_PREFETCH_K");
        if (env_pfk != nullptr) {
            prefetch_max_k = std::max(0, std::min(16, atoi(env_pfk)));
        }

#if defined(GGML_USE_IO_URING)
        // Use modern Linux SINGLE_ISSUER + DEFER_TASKRUN for lowest completion latency (falls back to 0)
        struct io_uring_params up = {};
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
        up.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
        if (io_uring_queue_init_params(256, &ring, &up) == 0) {
            ring_ready = true;
        } else
#endif
        if (io_uring_queue_init(256, &ring, 0) == 0) {
            ring_ready = true;
        } else {
            fprintf(stderr, "nvme_expert_cache: io_uring_queue_init failed (%s), falling back to pread\n", strerror(errno));
        }
#endif
        valid = true;
    }

    ~impl() {
        // Stop background repack workers BEFORE freeing L2 pool memory
        if (!repack_threads.empty()) {
            {
                std::lock_guard<std::mutex> lock(repack_mutex);
                repack_stop = true;
            }
            repack_cv.notify_all();
            for (auto & t : repack_threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
            repack_threads.clear();
        }

        if (print_stats && (stat_l2_hits + stat_l3_misses) > 0) {
            const uint64_t total = stat_l2_hits + stat_l3_misses;
            const uint64_t pp_tot = stat_pp_hits + stat_pp_misses;
            const uint64_t dec_tot = stat_dec_hits + stat_dec_misses;
            fprintf(stderr,
                    "nvme_expert_cache: summary: L2 hits=%llu (%.1f%%, repacked=%llu), L3 NVMe misses=%llu, "
                    "admits=%llu, repacked_slots=%llu, prefetch=%llu/%llu hits (%.1f%%), read=%.2f GB, total_wait=%.1f ms\n",
                    (unsigned long long) stat_l2_hits,
                    100.0 * (double) stat_l2_hits / (double) total,
                    (unsigned long long) stat_l2_repacked_hits,
                    (unsigned long long) stat_l3_misses,
                    (unsigned long long) stat_l2_admits,
                    (unsigned long long) stat_repacked_done.load(std::memory_order_relaxed),
                    (unsigned long long) stat_prefetch_hits,
                    (unsigned long long) stat_prefetch_issued,
                    stat_prefetch_issued > 0 ? 100.0 * (double) stat_prefetch_hits / (double) stat_prefetch_issued : 0.0,
                    (double) stat_bytes_read / 1e9,
                    stat_wait_ms);
            if (pp_tot > 0) {
                fprintf(stderr,
                        "nvme_expert_cache:   PP    : L2 hits=%llu (%.1f%%, repacked=%llu), L3 misses=%llu, read=%.2f GB, wait=%.1f ms\n",
                        (unsigned long long) stat_pp_hits,
                        100.0 * (double) stat_pp_hits / (double) pp_tot,
                        (unsigned long long) stat_pp_repacked_hits,
                        (unsigned long long) stat_pp_misses,
                        (double) stat_pp_bytes_read / 1e9,
                        stat_pp_wait_ms);
            }
            if (dec_tot > 0) {
                fprintf(stderr,
                        "nvme_expert_cache:   Decode: L2 hits=%llu (%.1f%%, repacked=%llu), L3 misses=%llu, read=%.2f GB, wait=%.1f ms\n",
                        (unsigned long long) stat_dec_hits,
                        100.0 * (double) stat_dec_hits / (double) dec_tot,
                        (unsigned long long) stat_dec_repacked_hits,
                        (unsigned long long) stat_dec_misses,
                        (double) stat_dec_bytes_read / 1e9,
                        stat_dec_wait_ms);
            }
        }
#if defined(GGML_USE_IO_URING)
        if (ring_ready) {
            io_uring_queue_exit(&ring);
            ring_ready = false;
        }
#endif
#if defined(__linux__)
        if (l2_pool_base != nullptr) {
            munmap(l2_pool_base, l2_pool_bytes);
            l2_pool_base = nullptr;
        }
        if (scratch_pool_base != nullptr) {
            munmap(scratch_pool_base, scratch_pool_bytes);
            scratch_pool_base = nullptr;
        }
        for (auto & kv : files) {
            if (kv.second.fd >= 0) {
                close(kv.second.fd);
            }
        }
#else
        free(l2_pool_base);
        free(scratch_pool_base);
#endif
    }

    int open_file(const char * path, bool & direct_io_out) {
        auto it = files.find(path);
        if (it != files.end()) {
            direct_io_out = it->second.direct_io;
            return it->second.fd;
        }
        nvme_file_handle h;
#if defined(__linux__)
        h.fd = open(path, O_RDONLY | O_DIRECT);
        if (h.fd >= 0) {
            h.direct_io = true;
        } else {
            fprintf(stderr, "nvme_expert_cache: O_DIRECT open failed for '%s' (%s), falling back to buffered pread\n",
                    path, strerror(errno));
            h.fd = open(path, O_RDONLY);
            h.direct_io = false;
        }
        if (h.fd >= 0) {
            struct stat st;
            if (fstat(h.fd, &st) == 0) {
                h.file_size = (size_t) st.st_size;
            }
        }
#endif
        files.emplace(path, h);
        direct_io_out = h.direct_io;
        return h.fd;
    }

    void register_tensor(const ggml_tensor * tensor, const char * file_path, uint64_t file_offset) {
        if (!tensor || !file_path) {
            return;
        }
        int layer_id = -1;
        const nvme_part part = parse_nvme_tensor(tensor->name, layer_id);
        if (part == nvme_part::none || layer_id < 0 || tensor->ne[2] <= 0) {
            return;
        }
        bool direct_io = false;
        int fd = open_file(file_path, direct_io);
        if (fd < 0) {
            fprintf(stderr, "nvme_expert_cache: failed to open '%s'\n", file_path);
            valid = false;
            return;
        }

        auto info = std::make_unique<nvme_tensor_info>();
        info->tensor = tensor;
        info->layer_id = layer_id;
        info->part = part;
        info->fd = fd;
        info->direct_io = direct_io;
        info->can_repack = ggml_cpu_nvme_can_repack(tensor);
        info->file_offset = file_offset;
        info->n_expert = tensor->ne[2];
        info->expert_bytes = tensor->nb[2];

        auto & layer_ptr = layers[layer_id];
        if (!layer_ptr) {
            layer_ptr = std::make_unique<nvme_layer>();
            layer_ptr->layer_id = layer_id;
            layer_ptr->n_expert = tensor->ne[2];
            layer_ptr->expert_to_slot.assign((size_t) tensor->ne[2], -1);
            layer_ptr->last_seen_step.assign((size_t) tensor->ne[2], -1000000);
            layer_ptr->active_l2_slots.assign((size_t) tensor->ne[2], -1);
            layer_ptr->active_scratch_slots.assign((size_t) tensor->ne[2], -1);
            layer_ptr->active_admit_l2.assign((size_t) tensor->ne[2], 0);
            layer_ptr->active_admit_from_scratch.assign((size_t) tensor->ne[2], 0);
            layer_ptr->active_is_repacked.assign((size_t) tensor->ne[2], 0);
            layer_ptr->active_can_repack_in_place.assign((size_t) tensor->ne[2], 0);
        }

        nvme_tensor_info * raw = info.get();
        raw->part_idx = (int) layer_ptr->parts.size();
        layer_ptr->parts.push_back(raw);
        layer_ptr->active_ptrs.emplace_back((size_t) tensor->ne[2], nullptr);
        layer_ptr->active_part_pending.emplace_back((size_t) tensor->ne[2], 0);
        layer_ptr->active_part_repacked.emplace_back((size_t) tensor->ne[2], 0);

        tensors[tensor] = std::move(info);
    }

    void register_router(int layer_id, const ggml_tensor * gate_inp, const ggml_tensor * exp_probs_b) {
        auto lit = layers.find(layer_id);
        if (lit == layers.end()) {
            return;
        }
        auto & layer = *lit->second;
        if (exp_probs_b != nullptr && exp_probs_b->buffer != nullptr &&
                exp_probs_b->ne[0] == layer.n_expert && exp_probs_b->type == GGML_TYPE_F32) {
            layer.router_b.resize((size_t) layer.n_expert);
            ggml_backend_tensor_get(exp_probs_b, layer.router_b.data(), 0, (size_t) layer.n_expert * sizeof(float));
        }
        if (!prefetch_enabled || gate_inp == nullptr || gate_inp->buffer == nullptr) {
            return;
        }
        if (gate_inp->ne[1] != layer.n_expert || gate_inp->ne[0] <= 0) {
            return;
        }
        const int n_embd = (int) gate_inp->ne[0];
        const size_t n_elem = (size_t) n_embd * (size_t) layer.n_expert;
        const auto * traits = ggml_get_type_traits(gate_inp->type);
        if (gate_inp->type != GGML_TYPE_F32 && (traits == nullptr || traits->to_float == nullptr)) {
            return;
        }

        layer.router_n_embd = n_embd;
        layer.router_w.resize(n_elem);
        if (gate_inp->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(gate_inp, layer.router_w.data(), 0, n_elem * sizeof(float));
        } else {
            std::vector<uint8_t> raw(ggml_nbytes(gate_inp));
            ggml_backend_tensor_get(gate_inp, raw.data(), 0, raw.size());
            traits->to_float(raw.data(), layer.router_w.data(), (int64_t) n_elem);
        }
    }

    void worker_main_repack() {
#if defined(__linux__)
        // Lower OS priority so background repacking uses idle/GPU/NVMe gaps without preempting compute threads
        setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 10);
#endif
        std::vector<uint8_t> bounce;
        for (;;) {
            repack_job job;
            {
                std::unique_lock<std::mutex> lock(repack_mutex);
                repack_cv.wait(lock, [this]() { return repack_stop || !repack_queue.empty(); });
                if (repack_stop) {
                    break;
                }
                if (repack_queue.empty()) {
                    continue;
                }
                job = repack_queue.front();
                repack_queue.pop_front();
            }

            nvme_layer * layer_ptr = nullptr;
            size_t total_native_bytes = 0;
            std::vector<const uint8_t *> native_ptrs;
            std::vector<size_t> part_bytes;
            std::vector<const ggml_tensor *> part_tensors;
            std::vector<size_t> part_dst_offs;

            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                if (job.slot < 0 || (uint32_t) job.slot >= l2_num_slots) {
                    continue;
                }
                auto & m = l2_slots[(size_t) job.slot];
                if (!m.valid || m.repacked || m.ref_count > 0 || m.generation != job.generation ||
                        m.layer_id != job.layer_id || m.expert != job.expert) {
                    if (m.generation == job.generation && m.layer_id == job.layer_id && m.expert == job.expert && !m.repacked) {
                        m.repack_queued = false;
                    }
                    continue;
                }
                auto lit = layers.find(job.layer_id);
                if (lit == layers.end() || !lit->second->can_repack_all) {
                    continue;
                }
                layer_ptr = lit->second.get();
                // Pin slot while we read native bytes into our bounce buffer
                m.ref_count++;

                uint8_t * slot_base = l2_pool_base + (size_t) job.slot * l2_slot_stride;
                const size_t np = layer_ptr->parts.size();
                native_ptrs.resize(np);
                part_bytes.resize(np);
                part_tensors.resize(np);
                part_dst_offs.resize(np);
                for (size_t p = 0; p < np; ++p) {
                    uint64_t aoff = 0; size_t alen = 0; size_t pad = 0;
                    calc_aligned_read(*layer_ptr->parts[p], job.expert, aoff, alen, pad);
                    native_ptrs[p] = slot_base + layer_ptr->part_slot_offsets[p] + pad;
                    part_bytes[p] = layer_ptr->parts[p]->expert_bytes;
                    part_tensors[p] = layer_ptr->parts[p]->tensor;
                    part_dst_offs[p] = layer_ptr->part_slot_offsets[p] + pad;
                    total_native_bytes += part_bytes[p];
                }
            }

            if (bounce.size() < total_native_bytes) {
                bounce.resize(total_native_bytes);
            }

            bool ok = true;
            size_t b_off = 0;
            for (size_t p = 0; p < part_tensors.size(); ++p) {
                if (!ggml_cpu_nvme_repack_expert(part_tensors[p], native_ptrs[p], bounce.data() + b_off, part_bytes[p])) {
                    ok = false;
                    break;
                }
                b_off += part_bytes[p];
            }

            bool swapped = false;
            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                auto & m = l2_slots[(size_t) job.slot];
                // Only overwrite in-place if nobody else is currently reading this slot (ref_count == 1)
                if (ok && m.valid && !m.repacked && m.generation == job.generation &&
                        m.layer_id == job.layer_id && m.expert == job.expert && m.ref_count == 1) {
                    uint8_t * slot_base = l2_pool_base + (size_t) job.slot * l2_slot_stride;
                    size_t off = 0;
                    for (size_t p = 0; p < part_bytes.size(); ++p) {
                        memcpy(slot_base + part_dst_offs[p], bounce.data() + off, part_bytes[p]);
                        off += part_bytes[p];
                    }
                    m.repacked = true;
                    swapped = true;
                }
                if (m.generation == job.generation && m.layer_id == job.layer_id && m.expert == job.expert && !m.repacked) {
                    // Allow retry on next hit if slot was busy (ref_count > 1)
                    m.repack_queued = false;
                }
                if (m.ref_count > 0) {
                    m.ref_count--;
                }
            }
            if (swapped) {
                stat_repacked_done.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void enqueue_repack(int32_t slot, int32_t layer_id, int32_t expert, uint64_t generation) {
        if (repack_threads.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(repack_mutex);
            if (repack_queue.size() < 4096) {
                repack_queue.push_back({ slot, layer_id, expert, generation });
            }
        }
        repack_cv.notify_one();
    }

    void ensure_pool_initialized() {
        if (pool_initialized) {
            return;
        }
        pool_initialized = true;

        size_t max_bundle = 0;
        bool any_repackable = false;
        for (auto & kv : layers) {
            auto & layer = *kv.second;
            size_t off = 0;
            layer.part_slot_offsets.resize(layer.parts.size());
            layer.part_alloc_bytes.resize(layer.parts.size());
            bool all_ok = !layer.parts.empty();
            for (size_t p = 0; p < layer.parts.size(); ++p) {
                const size_t eb = layer.parts[p]->expert_bytes;
                const size_t alloc_b = align_up(eb + 2 * NVME_ALIGN, NVME_ALIGN);
                layer.part_slot_offsets[p] = off;
                layer.part_alloc_bytes[p] = alloc_b;
                off += alloc_b;
                all_ok = all_ok && layer.parts[p]->can_repack;
            }
            layer.can_repack_all = all_ok;
            any_repackable = any_repackable || all_ok;
            layer.bundle_alloc_bytes = off;
            max_bundle = std::max(max_bundle, off);
        }

        if (max_bundle == 0) {
            return;
        }

        l2_slot_stride = align_up(max_bundle, NVME_ALIGN);
        scratch_slot_stride = l2_slot_stride;

        const uint32_t n_scratch = 64;
        scratch_pool_bytes = (size_t) n_scratch * scratch_slot_stride;
        if (scratch_pool_bytes > ram_budget_bytes / 2 && ram_budget_bytes >= 8 * scratch_slot_stride) {
            const uint32_t small_scratch = (uint32_t) std::max<size_t>(8, (ram_budget_bytes / 4) / scratch_slot_stride);
            scratch_pool_bytes = (size_t) small_scratch * scratch_slot_stride;
        }

        const size_t actual_scratch_slots = scratch_pool_bytes / scratch_slot_stride;

#if defined(__linux__)
        scratch_pool_base = (uint8_t *) mmap(nullptr, scratch_pool_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (scratch_pool_base == MAP_FAILED) {
            scratch_pool_base = nullptr;
            valid = false;
            return;
        }
        madvise(scratch_pool_base, scratch_pool_bytes, MADV_HUGEPAGE);
#else
        if (posix_memalign((void **) &scratch_pool_base, NVME_ALIGN, scratch_pool_bytes) != 0) {
            scratch_pool_base = nullptr;
            valid = false;
            return;
        }
#endif
        scratch_slots.resize(actual_scratch_slots);
        for (size_t i = 0; i < actual_scratch_slots; ++i) {
            scratch_slots[i].base = scratch_pool_base + i * scratch_slot_stride;
            scratch_slots[i].capacity = scratch_slot_stride;
            scratch_slots[i].in_use = false;
        }

        const size_t l2_budget = ram_budget_bytes > scratch_pool_bytes ? (ram_budget_bytes - scratch_pool_bytes) : 0;
        l2_num_slots = (uint32_t) (l2_budget / l2_slot_stride);
        if (l2_num_slots > 0) {
            l2_pool_bytes = (size_t) l2_num_slots * l2_slot_stride;
#if defined(__linux__)
            l2_pool_base = (uint8_t *) mmap(nullptr, l2_pool_bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (l2_pool_base == MAP_FAILED) {
                fprintf(stderr, "nvme_expert_cache: failed to allocate %.2f GiB L2 RAM pool\n",
                        (double) l2_pool_bytes / (1024.0 * 1024.0 * 1024.0));
                l2_pool_base = nullptr;
                l2_pool_bytes = 0;
                l2_num_slots = 0;
            } else {
                madvise(l2_pool_base, l2_pool_bytes, MADV_HUGEPAGE);
            }
#else
            if (posix_memalign((void **) &l2_pool_base, NVME_ALIGN, l2_pool_bytes) != 0) {
                l2_pool_base = nullptr;
                l2_pool_bytes = 0;
                l2_num_slots = 0;
            }
#endif
            l2_slots.resize(l2_num_slots);
        }

        if (l2_num_slots > 0 && any_repackable && repack_workers > 0) {
            repack_threads.reserve(repack_workers);
            for (uint32_t i = 0; i < repack_workers; ++i) {
                repack_threads.emplace_back([this]() { worker_main_repack(); });
            }
        }

        const size_t total_experts = layers.empty() ? 0 : layers.size() * (size_t) layers.begin()->second->n_expert;
        fprintf(stderr,
                "nvme_expert_cache: initialized L2 RAM pool = %.2f GiB (%u global slots of %.2f MiB, %.1f%% of %zu experts) "
                "+ %.1f MiB scratch (%zu slots), admit_window=%u, repack_workers=%zu, io_uring=%s\n",
                (double) l2_pool_bytes / (1024.0 * 1024.0 * 1024.0),
                l2_num_slots,
                (double) l2_slot_stride / (1024.0 * 1024.0),
                total_experts > 0 ? 100.0 * (double) l2_num_slots / (double) total_experts : 0.0,
                total_experts,
                (double) scratch_pool_bytes / (1024.0 * 1024.0),
                actual_scratch_slots,
                admit_window,
                repack_threads.size(),
#if defined(GGML_USE_IO_URING)
                ring_ready ? "on" : "off"
#else
                "off"
#endif
        );

#if defined(__linux__)
        bool prewarm = true;
        const char * env_pw = getenv("LLAMA_NVME_PREWARM");
        if (env_pw != nullptr) {
            prewarm = atoi(env_pw) != 0;
        }
        if (prewarm && l2_num_slots > 0 && !layers.empty()) {
            const auto t_pw0 = std::chrono::steady_clock::now();
            std::vector<int> layer_ids;
            layer_ids.reserve(layers.size());
            for (const auto & kv : layers) {
                layer_ids.push_back(kv.first);
            }
            std::sort(layer_ids.begin(), layer_ids.end());

            struct prewarm_item {
                int layer_id;
                int32_t expert;
                int32_t slot;
            };
            std::vector<prewarm_item> items;
            items.reserve(l2_num_slots);
            const size_t n_layers = layer_ids.size();
            uint32_t slot_cursor = 0;
            for (size_t li = 0; li < n_layers && slot_cursor < l2_num_slots; ++li) {
                auto & layer = *layers[layer_ids[li]];
                const uint32_t rem_slots = l2_num_slots - slot_cursor;
                const uint32_t rem_layers = (uint32_t) (n_layers - li);
                const uint32_t cap = std::min<uint32_t>((uint32_t) layer.n_expert, (rem_slots + rem_layers - 1) / rem_layers);
                std::vector<int32_t> chosen;
                chosen.reserve(cap);
                if (layer.router_b.size() == (size_t) layer.n_expert && cap < (uint32_t) layer.n_expert) {
                    std::vector<int32_t> order((size_t) layer.n_expert);
                    for (int32_t e = 0; e < layer.n_expert; ++e) {
                        order[(size_t) e] = e;
                    }
                    std::partial_sort(order.begin(), order.begin() + cap, order.end(),
                            [&](int32_t a, int32_t b) {
                                if (layer.router_b[(size_t) a] != layer.router_b[(size_t) b]) {
                                    return layer.router_b[(size_t) a] > layer.router_b[(size_t) b];
                                }
                                return a < b;
                            });
                    chosen.assign(order.begin(), order.begin() + cap);
                    std::sort(chosen.begin(), chosen.end());
                } else {
                    for (uint32_t e = 0; e < cap; ++e) {
                        chosen.push_back((int32_t) e);
                    }
                }
                for (int32_t e : chosen) {
                    if (slot_cursor >= l2_num_slots) break;
                    const int32_t s = (int32_t) slot_cursor++;
                    auto & m = l2_slots[(size_t) s];
                    m.layer_id = layer.layer_id;
                    m.expert = e;
                    m.generation = 1;
                    m.ref_count = 0;
                    m.second_chance = 0; // 0 so untouched pre-warmed slots are first to be replaced by hot misses
                    m.valid = true;
                    m.repacked = layer.can_repack_all;
                    m.repack_queued = false;
                    layer.expert_to_slot[(size_t) e] = s;
                    items.push_back({ layer.layer_id, e, s });
                }
            }
            l2_next_free = slot_cursor;

            const uint32_t n_pw_threads = std::min<uint32_t>(16, std::max<uint32_t>(1, std::thread::hardware_concurrency() / 2));
            std::atomic<size_t> next_item{0};
            std::vector<std::thread> pw_threads;
            pw_threads.reserve(n_pw_threads);
            for (uint32_t ti = 0; ti < n_pw_threads; ++ti) {
                pw_threads.emplace_back([&]() {
                    void * bounce = nullptr;
                    size_t bounce_cap = 0;
                    for (;;) {
                        const size_t idx = next_item.fetch_add(8, std::memory_order_relaxed);
                        if (idx >= items.size()) break;
                        const size_t end_idx = std::min(items.size(), idx + 8);
                        for (size_t k = idx; k < end_idx; ++k) {
                            const auto & it = items[k];
                            auto & layer = *layers[it.layer_id];
                            uint8_t * slot_base = l2_pool_base + (size_t) it.slot * l2_slot_stride;
                            for (size_t p = 0; p < layer.parts.size(); ++p) {
                                const auto & info = *layer.parts[p];
                                uint64_t aoff = 0; size_t alen = 0; size_t pad = 0;
                                calc_aligned_read(info, it.expert, aoff, alen, pad);
                                uint8_t * dst_part = slot_base + layer.part_slot_offsets[p];
                                if (layer.can_repack_all) {
                                    if (bounce_cap < alen) {
                                        free(bounce);
                                        if (posix_memalign(&bounce, NVME_ALIGN, alen) != 0) {
                                            bounce = nullptr; bounce_cap = 0; break;
                                        }
                                        bounce_cap = alen;
                                    }
                                    size_t done = 0;
                                    while (done < alen) {
                                        ssize_t r = pread(info.fd, (uint8_t *) bounce + done, alen - done, (off_t) (aoff + done));
                                        if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
                                        done += (size_t) r;
                                    }
                                    ggml_cpu_nvme_repack_expert(info.tensor, (const uint8_t *) bounce + pad, dst_part + pad, info.expert_bytes);
                                } else {
                                    size_t done = 0;
                                    while (done < alen) {
                                        ssize_t r = pread(info.fd, dst_part + done, alen - done, (off_t) (aoff + done));
                                        if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
                                        done += (size_t) r;
                                    }
                                }
                            }
                        }
                    }
                    free(bounce);
                });
            }
            for (auto & t : pw_threads) {
                t.join();
            }
            stat_repacked_done.store(slot_cursor, std::memory_order_relaxed);
            const auto t_pw1 = std::chrono::steady_clock::now();
            const double pw_s = std::chrono::duration<double>(t_pw1 - t_pw0).count();
            fprintf(stderr, "nvme_expert_cache: pre-warmed & AVX-512 repacked %u L2 slots (%.2f GiB) in %.2f s (%.2f GB/s)\n",
                    slot_cursor, (double) l2_pool_bytes / (1024.0 * 1024.0 * 1024.0),
                    pw_s, (double) l2_pool_bytes / 1e9 / std::max(1e-6, pw_s));
        }
#endif
    }

    int32_t alloc_scratch_slot() {
        for (size_t i = 0; i < scratch_slots.size(); ++i) {
            if (!scratch_slots[i].in_use) {
                scratch_slots[i].in_use = true;
                return (int32_t) i;
            }
        }
        return -1;
    }

    // Allocate or evict an L2 slot (caller holds pool_mutex). Returns -1 if all slots are pinned.
    int32_t claim_l2_slot_locked(int layer_id, int32_t expert, bool free_only = false) {
        if (l2_num_slots == 0) {
            return -1;
        }
        int32_t slot = -1;
        if (l2_next_free < l2_num_slots) {
            slot = (int32_t) l2_next_free++;
        } else if (!free_only) {
            // Aging-LFU Clock sweep: hot experts (high second_chance up to 15) survive multiple sweeps
            const uint32_t max_scans = l2_num_slots * 16;
            for (uint32_t scanned = 0; scanned < max_scans; ++scanned) {
                const uint32_t idx = l2_clock_hand;
                l2_clock_hand = (l2_clock_hand + 1) % l2_num_slots;
                auto & m = l2_slots[idx];
                if (m.ref_count > 0) {
                    continue;
                }
                if (m.second_chance > 0) {
                    m.second_chance >>= 1;
                    continue;
                }
                slot = (int32_t) idx;
                break;
            }
        }
        if (slot < 0) {
            return -1;
        }

        auto & m = l2_slots[(size_t) slot];
        if (m.valid && m.layer_id >= 0 && m.expert >= 0) {
            auto lit = layers.find(m.layer_id);
            if (lit != layers.end() && lit->second->expert_to_slot[(size_t) m.expert] == slot) {
                lit->second->expert_to_slot[(size_t) m.expert] = -1;
            }
        }
        m.layer_id = layer_id;
        m.expert = expert;
        m.generation++;
        m.ref_count = 1;
        m.second_chance = 1;
        m.valid = false;
        m.repacked = false;
        m.repack_queued = false;
        return slot;
    }

    static void calc_aligned_read(const nvme_tensor_info & info, int32_t expert,
            uint64_t & aligned_off, size_t & aligned_len, size_t & data_pad) {
        const uint64_t raw_off = info.file_offset + (uint64_t) expert * info.expert_bytes;
        aligned_off = align_down_u64(raw_off, NVME_ALIGN);
        data_pad = (size_t) (raw_off - aligned_off);
        aligned_len = align_up(data_pad + info.expert_bytes, NVME_ALIGN);
    }

    static uint64_t pack_user_data(int layer_id, int32_t expert, int part_idx) {
        return ((uint64_t) (uint16_t) layer_id << 32) | ((uint64_t) (uint32_t) expert << 8) | ((uint64_t) (uint8_t) part_idx);
    }

    static void unpack_user_data(uint64_t ud, int & layer_id, int32_t & expert, int & part_idx) {
        layer_id = (int) (uint16_t) (ud >> 32);
        expert   = (int32_t) (uint32_t) ((ud >> 8) & 0xffffffu);
        part_idx = (int) (ud & 0xffu);
    }

    void submit_part_read(nvme_layer & layer, int part_idx, int32_t expert, uint8_t * dst_buf) {
        const auto & info = *layer.parts[(size_t) part_idx];
        uint64_t aligned_off = 0;
        size_t aligned_len = 0;
        size_t data_pad = 0;
        calc_aligned_read(info, expert, aligned_off, aligned_len, data_pad);

        layer.active_ptrs[(size_t) part_idx][(size_t) expert] = dst_buf + data_pad;
        stat_bytes_read += aligned_len;
        if (layer.pp_mode) {
            stat_pp_bytes_read += aligned_len;
        } else {
            stat_dec_bytes_read += aligned_len;
        }

#if defined(GGML_USE_IO_URING)
        if (ring_ready) {
            struct io_uring_sqe * sqe = io_uring_get_sqe(&ring);
            if (sqe == nullptr) {
                io_uring_submit(&ring);
                sqe = io_uring_get_sqe(&ring);
            }
            if (sqe != nullptr) {
                io_uring_prep_read(sqe, info.fd, dst_buf, (unsigned) aligned_len, aligned_off);
                io_uring_sqe_set_data64(sqe, pack_user_data(layer.layer_id, expert, part_idx));
                layer.active_part_pending[(size_t) part_idx][(size_t) expert] = 1;
                ring_in_flight++;
                return;
            }
        }
#endif
#if defined(__linux__)
        size_t done = 0;
        while (done < aligned_len) {
            ssize_t r = pread(info.fd, dst_buf + done, aligned_len - done, (off_t) (aligned_off + done));
            if (r <= 0) {
                if (r < 0 && errno == EINTR) continue;
                break;
            }
            done += (size_t) r;
        }
#endif
    }

    void flush_ring_submissions() {
#if defined(GGML_USE_IO_URING)
        if (ring_ready && ring_in_flight > 0) {
            io_uring_submit(&ring);
        }
#endif
    }

    void reap_completions(bool block) {
#if defined(GGML_USE_IO_URING)
        if (!ring_ready || ring_in_flight == 0) {
            return;
        }
        struct io_uring_cqe * cqe = nullptr;
        if (block) {
            if (io_uring_wait_cqe(&ring, &cqe) < 0) {
                return;
            }
        } else {
            if (io_uring_peek_cqe(&ring, &cqe) < 0) {
                return;
            }
        }
        unsigned head = 0;
        unsigned count = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            int cqe_layer = -1;
            int32_t exp = -1;
            int part_idx = -1;
            unpack_user_data(io_uring_cqe_get_data64(cqe), cqe_layer, exp, part_idx);
            if (cqe->res < 0) {
                fprintf(stderr, "nvme_expert_cache: io_uring read error %d on layer %d expert %d part %d\n",
                        cqe->res, cqe_layer, exp, part_idx);
            }
            auto lit = layers.find(cqe_layer);
            if (lit != layers.end()) {
                auto & lyr = *lit->second;
                if (exp >= 0 && exp < lyr.n_expert &&
                        part_idx >= 0 && (size_t) part_idx < lyr.active_part_pending.size()) {
                    lyr.active_part_pending[(size_t) part_idx][(size_t) exp] = 0;
                }
            }
            count++;
            ring_in_flight--;
        }
        io_uring_cq_advance(&ring, count);
#else
        (void) block;
#endif
    }

    void drain_unused_prefetches() {
        if (prefetch_layer_id < 0 && prefetched.empty() && unused_prefetch_scratch.empty()) {
            return;
        }
#if defined(GGML_USE_IO_URING)
        while (ring_ready && ring_in_flight > 0) {
            reap_completions(/*block=*/ true);
        }
#endif
        if (prefetch_layer_id >= 0) {
            auto lit = layers.find(prefetch_layer_id);
            if (lit != layers.end()) {
                auto & lyr = *lit->second;
                for (const auto & pe : prefetched) {
                    if (!pe.adopted && pe.expert >= 0 && pe.expert < lyr.n_expert) {
                        for (size_t p = 0; p < lyr.parts.size(); ++p) {
                            lyr.active_ptrs[p][(size_t) pe.expert] = nullptr;
                            lyr.active_part_pending[p][(size_t) pe.expert] = 0;
                        }
                    }
                }
            }
        }
        for (const auto & pe : prefetched) {
            if (!pe.adopted && pe.scratch_slot >= 0) {
                scratch_slots[(size_t) pe.scratch_slot].in_use = false;
            }
        }
        for (int32_t sc : unused_prefetch_scratch) {
            if (sc >= 0) {
                scratch_slots[(size_t) sc].in_use = false;
            }
        }
        prefetched.clear();
        unused_prefetch_scratch.clear();
        prefetch_layer_id = -1;
    }

    void launch_next_layer_prefetch(int cur_layer_id) {
        if (!prefetch_enabled || prefetch_max_k <= 0 || last_hidden.empty() || last_n_ids <= 0) {
            return;
        }
        auto lit = layers.find(cur_layer_id + 1);
        if (lit == layers.end()) {
            return;
        }
        auto & next_layer = *lit->second;
        if (next_layer.router_w.empty() || (int) last_hidden.size() != next_layer.router_n_embd) {
            return;
        }

        const int n_exp = (int) next_layer.n_expert;
        const int n_embd = next_layer.router_n_embd;
        const float * x = last_hidden.data();
        const float * w = next_layer.router_w.data();
        const float * b = next_layer.router_b.empty() ? nullptr : next_layer.router_b.data();

        // Find top-K predicted experts for layer L+1
        const int top_k = std::min(n_exp, std::max(last_n_ids, prefetch_max_k));
        std::vector<std::pair<float, int32_t>> scores((size_t) n_exp);
        for (int e = 0; e < n_exp; ++e) {
            float s = dot_f32_fast(w + (size_t) e * (size_t) n_embd, x, n_embd);
            if (b != nullptr) {
                s += b[e];
            }
            scores[(size_t) e] = { s, e };
        }
        std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });

        // Submit io_uring reads ONLY for predicted experts that miss L2 RAM (up to prefetch_max_k)
        prefetched.clear();
        prefetch_layer_id = next_layer.layer_id;
        bool any_submitted = false;
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            for (int k = 0; k < top_k && (int) prefetched.size() < prefetch_max_k; ++k) {
                const int32_t e = scores[(size_t) k].second;
                const int32_t slot = next_layer.expert_to_slot[(size_t) e];
                if (slot >= 0 && l2_slots[(size_t) slot].valid) {
                    continue; // already in L2 RAM!
                }
                const int32_t sc = alloc_scratch_slot();
                if (sc < 0) {
                    break;
                }
                prefetched.push_back({ e, sc, false });
                stat_prefetch_issued++;
                uint8_t * bundle_base = scratch_slots[(size_t) sc].base;
                for (size_t p = 0; p < next_layer.parts.size(); ++p) {
                    submit_part_read(next_layer, (int) p, e, bundle_base + next_layer.part_slot_offsets[p]);
                }
                any_submitted = true;
            }
        }
        if (any_submitted) {
            flush_ring_submissions();
        } else {
            prefetch_layer_id = -1;
        }
    }

    bool begin_op(const ggml_tensor * op, const uint64_t * l1_ready_masks) {
        if (!valid || !op || !op->src[0] || !op->src[2]) {
            return false;
        }
        auto tit = tensors.find(op->src[0]);
        if (tit == tensors.end()) {
            return false;
        }
        ensure_pool_initialized();

        const auto & info = *tit->second;
        auto & layer = *layers[info.layer_id];
        const ggml_tensor * ids = op->src[2];

        // If this layer is already active with the same ids (e.g. up/down after gate_up/gate):
        // - In decode: all io_uring reads were already submitted at gate_up/gate; reap any ready CQEs.
        // - In PP: L2 hits and PP-warming L2 slots stay pinned across gate_up -> down!
        if (layer.active && layer.active_ids == ids) {
            if (!layer.pp_mode) {
                reap_completions(/*block=*/ false);
            }
            return true;
        }

        // If a previous prefetch targeted a DIFFERENT layer (or we're entering PP), drain it first.
        if (prefetch_layer_id >= 0 && (prefetch_layer_id != layer.layer_id || ids->ne[1] > 1)) {
            drain_unused_prefetches();
        }

        layer.active = true;
        layer.active_ids = ids;
        layer.active_experts.clear();

        const int64_t n_ids = ids->ne[0];
        const int64_t n_tokens = ids->ne[1];
        layer.pp_mode = (n_tokens > 1);

        if (layer.pp_mode) {
            // PP mode: pin L2 hits AND warm free/cold L2 slots with the most frequent prompt experts!
            std::vector<uint32_t> hit_freq((size_t) layer.n_expert, 0);
            std::vector<uint32_t> miss_freq((size_t) layer.n_expert, 0);
            std::lock_guard<std::mutex> lock(pool_mutex);
            for (int64_t t = 0; t < n_tokens; ++t) {
                for (int64_t i = 0; i < n_ids; ++i) {
                    if (l1_ready_masks != nullptr && (l1_ready_masks[(size_t) t] & (UINT64_C(1) << i))) {
                        continue;
                    }
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + i * ids->nb[0]);
                    if (e < 0 || e >= layer.n_expert) continue;
                    const int32_t slot = layer.expert_to_slot[(size_t) e];
                    if (slot >= 0 && l2_slots[(size_t) slot].valid) {
                        hit_freq[(size_t) e]++;
                    } else {
                        miss_freq[(size_t) e]++;
                    }
                }
            }
            const uint64_t avg_thresh = (uint64_t) n_tokens * (uint64_t) n_ids;
            for (int32_t e = 0; e < layer.n_expert; ++e) {
                if (hit_freq[(size_t) e] == 0) continue;
                const int32_t slot = layer.expert_to_slot[(size_t) e];
                auto & m = l2_slots[(size_t) slot];
                m.ref_count++;
                if ((uint64_t) hit_freq[(size_t) e] * (uint64_t) layer.n_expert >= avg_thresh) {
                    m.second_chance = (uint8_t) std::min<int>(15, (int) m.second_chance + 2);
                } else {
                    m.second_chance >>= 1;
                }
                const bool is_rep = m.repacked;
                layer.active_l2_slots[(size_t) e] = slot;
                layer.active_is_repacked[(size_t) e] = is_rep ? 1 : 0;
                layer.active_can_repack_in_place[(size_t) e] = (!is_rep && m.ref_count == 1 && layer.can_repack_all) ? 1 : 0;
                layer.active_experts.push_back(e);
                uint8_t * slot_base = l2_pool_base + (size_t) slot * l2_slot_stride;
                for (size_t p = 0; p < layer.parts.size(); ++p) {
                    uint64_t aoff = 0; size_t alen = 0; size_t pad = 0;
                    calc_aligned_read(*layer.parts[p], e, aoff, alen, pad);
                    layer.active_ptrs[p][(size_t) e] = slot_base + layer.part_slot_offsets[p] + pad;
                }
                const uint64_t n_parts = (uint64_t) layer.parts.size();
                stat_l2_hits += n_parts;
                stat_pp_hits += n_parts;
                if (is_rep) {
                    stat_l2_repacked_hits += n_parts;
                    stat_pp_repacked_hits += n_parts;
                }
            }

            // Warm free L2 slots OR replace untouched pre-warmed slots (second_chance == 0) in this layer
            if (l2_num_slots > 0 && !layers.empty()) {
                std::vector<int32_t> cold_layer_slots;
                for (int32_t e = 0; e < layer.n_expert; ++e) {
                    const int32_t s = layer.expert_to_slot[(size_t) e];
                    if (s >= 0 && l2_slots[(size_t) s].valid &&
                            l2_slots[(size_t) s].ref_count == 0 && l2_slots[(size_t) s].second_chance == 0) {
                        cold_layer_slots.push_back(s);
                    }
                }
                const uint32_t free_total = l2_next_free < l2_num_slots ? (l2_num_slots - l2_next_free) : 0;
                const uint32_t per_layer_cap = std::max<uint32_t>(1, l2_num_slots / (uint32_t) layers.size());
                const uint32_t budget = std::min(free_total, per_layer_cap) + (uint32_t) cold_layer_slots.size();
                if (budget > 0) {
                    std::vector<std::pair<uint32_t, int32_t>> cands;
                    cands.reserve((size_t) layer.n_expert);
                    for (int32_t e = 0; e < layer.n_expert; ++e) {
                        if (miss_freq[(size_t) e] > 0 && layer.active_l2_slots[(size_t) e] < 0) {
                            cands.emplace_back(miss_freq[(size_t) e], e);
                        }
                    }
                    const size_t n_warm = std::min<size_t>(budget, cands.size());
                    if (n_warm > 0) {
                        std::partial_sort(cands.begin(), cands.begin() + n_warm, cands.end(),
                                [](const auto & a, const auto & b) { return a.first > b.first; });
                        size_t cold_idx = 0;
                        for (size_t k = 0; k < n_warm; ++k) {
                            const int32_t e = cands[k].second;
                            int32_t slot = claim_l2_slot_locked(layer.layer_id, e, /*free_only=*/ true);
                            if (slot < 0 && cold_idx < cold_layer_slots.size()) {
                                slot = cold_layer_slots[cold_idx++];
                                auto & m = l2_slots[(size_t) slot];
                                if (m.valid && m.layer_id >= 0 && m.expert >= 0) {
                                    auto lit = layers.find(m.layer_id);
                                    if (lit != layers.end() && lit->second->expert_to_slot[(size_t) m.expert] == slot) {
                                        lit->second->expert_to_slot[(size_t) m.expert] = -1;
                                    }
                                }
                                m.layer_id = layer.layer_id;
                                m.expert = e;
                                m.generation++;
                                m.ref_count = 1;
                                m.second_chance = 1;
                                m.valid = false;
                                m.repacked = false;
                                m.repack_queued = false;
                            }
                            if (slot < 0) break;
                            layer.active_l2_slots[(size_t) e] = slot;
                            layer.active_admit_l2[(size_t) e] = 1;
                            layer.active_is_repacked[(size_t) e] = 0;
                            layer.active_can_repack_in_place[(size_t) e] = layer.can_repack_all ? 1 : 0;
                            layer.active_experts.push_back(e);
                            stat_l2_admits++;
                        }
                    }
                }
            }
            return true;
        }

        // Decode mode (n_tokens == 1): save input hidden state for layer L+1 prefetch, reap any ready prefetch CQEs,
        // adopt speculative hits, and submit remaining missing experts in PART-MAJOR order across all parts!
        last_n_ids = (int) n_ids;
        if (prefetch_enabled && op->src[1] != nullptr && op->src[1]->data != nullptr &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[1]->ne[0] > 0) {
            const size_t n_embd = (size_t) op->src[1]->ne[0];
            last_hidden.resize(n_embd);
            memcpy(last_hidden.data(), op->src[1]->data, n_embd * sizeof(float));
        }
        reap_completions(/*block=*/ false);

        struct decode_miss_submit {
            int32_t expert;
            uint8_t * bundle_base;
        };
        std::vector<decode_miss_submit> misses_to_submit;
        std::vector<repack_job> hot_repacks;
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            for (int64_t i = 0; i < n_ids; ++i) {
                if (l1_ready_masks != nullptr && (l1_ready_masks[0] & (UINT64_C(1) << i))) {
                    continue;
                }
                const int32_t e = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0]);
                if (e < 0 || e >= layer.n_expert) continue;
                if (layer.active_l2_slots[(size_t) e] >= 0 || layer.active_scratch_slots[(size_t) e] >= 0) {
                    continue;
                }

                layer.active_experts.push_back(e);
                const int32_t slot = layer.expert_to_slot[(size_t) e];
                if (slot >= 0 && l2_slots[(size_t) slot].valid) {
                    auto & m = l2_slots[(size_t) slot];
                    m.ref_count++;
                    m.second_chance = (uint8_t) std::min<int>(15, (int) m.second_chance + 2);
                    layer.active_l2_slots[(size_t) e] = slot;
                    const bool is_rep = m.repacked;
                    layer.active_is_repacked[(size_t) e] = is_rep ? 1 : 0;
                    const bool can_rep_ip = !is_rep && !m.repack_queued && m.ref_count == 1 && layer.can_repack_all;
                    layer.active_can_repack_in_place[(size_t) e] = can_rep_ip ? 1 : 0;
                    if (!is_rep && !can_rep_ip && !m.repack_queued && layer.can_repack_all) {
                        m.repack_queued = true;
                        hot_repacks.push_back({ slot, layer.layer_id, e, m.generation });
                    }
                    uint8_t * slot_base = l2_pool_base + (size_t) slot * l2_slot_stride;
                    for (size_t p = 0; p < layer.parts.size(); ++p) {
                        uint64_t aoff = 0; size_t alen = 0; size_t pad = 0;
                        calc_aligned_read(*layer.parts[p], e, aoff, alen, pad);
                        layer.active_ptrs[p][(size_t) e] = slot_base + layer.part_slot_offsets[p] + pad;
                    }
                    stat_l2_hits++;
                    stat_dec_hits++;
                    if (is_rep || can_rep_ip) {
                        stat_l2_repacked_hits++;
                        stat_dec_repacked_hits++;
                    }
                } else {
                    stat_l3_misses++;
                    stat_dec_misses++;
                    const bool has_free = (l2_next_free < l2_num_slots);
                    const bool recent = (admit_window == 0) ||
                        ((layer.step - layer.last_seen_step[(size_t) e]) <= (int64_t) admit_window);

                    // Check if cross-layer speculative prefetch ALREADY submitted/loaded this expert!
                    int pf_idx = -1;
                    if (prefetch_layer_id == layer.layer_id) {
                        for (size_t k = 0; k < prefetched.size(); ++k) {
                            if (prefetched[k].expert == e && !prefetched[k].adopted) {
                                pf_idx = (int) k;
                                break;
                            }
                        }
                    }
                    if (pf_idx >= 0) {
                        prefetched[(size_t) pf_idx].adopted = true;
                        layer.active_scratch_slots[(size_t) e] = prefetched[(size_t) pf_idx].scratch_slot;
                        layer.active_is_repacked[(size_t) e] = 0;
                        layer.active_can_repack_in_place[(size_t) e] = 0;
                        if (has_free || recent) {
                            layer.active_admit_from_scratch[(size_t) e] = 1;
                        }
                        stat_prefetch_hits++;
                        continue; // Already in-flight or completed in scratch_slot!
                    }

                    int32_t l2_slot = -1;
                    if (has_free || recent) {
                        l2_slot = claim_l2_slot_locked(layer.layer_id, e);
                    }
                    uint8_t * bundle_base = nullptr;
                    if (l2_slot >= 0) {
                        layer.active_l2_slots[(size_t) e] = l2_slot;
                        layer.active_admit_l2[(size_t) e] = 1;
                        layer.active_is_repacked[(size_t) e] = 0;
                        layer.active_can_repack_in_place[(size_t) e] = layer.can_repack_all ? 1 : 0;
                        bundle_base = l2_pool_base + (size_t) l2_slot * l2_slot_stride;
                        stat_l2_admits++;
                    } else {
                        const int32_t sc = alloc_scratch_slot();
                        GGML_ASSERT(sc >= 0 && "nvme_expert_cache: scratch ring exhausted");
                        layer.active_scratch_slots[(size_t) e] = sc;
                        layer.active_is_repacked[(size_t) e] = 0;
                        layer.active_can_repack_in_place[(size_t) e] = 0;
                        bundle_base = scratch_slots[(size_t) sc].base;
                    }
                    misses_to_submit.push_back({ e, bundle_base });
                }
            }
        }

        // Submit in PART-MAJOR order so all gate_up (part 0) reads complete before down (part 1) reads!
        if (!misses_to_submit.empty()) {
            for (size_t p = 0; p < layer.parts.size(); ++p) {
                for (const auto & ms : misses_to_submit) {
                    submit_part_read(layer, (int) p, ms.expert, ms.bundle_base + layer.part_slot_offsets[p]);
                }
            }
        }

        // Defer cleanup of any unused prefetches for this layer until end_op (so they don't stall begin_op)
        if (prefetch_layer_id == layer.layer_id) {
            for (const auto & pe : prefetched) {
                if (!pe.adopted && pe.scratch_slot >= 0) {
                    unused_prefetch_scratch.push_back(pe.scratch_slot);
                    if (pe.expert >= 0 && pe.expert < layer.n_expert) {
                        for (size_t p = 0; p < layer.parts.size(); ++p) {
                            layer.active_ptrs[p][(size_t) pe.expert] = nullptr;
                        }
                    }
                }
            }
            prefetched.clear();
            prefetch_layer_id = -1;
        }

        if (!misses_to_submit.empty()) {
            flush_ring_submissions();
        }
        for (const auto & job : hot_repacks) {
            enqueue_repack(job.slot, job.layer_id, job.expert, job.generation);
        }
        return true;
    }

    const void * get_expert(const ggml_tensor * op, int32_t expert, bool wait) {
        auto tit = tensors.find(op->src[0]);
        if (tit == tensors.end()) {
            return nullptr;
        }
        const auto & info = *tit->second;
        auto & layer = *layers[info.layer_id];

        if (expert < 0) {
            if (wait && ring_in_flight > 0) {
                const auto t0 = print_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                reap_completions(/*block=*/ true);
                if (print_stats) {
                    const auto t1 = std::chrono::steady_clock::now();
                    const double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    stat_wait_ms += dt;
                    if (layer.pp_mode) {
                        stat_pp_wait_ms += dt;
                    } else {
                        stat_dec_wait_ms += dt;
                    }
                }
            } else {
                reap_completions(/*block=*/ false);
            }
            return nullptr;
        }

        if (!wait) {
            if (layer.active_part_pending[(size_t) info.part_idx][(size_t) expert] != 0) {
                return nullptr;
            }
            return layer.active_ptrs[(size_t) info.part_idx][(size_t) expert];
        }

        if (layer.active_part_pending[(size_t) info.part_idx][(size_t) expert] != 0) {
            const auto t0 = print_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            while (layer.active_part_pending[(size_t) info.part_idx][(size_t) expert] != 0) {
                reap_completions(/*block=*/ true);
            }
            if (print_stats) {
                const auto t1 = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                stat_wait_ms += dt;
                if (layer.pp_mode) {
                    stat_pp_wait_ms += dt;
                } else {
                    stat_dec_wait_ms += dt;
                }
            }
        }
        return layer.active_ptrs[(size_t) info.part_idx][(size_t) expert];
    }

    int is_expert_repacked(const ggml_tensor * op, int32_t expert) {
        auto tit = tensors.find(op->src[0]);
        if (tit == tensors.end()) {
            return 0;
        }
        const auto & info = *tit->second;
        auto lit = layers.find(info.layer_id);
        if (lit == layers.end() || expert < 0 || expert >= lit->second->n_expert) {
            return 0;
        }
        auto & layer = *lit->second;
        const int l2_hit_flag = (layer.active_l2_slots[(size_t) expert] >= 0 && !layer.active_admit_l2[(size_t) expert]) ? 4 : 0;
        if (layer.active_is_repacked[(size_t) expert] != 0) {
            return l2_hit_flag | 1;
        }
        if (layer.active_can_repack_in_place[(size_t) expert] != 0) {
            layer.active_part_repacked[(size_t) info.part_idx][(size_t) expert] = 1;
            return l2_hit_flag | 2;
        }
        return l2_hit_flag;
    }

    const void * pp_prepare_expert(const ggml_tensor * op, const int32_t * active_experts, int active_idx, int n_active) {
        auto tit = tensors.find(op->src[0]);
        if (tit == tensors.end()) {
            return nullptr;
        }
        const auto & info = *tit->second;
        auto & layer = *layers[info.layer_id];
        const int p = info.part_idx;

        bool any_submitted = false;
        const int start_k = std::max(0, active_idx);
        const int win_end = std::min(n_active, start_k + PP_WINDOW);
        for (int k = start_k; k < win_end; ++k) {
            const int32_t e = active_experts[k];
            if (layer.active_ptrs[(size_t) p][(size_t) e] != nullptr) {
                continue; // L2 hit or already submitted!
            }
            // If this expert was selected for PP L2 warming, read directly into its claimed L2 slot!
            if (layer.active_l2_slots[(size_t) e] >= 0 && layer.active_admit_l2[(size_t) e]) {
                uint8_t * dst = l2_pool_base + (size_t) layer.active_l2_slots[(size_t) e] * l2_slot_stride +
                                layer.part_slot_offsets[(size_t) p];
                stat_l3_misses++;
                stat_pp_misses++;
                submit_part_read(layer, p, e, dst);
                any_submitted = true;
                continue;
            }
            const int32_t sc = alloc_scratch_slot();
            if (sc < 0) {
                break;
            }
            layer.active_scratch_slots[(size_t) e] = sc;
            stat_l3_misses++;
            stat_pp_misses++;
            submit_part_read(layer, p, e, scratch_slots[(size_t) sc].base);
            any_submitted = true;
        }
        if (any_submitted) {
            flush_ring_submissions();
        }
        if (active_idx < 0) {
            return nullptr;
        }

        const int32_t cur_e = active_experts[active_idx];
        return get_expert(op, cur_e, /*wait=*/ true);
    }

    void pp_release_expert(const ggml_tensor * op, int32_t expert) {
        auto tit = tensors.find(op->src[0]);
        if (tit == tensors.end()) {
            return;
        }
        const auto & info = *tit->second;
        auto & layer = *layers[info.layer_id];

        const int32_t sc = layer.active_scratch_slots[(size_t) expert];
        if (sc >= 0) {
            scratch_slots[(size_t) sc].in_use = false;
            layer.active_scratch_slots[(size_t) expert] = -1;
            layer.active_ptrs[(size_t) info.part_idx][(size_t) expert] = nullptr;
        }
    }

    void end_op(const ggml_tensor * op) {
        if (!valid || !op || !op->src[0]) {
            return;
        }
        auto tit = tensors.find(op->src[0]);
        if (tit == tensors.end()) {
            return;
        }
        const auto & info = *tit->second;
        auto & layer = *layers[info.layer_id];
        if (!layer.active) {
            return;
        }

        // Keep layer active & L2 slots pinned until `down` finishes (both decode AND PP!)
        if (info.part != nvme_part::down) {
            if (layer.pp_mode) {
#if defined(GGML_USE_IO_URING)
                while (ring_ready && ring_in_flight > 0) {
                    reap_completions(/*block=*/ true);
                }
#endif
                for (auto & sc : scratch_slots) {
                    sc.in_use = false;
                }
                std::fill(layer.active_scratch_slots.begin(), layer.active_scratch_slots.end(), -1);
                std::fill(layer.active_part_pending[(size_t) info.part_idx].begin(),
                          layer.active_part_pending[(size_t) info.part_idx].end(), 0);
            }
            return;
        }

#if defined(GGML_USE_IO_URING)
        while (ring_ready && ring_in_flight > 0) {
            reap_completions(/*block=*/ true);
        }
#endif

        // Release any unused speculative prefetch scratch slots from this layer now that CQEs are drained
        for (int32_t sc : unused_prefetch_scratch) {
            if (sc >= 0) {
                scratch_slots[(size_t) sc].in_use = false;
            }
        }
        unused_prefetch_scratch.clear();

        std::vector<repack_job> admit_repacks;
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            for (int32_t e : layer.active_experts) {
                const int32_t l2_slot = layer.active_l2_slots[(size_t) e];
                if (l2_slot >= 0) {
                    auto & m = l2_slots[(size_t) l2_slot];
                    if (layer.active_admit_l2[(size_t) e]) {
                        bool all_parts_read = true;
                        for (size_t p = 0; p < layer.parts.size(); ++p) {
                            if (layer.active_ptrs[p][(size_t) e] == nullptr) {
                                all_parts_read = false;
                                break;
                            }
                        }
                        if (all_parts_read) {
                            m.valid = true;
                            layer.expert_to_slot[(size_t) e] = l2_slot;
                        }
                        layer.active_admit_l2[(size_t) e] = 0;
                    }
                    if (m.valid && !m.repacked && layer.can_repack_all) {
                        bool all_parts_rep = true;
                        for (size_t p = 0; p < layer.parts.size(); ++p) {
                            if (!layer.active_part_repacked[p][(size_t) e]) {
                                all_parts_rep = false;
                                break;
                            }
                        }
                        if (all_parts_rep) {
                            m.repacked = true;
                            m.repack_queued = false;
                            stat_repacked_done.fetch_add(1, std::memory_order_relaxed);
                        } else if (!m.repack_queued) {
                            m.repack_queued = true;
                            admit_repacks.push_back({ l2_slot, layer.layer_id, e, m.generation });
                        }
                    }
                    if (m.ref_count > 0) {
                        m.ref_count--;
                    }
                    layer.active_l2_slots[(size_t) e] = -1;
                }
                const int32_t sc = layer.active_scratch_slots[(size_t) e];
                if (sc >= 0) {
                    if (layer.active_admit_from_scratch[(size_t) e]) {
                        const int32_t new_slot = claim_l2_slot_locked(layer.layer_id, e);
                        if (new_slot >= 0) {
                            uint8_t * dst_base = l2_pool_base + (size_t) new_slot * l2_slot_stride;
                            const uint8_t * src_base = scratch_slots[(size_t) sc].base;
                            memcpy(dst_base, src_base, layer.bundle_alloc_bytes);
                            auto & nm = l2_slots[(size_t) new_slot];
                            nm.valid = true;
                            nm.ref_count = 0;
                            layer.expert_to_slot[(size_t) e] = new_slot;
                            stat_l2_admits++;
                            if (layer.can_repack_all && !nm.repack_queued) {
                                nm.repack_queued = true;
                                admit_repacks.push_back({ new_slot, layer.layer_id, e, nm.generation });
                            }
                        }
                        layer.active_admit_from_scratch[(size_t) e] = 0;
                    }
                    scratch_slots[(size_t) sc].in_use = false;
                    layer.active_scratch_slots[(size_t) e] = -1;
                }
                layer.active_is_repacked[(size_t) e] = 0;
                layer.active_can_repack_in_place[(size_t) e] = 0;
                for (size_t p = 0; p < layer.parts.size(); ++p) {
                    layer.active_ptrs[p][(size_t) e] = nullptr;
                    layer.active_part_pending[p][(size_t) e] = 0;
                    layer.active_part_repacked[p][(size_t) e] = 0;
                }
                if (!layer.pp_mode) {
                    layer.last_seen_step[(size_t) e] = layer.step;
                }
            }
        }
        for (const auto & job : admit_repacks) {
            enqueue_repack(job.slot, job.layer_id, job.expert, job.generation);
        }

        const bool was_decode = !layer.pp_mode;
        if (was_decode) {
            layer.step++;
        } else {
            for (auto & sc : scratch_slots) {
                sc.in_use = false;
            }
            std::fill(layer.active_scratch_slots.begin(), layer.active_scratch_slots.end(), -1);
            for (size_t p = 0; p < layer.parts.size(); ++p) {
                std::fill(layer.active_ptrs[p].begin(), layer.active_ptrs[p].end(), nullptr);
                std::fill(layer.active_part_pending[p].begin(), layer.active_part_pending[p].end(), 0);
            }
        }
        layer.active_experts.clear();
        layer.active = false;
        layer.active_ids = nullptr;

        if (was_decode) {
            launch_next_layer_prefetch(layer.layer_id);
        }

        stat_ops++;
        if (print_stats && (stat_ops % 4800 == 0)) {
            const uint64_t total = stat_l2_hits + stat_l3_misses;
            fprintf(stderr,
                    "nvme_expert_cache: [ops=%llu] L2 hits=%llu (%.1f%%, repacked=%llu), L3 misses=%llu, "
                    "admits=%llu, repacked_slots=%llu, read=%.2f GB, wait=%.1f ms\n",
                    (unsigned long long) stat_ops,
                    (unsigned long long) stat_l2_hits,
                    total > 0 ? 100.0 * (double) stat_l2_hits / (double) total : 0.0,
                    (unsigned long long) stat_l2_repacked_hits,
                    (unsigned long long) stat_l3_misses,
                    (unsigned long long) stat_l2_admits,
                    (unsigned long long) stat_repacked_done.load(std::memory_order_relaxed),
                    (double) stat_bytes_read / 1e9,
                    stat_wait_ms);
        }
    }

    bool copy_expert_to_staging(const ggml_tensor * tensor, int32_t expert, void * dst, size_t bytes) {
        auto tit = tensors.find(tensor);
        if (tit == tensors.end() || !dst) {
            return false;
        }
        const auto & info = *tit->second;
        if (expert < 0 || expert >= info.n_expert || bytes != info.expert_bytes) {
            return false;
        }
        auto & layer = *layers[info.layer_id];

        int32_t pinned_slot = -1;
        bool was_repacked = false;
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            const int32_t slot = layer.expert_to_slot[(size_t) expert];
            if (slot >= 0 && l2_slots[(size_t) slot].valid) {
                l2_slots[(size_t) slot].ref_count++;
                pinned_slot = slot;
                was_repacked = l2_slots[(size_t) slot].repacked;
            }
        }
        if (pinned_slot >= 0) {
            uint64_t aoff = 0; size_t alen = 0; size_t pad = 0;
            calc_aligned_read(info, expert, aoff, alen, pad);
            const uint8_t * src = l2_pool_base + (size_t) pinned_slot * l2_slot_stride +
                                  layer.part_slot_offsets[(size_t) info.part_idx] + pad;
            bool ok = false;
            if (!was_repacked) {
                memcpy(dst, src, bytes);
                ok = true;
            } else {
                ok = ggml_cpu_nvme_unpack_znq_expert(tensor, src, dst, bytes);
            }
            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                l2_slots[(size_t) pinned_slot].ref_count--;
            }
            if (ok) {
                return true;
            }
        }
        if (l2_num_slots > 0) {
            // Strictly promote L2 RAM -> L1 VRAM so background L1 workers never contend with io_uring on the NVMe bus
            return false;
        }

        // Slow path (only when L2 RAM pool is disabled): aligned O_DIRECT pread into thread-local bounce buffer
#if defined(__linux__)
        uint64_t aligned_off = 0;
        size_t aligned_len = 0;
        size_t data_pad = 0;
        calc_aligned_read(info, expert, aligned_off, aligned_len, data_pad);

        thread_local void * tl_buf = nullptr;
        thread_local size_t tl_cap = 0;
        if (tl_cap < aligned_len) {
            free(tl_buf);
            tl_buf = nullptr;
            tl_cap = 0;
            if (posix_memalign(&tl_buf, NVME_ALIGN, aligned_len) != 0) {
                return false;
            }
            tl_cap = aligned_len;
        }

        size_t done = 0;
        while (done < aligned_len) {
            ssize_t r = pread(info.fd, (uint8_t *) tl_buf + done, aligned_len - done, (off_t) (aligned_off + done));
            if (r <= 0) {
                if (r < 0 && errno == EINTR) continue;
                return false;
            }
            done += (size_t) r;
        }
        memcpy(dst, (const uint8_t *) tl_buf + data_pad, bytes);
        return true;
#else
        return false;
#endif
    }
};

nvme_expert_cache::nvme_expert_cache(size_t ram_budget_bytes, uint32_t admit_window)
    : pimpl(std::make_unique<impl>(ram_budget_bytes, admit_window)) {}

nvme_expert_cache::~nvme_expert_cache() = default;

bool nvme_expert_cache::is_valid() const { return pimpl && pimpl->valid; }

size_t nvme_expert_cache::memory_bytes() const {
    return pimpl ? (pimpl->l2_pool_bytes + pimpl->scratch_pool_bytes) : 0;
}

void nvme_expert_cache::register_tensor(const ggml_tensor * tensor, const char * file_path, uint64_t file_offset) {
    if (pimpl) pimpl->register_tensor(tensor, file_path, file_offset);
}

void nvme_expert_cache::register_router(int layer_id, const ggml_tensor * gate_inp, const ggml_tensor * exp_probs_b) {
    if (pimpl) pimpl->register_router(layer_id, gate_inp, exp_probs_b);
}

bool nvme_expert_cache::has_tensor(const ggml_tensor * tensor) const {
    return pimpl && pimpl->tensors.count(tensor) != 0;
}

bool nvme_expert_cache::begin_op(const ggml_tensor * op, const uint64_t * l1_ready_masks) {
    return pimpl && pimpl->begin_op(op, l1_ready_masks);
}

void nvme_expert_cache::end_op(const ggml_tensor * op) {
    if (pimpl) pimpl->end_op(op);
}

const void * nvme_expert_cache::get_expert(const ggml_tensor * op, int32_t expert, bool wait) {
    return pimpl ? pimpl->get_expert(op, expert, wait) : nullptr;
}

int nvme_expert_cache::is_expert_repacked(const ggml_tensor * op, int32_t expert) {
    return pimpl ? pimpl->is_expert_repacked(op, expert) : 0;
}

const void * nvme_expert_cache::pp_prepare_expert(const ggml_tensor * op, const int32_t * active_experts, int active_idx, int n_active) {
    return pimpl ? pimpl->pp_prepare_expert(op, active_experts, active_idx, n_active) : nullptr;
}

void nvme_expert_cache::pp_release_expert(const ggml_tensor * op, int32_t expert) {
    if (pimpl) pimpl->pp_release_expert(op, expert);
}

bool nvme_expert_cache::copy_expert_to_staging(const ggml_tensor * tensor, int32_t expert, void * dst, size_t bytes) {
    return pimpl && pimpl->copy_expert_to_staging(tensor, expert, dst, bytes);
}
