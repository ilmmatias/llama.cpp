#include "ggml-backend-impl.h"
#include "qsa-kv.cuh"

#include <climits>
#include <map>

namespace {

constexpr int resolve_threads = 1024;
constexpr int control_ints    = 16;

struct cache_state {
    char * host;
    char * device_host;
    char * cache = nullptr;

    int * storage = nullptr;
    int * pages;
    int * owners;
    int * referenced;
    int * needed;
    int * misses;
    // [0] CLOCK hand, [1] placed misses, [2] selection epoch,
    // [3] latest hits, [4] latest misses, [5] latest evictions,
    // [6] latest overlap with prev epoch, [7] latest unique selected,
    // [8] latest misses in prev epoch, [9] latest misses seen before,
    // [10..15] cumulative totals for hits, misses, evictions, overlap_prev, coherence_denom, miss_in_prev.
    int * control;

    size_t bytes;
    size_t page_bytes;
    uint32_t page_shift = 0;
    int n_pages;
    int n_slots;

    cache_state * paired_peer    = nullptr;
    cache_state * paired_primary = nullptr;
};

struct buffer_context {
    void * host        = nullptr;
    void * device_host = nullptr;
    int device;
    uint32_t resident_tokens;

    std::vector<std::unique_ptr<cache_state>> tensors;
};

struct type_context {
    std::string name;
    int         device;
    uint32_t    resident_tokens;
};

static const char * type_name(ggml_backend_buffer_type_t buft) {
    return static_cast<type_context *>(buft->context)->name.c_str();
}

static buffer_context & context(ggml_backend_buffer_t buffer) {
    return *static_cast<buffer_context *>(buffer->context);
}

static cache_state * find_cache(const ggml_tensor * tensor) {
    if (!ggml_cuda_qsa_kv_is_paged(tensor)) {
        return nullptr;
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(tensor->data);

    for (const auto & state : context(tensor->buffer).tensors) {
        const uintptr_t start = reinterpret_cast<uintptr_t>(state->host);

        if (address >= start && address < start + state->bytes) {
            return state.get();
        }
    }

    return nullptr;
}

static size_t cache_metadata_size(size_t n_pages, size_t n_slots) {
    return (2*n_pages + 3*n_slots + control_ints)*sizeof(int);
}

static __device__ __forceinline__ int cache_page_for_byte(const cache_state & state, size_t byte) {
    return state.page_shift ? int(byte >> state.page_shift) : int(byte / state.page_bytes);
}

static void reset_cache(cache_state & state) {
    CUDA_CHECK(cudaMemset(state.pages, 0xff, (state.n_pages + state.n_slots) * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.referenced, 0, state.n_slots * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.control, 0, control_ints*sizeof(int)));
}

static void reset_cache_async(cache_state & state, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(state.pages, 0xff, (state.n_pages + state.n_slots) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(state.referenced, 0, state.n_slots * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(state.control, 0, control_ints*sizeof(int), stream));
}

static void unpair_state_sync(cache_state & state) {
    if (state.paired_peer) {
        cache_state * peer   = state.paired_peer;
        state.paired_peer    = nullptr;
        state.paired_primary = nullptr;
        peer->paired_peer    = nullptr;
        peer->paired_primary = nullptr;
        reset_cache(*peer);
    }
}

static void unpair_state_async(cache_state * state, cudaStream_t stream) {
    if (state && state->paired_peer) {
        cache_state * peer    = state->paired_peer;
        state->paired_peer    = nullptr;
        state->paired_primary = nullptr;
        peer->paired_peer     = nullptr;
        peer->paired_primary  = nullptr;
        reset_cache_async(*state, stream);
        reset_cache_async(*peer, stream);
    }
}

static void before_host_write(ggml_backend_buffer_t buffer, const ggml_tensor * tensor = nullptr) {
    auto & ctx = context(buffer);
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (tensor) {
        cache_state * state = find_cache(tensor);
        GGML_ASSERT(state);
        unpair_state_sync(*state);
        reset_cache(*state);
    } else {
        for (auto & state : ctx.tensors) {
            state->paired_peer    = nullptr;
            state->paired_primary = nullptr;
            reset_cache(*state);
        }
    }

    // Backend compute streams are non-blocking and need not wait for these
    // default-stream memsets. Finish invalidation before returning to callers.
    CUDA_CHECK(cudaDeviceSynchronize());
}

static int qsa_kv_stats_level() {
    static const int level = [] {
        const char * env = getenv("LLAMA_QSA_KV_STATS");
        return env ? atoi(env) : 0;
    }();
    return level;
}

static void free_buffer(ggml_backend_buffer_t buffer) {
    auto & ctx = context(buffer);
    ggml_cuda_set_device(ctx.device);

    if (qsa_kv_stats_level() > 0 && !ctx.tensors.empty()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx = 0;
        for (const auto & state : ctx.tensors) {
            int ctrl[control_ints] = {};
            CUDA_CHECK(cudaMemcpy(ctrl, state->control, control_ints * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctrl[2] > 0) {
                const int64_t total_req = int64_t(ctrl[10]) + int64_t(ctrl[11]);
                const double  hit_pct   = total_req > 0 ? 100.0 * double(ctrl[10]) / double(total_req) : 0.0;
                const double  coh       = ctrl[14] > 0 ? 100.0 * double(ctrl[13]) / double(ctrl[14]) : 0.0;
                fprintf(stderr, "qsa_kv[%02d] pages=%d slots=%d epochs=%d | last(hit=%d miss=%d evict=%d uniq=%d coh=%.1f%% miss_prev=%d miss_seen=%d) | tot(hits=%d misses=%d evict=%d hit=%.2f%% coh=%.2f%% [%d/%d] miss_prev=%d)\n",
                    idx, state->n_pages, state->n_slots, ctrl[2],
                    ctrl[3], ctrl[4], ctrl[5], ctrl[7],
                    ctrl[7] > 0 ? 100.0 * double(ctrl[6]) / double(ctrl[7]) : 0.0,
                    ctrl[8], ctrl[9],
                    ctrl[10], ctrl[11], ctrl[12], hit_pct, coh, ctrl[13], ctrl[14], ctrl[15]);
            }
            ++idx;
        }
    }

    for (auto & state : ctx.tensors) {
        CUDA_CHECK(cudaFree(state->cache));
        CUDA_CHECK(cudaFree(state->storage));
    }

    if (ctx.host) {
        CUDA_CHECK(cudaFreeHost(ctx.host));
    }

    delete &ctx;
}

static void * get_base(ggml_backend_buffer_t buffer) {
    return context(buffer).host;
}

static ggml_status init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src) {
        return GGML_STATUS_SUCCESS;
    }

    auto & ctx = context(buffer);
    ggml_cuda_set_device(ctx.device);

    GGML_ASSERT((tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_Q8_0) && ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[2] == 1 && tensor->ne[3] == 1 && tensor->nb[1] % 16 == 0);

    auto state         = std::make_unique<cache_state>();
    state->host        = static_cast<char *>(tensor->data);
    state->device_host = static_cast<char *>(ctx.device_host) + (state->host - static_cast<char *>(ctx.host));
    state->bytes       = ggml_nbytes(tensor);
    state->page_bytes  = 4 * tensor->nb[1];
    state->n_pages     = (tensor->ne[1] + 3) / 4;
    state->n_slots     = std::min<int>(state->n_pages, ctx.resident_tokens / 4);

    GGML_ASSERT(state->n_slots > 0);

    state->page_shift = 0;
    if ((state->page_bytes & (state->page_bytes - 1)) == 0) {
        for (size_t bytes = state->page_bytes; bytes > 1; bytes >>= 1) {
            ++state->page_shift;
        }
    }

    const size_t metadata_size = cache_metadata_size(state->n_pages, state->n_slots);
    cudaError_t  err = cudaMalloc(reinterpret_cast<void **>(&state->cache), size_t(state->n_slots) * state->page_bytes);
    if (err == cudaSuccess) {
        err = cudaMalloc(reinterpret_cast<void **>(&state->storage), metadata_size);
    }

    if (err != cudaSuccess) {
        (void) cudaGetLastError();

        if (state->cache) {
            CUDA_CHECK(cudaFree(state->cache));
        }

        return GGML_STATUS_ALLOC_FAILED;
    }

    state->pages      = state->storage;
    state->owners     = state->pages + state->n_pages;
    state->referenced = state->owners + state->n_slots;
    state->needed     = state->referenced + state->n_slots;
    state->misses     = state->needed + state->n_pages;
    state->control    = state->misses + state->n_slots;

    reset_cache(*state);
    CUDA_CHECK(cudaDeviceSynchronize());
    ctx.tensors.push_back(std::move(state));

    return GGML_STATUS_SUCCESS;
}

static void memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    before_host_write(buffer, tensor);
    memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    before_host_write(buffer, tensor);
    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_cuda_set_device(context(buffer).device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static void clear_buffer(ggml_backend_buffer_t buffer, uint8_t value) {
    before_host_write(buffer);

    if (buffer->size) {
        memset(context(buffer).host, value, buffer->size);
    }
}

static ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const auto & type    = *static_cast<type_context *>(buft->context);
    auto         ctx     = std::make_unique<buffer_context>();
    ctx->device          = type.device;
    ctx->resident_tokens = type.resident_tokens;

    ggml_cuda_set_device(ctx->device);
    if (size) {
        cudaError_t err = cudaHostAlloc(&ctx->host, size, cudaHostAllocMapped | cudaHostAllocPortable);

        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_ERROR("%s: pinned KV allocation failed: %s\n", __func__, cudaGetErrorString(err));
            return nullptr;
        }

        CUDA_CHECK(cudaHostGetDevicePointer(&ctx->device_host, ctx->host, 0));
    }

    ggml_backend_buffer_i iface = {};
    iface.free_buffer           = free_buffer;
    iface.get_base              = get_base;
    iface.init_tensor           = init_tensor;
    iface.memset_tensor         = memset_tensor;
    iface.set_tensor            = set_tensor;
    iface.get_tensor            = get_tensor;
    iface.clear                 = clear_buffer;

    return ggml_backend_buffer_init(buft, iface, ctx.release(), size);
}

static __device__ __forceinline__ void record_resolve_stats(
        const cache_state & state, int epoch,
        int n_protected, int n_misses, int n_evictions,
        int n_overlap_prev, int n_miss_in_prev, int n_miss_seen) {
    const int uniq    = n_protected + n_misses;
    state.control[3]  = n_protected;
    state.control[4]  = n_misses;
    state.control[5]  = n_evictions;
    state.control[6]  = n_overlap_prev;
    state.control[7]  = uniq;
    state.control[8]  = n_miss_in_prev;
    state.control[9]  = n_miss_seen;
    state.control[10] += n_protected;
    state.control[11] += n_misses;
    state.control[12] += n_evictions;
    if (epoch > 1) {
        state.control[13] += n_overlap_prev;
        state.control[14] += uniq;
        state.control[15] += n_miss_in_prev;
    }
}

static __device__ __forceinline__ int clock_place_evictions(
        const cache_state & state, int epoch, int need) {
    __shared__ int cut;
    __shared__ int warp_sums[resolve_threads / WARP_SIZE];
    __shared__ int n_evictions;

    if (threadIdx.x == 0) {
        n_evictions = 0;
    }

    __syncthreads();

    const int scan_width = min(resolve_threads, state.n_slots);
    const int lane       = threadIdx.x % WARP_SIZE;
    const int warp       = threadIdx.x / WARP_SIZE;

    int hand   = state.control[0];
    int placed = 0;

    // Sweep disjoint slots cooperatively instead of making every miss search
    // the same protected range. A prefix sum pairs victims with queued misses.
    for (int scanned = 0; placed < need && scanned < 2*state.n_slots; scanned += scan_width) {
        int slot = hand + threadIdx.x;

        if (slot >= state.n_slots) {
            slot -= state.n_slots;
        }

        const bool active         = threadIdx.x < scan_width;
        const int  old            = active ? state.owners[slot] : -1;
        const bool protected_page = old >= 0 && state.needed[old] == epoch;
        const bool candidate      = active && !protected_page && (old < 0 || state.referenced[slot] == 0);
        const int  inclusive      = warp_prefix_inclusive_sum(int(candidate));

        if (threadIdx.x == 0) {
            cut = scan_width;
        }
        if (lane == WARP_SIZE - 1) {
            warp_sums[warp] = inclusive;
        }

        __syncthreads();

        if (warp == 0) {
            warp_sums[lane] = warp_prefix_inclusive_sum(warp_sums[lane]);
        }

        __syncthreads();

        const int rank      = inclusive - int(candidate) + (warp > 0 ? warp_sums[warp - 1] : 0);
        const int total     = warp_sums[resolve_threads / WARP_SIZE - 1];
        const int remaining = need - placed;

        if (candidate && rank == remaining - 1) {
            cut = threadIdx.x + 1;
        }

        __syncthreads();

        if (candidate && rank < remaining) {
            const int page = state.misses[placed + rank];

            if (old >= 0) {
                state.pages[old] = -1;
                atomicAdd(&n_evictions, 1);
            }

            state.owners[slot]     = page;
            state.referenced[slot] = 1;
            state.pages[page]      = slot;
        } else if (active && threadIdx.x < cut && !protected_page) {
            state.referenced[slot] = 0;
        }

        placed += min(total, remaining);
        hand   += cut;

        if (hand >= state.n_slots) {
            hand -= state.n_slots;
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        state.control[0] = hand;
        state.control[1] = placed;
    }

    return n_evictions;
}

// Mark the entire selection before replacing anything: no page used by another
// query in this launch may be evicted. Oversized unions retain unmapped pages,
// which attention reads from authoritative RAM rather than stale cache slots.
static __global__ void resolve_pages(
        cache_state state_k, cache_state state_v, const int32_t * indices, int n_indices,
        size_t offset_k, size_t offset_v, size_t row_bytes_k, size_t row_bytes_v) {
    const cache_state & state = blockIdx.x == 0 ? state_k : state_v;
    const size_t offset      = blockIdx.x == 0 ? offset_k : offset_v;
    const size_t row_bytes   = blockIdx.x == 0 ? row_bytes_k : row_bytes_v;

    __shared__ int n_protected;
    __shared__ int n_misses;
    __shared__ int n_overlap_prev;
    __shared__ int n_miss_in_prev;
    __shared__ int n_miss_seen;
    __shared__ int epoch;

    if (threadIdx.x == 0) {
        n_protected    = 0;
        n_misses       = 0;
        n_overlap_prev = 0;
        n_miss_in_prev = 0;
        n_miss_seen    = 0;

        epoch = state.control[2] == INT_MAX ? 1 : state.control[2] + 1;
        state.control[2] = epoch;
    }

    __syncthreads();

    // Epoch tags avoid scanning the entire allocated context on every token.
    // Clear tags only on initialization, host reset, or epoch wraparound.
    if (epoch == 1) {
        for (int p = threadIdx.x; p < state.n_pages; p += blockDim.x) {
            state.needed[p] = 0;
        }
    }

    __syncthreads();

    int loc_prot   = 0;
    int loc_ov     = 0;
    int loc_m_prev = 0;
    int loc_m_seen = 0;

    for (int i = threadIdx.x; i < n_indices; i += blockDim.x) {
        const int row = indices ? indices[i] : i;

        if (row < 0) {
            continue;
        }

        const size_t byte = offset + size_t(row)*row_bytes;

        if (byte >= state.bytes) {
            continue;
        }

        const int page = cache_page_for_byte(state, byte);
        const int prev = atomicExch(state.needed + page, epoch);

        if (prev != epoch) {
            const bool in_prev = epoch > 1 && prev == epoch - 1;
            if (in_prev) {
                ++loc_ov;
            }

            const int slot = state.pages[page];

            if (slot >= 0) {
                state.referenced[slot] = 1;
                ++loc_prot;
            } else {
                if (in_prev) {
                    ++loc_m_prev;
                }
                if (epoch > 1 && prev > 0) {
                    ++loc_m_seen;
                }
                const int miss = atomicAdd(&n_misses, 1);

                if (miss < state.n_slots) {
                    state.misses[miss] = page;
                }
            }
        }
    }

    if (loc_prot > 0) {
        atomicAdd(&n_protected, loc_prot);
    }
    if (loc_ov > 0) {
        atomicAdd(&n_overlap_prev, loc_ov);
    }
    if (loc_m_prev > 0) {
        atomicAdd(&n_miss_in_prev, loc_m_prev);
    }
    if (loc_m_seen > 0) {
        atomicAdd(&n_miss_seen, loc_m_seen);
    }

    __syncthreads();

    // Admit only as many misses as can fit without evicting selected hits.
    const int need = min(n_misses, state.n_slots - n_protected);
    if (need <= 0) {
        if (threadIdx.x == 0) {
            state.control[1] = 0;
            record_resolve_stats(
                state, epoch, n_protected, n_misses, 0,
                n_overlap_prev, n_miss_in_prev, n_miss_seen);
        }
        return;
    }

    const int n_evictions = clock_place_evictions(state, epoch, need);

    if (threadIdx.x == 0) {
        record_resolve_stats(
            state, epoch, n_protected, n_misses, n_evictions,
            n_overlap_prev, n_miss_in_prev, n_miss_seen);
    }
}

static __global__ void resolve_begin(cache_state state_k, cache_state state_v) {
    const cache_state & state = blockIdx.x == 0 ? state_k : state_v;
    __shared__ int epoch;

    if (threadIdx.x == 0) {
        epoch            = state.control[2] == INT_MAX ? 1 : state.control[2] + 1;
        state.control[1] = 0;
        state.control[2] = epoch;
        state.control[3] = 0;
        state.control[4] = 0;
        state.control[5] = 0;
        state.control[6] = 0;
        state.control[7] = 0;
        state.control[8] = 0;
        state.control[9] = 0;
    }

    __syncthreads();

    if (epoch == 1) {
        for (int p = threadIdx.x; p < state.n_pages; p += blockDim.x) {
            state.needed[p] = 0;
        }
    }
}

static __global__ void resolve_mark_needed(
        cache_state state_k, cache_state state_v, const int32_t * indices, int n_indices,
        size_t offset_k, size_t offset_v, size_t row_bytes_k, size_t row_bytes_v) {
    const cache_state & state = blockIdx.y == 0 ? state_k : state_v;
    const size_t offset       = blockIdx.y == 0 ? offset_k : offset_v;
    const size_t row_bytes    = blockIdx.y == 0 ? row_bytes_k : row_bytes_v;
    const int    epoch        = state.control[2];

    __shared__ int sh_prot;
    __shared__ int sh_ov;
    __shared__ int sh_m_prev;
    __shared__ int sh_m_seen;
    if (threadIdx.x == 0) {
        sh_prot   = 0;
        sh_ov     = 0;
        sh_m_prev = 0;
        sh_m_seen = 0;
    }
    __syncthreads();

    int loc_prot   = 0;
    int loc_ov     = 0;
    int loc_m_prev = 0;
    int loc_m_seen = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_indices; i += gridDim.x * blockDim.x) {
        const int row = indices ? indices[i] : i;
        if (row < 0) {
            continue;
        }
        const size_t byte = offset + size_t(row) * row_bytes;
        if (byte >= state.bytes) {
            continue;
        }

        const int page = cache_page_for_byte(state, byte);
        const int prev = atomicExch(state.needed + page, epoch);

        if (prev != epoch) {
            const bool in_prev = epoch > 1 && prev == epoch - 1;
            if (in_prev) {
                ++loc_ov;
            }
            const int slot = state.pages[page];
            if (slot >= 0) {
                state.referenced[slot] = 1;
                ++loc_prot;
            } else {
                if (in_prev) {
                    ++loc_m_prev;
                }
                if (epoch > 1 && prev > 0) {
                    ++loc_m_seen;
                }
                const int miss = atomicAdd(state.control + 4, 1);
                if (miss < state.n_slots) {
                    state.misses[miss] = page;
                }
            }
        }
    }

    if (loc_prot > 0) {
        atomicAdd(&sh_prot, loc_prot);
    }
    if (loc_ov > 0) {
        atomicAdd(&sh_ov, loc_ov);
    }
    if (loc_m_prev > 0) {
        atomicAdd(&sh_m_prev, loc_m_prev);
    }
    if (loc_m_seen > 0) {
        atomicAdd(&sh_m_seen, loc_m_seen);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (sh_prot > 0) {
            atomicAdd(state.control + 3, sh_prot);
        }
        if (sh_ov > 0) {
            atomicAdd(state.control + 6, sh_ov);
        }
        if (sh_m_prev > 0) {
            atomicAdd(state.control + 8, sh_m_prev);
        }
        if (sh_m_seen > 0) {
            atomicAdd(state.control + 9, sh_m_seen);
        }
    }
}

static __global__ void resolve_evictions(cache_state state_k, cache_state state_v) {
    const cache_state & state = blockIdx.x == 0 ? state_k : state_v;
    const int epoch          = state.control[2];
    const int n_protected    = state.control[3];
    const int n_misses       = state.control[4];
    const int n_overlap_prev = state.control[6];
    const int n_miss_in_prev = state.control[8];
    const int n_miss_seen    = state.control[9];

    const int need = min(n_misses, state.n_slots - n_protected);
    if (need <= 0) {
        if (threadIdx.x == 0) {
            state.control[1] = 0;
            record_resolve_stats(
                state, epoch, n_protected, n_misses, 0,
                n_overlap_prev, n_miss_in_prev, n_miss_seen);
        }
        return;
    }

    const int n_evictions = clock_place_evictions(state, epoch, need);
    if (threadIdx.x == 0) {
        record_resolve_stats(
            state, epoch, n_protected, n_misses, n_evictions,
            n_overlap_prev, n_miss_in_prev, n_miss_seen);
    }
}

static __global__ void copy_pages(cache_state state_k, cache_state state_v, cache_state meta_k, cache_state meta_v) {
    const cache_state & data = blockIdx.y == 0 ? state_k : state_v;
    const cache_state & meta = blockIdx.y == 0 ? meta_k  : meta_v;

    for (int i = blockIdx.x; i < meta.control[1]; i += gridDim.x) {
        const int     page   = meta.misses[i];
        const int     slot   = meta.pages[page];
        const size_t  offset = size_t(page) * data.page_bytes;
        const size_t  bytes  = min(data.page_bytes, data.bytes - offset);
        const uint4 * src    = reinterpret_cast<const uint4 *>(data.device_host + offset);
        uint4 *       dst    = reinterpret_cast<uint4 *>(data.cache + size_t(slot) * data.page_bytes);

        for (size_t j = threadIdx.x; j < bytes / sizeof(uint4); j += blockDim.x) {
            dst[j] = src[j];
        }
    }
}

template <typename T>
static __global__ void sync_rows(cache_state state, cache_state meta, const T * indices, int n, size_t offset, size_t row_bytes) {
    const int i = blockIdx.x;
    if (i >= n) {
        return;
    }

    const int64_t row = int64_t(indices[i]);
    if (row < 0) {
        return;
    }

    const size_t byte = offset + size_t(row) * row_bytes;
    if (byte >= state.bytes) {
        return;
    }

    const int page = cache_page_for_byte(state, byte);
    const int slot = meta.pages[page];
    if (slot < 0) {
        return;
    }

    const size_t in_page = byte - size_t(page) * state.page_bytes;
    const size_t nbytes  = min(row_bytes, state.bytes - byte);
    if (in_page + nbytes > state.page_bytes) {
        if (threadIdx.x == 0) {
            const int evicted = atomicExch(meta.pages + page, -1);
            if (evicted >= 0) {
                meta.owners[evicted] = -1;
            }
        }
        return;
    }

    const char * src = state.device_host + byte;
    char *       dst = state.cache + size_t(slot) * state.page_bytes + in_page;
    if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15u) == 0) {
        const size_t  n16 = nbytes / sizeof(uint4);
        const uint4 * s16 = reinterpret_cast<const uint4 *>(src);
        uint4 *       d16 = reinterpret_cast<uint4 *>(dst);
        for (size_t j = threadIdx.x; j < n16; j += blockDim.x) {
            d16[j] = s16[j];
        }
        for (size_t j = n16 * sizeof(uint4) + threadIdx.x; j < nbytes; j += blockDim.x) {
            dst[j] = src[j];
        }
    } else {
        for (size_t j = threadIdx.x; j < nbytes; j += blockDim.x) {
            dst[j] = src[j];
        }
    }
}

template <typename T>
static __global__ void invalidate_rows(cache_state state, const T * indices, int n, size_t offset, size_t row_bytes) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const size_t byte = offset + size_t(indices[i]) * row_bytes;

        if (byte < state.bytes) {
            const int slot = atomicExch(state.pages + cache_page_for_byte(state, byte), -1);

            if (slot >= 0) {
                state.owners[slot] = -1;
            }
        }
    }
}

}  // namespace

bool ggml_backend_buft_is_cuda_qsa_kv(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == type_name;
}

bool ggml_cuda_qsa_kv_is_paged(const ggml_tensor * tensor) {
    // Allocation probes run before views inherit their source buffer.
    while (tensor && !tensor->buffer && tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor && tensor->buffer && ggml_backend_buft_is_cuda_qsa_kv(ggml_backend_buffer_get_type(tensor->buffer));
}

bool ggml_cuda_qsa_kv_fits(const ggml_tensor * tensor) {
    const ggml_tensor * storage = tensor;
    size_t offset = 0;
    while (storage->view_src) {
        offset += storage->view_offs;
        storage = storage->view_src;
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(storage->buffer);
    const auto & type = *static_cast<type_context *>(buft->context);
    const size_t page_bytes = 4 * storage->nb[1];
    const size_t n_pages = (storage->ne[1] + 3) / 4;
    const size_t n_slots = std::min<size_t>(n_pages, type.resident_tokens / 4);
    const size_t first_page = offset / page_bytes;
    const size_t last_page = (offset + ggml_nbytes(tensor) - 1) / page_bytes;

    return last_page - first_page + 1 <= n_slots;
}

ggml_backend_buffer_type_t ggml_backend_cuda_qsa_kv_buffer_type(ggml_backend_dev_t dev, uint32_t resident_tokens) {
    GGML_ASSERT(resident_tokens >= 4);

    static std::mutex mutex;
    static std::map<std::pair<ggml_backend_dev_t, uint32_t>, std::unique_ptr<ggml_backend_buffer_type>> types;

    std::lock_guard<std::mutex> lock(mutex);
    auto & buft = types[{ dev, resident_tokens }];

    if (!buft) {
        auto reg    = ggml_backend_dev_backend_reg(dev);
        int  device = 0;

        while (ggml_backend_reg_dev_get(reg, device) != dev) {
            ++device;
        }

        buft                    = std::make_unique<ggml_backend_buffer_type>();
        buft->iface             = ggml_backend_cpu_buffer_type()->iface;
        buft->iface.get_name     = type_name;
        buft->iface.alloc_buffer = alloc_buffer;
        buft->device            = dev;
        buft->context =
            new type_context{ std::string(ggml_backend_dev_name(dev)) + "_QSA_KV", device, resident_tokens };
    }

    return buft.get();
}

size_t ggml_backend_cuda_qsa_kv_tensor_device_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (!ggml_backend_buft_is_cuda_qsa_kv(buft) || tensor->view_src) {
        return 0;
    }

    const auto & type = *static_cast<type_context *>(buft->context);
    const size_t n_pages = (tensor->ne[1] + 3) / 4;
    const size_t n_slots = std::min<size_t>(n_pages, type.resident_tokens / 4);
    return n_slots * 4 * tensor->nb[1] + cache_metadata_size(n_pages, n_slots);
}

bool ggml_backend_cuda_qsa_kv_q8_supported() {
#ifdef GGML_USE_HIP
    return true;
#else
    return false;
#endif
}

void * ggml_cuda_qsa_kv_device_ptr(const ggml_tensor * tensor) {
    auto & ctx = context(tensor->buffer);
    return static_cast<char *>(ctx.device_host) +
           (static_cast<const char *>(tensor->data) - static_cast<const char *>(ctx.host));
}

bool ggml_cuda_qsa_kv_get_view(const ggml_tensor * tensor, ggml_cuda_qsa_kv_view & view) {
    cache_state * state = find_cache(tensor);
    if (!state) {
        return false;
    }

    cache_state * meta = state->paired_primary ? state->paired_primary : state;
    view = { state->device_host, state->cache, meta->pages, state->page_bytes, state->page_shift };
    return true;
}

static void launch_resolve_pages(
        cudaStream_t stream,
        const cache_state & state_0, const cache_state & state_1, int n_resolve,
        const int32_t * indices, int n_indices,
        size_t offset_0, size_t offset_1, size_t row_bytes_0, size_t row_bytes_1) {
    if (n_indices <= 4096) {
        resolve_pages<<<n_resolve, resolve_threads, 0, stream>>>(
                state_0, state_1, indices, n_indices, offset_0, offset_1, row_bytes_0, row_bytes_1);
    } else {
        const int mark_blocks = std::min(64, (n_indices + 1023) / 1024);
        resolve_begin<<<n_resolve, 256, 0, stream>>>(state_0, state_1);
        resolve_mark_needed<<<dim3(mark_blocks, n_resolve), 256, 0, stream>>>(
                state_0, state_1, indices, n_indices, offset_0, offset_1, row_bytes_0, row_bytes_1);
        resolve_evictions<<<n_resolve, resolve_threads, 0, stream>>>(state_0, state_1);
    }
}

void ggml_cuda_qsa_kv_prepare(
        ggml_backend_cuda_context & ctx, const ggml_tensor * K, const ggml_tensor * V,
        const int32_t * indices, int n_indices, ggml_cuda_qsa_kv_view & K_cache, ggml_cuda_qsa_kv_view & V_cache) {
    cache_state * state_k = find_cache(K);
    cache_state * state_v = find_cache(V);
    GGML_ASSERT(state_k && state_v);

    const size_t offset_k = static_cast<const char *>(K->data) - state_k->host;
    const size_t offset_v = static_cast<const char *>(V->data) - state_v->host;
    const int copy_blocks = n_indices <= 4096 ? 32 : 96;

    // Pair distinct symmetric (K, V) tensors onto one shared page table so
    // page resolution runs once and attention resolves K/V slots with 1 load.
    const bool want_pair =
        state_k != state_v &&
        state_k->bytes == state_v->bytes &&
        state_k->page_bytes == state_v->page_bytes &&
        state_k->n_pages == state_v->n_pages &&
        state_k->n_slots == state_v->n_slots &&
        offset_k == offset_v &&
        K->nb[1] == V->nb[1];

    cache_state * stats_state = state_k;

    if (want_pair) {
        if (state_k->paired_peer != state_v) {
            unpair_state_async(state_k, ctx.stream());
            unpair_state_async(state_v, ctx.stream());
            reset_cache_async(*state_k, ctx.stream());
            reset_cache_async(*state_v, ctx.stream());
            state_k->paired_peer    = state_v;
            state_k->paired_primary = nullptr;
            state_v->paired_peer    = state_k;
            state_v->paired_primary = state_k;
        }

        cache_state * primary = state_k->paired_primary ? state_k->paired_primary : state_k;
        stats_state = primary;

        launch_resolve_pages(ctx.stream(), *primary, *primary, 1, indices, n_indices, offset_k, offset_k, K->nb[1], K->nb[1]);
        copy_pages<<<dim3(copy_blocks, 2), 256, 0, ctx.stream()>>>(*state_k, *state_v, *primary, *primary);

        K_cache = { state_k->device_host, state_k->cache, primary->pages, state_k->page_bytes, state_k->page_shift };
        V_cache = { state_v->device_host, state_v->cache, primary->pages, state_v->page_bytes, state_v->page_shift };
    } else {
        unpair_state_async(state_k, ctx.stream());
        if (state_v != state_k) {
            unpair_state_async(state_v, ctx.stream());
        }

        const int n_caches = state_k == state_v ? 1 : 2;
        launch_resolve_pages(ctx.stream(), *state_k, *state_v, n_caches, indices, n_indices, offset_k, offset_v, K->nb[1], V->nb[1]);
        copy_pages<<<dim3(copy_blocks, n_caches), 256, 0, ctx.stream()>>>(*state_k, *state_v, *state_k, *state_v);

        // Aliased views must not update the same table concurrently. A second
        // selection can evict K pages safely: unmapped reads fall back to RAM.
        if (n_caches == 1 && (offset_k != offset_v || K->nb[1] != V->nb[1])) {
            launch_resolve_pages(ctx.stream(), *state_v, *state_v, 1, indices, n_indices, offset_v, offset_v, V->nb[1], V->nb[1]);
            copy_pages<<<copy_blocks, 256, 0, ctx.stream()>>>(*state_v, *state_v, *state_v, *state_v);
        }

        K_cache = { state_k->device_host, state_k->cache, state_k->pages, state_k->page_bytes, state_k->page_shift };
        V_cache = { state_v->device_host, state_v->cache, state_v->pages, state_v->page_bytes, state_v->page_shift };
    }

    CUDA_CHECK(cudaGetLastError());

    if (qsa_kv_stats_level() >= 2) {
#ifdef GGML_USE_HIP
        hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
        const bool can_sync = hipStreamIsCapturing(ctx.stream(), &cap_status) == hipSuccess &&
                              cap_status == hipStreamCaptureStatusNone;
#else
        cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
        const bool can_sync = cudaStreamIsCapturing(ctx.stream(), &cap_status) == cudaSuccess &&
                              cap_status == cudaStreamCaptureStatusNone;
#endif
        if (can_sync) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
            int ctrl[control_ints] = {};
            CUDA_CHECK(cudaMemcpy(ctrl, stats_state->control, control_ints * sizeof(int), cudaMemcpyDeviceToHost));
            fprintf(stderr, "qsa_step K=%p epoch=%d n_idx=%d | hit=%d miss=%d placed=%d evict=%d uniq=%d coh=%.1f%% (%d/%d) miss_prev=%d miss_seen=%d\n",
                (const void *) stats_state, ctrl[2], n_indices, ctrl[3], ctrl[4], ctrl[1], ctrl[5], ctrl[7],
                ctrl[7] > 0 ? 100.0 * double(ctrl[6]) / double(ctrl[7]) : 0.0, ctrl[6], ctrl[7], ctrl[8], ctrl[9]);
        }
    }
}

void ggml_cuda_qsa_kv_sync_rows(
        ggml_backend_cuda_context & ctx, const ggml_tensor * tensor, const ggml_tensor * indices) {
    cache_state * state = find_cache(tensor);

    if (!state) {
        return;
    }

    cache_state * meta = state->paired_primary ? state->paired_primary : state;
    GGML_ASSERT(ggml_is_contiguous(indices));
    const int    n      = ggml_nelements(indices);
    const size_t offset = static_cast<const char *>(tensor->data) - state->host;

    if (n <= 0) {
        return;
    }

    // Preserve mapped pages on decode/small writes by syncing written rows
    // directly into the resident slot; bulk prefill writes invalidate instead.
    if (n <= 64) {
        if (indices->type == GGML_TYPE_I64) {
            sync_rows<<<n, 256, 0, ctx.stream()>>>(
                    *state, *meta, static_cast<const int64_t *>(indices->data), n, offset, tensor->nb[1]);
        } else {
            GGML_ASSERT(indices->type == GGML_TYPE_I32);
            sync_rows<<<n, 256, 0, ctx.stream()>>>(
                    *state, *meta, static_cast<const int32_t *>(indices->data), n, offset, tensor->nb[1]);
        }
    } else if (indices->type == GGML_TYPE_I64) {
        invalidate_rows<<<(n + 255) / 256, 256, 0, ctx.stream()>>>(
                *meta, static_cast<const int64_t *>(indices->data), n, offset, tensor->nb[1]);
    } else {
        GGML_ASSERT(indices->type == GGML_TYPE_I32);
        invalidate_rows<<<(n + 255) / 256, 256, 0, ctx.stream()>>>(
                *meta, static_cast<const int32_t *>(indices->data), n, offset, tensor->nb[1]);
    }

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_qsa_kv_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor) {
    cache_state * state = find_cache(tensor);

    if (!state) {
        return;
    }

    cache_state * meta = state->paired_primary ? state->paired_primary : state;

    // Bulk device writes (copies and K-shifts) bypass SET_ROWS. Invalidate on
    // the same stream so subsequent attention cannot reuse pre-write pages.
    CUDA_CHECK(cudaMemsetAsync(meta->pages, 0xff, (meta->n_pages + meta->n_slots)*sizeof(int), ctx.stream()));
    CUDA_CHECK(cudaMemsetAsync(meta->referenced, 0, meta->n_slots*sizeof(int), ctx.stream()));
}
