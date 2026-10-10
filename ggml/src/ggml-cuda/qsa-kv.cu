#include "ggml-backend-impl.h"
#include "qsa-kv.cuh"

#include <climits>
#include <map>

namespace {

constexpr int resolve_threads = 1024;

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
    int * control; // CLOCK hand, miss count, selection epoch

    size_t bytes;
    size_t page_bytes;
    uint32_t page_shift = 0;
    int n_pages;
    int n_slots;
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
    return (2*n_pages + 3*n_slots + 3)*sizeof(int);
}

static __device__ __forceinline__ int cache_page_for_byte(const cache_state & state, size_t byte) {
    return state.page_shift ? int(byte >> state.page_shift) : int(byte / state.page_bytes);
}

static void reset_cache(cache_state & state) {
    CUDA_CHECK(cudaMemset(state.pages, 0xff, (state.n_pages + state.n_slots) * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.referenced, 0, state.n_slots * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.control, 0, 3*sizeof(int)));
}

static void before_host_write(ggml_backend_buffer_t buffer, const ggml_tensor * tensor = nullptr) {
    auto & ctx = context(buffer);
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (tensor) {
        cache_state * state = find_cache(tensor);
        GGML_ASSERT(state);
        reset_cache(*state);
    } else {
        for (auto & state : ctx.tensors) {
            reset_cache(*state);
        }
    }

    // Backend compute streams are non-blocking and need not wait for these
    // default-stream memsets. Finish invalidation before returning to callers.
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void free_buffer(ggml_backend_buffer_t buffer) {
    auto & ctx = context(buffer);
    ggml_cuda_set_device(ctx.device);

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
    __shared__ int cut;
    __shared__ int warp_sums[resolve_threads / WARP_SIZE];
    __shared__ int epoch;

    if (threadIdx.x == 0) {
        n_protected = 0;
        n_misses    = 0;

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

        if (atomicExch(state.needed + page, epoch) != epoch) {
            const int slot = state.pages[page];

            if (slot >= 0) {
                state.referenced[slot] = 1;
                atomicAdd(&n_protected, 1);
            } else {
                const int miss = atomicAdd(&n_misses, 1);

                if (miss < state.n_slots) {
                    state.misses[miss] = page;
                }
            }
        }
    }

    __syncthreads();

    // Admit only as many misses as can fit without evicting selected hits.
    const int need       = min(n_misses, state.n_slots - n_protected);
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
}

static __global__ void copy_pages(cache_state state_k, cache_state state_v) {
    const cache_state & state = blockIdx.y == 0 ? state_k : state_v;

    for (int i = blockIdx.x; i < state.control[1]; i += gridDim.x) {
        const int     page   = state.misses[i];
        const int     slot   = state.pages[page];
        const size_t  offset = size_t(page) * state.page_bytes;
        const size_t  bytes  = min(state.page_bytes, state.bytes - offset);
        const uint4 * src    = reinterpret_cast<const uint4 *>(state.device_host + offset);
        uint4 *       dst    = reinterpret_cast<uint4 *>(state.cache + size_t(slot) * state.page_bytes);

        for (size_t j = threadIdx.x; j < bytes / sizeof(uint4); j += blockDim.x) {
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

void ggml_cuda_qsa_kv_prepare(
        ggml_backend_cuda_context & ctx, const ggml_tensor * K, const ggml_tensor * V,
        const int32_t * indices, int n_indices, ggml_cuda_qsa_kv_view & K_cache, ggml_cuda_qsa_kv_view & V_cache) {
    cache_state * state_k = find_cache(K);
    cache_state * state_v = find_cache(V);
    GGML_ASSERT(state_k && state_v);

    const size_t offset_k = static_cast<const char *>(K->data) - state_k->host;
    const size_t offset_v = static_cast<const char *>(V->data) - state_v->host;
    const int n_caches = state_k == state_v ? 1 : 2;

    // K and V have independent page tables. Resolve and fill them in parallel
    // rather than serializing two otherwise identical launch sequences.
    resolve_pages<<<n_caches, resolve_threads, 0, ctx.stream()>>>(
            *state_k, *state_v, indices, n_indices, offset_k, offset_v, K->nb[1], V->nb[1]);
    copy_pages<<<dim3(96, n_caches), 128, 0, ctx.stream()>>>(*state_k, *state_v);

    // Aliased views must not update the same table concurrently. A second
    // selection can evict K pages safely: unmapped reads fall back to RAM.
    if (n_caches == 1 && (offset_k != offset_v || K->nb[1] != V->nb[1])) {
        resolve_pages<<<1, resolve_threads, 0, ctx.stream()>>>(
                *state_v, *state_v, indices, n_indices, offset_v, offset_v, V->nb[1], V->nb[1]);
        copy_pages<<<96, 128, 0, ctx.stream()>>>(*state_v, *state_v);
    }

    CUDA_CHECK(cudaGetLastError());

    K_cache = { state_k->device_host, state_k->cache, state_k->pages, state_k->page_bytes, state_k->page_shift };
    V_cache = { state_v->device_host, state_v->cache, state_v->pages, state_v->page_bytes, state_v->page_shift };
}

void ggml_cuda_qsa_kv_invalidate_rows(
        ggml_backend_cuda_context & ctx, const ggml_tensor * tensor, const ggml_tensor * indices) {
    cache_state * state = find_cache(tensor);

    if (!state) {
        return;
    }

    GGML_ASSERT(ggml_is_contiguous(indices));
    const int    n      = ggml_nelements(indices);
    const size_t offset = static_cast<const char *>(tensor->data) - state->host;

    if (indices->type == GGML_TYPE_I64) {
        invalidate_rows<<<(n + 255) / 256, 256, 0, ctx.stream()>>>(
                *state, static_cast<const int64_t *>(indices->data), n, offset, tensor->nb[1]);
    } else {
        GGML_ASSERT(indices->type == GGML_TYPE_I32);
        invalidate_rows<<<(n + 255) / 256, 256, 0, ctx.stream()>>>(
                *state, static_cast<const int32_t *>(indices->data), n, offset, tensor->nb[1]);
    }

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_qsa_kv_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor) {
    cache_state * state = find_cache(tensor);

    if (!state) {
        return;
    }

    // Bulk device writes (copies and K-shifts) bypass SET_ROWS. Invalidate on
    // the same stream so subsequent attention cannot reuse pre-write pages.
    CUDA_CHECK(cudaMemsetAsync(state->pages, 0xff, (state->n_pages + state->n_slots)*sizeof(int), ctx.stream()));
    CUDA_CHECK(cudaMemsetAsync(state->referenced, 0, state->n_slots*sizeof(int), ctx.stream()));
}
