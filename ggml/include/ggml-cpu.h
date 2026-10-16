#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    struct ggml_backend_cpu_expert_cache;
    typedef struct ggml_backend_cpu_expert_cache * ggml_backend_cpu_expert_cache_t;

    typedef ggml_backend_cpu_expert_cache_t (*ggml_backend_cpu_expert_cache_create_t)(void);
    typedef void (*ggml_backend_cpu_expert_cache_free_t)(ggml_backend_cpu_expert_cache_t cache);
    typedef bool (*ggml_backend_cpu_expert_cache_add_device_t)(
        ggml_backend_cpu_expert_cache_t cache, uint32_t slots, uint32_t admit_window, uint32_t convert_workers,
        ggml_backend_dev_t device);
    typedef void (*ggml_backend_cpu_expert_cache_register_t)(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_dev_t device, const struct ggml_tensor * weights);
    typedef void (*ggml_backend_cpu_set_expert_cache_t)(ggml_backend_t backend_cpu, ggml_backend_cpu_expert_cache_t cache);
    typedef const uint64_t * (*ggml_backend_cpu_expert_cache_begin_t)(
        ggml_backend_cpu_expert_cache_t cache, struct ggml_tensor * op);
    typedef void (*ggml_backend_cpu_expert_cache_end_t)(ggml_backend_cpu_expert_cache_t cache, struct ggml_tensor * op);
    typedef ggml_backend_dev_t (*ggml_backend_cpu_expert_cache_device_t)(
        ggml_backend_cpu_expert_cache_t cache, const struct ggml_tensor * op);
    typedef void (*ggml_backend_cpu_expert_cache_prepare_t)(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    typedef bool (*ggml_backend_cpu_expert_cache_copy_t)(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst);
    typedef size_t (*ggml_backend_cpu_expert_cache_memory_t)(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_dev_t device);
    typedef uint64_t (*ggml_backend_cpu_expert_cache_hits_t)(ggml_backend_cpu_expert_cache_t cache);
    typedef size_t (*ggml_backend_cpu_expert_cache_get_buffer_size_t)(
        ggml_backend_dev_t device, int64_t input_dim, int64_t ffn_dim, int64_t output_dim, int64_t n_expert_used);

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // use only reference implementations
        bool use_ref;

        // non-owning expert cache, shared by computations in this plan
        ggml_backend_cpu_expert_cache_t expert_cache;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
    GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
    GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
    GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

    // ggml_graph_plan() has to be called before ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
                  const struct ggml_cgraph * cgraph,
                                       int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                    struct ggml_threadpool * threadpool /* = NULL */ );
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

    // same as ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    GGML_BACKEND_API int ggml_cpu_has_sme2       (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);  // risc-v vector length in bytes
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);

    struct ggml_type_traits_cpu {
        ggml_from_float_t        from_float;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);

    GGML_BACKEND_API void ggml_cpu_init(void);

    //
    // CPU backend
    //

    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);

    GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
    GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
    GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
    GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

    GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);

    // The caller owns the cache and must keep it alive while attached to a backend or plan.
    GGML_BACKEND_API ggml_backend_cpu_expert_cache_t ggml_backend_cpu_expert_cache_create(void);
    GGML_BACKEND_API void ggml_backend_cpu_expert_cache_free(ggml_backend_cpu_expert_cache_t cache);
    GGML_BACKEND_API bool ggml_backend_cpu_expert_cache_add_device(
        ggml_backend_cpu_expert_cache_t cache, uint32_t slots, uint32_t admit_window, uint32_t convert_workers,
        ggml_backend_dev_t device);
    // Register borrowed host weights with their model-assigned device before computation.
    // GPU allocations are deferred until an op supplies the actual execution dimensions.
    GGML_BACKEND_API void ggml_backend_cpu_expert_cache_register(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_dev_t device, const struct ggml_tensor * weights);
    GGML_BACKEND_API void ggml_backend_cpu_set_expert_cache(ggml_backend_t backend_cpu, ggml_backend_cpu_expert_cache_t cache);

    // begin() freezes route-position masks, valid until end(), and starts ready GPU hits.
    // CPU consumers retain the host join; NULL masks mean CPU-only work for this op.
    GGML_BACKEND_API const uint64_t * ggml_backend_cpu_expert_cache_begin(
        ggml_backend_cpu_expert_cache_t cache, struct ggml_tensor * op);
    GGML_BACKEND_API void ggml_backend_cpu_expert_cache_end(
        ggml_backend_cpu_expert_cache_t cache, struct ggml_tensor * op);
    // Registered owner, including CPU-only batches.
    GGML_BACKEND_API ggml_backend_dev_t ggml_backend_cpu_expert_cache_device(
        ggml_backend_cpu_expert_cache_t cache, const struct ggml_tensor * op);
    // Cache offload eligibility for this operation and batch.
    GGML_BACKEND_API ggml_backend_dev_t ggml_backend_cpu_expert_cache_offload_device(
        ggml_backend_cpu_expert_cache_t cache, const struct ggml_tensor * op);
    GGML_BACKEND_API void ggml_backend_cpu_expert_cache_prepare(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_BACKEND_API bool ggml_backend_cpu_expert_cache_copy(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_t backend, const struct ggml_tensor * src, struct ggml_tensor * dst);
    // Current GPU allocations, including cache, workspace, merge and staging buffers.
    GGML_BACKEND_API size_t ggml_backend_cpu_expert_cache_memory(
        ggml_backend_cpu_expert_cache_t cache, ggml_backend_dev_t device);
    // Cumulative ready routes successfully submitted to a GPU across all cache devices.
    GGML_BACKEND_API uint64_t ggml_backend_cpu_expert_cache_hits(ggml_backend_cpu_expert_cache_t cache);
    GGML_BACKEND_API size_t ggml_backend_cpu_expert_cache_get_buffer_size(
        ggml_backend_dev_t device, int64_t input_dim, int64_t ffn_dim, int64_t output_dim, int64_t n_expert_used);

    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

    GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);
    GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t);
    GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t);

#ifdef __cplusplus
}
#endif
