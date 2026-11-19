#pragma once
#include "ggml-backend-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
#    include <vector>
extern "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);

// Single-expert repack/GEMV helpers for NVMe-backed L2 RAM cache
bool ggml_cpu_nvme_can_repack(const struct ggml_tensor * t);
int  ggml_cpu_nvme_repack_col_align(const struct ggml_tensor * t);
bool ggml_cpu_nvme_repack_expert(const struct ggml_tensor * t, const void * native_src, void * repacked_dst, size_t expert_bytes);
bool ggml_cpu_nvme_repack_gemv(const struct ggml_tensor * src0, const void * repacked_expert, bool is_repacked, bool repack_in_place,
        const void * src1_q, enum ggml_type src1_q_type, float * dst_row, int64_t col_start, int64_t col_end);
bool ggml_cpu_nvme_repack_gemm_rows(const struct ggml_tensor * src0, const void * expert_data, bool is_repacked, bool repack_in_place,
        const void * src1_q_base, size_t nb11_q, size_t nb12_q, int64_t ne11,
        enum ggml_type src1_q_type, const int32_t * row_map, int64_t cne1,
        float * dst_data, size_t dst_nb1, size_t dst_nb2, int64_t col_start, int64_t col_end);
bool ggml_cpu_nvme_unpack_znq_expert(const struct ggml_tensor * tensor, const void * repacked_src, void * dst, size_t dst_size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  public:
    virtual ~tensor_traits();
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
    virtual int  repack_col_align() const { return 0; }
    virtual int  repack_one(struct ggml_tensor * t, const void * data, size_t data_size) const {
        (void) t; (void) data; (void) data_size;
        return -1;
    }
    virtual bool gemv_one(const struct ggml_tensor * src0, const void * expert_data, bool is_repacked, bool repack_in_place,
                          const void * src1_q, enum ggml_type src1_q_type,
                          float * dst_row, int64_t col_start, int64_t col_end) const {
        (void) src0; (void) expert_data; (void) is_repacked; (void) repack_in_place; (void) src1_q; (void) src1_q_type;
        (void) dst_row; (void) col_start; (void) col_end;
        return false;
    }
    virtual bool gemm_rows(const struct ggml_tensor * src0, const void * expert_data, bool is_repacked, bool repack_in_place,
                           const void * src1_q_base, size_t nb11_q, size_t nb12_q, int64_t ne11,
                           enum ggml_type src1_q_type, const int32_t * row_map, int64_t cne1,
                           float * dst_data, size_t dst_nb1, size_t dst_nb2,
                           int64_t col_start, int64_t col_end) const {
        (void) src0; (void) expert_data; (void) is_repacked; (void) repack_in_place;
        (void) src1_q_base; (void) nb11_q; (void) nb12_q; (void) ne11;
        (void) src1_q_type; (void) row_map; (void) cne1;
        (void) dst_data; (void) dst_nb1; (void) dst_nb2;
        (void) col_start; (void) col_end;
        return false;
    }
};

class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
