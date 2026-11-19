#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "qsa-kv.cuh"

void * ggml_cuda_qsa_host_device_ptr(const ggml_tensor * tensor);

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <int mask>
static __device__ __forceinline__ float set_rows_dpp_xor(float x) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x160 | mask, 0xf, 0xf, true));
#else
    return __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, x, mask, 32);
#endif
}

// Coalesced Wave32 Q8_0 set_rows: 1 warp quantizes 4 consecutive blocks (128 floats = 136 B)
// via float4 loads, 8-lane register max, and aligned 32-bit stores across PCIe/VRAM.
template <int WARPS, typename idx_t>
static __global__ void k_set_rows_q8_0_wave32(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        char * __restrict__ dst,
        const int64_t n_chunks,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const uint3 ne00_chunks,
        const uint3 ne01,
        const uint3 ne02,
        const uint3 ne11_fd,
        const uint3 ne12_fd) {
    __shared__ uint32_t smem[WARPS][36];

    const int warp = WARPS == 1 ? 0 : (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int64_t chunk_idx = int64_t(blockIdx.x) * WARPS + warp;
    if (chunk_idx >= n_chunks) {
        return;
    }

    uint32_t tmp = (uint32_t) chunk_idx;
    uint2 div_mod = fast_div_modulo(tmp, ne00_chunks);
    const int64_t i_chunk = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10 * s10 + i11 * s11 + i12 * s12);

    const float * src_p = src0 + i01 * s01 + i02 * s02 + i03 * s03 + i_chunk * 128;
    const float4 v = reinterpret_cast<const float4 *>(src_p)[lane];

    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, set_rows_dpp_xor<4>(amax));
    amax = fmaxf(amax, set_rows_dpp_xor<2>(amax));
    amax = fmaxf(amax, set_rows_dpp_xor<1>(amax));

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    const uint32_t q0 = (uint8_t) (int8_t) roundf(v.x * id);
    const uint32_t q1 = (uint8_t) (int8_t) roundf(v.y * id);
    const uint32_t q2 = (uint8_t) (int8_t) roundf(v.z * id);
    const uint32_t q3 = (uint8_t) (int8_t) roundf(v.w * id);
    const uint32_t q4 = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);

    const int grp = lane >> 3;
    const int sub = lane & 7;
    uint16_t * smem_u16 = reinterpret_cast<uint16_t *>(&smem[warp][0]);
    if (sub == 0) {
        const half dh = __float2half_rn(d);
        smem_u16[grp * 17] = *reinterpret_cast<const uint16_t *>(&dh);
    }
    const int q_off = grp * 17 + 1 + sub * 2;
    smem_u16[q_off + 0] = (uint16_t) (q4 & 0xffffu);
    smem_u16[q_off + 1] = (uint16_t) (q4 >> 16);

    __syncwarp();

    uint32_t * dst_u32 = reinterpret_cast<uint32_t *>(
        dst + dst_row * s1 + i02 * s2 + i03 * s3 + size_t(i_chunk) * 136);
    dst_u32[lane] = smem[warp][lane];
    if (lane < 2) {
        dst_u32[32 + lane] = smem[warp][32 + lane];
    }
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        if constexpr (std::is_same_v<block_type, block_q8_0>) {
            const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
            const bool aligned_src = ((reinterpret_cast<uintptr_t>(src0_d) & 15) == 0) &&
                ((nb01 & 15) == 0) && ((nb02 & 15) == 0) && ((nb03 & 15) == 0);
            const bool aligned_dst = ((reinterpret_cast<uintptr_t>(dst_d) & 3) == 0) &&
                ((nb1 & 3) == 0) && ((nb2 & 3) == 0) && ((nb3 & 3) == 0);
            if (warp_size == 32 && (ne00 & 127) == 0 && aligned_src && aligned_dst) {
                const int64_t chunks_per_row = ne00 / 128;
                const int64_t n_chunks = chunks_per_row * ne01 * ne02 * ne03;
                const uint3 ne00_chunks_fd = init_fastdiv_values((uint32_t) chunks_per_row);
                if (n_chunks <= 8) {
                    k_set_rows_q8_0_wave32<1, idx_t><<<dim3(n_chunks), dim3(32), 0, stream>>>(
                        src0_d, src1_d, reinterpret_cast<char *>(dst_d), n_chunks,
                        s01, s02, s03, s10, s11, s12, s1, s2, s3,
                        ne00_chunks_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                } else {
                    k_set_rows_q8_0_wave32<2, idx_t><<<dim3((n_chunks + 1) / 2), dim3(64), 0, stream>>>(
                        src0_d, src1_d, reinterpret_cast<char *>(dst_d), n_chunks,
                        s01, s02, s03, s10, s11, s12, s1, s2, s3,
                        ne00_chunks_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
                }
                GGML_UNUSED(ne10);
                GGML_UNUSED(ne13);
                return;
            }
        }

        const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
        const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
        const dim3 grid_size(num_blocks);
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    if constexpr (std::is_same_v<dst_t, src_t>) {
        dst_row_ptr[i00] = src0_row[i00];
    } else {
        dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F8_E4M3) {
        set_rows_cuda(
            src0_d, src1_d, (ggml_fp8_e4m3_t *) dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    // SET_ROWS returns a view of the destination cache tensor.  Keep the tensor
    // metadata intact but hand the kernel the GPU-visible alias of mapped host
    // storage when this is the QSA raw-indexer cache.
    ggml_tensor dst_mapped = *dst;
    dst_mapped.data = ggml_cuda_qsa_host_device_ptr(dst);

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, &dst_mapped);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, &dst_mapped);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, &dst_mapped);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, &dst_mapped);
        }
    } else if (src0->type == GGML_TYPE_I32) {
        GGML_TENSOR_BINARY_OP_LOCALS
        cudaStream_t stream = ctx.stream();
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda((const int32_t *) src0->data, (const int64_t *) src1->data, (int32_t *) dst_mapped.data,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,
                nb01, nb02, nb03, nb10, nb11, nb12, dst_mapped.nb[1], dst_mapped.nb[2], dst_mapped.nb[3], stream);
        } else {
            set_rows_cuda((const int32_t *) src0->data, (const int32_t *) src1->data, (int32_t *) dst_mapped.data,
                ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13,
                nb01, nb02, nb03, nb10, nb11, nb12, dst_mapped.nb[1], dst_mapped.nb[2], dst_mapped.nb[3], stream);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }

    ggml_cuda_qsa_kv_sync_rows(ctx, dst, src1);
}
