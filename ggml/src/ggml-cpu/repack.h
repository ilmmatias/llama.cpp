#pragma once

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "traits.h"
#include "ggml.h"

// GGML internal header

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

template <int K> constexpr int QK_0() {
    if constexpr (K == 1) {
        return QK1_0;
    }
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<1, 4>) == 4 * sizeof(ggml_half) + QK1_0 / 2, "wrong block<1,4> size/padding");
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<4, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<4,16> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");
static_assert(sizeof(block<8, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 16, "wrong block<8,16> size/padding");

using block_q1_0x4 = block<1, 4>;
using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q4_0x16 = block<4, 16>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
using block_q8_0x16 = block<8, 16>;

struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

static_assert(sizeof(block_q4_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4, "wrong q4_K block size/padding");
struct block_q4_Kx16 {
    ggml_half d[16];      // super-block scale for quantized scales
    ggml_half dmin[16];   // super-block scale for quantized mins
    uint8_t scales[192];  // scales and mins, quantized with 6 bits
    uint8_t qs[2048];    // 4--bit quants
};

static_assert(sizeof(block_q4_Kx16) == sizeof(ggml_half) * 32 + K_SCALE_SIZE * 16 + QK_K * 8, "wrong q4_K block size/padding");
struct block_q2_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[128];  // scales and mins, quantized with 4 bits
    uint8_t qs[512];    // 2--bit quants
};

static_assert(sizeof(block_q2_Kx8) == sizeof(ggml_half) * 16 + QK_K/2 + QK_K * 2, "wrong q2_K block size/padding");
struct block_q2_Kx16 {
    ggml_half d[16];       // Super-block scale for quantized scales
    ggml_half dmin[16];    // Super-block scale for quantized mins
    uint8_t   scales[256]; // Sub-block scales (16 cols * 16 sub-blocks)
    uint8_t   qs[1024];    // Data (16 cols * 64 bytes per block)
};
static_assert(sizeof(block_q2_Kx16) == sizeof(ggml_half) * 32 + QK_K + QK_K * 4, "wrong q2_K block size/padding");

struct block_q5_Kx8 {
    ggml_half d[8];              // super-block scale for quantized scales
    ggml_half dmin[8];           // super-block scale for quantized mins
    uint8_t   scales[96];        // scales and mins, quantized with 6 bits
    uint8_t   qh[QK_K * 8 / 8];  // high bits of 5-bit quants
    uint8_t   qs[QK_K * 8 / 2];  // low bits of 5-bit quants (in groups of 4)
};

static_assert(sizeof(block_q5_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 5,
              "wrong q5_K block size/padding");

struct block_q6_Kx8 {
    ggml_half d[8];
    int8_t    scales[QK_K / 16 * 8];
    uint8_t   ql[QK_K / 2 * 8];  // low bits of 6-bit quants (groups of 2)
    uint8_t   qh[QK_K / 4 * 8];  // high bits of 6-bit quants (groups of 4)
};

static_assert(sizeof(block_q6_Kx8) == sizeof(ggml_half) * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8,
              "wrong q6_K block size/padding");

struct block_q8_Kx4 {
    float d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

static_assert(sizeof(block_q8_Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t), "wrong q8_K block size/padding");

struct block_iq4_nlx4 {
    ggml_half d[4];            // deltas for 4 iq4_nl blocks
    uint8_t   qs[QK4_NL * 2];  // nibbles / quants for 4 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx4) == 4 * sizeof(ggml_half) + QK4_NL * 2, "wrong iq4_nlx4 block size/padding");

struct block_iq4_nlx8 {
    ggml_half d[8];            // deltas for 8 iq4_nl blocks
    uint8_t   qs[QK4_NL * 4];  // nibbles / quants for 8 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx8) == 8 * sizeof(ggml_half) + QK4_NL * 4, "wrong iq4_nlx8 block size/padding");

struct block_znq2x8 {
    uint8_t d[8];
    uint8_t books[8];
    // Packed 2-bit codes interleaved by 4-weight group across eight rows.
    uint8_t qs[QK_ZNQ / 4][8];
};

static_assert(sizeof(block_znq2x8) == 8 * sizeof(block_znq2), "wrong znq2x8 block size/padding");

struct block_znq3x8 {
    uint8_t d[8];
    uint8_t books[8];
    // Three bitplanes per 4-weight group, interleaved across eight rows.
    uint32_t planes[QK_ZNQ / 4][3];
};

static_assert(sizeof(block_znq3x8) == 8 * sizeof(block_znq3), "wrong znq3x8 block size/padding");

struct block_znq4x8 {
    uint8_t d[8];
    uint8_t books[8];
    uint8_t qs[QK_ZNQ * 4];  // 8 rows x 16 packed-nibble bytes, interleaved in 4-weight groups
};

static_assert(sizeof(block_znq4x8) == 8 * sizeof(block_znq4), "wrong znq4x8 block size/padding");

struct block_iq4_nlx16 {
    ggml_half d[16];            // deltas for 16 iq4_nl blocks
    uint8_t   qs[QK4_NL * 8];  // nibbles / quants for 16 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx16) == 16 * sizeof(ggml_half) + QK4_NL * 8, "wrong iq4_nlx16 block size/padding");
struct block_mxfp4x4 {
    uint8_t e[4];
    uint8_t qs[QK_MXFP4 * 2];
};
static_assert(sizeof(block_mxfp4x4) == 4 + QK_MXFP4 * 2, "wrong mxfp4x4 block size/padding");

struct block_mxfp4x8 {
    uint8_t e[8];
    uint8_t qs[QK_MXFP4 * 4];
};
static_assert(sizeof(block_mxfp4x8) == 8 + QK_MXFP4 * 4, "wrong mxfp4x8 block size/padding");

// Q8 panel for grid IQ types. Eight rows are decoded to signed 8-bit values
// with integer sub-block scales. Sub-blocks are 16 elements wide.
//
// qs[sb * 128 + g * 32 + row * 4 + k] stores column
// sb * 16 + g * 4 + k. bias[] removes the unsigned-activation correction.
#define IQP_SB_SIZE 16                    // weights per sub-block
#define IQP_NSB     (QK_K / IQP_SB_SIZE)  // sub-blocks per super-block

struct block_iqp_x8 {
    float   dfac[8];               // dfac[row]          : f32 super-block scale, d * 2^-k
    int32_t bias[8];               // bias[row]          : unsigned-activation correction, see above
    int8_t  iscales[IQP_NSB * 8];  // iscales[sb*8 + row]: integer scale of sub-block sb, in [-32, 31]
    int8_t  qs[QK_K * 8];          // decoded signed values, interleaved (see above)
};

static_assert(sizeof(block_iqp_x8) == 8 * sizeof(float) + 8 * sizeof(int32_t) + IQP_NSB * 8 + QK_K * 8,
              "wrong iqp_x8 block size/padding");

// VNNI uses unsigned activations and subtracts bias[]. Other paths use signed
// activations and leave bias[] unused.
#if defined(__AVX2__) && ((defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__))
#define GGML_IQP_USE_BIAS 1
#else
#define GGML_IQP_USE_BIAS 0
#endif

// Keep the decoder and x86 kernels on the same bias convention.
#if GGML_IQP_USE_BIAS
#    define ggml_gemv_iqp_8x8_q8_K ggml_gemv_iqp_8x8_q8_K_bias
#    define ggml_gemm_iqp_8x8_q8_K ggml_gemm_iqp_8x8_q8_K_bias
#    define ggml_gemm_iqp_8x8_q8_K_p4 ggml_gemm_iqp_8x8_q8_K_p4_bias
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q1_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q1_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_znq2_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_znq2_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_znq3_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_znq3_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
// Quantize four non-contiguous routed activation rows directly into the 4x8
// Q8_0 interleave consumed by the ZNQ GEMM kernels.
void ggml_quantize_mat_znq4_q8_0_4x8(const float * x0, const float * x1,
        const float * x2, const float * x3, void * vy, int64_t k);
// Direct-scatter MoE GEMM entry points. row_map is a packed {route, token}
// int32 pair per activation row; destination strides are in float elements.
void ggml_gemm_znq2_8x8_q8_0_moe(
        int n, float * GGML_RESTRICT s, size_t dst_bs1, size_t dst_bs2, const int32_t * row_map,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_znq3_8x8_q8_0_moe(
        int n, float * GGML_RESTRICT s, size_t dst_bs1, size_t dst_bs2, const int32_t * row_map,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_znq4_8x8_q8_0_moe(
        int n, float * GGML_RESTRICT s, size_t dst_bs1, size_t dst_bs2, const int32_t * row_map,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#endif
void ggml_gemv_znq4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q1_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q1_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_znq4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iqp_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iqp_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
// same as ggml_gemm_iqp_8x8_q8_K with nr = 4, except the four activation rows are passed as four independent pointers instead of one contiguous base - for MUL_MAT_ID, where the routed rows are scattered over the q8_K conversion area
void ggml_gemm_iqp_8x8_q8_K_p4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * const * GGML_RESTRICT vy, int nc);
#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#endif

// Native implementations
void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q1_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q1_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q1_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q1_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iqp_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iqp_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iqp_8x8_q8_K_p4_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * const * GGML_RESTRICT vy, int nc);
#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#endif

#if defined(__cplusplus)
} // extern "C"
#endif
