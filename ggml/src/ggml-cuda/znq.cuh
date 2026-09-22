#pragma once

#include "common.cuh"

static __device__ __forceinline__ float znq_scale_cuda(uint8_t code) {
    const uint32_t e = code >> 3;
    const uint32_t m = code & 7;
    return e == 0 ? float(m) * (1.0f / 131072.0f) : __uint_as_float(((e + 112u) << 23) | (m << 20));
}

template <int bits>
static __device__ __forceinline__ int znq_value_cuda(const uint8_t * block, int j) {
    const int offset = bits*j;
    const int shift = offset % 8;
    unsigned code = block[2 + offset/8] >> shift;
    if constexpr (bits == 3) {
        if (shift > 5) {
            code |= unsigned(block[3 + offset/8]) << (8 - shift);
        }
    }
    code &= (1u << bits) - 1;
    const int book = (block[1] >> (4*(j/16))) & 15;
    if constexpr (bits == 2) {
        return kvalues_znq2[book*4 + code];
    } else if constexpr (bits == 3) {
        return kvalues_znq3[book*8 + code];
    } else {
        return kvalues_znq4[book*16 + code];
    }
}

template <int bits, bool reuse_book = false>
static __device__ __forceinline__ int znq_pack4_cuda(const uint8_t * block, int j) {
    uint32_t codes;
    if constexpr (bits == 2) {
        // Callers pass four-aligned indices, so one quad fits in one payload byte.
        codes = block[2 + j/4];
    } else if constexpr (bits == 3) {
        const int offset = 3*j;
        codes = (uint32_t(block[2 + offset/8]) | (uint32_t(block[3 + offset/8]) << 8)) >> (offset%8);
    } else {
        codes = uint32_t(block[2 + j/2]) | (uint32_t(block[3 + j/2]) << 8);
    }
    const int book = (block[1] >> (4*(j/16))) & 15;

#if defined(GGML_USE_HIP)
    // Larger books pay off when one lane decodes several quads from the same book.
    if constexpr (bits == 2 || reuse_book) {
        uint32_t indices = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            indices |= ((codes >> (bits*k)) & ((1u << bits) - 1)) << (8*k);
        }
        if constexpr (bits == 2) {
            const uint32_t * table = (const uint32_t *) (kvalues_znq2 + 4*book);
            return __builtin_amdgcn_perm(0, table[0], indices);
        } else if constexpr (bits == 3) {
            const uint32_t * table = (const uint32_t *) (kvalues_znq3 + 8*book);
            return __builtin_amdgcn_perm(table[1], table[0], indices);
        } else {
            const uint32_t * table = (const uint32_t *) (kvalues_znq4 + 16*book);
            const uint32_t lo = __builtin_amdgcn_perm(table[1], table[0], indices & 0x07070707);
            const uint32_t hi = __builtin_amdgcn_perm(table[3], table[2], indices & 0x07070707);
            return __builtin_amdgcn_perm(hi, lo, 0x03020100 | ((indices & 0x08080808) >> 1));
        }
    }
#endif
    uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int code = (codes >> (bits*k)) & ((1u << bits) - 1);
        if constexpr (bits == 2) {
            packed |= uint32_t(uint8_t(kvalues_znq2[book*4 + code])) << (8*k);
        } else if constexpr (bits == 3) {
            packed |= uint32_t(uint8_t(kvalues_znq3[book*8 + code])) << (8*k);
        } else {
            packed |= uint32_t(uint8_t(kvalues_znq4[book*16 + code])) << (8*k);
        }
    }
    return int(packed);
}
