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

template <int bits>
static __device__ __forceinline__ int znq_pack4_cuda(const uint8_t * block, int j) {
    uint32_t packed = 0;
    if constexpr (bits == 2) {
        // Callers pass four-aligned indices, so one quad fits in one payload byte.
        const uint32_t codes = block[2 + j/4];
        const int book = (block[1] >> (4*(j/16))) & 15;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            packed |= uint32_t(uint8_t(kvalues_znq2[book*4 + ((codes >> (2*k)) & 3)])) << (8*k);
        }
    } else if constexpr (bits == 3) {
        const int offset = 3*j;
        const uint32_t codes = (uint32_t(block[2 + offset/8]) | (uint32_t(block[3 + offset/8]) << 8)) >> (offset%8);
        const int book = (block[1] >> (4*(j/16))) & 15;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            packed |= uint32_t(uint8_t(kvalues_znq3[book*8 + ((codes >> (3*k)) & 7)])) << (8*k);
        }
    } else if constexpr (bits == 4) {
        const uint32_t codes = uint32_t(block[2 + j/2]) | (uint32_t(block[3 + j/2]) << 8);
        const int book = (block[1] >> (4*(j/16))) & 15;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            packed |= uint32_t(uint8_t(kvalues_znq4[book*16 + ((codes >> (4*k)) & 15)])) << (8*k);
        }
    }
    return int(packed);
}
