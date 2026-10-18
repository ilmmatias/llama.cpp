#pragma once

#include "common.cuh"

static __device__ __forceinline__ float znq_scale_cuda(uint8_t code) {
    const uint32_t e = code >> 3;
    const uint32_t m = code & 7;
    return e == 0 ? float(m) * (1.0f / 131072.0f) : __uint_as_float(((e + 112u) << 23) | (m << 20));
}

template <int bits, int group = 16>
static __host__ __device__ constexpr int znq_header_cuda() { return group == 16 ? 2 : (group == 8 ? 3 : 4); }

template <int bits, int group>
static __device__ __forceinline__ const int8_t * znq_values_cuda() {
    if constexpr (group == 4) {
        return bits == 2 ? kvalues_znq3_s : kvalues_znq4_s;
    } else if constexpr (group == 8) {
        return bits == 2 ? kvalues_znq2_l : kvalues_znq3_l;
    } else if constexpr (bits == 2) {
        return kvalues_znq2_m;
    } else if constexpr (bits == 3) {
        return kvalues_znq3_m;
    } else {
        return kvalues_znq4_m;
    }
}

template <int group>
static __device__ __forceinline__ int znq_book_cuda(const uint8_t * block, int j) {
    if constexpr (group == 16) {
        return (block[1] >> (4*(j/16))) & 15;
    } else if constexpr (group == 8) {
        return ((uint32_t(block[1]) | (uint32_t(block[2]) << 8)) >> (4*(j/8))) & 15;
    } else {
        // The header is four bytes in both local-book formats.
        uint32_t header;
        memcpy(&header, block, sizeof(header));
        return (header >> (8 + 3*(j/4))) & 7;
    }
}

template <int bits, int group = 16>
static __device__ __forceinline__ int znq_value_cuda(const uint8_t * block, int j) {
    const int offset = bits*j;
    const int shift = offset % 8;
    unsigned code = block[znq_header_cuda<bits, group>() + offset/8] >> shift;
    if constexpr (bits == 3) {
        if (shift > 5) {
            code |= unsigned(block[znq_header_cuda<bits, group>() + 1 + offset/8]) << (8 - shift);
        }
    }
    code &= (1u << bits) - 1;
    const int book = znq_book_cuda<group>(block, j);
    if constexpr (bits == 2) {
        return znq_values_cuda<bits, group>()[book*4 + code];
    } else if constexpr (bits == 3) {
        return znq_values_cuda<bits, group>()[book*8 + code];
    } else {
        return kvalues_znq4_m[book*16 + code];
    }
}

template <int bits, bool reuse_book = false, int group = 16>
static __device__ __forceinline__ int znq_pack4_cuda(const uint8_t * block, int j) {
    uint32_t codes;
    if constexpr (bits == 2) {
        // Callers pass four-aligned indices, so one quad fits in one payload byte.
        codes = block[znq_header_cuda<bits, group>() + j/4];
    } else if constexpr (bits == 3) {
        const int offset = 3*j;
        codes = (uint32_t(block[znq_header_cuda<bits, group>() + offset/8]) | (uint32_t(block[znq_header_cuda<bits, group>() + 1 + offset/8]) << 8)) >> (offset%8);
    } else {
        codes = uint32_t(block[znq_header_cuda<bits, group>() + j/2]) | (uint32_t(block[znq_header_cuda<bits, group>() + 1 + j/2]) << 8);
    }
    const int book = znq_book_cuda<group>(block, j);

#if defined(GGML_USE_HIP)
    // Larger books pay off when one lane decodes several quads from the same book.
    if constexpr (bits == 2 || reuse_book) {
        uint32_t indices = 0;
        if constexpr (bits == 4) {
            // Selectors 0, 4, 1, 5 put the four nibble codes in separate bytes.
            indices = __builtin_amdgcn_perm(codes >> 4, codes, 0x05010400) & 0x0f0f0f0f;
        } else if constexpr (bits == 3) {
            // Spread four 3-bit codes into separate byte selectors.
            indices = (codes | (codes << 10)) & 0x003f003f;
            indices = (indices | (indices << 5)) & 0x07070707;
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                indices |= ((codes >> (bits*k)) & ((1u << bits) - 1)) << (8*k);
            }
        }
        if constexpr (bits == 2) {
            const uint32_t * table = (const uint32_t *) (znq_values_cuda<bits, group>() + 4*book);
            return __builtin_amdgcn_perm(0, table[0], indices);
        } else if constexpr (bits == 3) {
            const uint32_t * table = (const uint32_t *) (znq_values_cuda<bits, group>() + 8*book);
            return __builtin_amdgcn_perm(table[1], table[0], indices);
        } else {
            const uint32_t * table = (const uint32_t *) (kvalues_znq4_m + 16*book);
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
            packed |= uint32_t(uint8_t(znq_values_cuda<bits, group>()[book*4 + code])) << (8*k);
        } else if constexpr (bits == 3) {
            packed |= uint32_t(uint8_t(znq_values_cuda<bits, group>()[book*8 + code])) << (8*k);
        } else {
            packed |= uint32_t(uint8_t(kvalues_znq4_m[book*16 + code])) << (8*k);
        }
    }
    return int(packed);
}
