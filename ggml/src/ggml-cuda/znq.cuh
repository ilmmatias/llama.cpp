#pragma once

#include "common.cuh"

static __host__ __device__ constexpr int znq_bits_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_ZNQ2_M:
        case GGML_TYPE_ZNQ2_L:
        case GGML_TYPE_ZNQ3_S: return 2;
        case GGML_TYPE_ZNQ3_M:
        case GGML_TYPE_ZNQ3_L:
        case GGML_TYPE_ZNQ4_S: return 3;
        case GGML_TYPE_ZNQ4_M:
        case GGML_TYPE_ZNQ4_L:
        case GGML_TYPE_ZNQ5_S: return 4;
        case GGML_TYPE_ZNQ5_M:
        case GGML_TYPE_ZNQ5_L:
        case GGML_TYPE_ZNQ6_S: return 5;
        case GGML_TYPE_ZNQ6_M:
        case GGML_TYPE_ZNQ6_L:
        case GGML_TYPE_ZNQ7_S: return 6;
        case GGML_TYPE_ZNQ7_M:
        case GGML_TYPE_ZNQ7_L: return 7;
        default: return 0;
    }
}

static __host__ __device__ constexpr int znq_group_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_ZNQ2_L:
        case GGML_TYPE_ZNQ3_L:
        case GGML_TYPE_ZNQ4_L:
        case GGML_TYPE_ZNQ5_L:
        case GGML_TYPE_ZNQ6_L:
        case GGML_TYPE_ZNQ7_L: return 8;
        case GGML_TYPE_ZNQ3_S:
        case GGML_TYPE_ZNQ4_S:
        case GGML_TYPE_ZNQ5_S:
        case GGML_TYPE_ZNQ6_S:
        case GGML_TYPE_ZNQ7_S: return 4;
        default: return 16;
    }
}

static __device__ __forceinline__ float znq_scale_cuda(uint8_t code) {
    const uint32_t e = code >> 3;
    const uint32_t m = code & 7;

    return e == 0 ? float(m) * (1.0f / 131072.0f) : __uint_as_float(((e + 112u) << 23) | (m << 20));
}

template <int bits, int group = 16>
static __host__ __device__ constexpr int znq_header_cuda() {
    return group == 16 ? 2 : (group == 8 ? 3 : 4);
}

template <int bits, int group>
static __device__ __forceinline__ const int8_t * znq_values_cuda() {
    if constexpr (group == 4) {
        return bits == 2 ? kvalues_znq3_s : bits == 3 ? kvalues_znq4_s : bits == 4 ? kvalues_znq5_s : bits == 5 ? kvalues_znq6_s : kvalues_znq7_s;
    } else if constexpr (group == 8) {
        return bits == 2 ? kvalues_znq2_l : bits == 3 ? kvalues_znq3_l : bits == 4 ? kvalues_znq4_l : bits == 5 ? kvalues_znq5_l : bits == 6 ? kvalues_znq6_l : kvalues_znq7_l;
    } else if constexpr (bits == 2) {
        return kvalues_znq2_m;
    } else if constexpr (bits == 3) {
        return kvalues_znq3_m;
    } else {
        return bits == 4 ? kvalues_znq4_m : bits == 5 ? kvalues_znq5_m : bits == 6 ? kvalues_znq6_m : kvalues_znq7_m;
    }
}

template <int group>
static __device__ __forceinline__ int znq_book_cuda(const uint8_t * block, int j) {
    if constexpr (group == 16) {
        return (block[1] >> (4 * (j / 16))) & 15;
    } else if constexpr (group == 8) {
        return ((uint32_t(block[1]) | (uint32_t(block[2]) << 8)) >> (4 * (j / 8))) & 15;
    } else {
        // The header is four bytes in both local-book formats.
        uint32_t header;
        memcpy(&header, block, sizeof(header));

        return (header >> (8 + 3 * (j / 4))) & 7;
    }
}

template <int bits, int group = 16>
static __device__ __forceinline__ int znq_value_cuda(const uint8_t * block, int j) {
    const int offset = bits * j;
    const int shift = offset % 8;
    unsigned code = block[znq_header_cuda<bits, group>() + offset / 8] >> shift;
    if constexpr (bits == 3 || bits >= 5) {
        if (shift + bits > 8) {
            code |= unsigned(block[znq_header_cuda<bits, group>() + 1 + offset / 8]) << (8 - shift);
        }
    }

    code &= (1u << bits) - 1;
    const int book = znq_book_cuda<group>(block, j);
    if constexpr (bits == 2) {
        return znq_values_cuda<bits, group>()[book * 4 + code];
    } else if constexpr (bits == 3) {
        return znq_values_cuda<bits, group>()[book * 8 + code];
    } else {
        return znq_values_cuda<bits, group>()[book * (1 << bits) + code];
    }
}

template <int bits, bool reuse_book = false, int group = 16>
static __device__ __forceinline__ int znq_pack4_cuda(const uint8_t * block, int j) {
    uint32_t codes;
    if constexpr (bits == 2) {
        // Callers pass four-aligned indices, so one quad fits in one payload byte.
        codes = block[znq_header_cuda<bits, group>() + j / 4];
    } else if constexpr (bits == 3) {
        const int offset = 3 * j;
        codes = (uint32_t(block[znq_header_cuda<bits, group>() + offset / 8]) | (uint32_t(block[znq_header_cuda<bits, group>() + 1 + offset / 8]) << 8)) >> (offset % 8);
    } else if constexpr (bits >= 5) {
        const int offset = bits * j;
#if defined(GGML_USE_HIP)
        // Keep the final word inside the payload, including an unpadded last block.
        const int last_word = 4*bits - int(sizeof(codes));
        const int word_offset = offset/8 < last_word ? offset/8 : last_word;
        memcpy(&codes, block + znq_header_cuda<bits, group>() + word_offset, sizeof(codes));
        codes >>= offset - 8*word_offset;
#else
        const uint8_t * qs = block + znq_header_cuda<bits, group>() + offset / 8;
        codes = uint32_t(qs[0]) | (uint32_t(qs[1]) << 8) | (uint32_t(qs[2]) << 16);
        if constexpr (bits == 7) {
            codes |= uint32_t(qs[3]) << 24;
        }
        codes >>= offset % 8;
#endif
    } else {
        codes = uint32_t(block[znq_header_cuda<bits, group>() + j / 2]) | (uint32_t(block[znq_header_cuda<bits, group>() + 1 + j / 2]) << 8);
    }

    const int book = znq_book_cuda<group>(block, j);

#if defined(GGML_USE_HIP)
    // Loading a whole book pays off when one lane reuses it across several quads.
    // MMQ's single-quad lanes keep direct lookups for books larger than four bytes.
    if constexpr (bits == 2 || (reuse_book && bits < 6 && (bits != 5 || group != 4))) {
        uint32_t indices = 0;
        if constexpr (bits == 4) {
            // Selectors 0, 4, 1, 5 put the four nibble codes in separate bytes.
            indices = ggml_hip_perm(codes >> 4, codes, 0x05010400) & 0x0f0f0f0f;
        } else if constexpr (bits == 5) {
            indices = (codes & 0x000003ff) | ((codes & 0x000ffc00) << 6);
            indices = (indices & 0x001f001f) | ((indices & 0x03e003e0) << 3);
        } else if constexpr (bits == 3) {
            // Spread four 3-bit codes into separate byte selectors.
            indices = (codes | (codes << 10)) & 0x003f003f;
            indices = (indices | (indices << 5)) & 0x07070707;
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                indices |= ((codes >> (bits * k)) & ((1u << bits) - 1)) << (8 * k);
            }
        }

        if constexpr (bits == 2) {
            const uint32_t * table = (const uint32_t *) (znq_values_cuda<bits, group>() + 4 * book);

            return ggml_hip_perm(0, table[0], indices);
        } else if constexpr (bits == 3) {
            const uint32_t * table = (const uint32_t *) (znq_values_cuda<bits, group>() + 8 * book);

            return ggml_hip_perm(table[1], table[0], indices);
        } else if constexpr (bits == 5) {
            const uint32_t * table = (const uint32_t *) (znq_values_cuda<bits, group>() + 32 * book);
            const uint32_t select = 0x03020100 | ((indices & 0x08080808) >> 1);
            const uint32_t lo0 = ggml_hip_perm(table[1], table[0], indices & 0x07070707);
            const uint32_t lo1 = ggml_hip_perm(table[3], table[2], indices & 0x07070707);
            const uint32_t hi0 = ggml_hip_perm(table[5], table[4], indices & 0x07070707);
            const uint32_t hi1 = ggml_hip_perm(table[7], table[6], indices & 0x07070707);
            const uint32_t lo = ggml_hip_perm(lo1, lo0, select);
            const uint32_t hi = ggml_hip_perm(hi1, hi0, select);

            return ggml_hip_perm(hi, lo, 0x03020100 | ((indices & 0x10101010) >> 2));
        } else {
            const uint32_t * table = (const uint32_t *) (znq_values_cuda<bits, group>() + 16 * book);
            const uint32_t lo = ggml_hip_perm(table[1], table[0], indices & 0x07070707);
            const uint32_t hi = ggml_hip_perm(table[3], table[2], indices & 0x07070707);

            return ggml_hip_perm(hi, lo, 0x03020100 | ((indices & 0x08080808) >> 1));
        }
    }
#endif
    uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int code = (codes >> (bits * k)) & ((1u << bits) - 1);
        if constexpr (bits == 2) {
            packed |= uint32_t(uint8_t(znq_values_cuda<bits, group>()[book * 4 + code])) << (8 * k);
        } else if constexpr (bits == 3) {
            packed |= uint32_t(uint8_t(znq_values_cuda<bits, group>()[book * 8 + code])) << (8 * k);
        } else {
            packed |= uint32_t(uint8_t(znq_values_cuda<bits, group>()[book * (1 << bits) + code])) << (8 * k);
        }
    }

    return int(packed);
}
