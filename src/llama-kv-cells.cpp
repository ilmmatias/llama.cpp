#include "llama-kv-cache.h"
#include <cstdint>

void llama_kv_cells::seqS_add(uint32_t i, int32_t n, llama_seq_id *_seq) {
    llama_seq_id seq_id{};
    assert(i < pos.size());
    assert(pos[i] != -1);
    for (int32_t s = 0; s < n; s++) {
        seq_id = _seq[s];
        assert(!seq[i].test(seq_id));
        seq[i].set(seq_id);
        seq_pos_inc(seq_id, pos[i], i);
    }
}

void llama_kv_cells::compact(llama_seq_id s) {
    auto & v = seq_pos[s];

    if (v.total == 0) {
        v.clear();
        return;
    }

    const uint32_t h = v.head;
    const uint32_t t = v.tail;

    if (h == 0 && t + 1 == v.cnt.size()) {
        return;
    }

    if (h > 0) {
        v.cnt.erase(v.cnt.begin(), v.cnt.begin() + h);
        v.row_max.erase(v.row_max.begin(), v.row_max.begin() + h);
    }
    v.cnt.resize(t + 1 - h);
    v.row_max.resize(t + 1 - h);

    v.base += (llama_pos)h;
    v.head  = 0;
    v.tail  = t - h;
}

uint32_t llama_kv_cells::nextHead(int32_t seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size();

    for (size_t w = 0; w < used_bits.size(); ++w) {
        uint64_t mask = used_bits[w];
        while (mask) {
            const int bit = llama_bits::countr_zero64(mask);
            const uint32_t i = (uint32_t)(w * 64 + bit);
            mask &= mask - 1;

            const llama_pos p = pos[i];
            if (p < p0 || p >= p1) continue;

            if (seq_has(i, seq_id) && seq_rm(i, seq_id)) {
                if (new_head == size()) {
                    new_head = i;
                }
            }
        }
    }
    compact(seq_id);
    return new_head;
}

void llama_kv_cells::set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
    assert(idxs.size() == other.pos.size());

    for (uint32_t j = 0; j < other.pos.size(); ++j) {
        const auto idx = idxs[j];

        if (pos[idx] == other.pos[j] && seq[idx] == other.seq[j]) {
            ext[idx] = other.ext[j];
            assert(shift[idx] == 0);
            continue;
        }

        if (pos[idx] == -1 && other.pos[j] != -1) {
            used_insert(idx);
        }
        if (pos[idx] != -1 && other.pos[j] == -1) {
            used_erase(idx);
        }
        if (pos[idx] != -1) {
            seq_pos_rm(idx);
        }

        pos[idx] = other.pos[j];
        ext[idx] = other.ext[j];
        seq[idx] = other.seq[j];

        if (pos[idx] != -1) {
            seq_pos_add(idx);
        }

        assert(shift[idx] == 0);
    }
}

void llama_kv_cells::seq_pos_dec(llama_seq_id s, llama_pos p, uint32_t row) {
    auto & v = seq_pos[s];

    assert(v.total > 0);
    assert(row < pos.size());
    const uint32_t idx = (uint32_t) (p - v.base);
    assert(idx < v.cnt.size());
    assert(v.cnt[idx] > 0);
    assert(v.row_max[idx] != UINT32_MAX);

    --v.cnt[idx];
    --v.total;

    if (v.total == 0) {
        v.clear();
        return;
    }

    if (v.cnt[idx] == 0) {
        v.row_max[idx] = UINT32_MAX;
    } else if (v.row_max[idx] == row) {
        uint32_t new_max = UINT32_MAX;

        for (size_t w = used_bits.size(); w-- > 0;) {
            uint64_t mask = used_bits[w];

            while (mask) {
                const int bit = 63 - llama_bits::countl_zero64(mask);
                const uint32_t i = (uint32_t) (w * 64 + bit);
                mask &= ~(UINT64_C(1) << bit);
                if (i >= pos.size() || i == row) {
                    continue;
                }

                if (pos[i] == p && seq[i].test(s)) {
                    new_max = i;
                    break;
                }
            }

            if (new_max != UINT32_MAX) {
                break;
            }
        }

        assert(new_max != UINT32_MAX);
        v.row_max[idx] = new_max;
    }

    if (idx == v.head) {
        while (v.cnt[v.head] == 0) {
            ++v.head;
        }
    } else if (idx == v.tail) {
        while (v.cnt[v.tail] == 0) {
            --v.tail;
        }
    }
}

void llama_kv_cells::seq_pos_inc(llama_seq_id s, llama_pos p, uint32_t row) {
    auto & v = seq_pos[s];

    if (v.total == 0) {
        v.base = p;
        v.cnt.assign(1, 1);
        v.row_max.assign(1, row);
        v.head = v.tail = 0;
        v.total         = 1;
        return;
    }

    if (p >= v.base) {
        const uint32_t idx = (uint32_t) (p - v.base);
        if (idx >= v.cnt.size()) {
            v.cnt.resize(idx + 1, 0);
            v.row_max.resize(idx + 1, UINT32_MAX);
        }
        if (++v.cnt[idx] == 1) {
            v.row_max[idx] = row;
            if (idx < v.head) {
                v.head = idx;
            }
            if (idx > v.tail) {
                v.tail = idx;
            }
        } else {
            assert(v.row_max[idx] != UINT32_MAX);
            if (row > v.row_max[idx]) {
                v.row_max[idx] = row;
            }
        }
    } else {
        // rary
        const uint32_t pre = (uint32_t) (v.base - p);
        v.cnt.insert(v.cnt.begin(), pre, 0);
        v.row_max.insert(v.row_max.begin(), pre, UINT32_MAX);
        v.cnt[0] = 1;
        v.row_max[0] = row;
        v.base   = p;
        v.head   = 0;
        v.tail += pre;
    }

    ++v.total;
}

void llama_kv_cells::rm_single(uint32_t i, llama_seq_id seq_id) {  // need compact after some seq_pos_dec
    assert(i < pos.size());
    assert(pos[i] != -1);
    assert(seq[i].count() == 1);
    assert(seq[i].test(seq_id));

    seq_pos_dec(seq_id, pos[i], i);

    seq[i].reset();

    pos[i] = -1;
    ext[i].reset();
    shift[i] = 0;

    used_erase(i);
}

