#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <cassert>
#include <iterator>
#include <limits>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    qsa_block_capacity(filter_idx == nullptr || model.arch != LLM_ARCH_QWEN4EXP ? 0 : [&] {
        uint32_t min_ratio = std::numeric_limits<uint32_t>::max();
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (!filter_idx(il)) {
                continue;
            }

            const uint32_t ratio = model.hparams.dsv4_compress_ratios[il];
            if (ratio > 0) {
                min_ratio = std::min(min_ratio, ratio);
            }
        }

        if (min_ratio == std::numeric_limits<uint32_t>::max()) {
            return 0u;
        }

        const uint64_t n_stream = unified ? 1 : n_seq_max;
        const uint64_t n_cells = (uint64_t) kv_size*n_stream;
        // Complete blocks partition the physical cells. The small per-sequence margin
        // covers branch/tail boundaries without giving the block cache token-cache scale.
        const uint64_t n_blocks = (n_cells + min_ratio - 1)/min_ratio + std::max(1u, n_seq_max);
        GGML_ASSERT(n_blocks <= std::numeric_limits<uint32_t>::max());
        return (uint32_t) n_blocks;
    }()),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // GLM's k-pool indexer caches key | gate | pooled side by side.
        // Qwen uses raw key rows and a separate compact block cache.
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // Indexer storage is K-only; pooled GLM keys share each raw-key row,
        // while Qwen stores its derived keys in the compact cache below.
        hparams_idx.n_embd_head_k_mla_impl = hparams_idx.n_embd_head_k_full;
        hparams_idx.n_embd_head_v_mla_impl = hparams_idx.n_embd_head_k_full;

        LLAMA_LOG_INFO("%s: creating indexer K cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, false, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()),
    mem_qsa_blocks(filter_idx == nullptr || qsa_block_capacity == 0 ? nullptr : [&] {
        LLAMA_LOG_INFO("%s: creating compact QSA block cache, size = %u blocks\n", __func__, qsa_block_capacity);

        // This cache is storage only: one F32 K row per derived block, shared across
        // sequences by our slot table. It is always one stream and never stores V.
        return new llama_kv_cache(
            model, hparams_idx, GGML_TYPE_F32, GGML_TYPE_F32, false, offload, true,
            qsa_block_capacity, 1, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, filter_idx, nullptr, nullptr, "idx_blk_");
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (get_mem_recr()->has_state() && !get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }
    if (mem_qsa_blocks) {
        mem_qsa_blocks->clear(data);
    }

    qsa_reset();
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        qsa_reset();
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        bool tail = seq_id >= 0 && p0 > mem_idx->seq_pos_min(seq_id) &&
            (p1 < 0 || p1 > mem_idx->seq_pos_max(seq_id));
        if (tail && mem_idx->get_n_stream() == 1) {
            const auto & cells = mem_idx->get_cells(seq_id);
            for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (s != seq_id && cells.seq_pos_count(s) > 0) {
                    tail = false;
                    break;
                }
            }
        }
        // Shared-stream edits can change ranking; ordinary tails keep their prefix grouping.
        if (tail) {
            qsa_trim(seq_id, p0);
        } else {
            qsa_reset();
        }
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool grouping
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
        // pooled keys, the destination rebuilds its layout and re-pools into the same rows
        mem_idx_stale_set(seq_id_dst, 0);
    }

    qsa_reset();
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }

    qsa_reset();
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }

    qsa_reset();
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }

    qsa_reset();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    if (mem_qsa_blocks) {
        for (const auto & buft_size : mem_qsa_blocks->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                // the restore rewrites the cells behind the pool layout's back
                mem_idx_stale_set(seq_id, 0);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    qsa_reset();
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }

    qsa_reset();
}

void llama_memory_hybrid_idx::qsa_reset(uint32_t ratio) const {
    if (ratio != 0) {
        qsa_blocks.erase(ratio);
        qsa_slots.erase(ratio);
        return;
    }
    qsa_blocks.clear();
    qsa_slots.clear();
}

void llama_memory_hybrid_idx::qsa_trim(llama_seq_id seq_id, llama_pos p0) const {
    const auto & cells = mem_idx->get_cells(seq_id);
    const uint32_t size = mem_idx->get_size();
    for (auto & [ratio, by_seq] : qsa_blocks) {
        const auto it = by_seq.find(seq_id);
        if (it == by_seq.end()) {
            continue;
        }
        auto & blocks = it->second;
        size_t keep = blocks.size();
        for (size_t b = 0; b < blocks.size(); ++b) {
            for (const uint32_t row : blocks[b].cells) {
                const uint32_t cell = row % size;
                if (cells.is_empty(cell) || cells.pos_get(cell) >= p0) {
                    keep = b;
                    break;
                }
            }
            if (keep != blocks.size()) {
                break;
            }
        }
        for (size_t b = keep; b < blocks.size(); ++b) {
            qsa_release_slot(ratio, blocks[b]);
        }
        blocks.resize(keep);
    }
}

void llama_memory_hybrid_idx::qsa_invalidate_rows(const llama_kv_cache::slot_info & sinfo) const {
    if (qsa_blocks.empty()) {
        return;
    }
    std::vector<uint32_t> rows;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint64_t offset = (uint64_t) sinfo.strm[s]*mem_idx->get_size();
        const auto & cells = mem_idx->get_cells(sinfo.strm[s]);
        for (const uint32_t cell : sinfo.idxs[s]) {
            if (cells.is_empty(cell)) {
                continue;
            }
            GGML_ASSERT(offset + cell <= std::numeric_limits<uint32_t>::max());
            rows.push_back((uint32_t) (offset + cell));
        }
    }
    if (rows.empty()) {
        return;
    }
    std::sort(rows.begin(), rows.end());
    // Invalidate every alias before raw keys are rewritten, even when cell ids stay the same.
    for (auto & [ratio, by_seq] : qsa_blocks) {
        for (auto & [seq_id, blocks] : by_seq) {
            for (auto & block : blocks) {
                if (!block.valid) {
                    continue;
                }
                for (const uint32_t row : block.cells) {
                    if (std::binary_search(rows.begin(), rows.end(), row)) {
                        qsa_release_slot(ratio, block);
                        block.dirty = true;
                        break;
                    }
                }
            }
        }
    }
}

uint32_t llama_memory_hybrid_idx::qsa_acquire_slot(
        uint32_t ratio, const std::vector<uint32_t> & cells, bool & is_new) const {
    GGML_ASSERT(mem_qsa_blocks != nullptr);

    auto & state = qsa_slots[ratio];
    const auto found = state.by_cells.find(cells);
    if (found != state.by_cells.end()) {
        const uint32_t slot = found->second;
        GGML_ASSERT(slot < state.refs.size() && state.refs[slot] > 0);
        state.refs[slot]++;
        is_new = false;
        return slot;
    }

    uint32_t slot;
    if (!state.free_slots.empty()) {
        slot = state.free_slots.back();
        state.free_slots.pop_back();
    } else {
        slot = state.next_slot++;
    }

    if (slot >= qsa_block_capacity) {
        throw std::runtime_error("QSA compact block cache exhausted");
    }

    if (state.refs.size() <= slot) {
        state.refs.resize(slot + 1, 0);
    }
    GGML_ASSERT(state.refs[slot] == 0);
    state.refs[slot] = 1;
    state.by_cells.emplace(cells, slot);
    is_new = true;
    return slot;
}

void llama_memory_hybrid_idx::qsa_release_slot(uint32_t ratio, qsa_block & block) const {
    if (!block.valid) {
        return;
    }

    const auto state_it = qsa_slots.find(ratio);
    GGML_ASSERT(state_it != qsa_slots.end());
    auto & state = state_it->second;
    GGML_ASSERT(block.cache_slot < state.refs.size() && state.refs[block.cache_slot] > 0);

    if (--state.refs[block.cache_slot] == 0) {
        const auto key_it = state.by_cells.find(block.cells);
        GGML_ASSERT(key_it != state.by_cells.end() && key_it->second == block.cache_slot);
        state.by_cells.erase(key_it);
        state.free_slots.push_back(block.cache_slot);
    }

    block = {};
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

uint32_t llama_memory_hybrid_idx::get_qsa_update_capacity(
        const llama_ubatch & ubatch,
        uint32_t            ratio,
        uint32_t            n_blocks) const {
    GGML_ASSERT(mem_idx != nullptr);
    GGML_ASSERT(ratio > 0);

    bool cold = qsa_blocks.find(ratio) == qsa_blocks.end();
    if (!cold) {
        const auto & by_seq = qsa_blocks.at(ratio);
        for (uint32_t i = 0; i < ubatch.n_tokens && !cold; ++i) {
            for (int32_t is = 0; is < ubatch.n_seq_id[i]; ++is) {
                if (by_seq.find(ubatch.seq_id[i][is]) == by_seq.end()) {
                    cold = true;
                    break;
                }
            }
        }
    }

    uint64_t result;
    if (cold) {
        // First graph after clear/sequence edits/state restore must be able to rebuild
        // every complete block for every active sequence.
        result = (uint64_t) n_blocks*std::max(1u, ubatch.n_seqs_unq);
    } else {
        // During steady decode each active stream can complete at most one extra block
        // beyond the obvious n_tokens/ratio quotient.
        result = (uint64_t) ubatch.n_tokens/ratio + std::max(1u, ubatch.n_seqs_unq);
        const auto & by_seq = qsa_blocks.at(ratio);
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            for (const auto & block : by_seq.at(ubatch.seq_id_unq[s])) {
                result += block.dirty;
            }
        }
    }

    result = std::max<uint64_t>(1, result);
    GGML_ASSERT(result <= std::numeric_limits<uint32_t>::max());
    return (uint32_t) result;
}


//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> rep_gen; // per global cell, the generation that last marked a pool with that rep
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const size_t n_cells = cells.seq_pos_count(s);
        const llama_pos pos_min = cells.seq_pos_min(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && n_cells > 0 &&
                sq.pos_min == pos_min) {
            cells.seq_pos_append(s, sq.cells.back(), sq.cells);
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != n_cells || mem_idx_stale[s] != POS_CLEAN) {
            sq.cells.clear();
            sq.cells.reserve(n_cells);
            cells.seq_pos_append(s, { -1, 0 }, sq.cells);
            sq.pools.clear();
            sq.j_next  = 0;
            sq.pos_min = n_cells == 0 ? 0 : pos_min;
        }

        // Pools start at the first valid token
        size_t j = sq.j_next;
        if (hparams_idx.indexer_kpool_by_order) {
            // consecutive cells in sequence order, whatever their positions
            for (; j + kpool <= sq.cells.size(); j += kpool) {
                sq.pools.push_back((uint32_t) j);
            }
        } else {
            while (j + kpool <= sq.cells.size()) {
                const llama_pos p0 = sq.cells[j].first;
                if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                    ++j;
                    continue;
                }
                bool ok = true;
                for (uint32_t k = 1; k < kpool; ++k) {
                    if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    sq.pools.push_back((uint32_t) j);
                    j += kpool;
                } else {
                    ++j;
                }
            }
        }
        sq.j_next = j;

        lay.n_pool_real += (uint32_t) sq.pools.size();
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->mem_qsa_blocks != nullptr ||
        (mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order()) ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    if (mem != nullptr && mem->mem_qsa_blocks != nullptr && !sinfos_kpool.empty()) {
        mem->qsa_invalidate_rows(sinfos_kpool[i_cur]);
    }
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

ggml_tensor * llama_memory_hybrid_idx_context::get_qsa_block_storage(int32_t il) const {
    GGML_ASSERT(mem != nullptr && mem->mem_qsa_blocks != nullptr);
    return mem->mem_qsa_blocks->get_k_storage(il);
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

uint32_t llama_memory_hybrid_idx_context::get_qsa_update_capacity(
        const llama_ubatch & ubatch,
        uint32_t            ratio,
        uint32_t            n_blocks) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    return mem->get_qsa_update_capacity(ubatch, ratio, n_blocks);
}

void llama_memory_hybrid_idx_context::reset_qsa(uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);
    mem->qsa_reset(ratio);
}


void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * block_cells,
        ggml_tensor * block_cell_bias,
        ggml_tensor * bias,
        ggml_tensor * block_key_cells,
        ggml_tensor * update_cells,
        ggml_tensor * update_pos,
        ggml_tensor * update_idxs,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    // The generic path uploads cell -> block. The compact decode path uploads
    // block -> cell instead, so do not pay for both O(n_kv) maps on every token.
    GGML_ASSERT((cell_blk == nullptr) != (block_cells == nullptr));
    if (cell_blk != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));
        GGML_ASSERT(block_cell_bias == nullptr);
    } else {
        GGML_ASSERT(block_cells != nullptr && block_cell_bias != nullptr);
        GGML_ASSERT(ggml_backend_buffer_is_host(block_cells->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(block_cell_bias->buffer));
    }
    GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(block_key_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(update_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(update_pos->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(update_idxs->buffer));

    const auto * mem_idx = mem->get_mem_idx();
    const auto * idx_ctx = get_idx();
    GGML_ASSERT(idx_ctx != nullptr);

    const int64_t n_kv      = idx_ctx->get_n_kv();
    const int64_t n_ns      = get_n_stream();
    const int64_t n_tokens  = ubatch->n_tokens;
    const int64_t n_updates = update_idxs->ne[0];
    const int64_t r         = ratio;
    const int64_t n_blocks  = (n_kv + r - 1)/r;

    GGML_ASSERT(n_tokens % n_ns == 0);
    GGML_ASSERT(n_updates > 0);
    GGML_ASSERT(block_key_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(block_key_cells->ne[0] == n_blocks && block_key_cells->ne[1] == n_ns);
    GGML_ASSERT(update_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(update_cells->ne[0] == r && update_cells->ne[1] == n_updates);
    GGML_ASSERT(update_pos->type == GGML_TYPE_I32 && update_pos->ne[0] == 4*n_updates);
    GGML_ASSERT(update_idxs->type == GGML_TYPE_I64);
    if (cell_blk != nullptr) {
        GGML_ASSERT(cell_blk->type == GGML_TYPE_I32);
        GGML_ASSERT(cell_blk->ne[0] == n_kv && cell_blk->ne[1] == n_ns);
    } else {
        GGML_ASSERT(n_ns == 1); // stage-3 fast path is intentionally single-stream
        GGML_ASSERT(block_cells->type == GGML_TYPE_I32);
        GGML_ASSERT(block_cells->ne[0] == r && block_cells->ne[1] == n_blocks && block_cells->ne[2] == n_ns);
        GGML_ASSERT(block_cell_bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_are_same_shape(block_cells, block_cell_bias));
    }
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream
    int32_t * dst_cell_blk = cell_blk != nullptr ? (int32_t *) cell_blk->data : nullptr;
    float   * dst_bias      = (float   *) bias->data;
    int32_t * dst_block_cells = block_cells != nullptr ? (int32_t *) block_cells->data : nullptr;
    float * dst_block_cell_bias = block_cell_bias != nullptr ? (float *) block_cell_bias->data : nullptr;

    int32_t * dst_block_key_cell = (int32_t *) block_key_cells->data;
    int32_t * dst_update_cells   = (int32_t *) update_cells->data;
    int32_t * dst_update_pos     = (int32_t *) update_pos->data;
    int64_t * dst_update_idxs    = (int64_t *) update_idxs->data;

    std::fill(dst_block_key_cell, dst_block_key_cell + ggml_nelements(block_key_cells), 0);
    std::fill(dst_update_cells, dst_update_cells + ggml_nelements(update_cells), 0);
    std::fill(dst_update_pos, dst_update_pos + ggml_nelements(update_pos), 0);
    std::fill(dst_update_idxs, dst_update_idxs + ggml_nelements(update_idxs), 0);


    // one pass per stream: cell j is a different token in each, so no mapping is shared
    std::vector<int32_t> blk_of(n_kv);
    std::vector<int32_t> filled(n_blocks);

    std::vector<int32_t> cur_blk_cells(r*n_blocks);

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    int64_t n_update = 0;
    bool have_fallback = false;
    std::vector<int32_t> fallback_cells(r, 0);
    int32_t fallback_pos[4] = { 0, 0, 0, 0 };
    int64_t fallback_idx = 0;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem_idx->get_cells(seq_of_stream);
        int32_t * cur_cell_blk = dst_cell_blk != nullptr ? dst_cell_blk + s*n_kv : nullptr;
        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // an incomplete block cannot be pooled; the bias below forces those tail cells in
            // -1 means no usable block, and block 0 only keeps the gather in range
            std::fill(blk_of.begin(),  blk_of.end(),  -1);
            std::fill(filled.begin(),  filled.end(),   0);
            std::fill(cur_blk_cells.begin(), cur_blk_cells.end(), -1);

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t b   = idx/r;

                if (b >= n_blocks) {
                    oor = true;
                    continue;
                }

                const int64_t slot = b*r + (idx%r);

                const bool slot_filled = cur_blk_cells[slot] >= 0;
                dup |= slot_filled;

                blk_of[j] = (int32_t) b;
                cur_blk_cells[slot] = (int32_t) j;
                // Shared sequences can have distinct cells at the same position.
                // A pool is complete only when every position has a source cell.
                filled[b] += !slot_filled;
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");


        // Raw indexer K keeps token-cell addressing, while derived keys live in a
        // compact one-row-per-block cache. Global raw-K row ids make shared prefixes
        // converge on the same compact slot even when several sequences reference them.
        const int64_t cache_stream = mem_idx->get_n_stream() == 1 ? 0 : seq_of_stream;
        const int64_t stream_off = cache_stream*(int64_t) mem_idx->get_size();
        auto & cached = mem->qsa_blocks[ratio][seq_of_stream];
        if (cached.size() < (size_t) n_blocks) {
            cached.resize(n_blocks);
        }

        for (int64_t b = 0; b < n_blocks; ++b) {
            // Incomplete/padding blocks are masked or force-selected as tail, so their
            // dot product is irrelevant. Keep the compact gather index in range.
            dst_block_key_cell[s*n_blocks + b] = 0;

            if (filled[b] != r) {
                mem->qsa_release_slot(ratio, cached[b]);
                cached[b].dirty = false;
                continue;
            }

            int32_t sec_pos[4] = { (int32_t) (b*r), (int32_t) (b*r), (int32_t) (b*r), (int32_t) (b*r) };

            if (ranked) {
                const int32_t   c = cur_blk_cells[b*r];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            llama_memory_hybrid_idx::qsa_block now;
            now.cells.resize(r);
            for (int64_t ir = 0; ir < r; ++ir) {
                const int64_t global_src = stream_off + cur_blk_cells[b*r + ir];
                GGML_ASSERT(global_src <= std::numeric_limits<int32_t>::max());
                now.cells[ir] = (uint32_t) global_src;
            }

            bool refresh = false;
            if (!cached[b].valid || cached[b].cells != now.cells) {
                mem->qsa_release_slot(ratio, cached[b]);

                bool is_new = false;
                now.cache_slot = mem->qsa_acquire_slot(ratio, now.cells, is_new);
                now.valid = true;
                cached[b] = std::move(now);
                refresh = is_new;
            }

            const uint32_t cache_slot = cached[b].cache_slot;
            GGML_ASSERT(cache_slot <= (uint32_t) std::numeric_limits<int32_t>::max());
            dst_block_key_cell[s*n_blocks + b] = (int32_t) cache_slot;

            if (!have_fallback) {
                have_fallback = true;
                for (int64_t sec = 0; sec < 4; ++sec) {
                    fallback_pos[sec] = sec_pos[sec];
                }
                fallback_idx = cache_slot;
                for (int64_t ir = 0; ir < r; ++ir) {
                    fallback_cells[ir] = (int32_t) cached[b].cells[ir];
                }
            }

            // A matching physical block may already be referenced by another sequence.
            // In that case its derived rows are already valid in every QSA layer.
            if (!refresh) {
                continue;
            }

            GGML_ASSERT(n_update < n_updates);
            for (int64_t ir = 0; ir < r; ++ir) {
                dst_update_cells[n_update*r + ir] = (int32_t) cached[b].cells[ir];
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_update_pos[sec*n_updates + n_update] = sec_pos[sec];
            }
            dst_update_idxs[n_update++] = cache_slot;
        }

        if (dst_block_cells != nullptr) {
            int32_t * out_cells = dst_block_cells + s*(r*n_blocks);
            float * out_bias = dst_block_cell_bias + s*(r*n_blocks);
            for (int64_t bi = 0; bi < r*n_blocks; ++bi) {
                const int32_t cell = cur_blk_cells[bi];
                out_cells[bi] = cell >= 0 ? cell : 0;
                out_bias[bi] = cell >= 0 ? 0.0f : -INFINITY;
            }
        }

        // per-block mode keeps an unpooled cell's real block, so the block's own -inf reaches it
        // per-cell mode carries that -inf itself and only needs the gather in range
        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 && filled[blk_of[j]] < r && !blk_bias) {
                blk_of[j] = -1;
            }
            if (cur_cell_blk != nullptr) {
                cur_cell_blk[j] = blk_of[j] < 0 ? 0 : blk_of[j];
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // Block-first compact selection performs its top-k before the per-cell
                // attention mask is gathered. Reject future blocks here so they cannot
                // consume the shortlist; only the current partial tail is force-selected.
                // The legacy expanded path still gets its future masking before token top-k.
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    const int64_t b0 = b*r;
                    if (dst_block_cells != nullptr && b0 > q) {
                        cur_blk_bias[b] = -INFINITY;
                    } else if (b0 >= tail_start) {
                        // finite, so it can never meet a -inf and produce a nan
                        cur_blk_bias[b] = 1e9f;
                    } else {
                        cur_blk_bias[b] = filled[b] < r ? -INFINITY : 0.0f;
                    }
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }

    if (n_update == 0) {
        // ggml tensors cannot have a zero-sized update dimension. Recompute one already
        // cached block when possible; before the first complete block, row zero is harmless
        // because every incomplete block is excluded from scoring or force-selected as tail.
        for (int64_t ir = 0; ir < r; ++ir) {
            dst_update_cells[ir] = have_fallback ? fallback_cells[ir] : 0;
        }
        for (int64_t sec = 0; sec < 4; ++sec) {
            dst_update_pos[sec*n_updates] = have_fallback ? fallback_pos[sec] : 0;
        }
        dst_update_idxs[0] = have_fallback ? fallback_idx : 0;
        n_update = 1;
    }

    // Capacity is an upper bound chosen to keep the graph shape reusable. Duplicate the
    // first refresh into unused rows; repeated writes of the same derived key are harmless.
    while (n_update < n_updates) {
        for (int64_t ir = 0; ir < r; ++ir) {
            dst_update_cells[n_update*r + ir] = dst_update_cells[ir];
        }
        for (int64_t sec = 0; sec < 4; ++sec) {
            dst_update_pos[sec*n_updates + n_update] = dst_update_pos[sec*n_updates];
        }
        dst_update_idxs[n_update] = dst_update_idxs[0];
        ++n_update;
    }

}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(),  st.is_new.end(),  0);
        std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);
    st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
    // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const auto & sq  = lay.seqs[s];
        const size_t rep = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + kpool - 1].second;
        if (st.rep_gen[rep] != st.generation) {
            st.rep_gen[rep] = st.generation;
            st.is_new[pool_start[s] + k] = st.generation;
            ++st.n_new;
        }
    };

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_pad(kpool_cur().n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            pcell[ip] = (int32_t) gcell(sq, rep);

            for (uint32_t k = 0; k < kpool; ++k) {
                pidx[(size_t) ip*kpool + k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                nrep[i_new] = gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        std::vector<int64_t> reps(pcell, pcell + pool_end.size());
        std::sort(reps.begin(), reps.end());

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (std::binary_search(reps.begin(), reps.end(), pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            std::fill(row + p0, row + p0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}
