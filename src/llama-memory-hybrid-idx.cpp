#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        hparams_idx.n_embd_head_v_full = model.hparams.indexer_head_size;
        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, GGML_TYPE_F32, false, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
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
        if (!get_mem_recr()->prepare(ubatches)) {
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
    }

    qsa_blocks.clear();
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        qsa_blocks.clear();
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    qsa_blocks.clear();

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    qsa_blocks.clear();
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    qsa_blocks.clear();
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    qsa_blocks.clear();
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    qsa_blocks.clear();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
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
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    qsa_blocks.clear();
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    qsa_blocks.clear();
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
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
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

    int64_t n_update = 0;
    bool have_fallback = false;
    std::vector<int32_t> fallback_cells(r, 0);
    int32_t fallback_pos = 0;
    int64_t fallback_idx = 0;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem_idx->get_cells(seq_of_stream);
        int32_t * cur_cell_blk = dst_cell_blk != nullptr ? dst_cell_blk + s*n_kv : nullptr;
        // an incomplete block cannot be pooled; the bias below forces those tail cells in
        // -1 means no usable block, and block 0 only keeps the gather in range
        std::fill(blk_of.begin(),  blk_of.end(),  -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(cur_blk_cells.begin(), cur_blk_cells.end(), -1);
        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t   b = p/r;

            if (b >= n_blocks) {
                oor = true;
                continue;
            }

            blk_of[j] = (int32_t) b;
            cur_blk_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");


        // Derived block keys live in the row-major F32 V side of the indexer cache.
        // K and V use the same physical row id, so pick one member row as the block's
        // persistent destination and refresh it only when the block membership changes.
        const int64_t cache_stream = mem_idx->get_n_stream() == 1 ? 0 : seq_of_stream;
        const int64_t stream_off = cache_stream*(int64_t) mem_idx->get_size();
        auto & cached = mem->qsa_blocks[ratio][seq_of_stream];
        if (cached.size() < (size_t) n_blocks) {
            cached.resize(n_blocks);
        }

        for (int64_t b = 0; b < n_blocks; ++b) {
            // Incomplete/padding blocks are masked or force-selected as tail, so their
            // dot product is irrelevant. Keep the gather index in range.
            dst_block_key_cell[s*n_blocks + b] = (int32_t) stream_off;

            if (filled[b] != r) {
                continue;
            }

            llama_memory_hybrid_idx::qsa_block now;
            now.cells.assign(cur_blk_cells.begin() + b*r, cur_blk_cells.begin() + (b + 1)*r);

            uint32_t cache_cell = (uint32_t) cur_blk_cells[(b + 1)*r - 1];
            for (int64_t ir = 0; ir < r; ++ir) {
                const uint32_t cell = (uint32_t) cur_blk_cells[b*r + ir];
                if (cells.seq_count(cell) == 1) {
                    cache_cell = cell;
                    break;
                }
            }
            now.cache_cell = cache_cell;

            const int64_t global_dst = stream_off + cache_cell;
            GGML_ASSERT(global_dst <= std::numeric_limits<int32_t>::max());
            dst_block_key_cell[s*n_blocks + b] = (int32_t) global_dst;

            if (!have_fallback) {
                have_fallback = true;
                fallback_pos = (int32_t) (b*r);
                fallback_idx = global_dst;
                for (int64_t ir = 0; ir < r; ++ir) {
                    const int64_t global_src = stream_off + cur_blk_cells[b*r + ir];
                    GGML_ASSERT(global_src <= std::numeric_limits<int32_t>::max());
                    fallback_cells[ir] = (int32_t) global_src;
                }
            }

            if (cached[b] == now) {
                continue;
            }
            cached[b] = std::move(now);

            GGML_ASSERT(n_update < n_updates);
            for (int64_t ir = 0; ir < r; ++ir) {
                const int64_t global_src = stream_off + cur_blk_cells[b*r + ir];
                GGML_ASSERT(global_src <= std::numeric_limits<int32_t>::max());
                dst_update_cells[n_update*r + ir] = (int32_t) global_src;
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_update_pos[sec*n_updates + n_update] = (int32_t) (b*r);
            }
            dst_update_idxs[n_update++] = global_dst;
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
            const llama_pos    q      = ubatch->pos[i];

            // the tail is an incomplete block and is always visible, as in the reference
            const llama_pos tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = b*r >= tail_start ? 1e9f : (filled[b] < r ? -INFINITY : 0.0f);
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    v = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
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
            dst_update_pos[sec*n_updates] = have_fallback ? fallback_pos : 0;
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
