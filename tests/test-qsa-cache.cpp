#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

static llama_ubatch make_batch(uint32_t n, llama_pos p0, llama_seq_id seq_id = 0) {
    llama_ubatch batch = {};
    batch.b_equal_seqs = 1;
    batch.n_tokens = batch.n_seq_tokens = n;
    batch.n_seqs = batch.n_seqs_unq = 1;
    batch.n_pos = 4;
    batch.data = std::make_shared<llama_ubatch::data_t>();
    auto & data = *batch.data;
    data.token.assign(n, 1);
    data.pos.resize(4*n);
    data.n_seq_id.assign(n, 1);
    data.seq_id_data.assign(n, seq_id);
    data.seq_id.resize(n);
    data.seq_id_unq = {seq_id};
    data.output.assign(n, 0);
    for (uint32_t i = 0; i < n; ++i) {
        data.seq_id[i] = &data.seq_id_data[i];
        for (uint32_t section = 0; section < 4; ++section) {
            data.pos[section*n + i] = p0 + i;
        }
    }
    batch.token = data.token.data();
    batch.pos = data.pos.data();
    batch.n_seq_id = data.n_seq_id.data();
    batch.seq_id = data.seq_id.data();
    batch.seq_id_unq = data.seq_id_unq.data();
    batch.output = data.output.data();
    return batch;
}

struct fixture {
    std::unique_ptr<llama_model> model;
    std::unique_ptr<llama_memory_hybrid_idx> mem;

    explicit fixture(bool unified = true, ggml_type type = GGML_TYPE_F16) :
        model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params())) {
        auto & hp = model->hparams;
        hp.n_layer_all = 1;
        hp.n_embd = hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
        hp.n_head_arr[0] = hp.n_head_kv_arr[0] = 1;
        hp.indexer_head_size = 32;
        hp.dsv4_compress_ratios[0] = 4;
        mem.reset(new llama_memory_hybrid_idx(*model,
                type, type, false, 512, 1, 0, LLAMA_SWA_TYPE_NONE,
                GGML_TYPE_F32, GGML_TYPE_F32, 2, 2, 0, false, unified,
                [](uint32_t) { return true; }, [](uint32_t) { return false; }, [](uint32_t) { return true; }));
    }

    std::unique_ptr<llama_memory_hybrid_idx_context> append(const llama_ubatch & batch,
            const llama_kv_cache::slot_info * forced = nullptr) {
        auto sinfo = forced ? *forced : mem->get_mem_attn()->find_slot(batch, false);
        GGML_ASSERT(!sinfo.empty());
        GGML_ASSERT(mem->get_mem_recr()->prepare({batch}));
        auto ctx = std::make_unique<llama_memory_hybrid_idx_context>(mem.get(),
                llama_kv_cache::slot_info_vec_t{sinfo}, llama_kv_cache::slot_info_vec_t{sinfo},
                std::vector<llama_ubatch>{batch});
        GGML_ASSERT(ctx->apply());
        return ctx;
    }
};

struct inputs {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * cells;
    ggml_tensor * bias;
    ggml_tensor * keys;
    ggml_tensor * update_cells;
    ggml_tensor * update_pos;
    ggml_tensor * update_idxs;

    inputs(const llama_memory_hybrid_idx_context & mctx, const llama_ubatch & batch, uint32_t ratio = 4) {
        const uint32_t n_kv = mctx.get_idx()->get_n_kv();
        const uint32_t n_blocks = (n_kv + ratio - 1)/ratio;
        const uint32_t n_updates = mctx.get_qsa_update_capacity(batch, ratio, n_blocks);
        ctx = ggml_init({16*ggml_tensor_overhead(), nullptr, true});
        cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, 1);
        bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, batch.n_tokens, 1);
        keys = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_blocks, 1);
        update_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ratio, n_updates);
        update_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4*n_updates);
        update_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_updates);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(
                ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)));
        GGML_ASSERT(buffer);
        mctx.set_input_qsa(cells, nullptr, nullptr, bias, keys, update_cells, update_pos, update_idxs, &batch, ratio, true);
    }

    ~inputs() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    uint32_t capacity() const { return update_idxs->ne[0]; }

    std::vector<int32_t> key_slots(uint32_t n) const {
        const auto * data = (const int32_t *) keys->data;
        return {data, data + n};
    }

    std::vector<int32_t> refreshed_positions() const {
        const auto * data = (const int32_t *) update_pos->data;
        std::vector<int32_t> positions(data, data + capacity());
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
        return positions;
    }
};

static void test_tail(bool unified, ggml_type type) {
    fixture f(unified, type);
    const llama_seq_id seq_id = unified ? 0 : 1;
    auto batch = make_batch(128, 0, seq_id);
    auto ctx = f.append(batch);
    inputs cold(*ctx, batch);
    const auto prefix = cold.key_slots(16);
    GGML_ASSERT(cold.refreshed_positions().size() == 32);

    GGML_ASSERT(f.mem->seq_rm(seq_id, 66, -1));
    batch = make_batch(1, 66, seq_id);
    ctx = f.append(batch);
    inputs tail(*ctx, batch);
    GGML_ASSERT(tail.capacity() == 1);
    GGML_ASSERT(tail.key_slots(16) == prefix);

    batch = make_batch(1, 67, seq_id);
    ctx = f.append(batch);
    inputs complete(*ctx, batch);
    GGML_ASSERT(complete.capacity() == 1);
    GGML_ASSERT(complete.key_slots(16) == prefix);
    GGML_ASSERT(complete.refreshed_positions() == std::vector<int32_t>{64});

    for (int i = 0; i < 8; ++i) {
        GGML_ASSERT(f.mem->seq_rm(seq_id, 67, -1));
        ctx = f.append(batch);
        inputs replay(*ctx, batch);
        GGML_ASSERT(replay.capacity() == 1);
        GGML_ASSERT(replay.key_slots(16) == prefix);
        GGML_ASSERT(replay.refreshed_positions() == std::vector<int32_t>{64});
    }

    // A no-op tail removal must not turn the prefix cold either.
    GGML_ASSERT(f.mem->seq_rm(seq_id, 68, -1));
    batch = make_batch(1, 68, seq_id);
    ctx = f.append(batch);
    inputs noop(*ctx, batch);
    GGML_ASSERT(noop.capacity() == 1);
    GGML_ASSERT(noop.key_slots(16) == prefix);

    // Prefix edits still reset grouping.
    GGML_ASSERT(f.mem->seq_rm(seq_id, 0, 4));
    GGML_ASSERT(ctx->get_qsa_update_capacity(batch, 4, 64) == 64);
}

static void test_deferred() {
    fixture f;
    auto batch = make_batch(64, 0);
    auto ctx = f.append(batch);
    inputs ratio4(*ctx, batch, 4);
    inputs ratio2(*ctx, batch, 2);
    ctx->reset_qsa(4);
    GGML_ASSERT(ctx->get_qsa_update_capacity(batch, 4, 64) == 64);
    GGML_ASSERT(ctx->get_qsa_update_capacity(batch, 2, 128) < 128);
    batch = make_batch(64, 64);
    ctx = f.append(batch);
    inputs resumed(*ctx, batch, 4);
    GGML_ASSERT(resumed.refreshed_positions().size() == 32);
}

static void test_rewrite() {
    fixture f;
    auto batch = make_batch(64, 0);
    auto ctx = f.append(batch);
    inputs initial(*ctx, batch);
    f.mem->seq_cp(0, 1, -1, -1);
    inputs shared0(*ctx, batch);
    auto other = make_batch(1, 63, 1);
    inputs shared1(*ctx, other);
    GGML_ASSERT(shared0.key_slots(16) == shared1.key_slots(16));

    // Keep a cached alias while arranging an occupied row rewrite in the other sequence.
    f.mem->get_mem_attn()->seq_rm(1, -1, -1);
    f.mem->get_mem_idx()->seq_rm(1, -1, -1);
    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = sinfo.s1 = 0;
    sinfo.strm = {0};
    sinfo.idxs = {{63}};
    batch = make_batch(1, 63);
    ctx = f.append(batch, &sinfo);
    GGML_ASSERT(ctx->get_qsa_update_capacity(batch, 4, 64) == 2);
    GGML_ASSERT(f.mem->get_qsa_update_capacity(other, 4, 64) == 2);
}

static void test_shared_tail() {
    fixture f;
    auto batch = make_batch(64, 0);
    auto ctx = f.append(batch);
    f.mem->seq_cp(0, 1, -1, -1);
    inputs prefix(*ctx, batch);
    GGML_ASSERT(f.mem->seq_rm(0, 60, -1));
    GGML_ASSERT(f.mem->get_qsa_update_capacity(batch, 4, 64) == 64);
}

int main() {
    llama_backend_init();
    test_tail(true, GGML_TYPE_F16);
    test_tail(false, GGML_TYPE_F16);
    test_tail(true, GGML_TYPE_Q8_0);
    test_tail(false, GGML_TYPE_Q8_0);
    test_deferred();
    test_rewrite();
    test_shared_tail();
    llama_backend_free();
    puts("QSA cache: PASS");
    return 0;
}
