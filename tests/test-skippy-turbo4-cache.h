#pragma once

#include "../src/llama-io.h"
#include <cstring>
#include <stdexcept>

static_assert(SKIPPY_FEATURE_TURBO4_KV == (UINT64_C(1) << 41), "Q8_0/Turbo4 capability bit changed");
static_assert(SKIPPY_FEATURE_TURBO4_F16_KV == (UINT64_C(1) << 42), "F16/Turbo4 capability bit changed");

class turbo4_state_writer final : public llama_io_write_i {
public:
    std::vector<uint8_t> data;
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t start = data.size();
        data.resize(start + size);
        ggml_backend_tensor_get(tensor, data.data() + start, offset, size);
    }
    size_t n_bytes() override { return data.size(); }
};

class turbo4_state_reader final : public llama_io_read_i {
public:
    explicit turbo4_state_reader(const std::vector<uint8_t> & data) : data(data) {}
    void read(void * dst, size_t size) override {
        require(size);
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }
    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        require(size);
        ggml_backend_tensor_set(tensor, data.data() + offset, tensor_offset, size);
        offset += size;
    }
    size_t n_bytes() override { return offset; }
private:
    const std::vector<uint8_t> & data;
    size_t offset = 0;
    void require(size_t size) const {
        if (size > data.size() - offset) {
            throw std::runtime_error("truncated Turbo4 state fixture");
        }
    }
};

static bool turbo4_graph_capabilities() {
    ggml_init_params params = {ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(32, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 1);
    auto * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    auto * state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_TURBO4_0, 128, 4);
    ggml_set_input(input);
    ggml_set_input(rows);
    auto * write = ggml_set_rows(ctx.get(), state, input, rows);
    write->op_params[0] = 128;
    auto * output = ggml_turbo_wht(ctx.get(), input, 1, 128, nullptr);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, write);
    ggml_build_forward_expand(graph, output);
    llm_graph_result result(32);
    result.replace_with_stage_graph(graph, {}, {}, {
        {skippy_state_kind::kv_value, skippy_state_access::read, 0, state, nullptr},
        {skippy_state_kind::kv_value, skippy_state_access::write, 0, state, write, 0},
    });
    result.block_boundaries = {
        {skippy_block_boundary_kind::entry, 0, input},
        {skippy_block_boundary_kind::exit, 0, output},
    };
    const uint32_t expected = SKIPPY_MEMORY_CACHE_RESIDENT | SKIPPY_MEMORY_CACHE_KV_RECURRENT |
        SKIPPY_MEMORY_CACHE_FULL_STATE;
    if (skippy_graph_cache_capabilities(result) != expected) { return false; }
    result.state_effects.back().storage = input;
    if (skippy_graph_cache_capabilities(result) != 0) { return false; }
    result.state_effects.back().storage = state;
    result.block_boundaries.clear();
    return skippy_graph_cache_capabilities(result) == 0;
}

static bool turbo4_cache_lifecycle(ggml_type key_type) {
    if (SKIPPY_ABI_VERSION_PATCH != 73 || !(skippy_abi_features() & SKIPPY_FEATURE_TURBO4_KV) ||
            !(skippy_abi_features() & SKIPPY_FEATURE_TURBO4_F16_KV) ||
            !turbo4_graph_capabilities()) {
        return false;
    }
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    if (!model) { return false; }
    llama_hparams hp = {};
    hp.n_layer_all = 1;
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 128;
    hp.n_embd_head_k_swa = hp.n_embd_head_v_swa = 128;
    hp.n_head_arr[0] = 4;
    hp.n_head_kv_arr[0] = 2;
    const auto make_cache = [&](const llama_hparams & h, ggml_type k, ggml_type v, bool trans) {
        return std::make_unique<llama_kv_cache>(*model, h, k, v, trans, false, true, 4, 2, 1,
                0, LLAMA_SWA_TYPE_NONE, nullptr, llama_kv_cache::layer_filter_cb{},
                llama_kv_cache::layer_reuse_cb{}, llama_kv_cache::layer_share_cb{});
    };
    auto cache = make_cache(hp, key_type, GGML_TYPE_TURBO4_0, false);
    auto restore = make_cache(hp, key_type, GGML_TYPE_TURBO4_0, false);
    const uint32_t expected = SKIPPY_MEMORY_CACHE_RESIDENT | SKIPPY_MEMORY_CACHE_KV_RECURRENT |
        SKIPPY_MEMORY_CACHE_FULL_STATE;
    if (skippy_memory_cache_capabilities(cache.get()) != expected) { return false; }

    llama_pos positions[2] = {0, 1};
    llama_seq_id seq = 0;
    llama_seq_id * seqs[2] = {&seq, &seq};
    int32_t counts[2] = {1, 1};
    llama_ubatch batch = {};
    batch.n_tokens = 2; batch.n_pos = 1;
    batch.pos = positions; batch.n_seq_id = counts; batch.seq_id = seqs;
    llama_kv_cache::slot_info slot = {};
    slot.s0 = slot.s1 = 0; slot.strm = {0}; slot.idxs = {{1, 3}};
    cache->apply_ubatch(slot, batch);
    for (auto * tensor : {cache->get_k_storage(0), cache->get_v_storage(0)}) {
        std::vector<float> input(ggml_nelements(tensor));
        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 127) - 63)/64.0f; }
        std::vector<uint8_t> encoded(ggml_nbytes(tensor));
        if (ggml_quantize_chunk(tensor->type, input.data(), encoded.data(), 0, 4, 256, nullptr) != encoded.size()) {
            return false;
        }
        ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    }
    skippy_kv_page_desc desc = {};
    size_t bytes = 0;
    std::string error;
    const size_t key_row_bytes = key_type == GGML_TYPE_F16 ? 512 : 272;
    if (!cache->stage_export_kv_page(0, 0, 1, 0, 2, &desc, nullptr, 0, &bytes, error) ||
            desc.k_type != key_type || desc.v_type != GGML_TYPE_TURBO4_0 ||
            desc.k_row_bytes != key_row_bytes || desc.v_row_bytes != 132 || desc.flags != 0 ||
            bytes != 2*(key_row_bytes + 132)) {
        return false;
    }
    std::vector<uint8_t> page(bytes), roundtrip(bytes);
    if (!cache->stage_export_kv_page(0, 0, 1, 0, 2, &desc, page.data(), bytes, &bytes, error) ||
            !restore->stage_import_kv_page(0, desc, page.data(), bytes, error)) {
        return false;
    }
    const auto same_page = [&]() {
        skippy_kv_page_desc actual = {};
        size_t size = 0;
        return restore->stage_export_kv_page(0, 0, 1, 0, 2, &actual, roundtrip.data(), roundtrip.size(), &size, error) &&
            size == page.size() && roundtrip == page && actual.k_type == desc.k_type &&
            actual.k_row_bytes == desc.k_row_bytes &&
            actual.v_type == desc.v_type && actual.v_row_bytes == desc.v_row_bytes;
    };
    if (!same_page()) { return false; }
    for (int mutation = 0; mutation < 4; ++mutation) {
        auto invalid = desc;
        if (mutation == 0) { invalid.v_type = GGML_TYPE_Q8_0; }
        if (mutation == 1) { ++invalid.v_row_bytes; }
        if (mutation == 2) { ++invalid.layer_count; }
        if (mutation == 3) { invalid.k_type = key_type == GGML_TYPE_F16 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }
        if (restore->stage_import_kv_page(0, invalid, page.data(), bytes, error) || !same_page()) { return false; }
    }
    restore->seq_cp(0, 1, 0, 2);
    if (!restore->seq_rm(0, 0, 2) || restore->seq_pos_min(1) != 0) { return false; }
    restore->seq_cp(1, 0, 0, 2);
    if (!same_page()) { return false; }

    turbo4_state_writer snapshot;
    restore->state_write(snapshot);
    restore->clear(true);
    turbo4_state_reader reader(snapshot.data);
    restore->state_read(reader);
    if (reader.n_bytes() != snapshot.data.size() || !same_page() || restore->seq_pos_max(1) != 1) { return false; }
    restore->clear(true);
    if (restore->stage_export_kv_page(0, 0, 1, 0, 2, &desc, nullptr, 0, &bytes, error) ||
            !restore->stage_import_kv_page(0, desc, page.data(), page.size(), error) || !same_page()) {
        return false;
    }
    for (int invalid = 0; invalid < 5; ++invalid) {
        auto bad_hp = hp;
        auto k = key_type;
        bool trans = false;
        if (invalid == 0) { k = GGML_TYPE_TURBO4_0; }
        if (invalid == 1) { k = GGML_TYPE_F32; }
        if (invalid == 2) { trans = true; }
        if (invalid == 3) { bad_hp.n_embd_head_v_full = 64; }
        if (invalid == 4) { bad_hp.n_embd_head_k_mla_impl = bad_hp.n_embd_head_v_mla_impl = 128; }
        try {
            auto unexpected = make_cache(bad_hp, k, GGML_TYPE_TURBO4_0, trans);
            return false;
        } catch (const std::runtime_error &) {}
    }
    auto split_params = llama_model_default_params();
    split_params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    std::unique_ptr<llama_model> split_model(llama_model_create(LLM_ARCH_LLAMA, split_params));
    try {
        llama_kv_cache invalid(*split_model, hp, key_type, GGML_TYPE_TURBO4_0,
                false, false, true, 4, 2, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr, {}, {}, {});
        return false;
    } catch (const std::runtime_error &) {}
    return true;
}
