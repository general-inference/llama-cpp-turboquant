#include "skippy.h"
#include "../src/skippy/state_memory.h"

#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "../src/llama-graph.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "test-skippy-turbo4-cache.h"

// An unrecognized derived memory may add state that the base adapter omits.
class unknown_kv_memory : public llama_kv_cache {
public:
    using llama_kv_cache::llama_kv_cache;
};

static bool graph_cache_capabilities_cover_complete_state() {
    ggml_init_params params = {ggml_tensor_overhead() * 16 + 2 * ggml_graph_overhead_custom(32, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * state = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(input);
    auto * output = ggml_cpy(ctx.get(), input, state);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    llm_graph_result result(32);
    result.block_boundaries = {
        {skippy_block_boundary_kind::entry, 0, input},
        {skippy_block_boundary_kind::exit, 0, output},
    };
    const auto capabilities = [&](skippy_state_kind kind) {
        result.replace_with_stage_graph(graph, {}, {}, {
            {kind, skippy_state_access::read, 0, state, nullptr},
            {kind, skippy_state_access::write, 0, state, output, 0},
        });
        return skippy_graph_cache_capabilities(result);
    };
    const uint32_t dense = SKIPPY_MEMORY_CACHE_RESIDENT | SKIPPY_MEMORY_CACHE_KV_RECURRENT |
            SKIPPY_MEMORY_CACHE_FULL_STATE;
    if (capabilities(skippy_state_kind::kv_key) != dense ||
            capabilities(skippy_state_kind::kv_value) != dense ||
            capabilities(skippy_state_kind::recurrent_conv) != (SKIPPY_MEMORY_CACHE_KV_RECURRENT | SKIPPY_MEMORY_CACHE_FULL_STATE) ||
            capabilities(skippy_state_kind::recurrent_ssm) != (SKIPPY_MEMORY_CACHE_KV_RECURRENT | SKIPPY_MEMORY_CACHE_FULL_STATE) ||
            capabilities(skippy_state_kind::derived_persistent) != SKIPPY_MEMORY_CACHE_FULL_STATE) {
        return false;
    }
    capabilities(skippy_state_kind::kv_key);
    // Malformed ownership and missing graph admission must never advertise a
    // partial exporter, even when the underlying memory supports KV pages.
    result.state_effects.back().storage = input;
    if (skippy_graph_cache_capabilities(result) != 0) { return false; }
    capabilities(skippy_state_kind::kv_key);
    result.block_boundaries.clear();
    if (skippy_graph_cache_capabilities(result) != 0) { return false; }

    // An independently built stateless phase is compatible with every
    // admitted exporter; absence of effects is not a failure of ownership.
    auto * stateless_output = ggml_scale(ctx.get(), input, 2.0f);
    auto * stateless_graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(stateless_graph, stateless_output);
    result.replace_with_stage_graph(stateless_graph, {}, {}, {});
    result.block_boundaries = {
        {skippy_block_boundary_kind::entry, 0, input},
        {skippy_block_boundary_kind::exit, 0, stateless_output},
    };
    if (skippy_graph_cache_capabilities(result) != dense) { return false; }

    // Context-owned cross-attention state is missing even from full memory
    // snapshots. Each input owner declares that fact without a family lookup.
    result.add_input(std::make_unique<llm_graph_input_cross_embd>(nullptr));
    if (skippy_graph_cache_capabilities(result) != 0) { return false; }
    result.inputs.clear();
    result.add_input(std::make_unique<llm_graph_input_attn_cross>(nullptr));
    return skippy_graph_cache_capabilities(result) == 0;
}

int main() {
    if (!turbo4_cache_lifecycle(GGML_TYPE_Q8_0) || !turbo4_cache_lifecycle(GGML_TYPE_F16)) {
        fprintf(stderr, "Turbo4 cache lifecycle or graph capability contract failed\n");
        return 1;
    }
    fprintf(stderr, "Turbo4 cache lifecycle and graph capabilities: OK (ABI 0.1.73, feature bits 41/42, Q8_0/F16 K)\n");
    if (!graph_cache_capabilities_cover_complete_state()) {
        fprintf(stderr, "graph cache capabilities do not cover complete state\n");
        return 1;
    }
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    if (!model) {
        fprintf(stderr, "failed to create test model\n");
        return 1;
    }
    llama_hparams hparams = {};
    hparams.n_layer_all = 2;
    hparams.n_layer_nextn = 1;
    hparams.n_embd_head_k_full = 4;
    hparams.n_embd_head_v_full = 4;
    hparams.n_embd_head_k_swa = 4;
    hparams.n_embd_head_v_swa = 4;
    hparams.n_embd_head_k_mla_impl = 4;
    hparams.n_embd_head_v_mla_impl = 4;
    hparams.n_head_arr[0] = 1;
    hparams.n_head_kv_arr[0] = 1;

    llama_kv_cache kv(
            *model,
            hparams,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            true,
            false,
            true,
            1,
            1,
            1,
            0,
            LLAMA_SWA_TYPE_NONE,
            nullptr,
            [](int32_t il) { return il == 0; },
            {},
            {});

    if (skippy_memory_cache_capabilities(&kv) !=
            (SKIPPY_MEMORY_CACHE_RESIDENT | SKIPPY_MEMORY_CACHE_KV_RECURRENT | SKIPPY_MEMORY_CACHE_FULL_STATE) ||
            skippy_model_memory_cache_capabilities(nullptr) != 0) {
        return 1;
    }
    unknown_kv_memory unknown(*model, hparams, GGML_TYPE_F32, GGML_TYPE_F32,
            true, false, true, 1, 1, 1, 0, LLAMA_SWA_TYPE_NONE, nullptr,
            [](int32_t il) { return il == 0; }, {}, {});
    if (skippy_memory_cache_capabilities(&unknown) != SKIPPY_MEMORY_CACHE_FULL_STATE) {
        return 1;
    }

    llama_pos pos = 0;
    llama_seq_id seq_id = 0;
    llama_seq_id * seq_ids = &seq_id;
    int32_t n_seq_id = 1;
    llama_ubatch ubatch = {};
    ubatch.n_tokens = 1;
    ubatch.n_pos = 1;
    ubatch.pos = &pos;
    ubatch.n_seq_id = &n_seq_id;
    ubatch.seq_id = &seq_ids;

    llama_kv_cache::slot_info slot = {};
    slot.s0 = 0;
    slot.s1 = 0;
    slot.strm = { 0 };
    slot.idxs = {{ 0 }};
    kv.apply_ubatch(slot, ubatch);

    skippy_kv_page_desc desc = {};
    size_t payload_bytes = 0;
    std::string error;
    if (!kv.stage_export_kv_page(
                0,
                0,
                1,
                0,
                1,
                &desc,
                nullptr,
                0,
                &payload_bytes,
                error)) {
        fprintf(stderr, "K-only KV page sizing failed: %s\n", error.c_str());
        return 1;
    }
    if ((desc.flags & SKIPPY_KV_PAGE_FLAG_V_TRANSPOSED) == 0 ||
            desc.v_row_bytes != 0 ||
            desc.v_element_bytes != 0 ||
            payload_bytes == 0) {
        fprintf(stderr, "unexpected K-only transposed KV page descriptor\n");
        return 1;
    }

    std::vector<uint8_t> payload(payload_bytes);
    if (!kv.stage_export_kv_page(
                0,
                0,
                1,
                0,
                1,
                &desc,
                payload.data(),
                payload.size(),
                &payload_bytes,
                error)) {
        fprintf(stderr, "K-only KV page export failed: %s\n", error.c_str());
        return 1;
    }

    if (payload_bytes != payload.size()) {
        fprintf(stderr, "K-only KV page export byte count mismatch\n");
        return 1;
    }

    llama_kv_cache restore_kv(
            *model,
            hparams,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            true,
            false,
            true,
            1,
            1,
            1,
            0,
            LLAMA_SWA_TYPE_NONE,
            nullptr,
            [](int32_t il) { return il == 0; },
            {},
            {});
    restore_kv.apply_ubatch(slot, ubatch);

    desc.layer_end = 2;
    if (!restore_kv.stage_import_kv_page(
                0,
                desc,
                payload.data(),
                payload.size(),
                error)) {
        fprintf(stderr, "K-only KV page import failed: %s\n", error.c_str());
        return 1;
    }

    uint8_t empty_payload = 0;
    skippy_kv_page_desc invalid_range_desc = desc;
    invalid_range_desc.layer_start = 1;
    invalid_range_desc.layer_end = 2;
    invalid_range_desc.layer_count = 0;
    invalid_range_desc.payload_bytes = 0;
    if (restore_kv.stage_import_kv_page(
                0,
                invalid_range_desc,
                &empty_payload,
                0,
                error)) {
        fprintf(stderr, "out-of-bounds KV page layer range was accepted\n");
        return 1;
    }
    if (restore_kv.seq_pos_min(0) != 0) {
        fprintf(stderr, "out-of-bounds KV page import mutated the existing page\n");
        return 1;
    }
    error.clear();

    llama_kv_cache filtered_kv(
            *model,
            hparams,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            true,
            false,
            true,
            1,
            1,
            1,
            0,
            LLAMA_SWA_TYPE_NONE,
            nullptr,
            [](int32_t) { return false; },
            {},
            {});
    filtered_kv.apply_ubatch(slot, ubatch);

    skippy_kv_page_desc filtered_export_desc = {};
    size_t filtered_export_bytes = 0;
    if (filtered_kv.stage_export_kv_page(
                0,
                1,
                2,
                1,
                1,
                &filtered_export_desc,
                nullptr,
                0,
                &filtered_export_bytes,
                error)) {
        fprintf(stderr, "empty KV page layer selection was exported\n");
        return 1;
    }
    if (error != "no KV cache layers selected by layer range") {
        fprintf(stderr, "empty KV page layer selection returned the wrong error: %s\n", error.c_str());
        return 1;
    }
    error.clear();

    skippy_kv_page_desc empty_range_desc = desc;
    empty_range_desc.layer_count = 0;
    empty_range_desc.payload_bytes = 0;
    if (filtered_kv.stage_import_kv_page(
                0,
                empty_range_desc,
                &empty_payload,
                0,
                error)) {
        fprintf(stderr, "empty KV page layer selection was accepted\n");
        return 1;
    }
    if (filtered_kv.seq_pos_min(0) != 0) {
        fprintf(stderr, "empty KV page layer selection mutated the existing page\n");
        return 1;
    }
    error.clear();

    restore_kv.seq_cp(0, 1, 0, 1);
    if (restore_kv.stage_import_kv_page(
                0,
                desc,
                payload.data(),
                payload.size(),
                error)) {
        fprintf(stderr, "KV page import unexpectedly allocated a shared occupied cell\n");
        return 1;
    }
    if (restore_kv.seq_pos_min(0) != 0 || restore_kv.seq_pos_min(1) != 0) {
        fprintf(stderr, "failed KV page allocation did not preserve the existing page\n");
        return 1;
    }
    error.clear();

    skippy_kv_page_desc roundtrip_desc = {};
    size_t roundtrip_bytes = 0;
    if (!restore_kv.stage_export_kv_page(
                0,
                0,
                1,
                0,
                1,
                &roundtrip_desc,
                nullptr,
                0,
                &roundtrip_bytes,
                error)) {
        fprintf(stderr, "K-only KV page round-trip sizing failed: %s\n", error.c_str());
        return 1;
    }
    if (roundtrip_bytes != payload.size() ||
            roundtrip_desc.flags != desc.flags ||
            roundtrip_desc.k_row_bytes != desc.k_row_bytes ||
            roundtrip_desc.v_row_bytes != 0 ||
            roundtrip_desc.v_element_bytes != 0) {
        fprintf(stderr, "unexpected K-only transposed KV page round-trip descriptor\n");
        return 1;
    }
    std::vector<uint8_t> roundtrip_payload(roundtrip_bytes);
    if (!restore_kv.stage_export_kv_page(
                0,
                0,
                1,
                0,
                1,
                &roundtrip_desc,
                roundtrip_payload.data(),
                roundtrip_payload.size(),
                &roundtrip_bytes,
                error)) {
        fprintf(stderr, "K-only KV page round-trip export failed: %s\n", error.c_str());
        return 1;
    }
    if (roundtrip_payload != payload) {
        fprintf(stderr, "K-only KV page payload changed after round trip\n");
        return 1;
    }

    return 0;
}
