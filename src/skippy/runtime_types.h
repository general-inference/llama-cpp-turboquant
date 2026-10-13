#pragma once

#include "skippy.h"

#include "../llama-context.h"
#include "../llama-model.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

enum skippy_glm_dsa_op_kind {
    SKIPPY_GLM_DSA_OP_INDEXER = 0,
    SKIPPY_GLM_DSA_OP_TOP_K = 1,
    SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL = 2,
    SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK = 3,
    SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD = 4,
    SKIPPY_GLM_DSA_OP_DSA_SPARSE_ATTN = 5,
    SKIPPY_GLM_DSA_OP_COMPACT_GET_ROWS = 6,
    SKIPPY_GLM_DSA_OP_MLA_ATTENTION = 7,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE = 8,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE_UP = 9,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE = 10,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_UP = 11,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_ACT = 12,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_DOWN = 13,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_WEIGHTED = 14,
    SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE = 15,
    SKIPPY_GLM_DSA_OP_SHARED_EXPERT = 16,
    SKIPPY_GLM_DSA_OP_COUNT = 17,
    SKIPPY_GLM_DSA_OP_UNKNOWN = 255,
};

struct skippy_glm_dsa_op_stat {
    uint64_t nodes = 0;
    int64_t elapsed_us = 0;
};

static constexpr size_t SKIPPY_GLM_DSA_TIMING_GROUP_LIMIT = 96;
static constexpr size_t SKIPPY_GLM_DSA_TIMING_GROUP_NAME_SIZE = 32;
static constexpr size_t SKIPPY_GLM_DSA_HOT_TENSOR_LIMIT = 24;
static constexpr size_t SKIPPY_GLM_DSA_HOT_TENSOR_NAME_SIZE = 64;

struct skippy_glm_dsa_op_group {
    bool used = false;
    char name[SKIPPY_GLM_DSA_TIMING_GROUP_NAME_SIZE] = {};
    skippy_glm_dsa_op_stat stats[SKIPPY_GLM_DSA_OP_COUNT] = {};
};

struct skippy_glm_dsa_hot_tensor {
    bool used = false;
    skippy_glm_dsa_op_kind kind = SKIPPY_GLM_DSA_OP_UNKNOWN;
    enum ggml_op op = GGML_OP_NONE;
    int64_t elapsed_us = 0;
    int64_t ne[4] = {};
    char name[SKIPPY_GLM_DSA_HOT_TENSOR_NAME_SIZE] = {};
};

struct skippy_pending_activation_part {
    skippy_activation_part_desc desc = {};
    std::vector<uint8_t> payload;
};

struct skippy_glm_dsa_op_timing {
    bool enabled = false;
    bool print_timing = false;
    bool trace_tensors = false;
    bool trace_stats = false;
    int32_t stage_index = -1;
    size_t token_count = 0;
    skippy_glm_dsa_op_kind pending_kind = SKIPPY_GLM_DSA_OP_UNKNOWN;
    int64_t pending_start_us = 0;
    uint32_t trace_value_limit = 8;
    uint32_t trace_node_limit = 32;
    uint32_t trace_stats_max_bytes = 32 * 1024 * 1024;
    uint32_t trace_nodes = 0;
    std::string trace_filter;
    skippy_glm_dsa_op_stat stats[SKIPPY_GLM_DSA_OP_COUNT] = {};
    skippy_glm_dsa_op_group groups[SKIPPY_GLM_DSA_TIMING_GROUP_LIMIT] = {};
    skippy_glm_dsa_hot_tensor hot_tensors[SKIPPY_GLM_DSA_HOT_TENSOR_LIMIT] = {};
};

struct skippy_model {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_model * mtp_model = nullptr;
    llama_context * mtp_ctx = nullptr;
    std::atomic<bool> context_work_pending{false};
    skippy_runtime_config config = {};
    std::vector<std::string> activation_import_identities;
    std::vector<const char *> activation_import_identity_ptrs;
    std::vector<std::string> activation_import_bindings;
    std::vector<const char *> activation_import_binding_ptrs;
    std::vector<std::string> activation_export_identities;
    std::vector<const char *> activation_export_identity_ptrs;
    std::vector<std::string> activation_export_bindings;
    std::vector<const char *> activation_export_binding_ptrs;
    std::vector<std::string> resident_tensor_names;
    std::string execution_contract;
    std::vector<const char *> resident_tensor_name_ptrs;
    skippy_glm_dsa_op_timing glm_dsa_timing = {};
    bool executable = true;
    bool has_input_activation_boundary = false;
    skippy_activation_boundary_desc input_activation_boundary = {};
    bool has_output_activation_boundary = false;
    skippy_activation_boundary_desc output_activation_boundary = {};
    uint32_t lane_count = 1;
    std::mutex lane_mutex;
    std::vector<bool> lane_in_use;
    std::vector<std::vector<llama_token>> lane_resident_prefix_tokens;
    std::mutex chat_templates_mutex;
    void * chat_templates = nullptr;
    void (*chat_templates_free)(void *) = nullptr;
};

struct skippy_verify_checkpoint {
    bool valid = false;
    int32_t token_start = 0;
    size_t token_count = 0;
    std::vector<uint8_t> state;
    std::vector<llama_token> token_ids;
    bool has_activation_input = false;
    skippy_activation_desc input_desc = {};
    std::vector<uint8_t> input_payload;
};

struct skippy_session {
    skippy_model * stage_model = nullptr;
    llama_context * ctx = nullptr;
    int32_t seq_id = 0;
    int32_t n_past = 0;
    bool borrowed_sequence = false;
    int32_t borrowed_prefix_tokens = 0;
    bool preserve_prefix_on_free = false;
    int32_t preserve_prefix_tokens = 0;
    std::vector<llama_token> token_history;
    std::vector<skippy_token_signal> signal_history;
    std::vector<llama_pos> prefill_positions;
    std::vector<int32_t> prefill_n_seq_ids;
    std::vector<llama_seq_id> prefill_seq_ids;
    std::vector<llama_seq_id *> prefill_seq_id_ptrs;
    std::vector<int8_t> prefill_logits;
    llama_sampler * sampling_chain = nullptr;
    llama_sampler * grammar_sampler = nullptr;
    llama_sampler * reasoning_budget_sampler = nullptr;
    bool grammar_lazy = false;
    bool sampling_backend_enabled = false;
    // A sampler chain may be backend-initialized at most once
    // (llama_sampler_chain_backend_init asserts on reuse). Once a chain is
    // detached -- e.g. so a sampled verify window can output every row --
    // it must never be re-attached; only building a fresh chain clears this.
    bool sampling_backend_retired = false;
    uint64_t sampling_reuse_hit_count = 0;
    uint64_t sampling_rebuild_count = 0;
    skippy_sampling_config sampling_config = {};
    bool sampling_config_valid = false;
    std::string chat_sampling_metadata;
    uint64_t grammar_generated_start = 0;
    size_t sampling_accepted_token_count = 0;
    int32_t mtp_next_pos = 0;
    bool mtp_prefix_valid = true;
    bool mtp_has_pending_h = false;
    std::vector<float> mtp_pending_h;
    bool mtp_has_pending_draft = false;
    llama_pos mtp_pending_draft_pos = 0;
    llama_token mtp_pending_draft_token = -1;
    // One past the highest position the last proposal wrote into a sidecar
    // cache of its own. Only the first drafted token is recorded above, so this
    // is the boundary the next sync prunes back to once the accepted prefix is
    // known.
    llama_pos mtp_pending_draft_end = 0;
    std::vector<skippy_pending_activation_part> pending_activation_exports;
    bool reset_required = false;
    std::vector<skippy_verify_checkpoint> verify_checkpoints;
};
