#pragma once

#include "skippy/common.h"
#include "skippy/runtime.h"

#include "llama-memory.h"

#include <cstdint>

struct llama_model;
struct skippy_model;
struct skippy_session;

inline bool skippy_runtime_has_stage_plan(const skippy_runtime_config * config) {
    return config != nullptr && config->resident_tensor_name_count > 0;
}

inline bool skippy_runtime_is_source_stage(const skippy_runtime_config * config) {
    return !skippy_runtime_has_stage_plan(config) || config->activation_import_identity_count == 0;
}

inline bool skippy_runtime_is_terminal_stage(const skippy_runtime_config * config) {
    return !skippy_runtime_has_stage_plan(config) || config->activation_export_identity_count == 0;
}

void skippy_disable_chat_grammar_after_exception(
        skippy_session * session,
        const char * operation,
        const char * message);

enum skippy_status skippy_require_usable_session(
        const skippy_session * session,
        skippy_error ** out_error);
int32_t skippy_stage_layer_count(const llama_model * model);
int32_t skippy_claim_execution_lane(skippy_model * model);
void skippy_release_execution_lane(skippy_model * model, int32_t seq_id);
void skippy_mark_context_work_pending(skippy_model * model);
void skippy_mark_context_synchronized(skippy_model * model);
void skippy_synchronize_context_if_pending(skippy_model * model);
bool skippy_memory_needs_verify_checkpoint(llama_memory_t memory);
void skippy_mtp_clear_session_state(skippy_session * session);
void skippy_mtp_trim_session_state(skippy_session * session, llama_pos token_count);
void skippy_clear_chat_sampling(skippy_session * session);
bool skippy_verify_window_can_roll_back_in_place(
        const skippy_session * session,
        size_t token_count);
enum skippy_status skippy_restore_verify_prefix(
        skippy_session * session,
        uint64_t token_count,
        bool * out_restored,
        skippy_error ** out_error);
