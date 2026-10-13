#pragma once

#include "skippy/common.h"
#include "skippy/sampling.h"

#include "llama.h"
#include "../vendor/nlohmann/json_fwd.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_context;
struct llama_model;
struct llama_sampler;
struct llama_vocab;
struct skippy_session;

void skippy_record_tokens(skippy_session * session, const llama_token * token_ids, size_t token_count);
void skippy_record_signal(skippy_session * session, int32_t logits_index);
enum skippy_status skippy_decode_tokens(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        bool request_logits,
        int32_t glm_dsa_phase_hint,
        skippy_error ** out_error);
enum skippy_status skippy_verify_token_batch(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        skippy_error ** out_error);
llama_token skippy_greedy_sample_context(const llama_model * model, llama_context * ctx, int32_t index);
llama_token skippy_greedy_sample_allowed_logits(
        const float * logits,
        const llama_vocab * vocab,
        int32_t n_vocab,
        const llama_token * suppress_tokens,
        int32_t suppress_token_count,
        bool ignore_eog);
llama_token skippy_greedy_sample_ith(skippy_session * session, int32_t index);
// Keep exhausted sampling out of public outputs and subsequent decode inputs.
enum skippy_status skippy_store_sampled_token(
        llama_token token, llama_token * output, skippy_error ** out_error);
llama_sampler * skippy_build_sampling_chain(skippy_session * session, const skippy_sampling_config * sampling);
std::vector<llama_logit_bias> skippy_add_vocab_suppress_token_biases(
        std::vector<llama_logit_bias> biases,
        const llama_token * suppress_tokens,
        int32_t suppress_token_count);
void skippy_refresh_backend_sampling(skippy_session * session, const skippy_sampling_config * sampling);
bool skippy_reuse_plain_chat_sampling(
        skippy_session * session,
        const skippy_sampling_config * sampling,
        const char * metadata_json,
        uint64_t prompt_token_count);
bool skippy_reset_reusable_sampling(skippy_session * session);
bool skippy_init_chat_grammar_sampler(
        skippy_session * session,
        const nlohmann::ordered_json & metadata,
        skippy_error ** out_error);
bool skippy_init_chat_reasoning_budget_sampler(
        skippy_session * session,
        const skippy_sampling_config * sampling,
        const nlohmann::ordered_json & metadata,
        skippy_error ** out_error);
llama_token skippy_sample_token_ith(
        skippy_session * session,
        const skippy_sampling_config * sampling,
        int32_t logits_index);
llama_token skippy_sample_token(skippy_session * session, const skippy_sampling_config * sampling);
