#include "skippy/sampling.h"
#include "skippy/errors.h"
#include "skippy/execution_internal.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"
#include "../../common/reasoning-budget.h"
#include "skippy/sampling_internal.h"
#include "skippy/speculative_internal.h"

#include "llama-impl.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

using json = nlohmann::ordered_json;

void skippy_record_tokens(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count) {
    if (session != nullptr && token_ids != nullptr && token_count > 0) {
        session->token_history.insert(session->token_history.end(), token_ids, token_ids + token_count);
    }
}

void skippy_record_signal(skippy_session * session, int32_t logits_index);

enum skippy_status skippy_decode_tokens(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        bool request_logits,
        int32_t glm_dsa_phase_hint,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || token_ids == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require activation input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    if (n_tokens == 1) {
        llama_token token = token_ids[0];
        llama_pos pos = session->n_past;
        int32_t n_seq_id = 1;
        llama_seq_id seq_id = session->seq_id;
        llama_seq_id * seq_ids = &seq_id;
        int8_t logits = request_logits ? 1 : 0;
        llama_batch batch = {
            /*n_tokens =*/ 1,
            /*token    =*/ &token,
            /*embd     =*/ nullptr,
            /*pos      =*/ &pos,
            /*n_seq_id =*/ &n_seq_id,
            /*seq_id   =*/ &seq_ids,
            /*logits   =*/ &logits,
        };
        enum skippy_status status = skippy_decode_batch(session, batch, 1, glm_dsa_phase_hint, out_error);
        if (status == SKIPPY_STATUS_OK) {
            skippy_record_tokens(session, token_ids, token_count);
            status = skippy_mtp_sync_target_inputs(session, token_ids, nullptr, token_count, pos, out_error);
        }
        return status;
    }

    const llama_pos token_start = session->n_past;
    session->prefill_positions.resize(token_count);
    session->prefill_n_seq_ids.resize(token_count, 1);
    session->prefill_seq_ids.resize(token_count, session->seq_id);
    session->prefill_seq_id_ptrs.resize(token_count);
    session->prefill_logits.resize(token_count, 0);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const size_t index = static_cast<size_t>(i);
        session->prefill_positions[index] = session->n_past + i;
        session->prefill_n_seq_ids[index] = 1;
        session->prefill_seq_ids[index] = session->seq_id;
        session->prefill_seq_id_ptrs[index] = &session->prefill_seq_ids[index];
        session->prefill_logits[index] = request_logits && i == n_tokens - 1 ? 1 : 0;
    }
    llama_batch batch = {
        /*n_tokens =*/ n_tokens,
        /*token    =*/ const_cast<llama_token *>(token_ids),
        /*embd     =*/ nullptr,
        /*pos      =*/ session->prefill_positions.data(),
        /*n_seq_id =*/ session->prefill_n_seq_ids.data(),
        /*seq_id   =*/ session->prefill_seq_id_ptrs.data(),
        /*logits   =*/ session->prefill_logits.data(),
    };

    enum skippy_status status = skippy_decode_batch(session, batch, token_count, glm_dsa_phase_hint, out_error);
    if (status == SKIPPY_STATUS_OK) {
        skippy_record_tokens(session, token_ids, token_count);
        status = skippy_mtp_sync_target_inputs(session, token_ids, nullptr, token_count, token_start, out_error);
    }
    return status;
}

enum skippy_status skippy_verify_token_batch(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || token_ids == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require activation input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (session->sampling_backend_enabled) {
        // Verification requests logits for every row and then consumes them on
        // the CPU. An attached backend sampler permits only one output row per
        // sequence, so retire its one-shot backend initialization before
        // building the verification batch.
        llama_set_sampler(session->ctx, session->seq_id, nullptr);
        session->sampling_backend_enabled = false;
        session->sampling_backend_retired = true;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = token_ids[i];
        batch.pos[i] = session->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = session->seq_id;
        batch.logits[i] = 1;
    }

    const llama_pos token_start = session->n_past;
    enum skippy_status status = skippy_decode_batch(
            session,
            batch,
            token_count,
            SKIPPY_GLM_DSA_PHASE_HINT_VERIFY,
            out_error);
    llama_batch_free(batch);
    if (status == SKIPPY_STATUS_OK) {
        skippy_record_tokens(session, token_ids, token_count);
        status = skippy_mtp_sync_target_inputs(session, token_ids, nullptr, token_count, token_start, out_error);
    }
    return status;
}

static llama_token skippy_greedy_sample_context_with_eog_policy(
        const llama_model * model,
        llama_context * ctx,
        int32_t index,
        bool ignore_eog) {
    if (model == nullptr || ctx == nullptr) {
        return 0;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t suppress_token_count = 0;
    const llama_token * suppress_tokens = llama_vocab_get_suppress_tokens(vocab, &suppress_token_count);
    llama_synchronize(ctx);
    const float * logits = llama_get_logits_ith(ctx, index);
    if (logits == nullptr) {
        return 0;
    }

#if defined(__APPLE__)
    // Accelerate preserves the scalar path's first-index tie behavior while
    // avoiding a branch-heavy full-vocabulary scan on every generated row.
    float accelerated_best_logit = -std::numeric_limits<float>::infinity();
    vDSP_Length best_index = 0;
    vDSP_maxvi(
            logits,
            1,
            &accelerated_best_logit,
            &best_index,
            static_cast<vDSP_Length>(n_vocab));
    const bool accelerated_suppressed = suppress_tokens != nullptr && suppress_token_count > 0 &&
            std::find(suppress_tokens, suppress_tokens + suppress_token_count,
                    static_cast<llama_token>(best_index)) != suppress_tokens + suppress_token_count;
    if (!accelerated_suppressed &&
            (!ignore_eog || !llama_vocab_is_eog(vocab, static_cast<llama_token>(best_index)))) {
        return static_cast<llama_token>(best_index);
    }
#endif

    return skippy_greedy_sample_allowed_logits(
            logits, vocab, n_vocab, suppress_tokens, suppress_token_count, ignore_eog);
}

llama_token skippy_greedy_sample_context(
        const llama_model * model,
        llama_context * ctx,
        int32_t index) {
    return skippy_greedy_sample_context_with_eog_policy(model, ctx, index, false);
}

llama_token skippy_greedy_sample_ith(skippy_session * session, int32_t index) {
    return skippy_greedy_sample_context(session->stage_model->model, session->ctx, index);
}

static llama_token skippy_greedy_sample(skippy_session * session) {
    return skippy_greedy_sample_ith(session, -1);
}

static bool skippy_compute_token_signal_context(
        const llama_model * model,
        llama_context * ctx,
        int32_t logits_index,
        skippy_token_signal * out_signal) {
    if (model == nullptr || ctx == nullptr || out_signal == nullptr) {
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, logits_index);
    if (logits == nullptr || n_vocab <= 0) {
        return false;
    }

    int32_t top_token = 0;
    int32_t second_token = 0;
    float top_logit = -std::numeric_limits<float>::infinity();
    float second_logit = -std::numeric_limits<float>::infinity();
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < n_vocab; ++token) {
        const float logit = logits[token];
        if (logit > max_logit) {
            max_logit = logit;
        }
        if (logit > top_logit) {
            second_logit = top_logit;
            second_token = top_token;
            top_logit = logit;
            top_token = token;
        } else if (logit > second_logit) {
            second_logit = logit;
            second_token = token;
        }
    }

    if (!std::isfinite(max_logit)) {
        return false;
    }

    double sum_exp = 0.0;
    double entropy = 0.0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        sum_exp += std::exp(static_cast<double>(logits[token] - max_logit));
    }
    if (sum_exp <= 0.0 || !std::isfinite(sum_exp)) {
        return false;
    }
    const double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);
    for (int32_t token = 0; token < n_vocab; ++token) {
        const double logprob = static_cast<double>(logits[token]) - log_sum_exp;
        const double prob = std::exp(logprob);
        entropy -= prob * logprob;
    }

    const float top_logprob = static_cast<float>(static_cast<double>(top_logit) - log_sum_exp);
    const float second_logprob = static_cast<float>(static_cast<double>(second_logit) - log_sum_exp);
    *out_signal = {};
    out_signal->entropy = static_cast<float>(entropy);
    out_signal->top_logprob = top_logprob;
    out_signal->second_logprob = second_logprob;
    out_signal->margin = top_logprob - second_logprob;
    out_signal->top_token = top_token;
    out_signal->second_token = second_token;
    return true;
}

static bool skippy_compute_token_signal(
        skippy_session * session,
        int32_t logits_index,
        skippy_token_signal * out_signal) {
    if (session == nullptr || session->stage_model == nullptr) {
        return false;
    }
    return skippy_compute_token_signal_context(
            session->stage_model->model,
            session->ctx,
            logits_index,
            out_signal);
}

void skippy_record_signal(skippy_session * session, int32_t logits_index) {
    skippy_token_signal signal = {};
    if (skippy_compute_token_signal(session, logits_index, &signal)) {
        session->signal_history.push_back(signal);
    }
}

void skippy_clear_chat_sampling(skippy_session * session) {
    if (session == nullptr) {
        return;
    }
    if (session->ctx != nullptr) {
        llama_set_sampler(session->ctx, session->seq_id, nullptr);
    }
    llama_sampler_free(session->sampling_chain);
    llama_sampler_free(session->grammar_sampler);
    llama_sampler_free(session->reasoning_budget_sampler);
    session->sampling_chain = nullptr;
    session->grammar_sampler = nullptr;
    session->reasoning_budget_sampler = nullptr;
    session->grammar_lazy = false;
    session->sampling_backend_enabled = false;
    session->sampling_backend_retired = false;
    session->sampling_config = {};
    session->sampling_config_valid = false;
    session->chat_sampling_metadata.clear();
    session->grammar_generated_start = 0;
    session->sampling_accepted_token_count = 0;
}

static std::string skippy_regex_escape(const std::string & value) {
    static const std::regex special_chars(R"([-[\]{}()*+?.,\^$|#\s])");
    return std::regex_replace(value, special_chars, R"(\$&)");
}

static std::vector<llama_token> skippy_tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special) {
    if (vocab == nullptr || text.empty()) {
        return {};
    }
    const int32_t count = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (count <= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<size_t>(count));
    const int32_t written = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), count, add_special, parse_special);
    if (written < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(written));
    return tokens;
}

static bool skippy_sampling_enabled(const skippy_sampling_config * sampling) {
    if (sampling == nullptr || sampling->version == 0 ||
            (sampling->flags & SKIPPY_SAMPLING_FLAG_ENABLED) == 0) {
        return false;
    }
    return true;
}

static bool skippy_sampling_is_greedy_equivalent(const skippy_sampling_config * sampling) {
    if (!skippy_sampling_enabled(sampling)) {
        return true;
    }
    const float repeat_penalty = sampling->repeat_penalty == 0.0f ? 1.0f : sampling->repeat_penalty;
    const bool ignore_eos = sampling->ignore_eos != 0 ||
            (sampling->flags & SKIPPY_SAMPLING_FLAG_IGNORE_EOS) != 0;
    return sampling->temperature <= 0.0f &&
           !ignore_eos &&
           sampling->presence_penalty == 0.0f &&
           sampling->frequency_penalty == 0.0f &&
           repeat_penalty == 1.0f &&
           sampling->logit_bias_count == 0;
}

static bool skippy_sampling_configs_equal(
        const skippy_sampling_config & lhs,
        const skippy_sampling_config & rhs) {
    if (lhs.version != rhs.version ||
            lhs.flags != rhs.flags ||
            lhs.seed != rhs.seed ||
            lhs.top_k != rhs.top_k ||
            lhs.penalty_last_n != rhs.penalty_last_n ||
            lhs.temperature != rhs.temperature ||
            lhs.top_p != rhs.top_p ||
            lhs.presence_penalty != rhs.presence_penalty ||
            lhs.frequency_penalty != rhs.frequency_penalty ||
            lhs.repeat_penalty != rhs.repeat_penalty ||
            lhs.logit_bias_count != rhs.logit_bias_count ||
            lhs.min_p != rhs.min_p ||
            lhs.typical_p != rhs.typical_p ||
            lhs.top_nsigma != rhs.top_nsigma ||
            lhs.dynatemp_range != rhs.dynatemp_range ||
            lhs.dynatemp_exponent != rhs.dynatemp_exponent ||
            lhs.dry_multiplier != rhs.dry_multiplier ||
            lhs.dry_base != rhs.dry_base ||
            lhs.dry_allowed_length != rhs.dry_allowed_length ||
            lhs.dry_penalty_last_n != rhs.dry_penalty_last_n ||
            lhs.xtc_probability != rhs.xtc_probability ||
            lhs.xtc_threshold != rhs.xtc_threshold ||
            lhs.mirostat_mode != rhs.mirostat_mode ||
            lhs.mirostat_entropy != rhs.mirostat_entropy ||
            lhs.mirostat_learning_rate != rhs.mirostat_learning_rate ||
            lhs.sampler_count != rhs.sampler_count ||
            lhs.reasoning_budget_tokens != rhs.reasoning_budget_tokens ||
            lhs.ignore_eos != rhs.ignore_eos ||
            lhs.dry_sequence_breaker_count != rhs.dry_sequence_breaker_count) {
        return false;
    }

    const uint32_t sampler_count = std::min<uint32_t>(lhs.sampler_count, SKIPPY_MAX_SAMPLERS);
    for (uint32_t i = 0; i < sampler_count; ++i) {
        if (lhs.samplers[i] != rhs.samplers[i]) {
            return false;
        }
    }
    const uint32_t breaker_count = std::min<uint32_t>(
            lhs.dry_sequence_breaker_count,
            SKIPPY_MAX_DRY_SEQUENCE_BREAKERS);
    for (uint32_t i = 0; i < breaker_count; ++i) {
        if (std::strncmp(
                    lhs.dry_sequence_breakers[i],
                    rhs.dry_sequence_breakers[i],
                    SKIPPY_MAX_DRY_SEQUENCE_BREAKER_BYTES) != 0) {
            return false;
        }
    }

    const uint32_t logit_bias_count = std::min<uint32_t>(lhs.logit_bias_count, SKIPPY_MAX_LOGIT_BIAS);
    for (uint32_t i = 0; i < logit_bias_count; ++i) {
        if (lhs.logit_bias[i].token != rhs.logit_bias[i].token ||
                lhs.logit_bias[i].bias != rhs.logit_bias[i].bias) {
            return false;
        }
    }
    return true;
}

static bool skippy_sampling_chain_is_reusable_for(
        const skippy_session * session,
        const skippy_sampling_config * sampling) {
    if (session == nullptr || !session->sampling_config_valid ||
            session->sampling_backend_retired) {
        return false;
    }
    if (sampling != nullptr &&
            skippy_sampling_configs_equal(session->sampling_config, *sampling)) {
        return true;
    }
    // Greedy-equivalent configurations all build the same one-node greedy
    // chain. Fields such as seed and top_p are semantically inactive while
    // temperature is zero, so changing them must not detach the sampler and
    // force an otherwise identical scheduler reservation.
    return skippy_sampling_is_greedy_equivalent(&session->sampling_config) &&
           skippy_sampling_is_greedy_equivalent(sampling);
}

static void skippy_store_sampling_config(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    session->sampling_config = sampling != nullptr ?
            *sampling : skippy_sampling_config{};
    session->sampling_config_valid = true;
}

static void skippy_log_sampling_cache_metrics(
        const skippy_session * session,
        const char * action) {
    if (session == nullptr || session->ctx == nullptr) {
        return;
    }
    LLAMA_LOG_INFO(
            "%s: seq=%d action=%s reuse_hits=%" PRIu64 " rebuilds=%" PRIu64 " scheduler_reserves=%" PRIu64 "\n",
            __func__,
            session->seq_id,
            action,
            session->sampling_reuse_hit_count,
            session->sampling_rebuild_count,
            session->ctx->get_sched_reserve_count());
}

bool skippy_reuse_plain_chat_sampling(
        skippy_session * session,
        const skippy_sampling_config * sampling,
        const char * metadata_json,
        uint64_t prompt_token_count) {
    if (session == nullptr || metadata_json == nullptr ||
            std::strcmp(metadata_json, "{\"grammar\":\"\"}") != 0 ||
            session->sampling_chain == nullptr || session->grammar_sampler != nullptr ||
            session->reasoning_budget_sampler != nullptr ||
            !session->sampling_config_valid || session->sampling_config.logit_bias_count != 0 ||
            (sampling != nullptr && sampling->logit_bias_count != 0) ||
            !skippy_sampling_chain_is_reusable_for(session, sampling)) {
        return false;
    }
    llama_sampler_reset(session->sampling_chain);
    skippy_store_sampling_config(session, sampling);
    session->chat_sampling_metadata = metadata_json;
    session->grammar_generated_start = prompt_token_count;
    session->sampling_accepted_token_count = 0;
    skippy_refresh_backend_sampling(session, sampling);
    ++session->sampling_reuse_hit_count;
    skippy_log_sampling_cache_metrics(session, "reuse");
    return true;
}

bool skippy_reset_reusable_sampling(skippy_session * session) {
    if (session == nullptr || session->sampling_chain == nullptr ||
            session->grammar_sampler != nullptr || !session->sampling_config_valid ||
            session->reasoning_budget_sampler != nullptr ||
            session->sampling_backend_retired ||
            session->sampling_config.logit_bias_count != 0) {
        return false;
    }
    llama_sampler_reset(session->sampling_chain);
    session->chat_sampling_metadata.clear();
    session->grammar_generated_start = 0;
    session->sampling_accepted_token_count = 0;
    skippy_refresh_backend_sampling(session, &session->sampling_config);
    ++session->sampling_reuse_hit_count;
    skippy_log_sampling_cache_metrics(session, "reset-reuse");
    return true;
}

static std::vector<llama_logit_bias> skippy_sampling_logit_biases(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    std::vector<llama_logit_bias> biases;
    if (sampling != nullptr) {
        const uint32_t count = std::min<uint32_t>(sampling->logit_bias_count, SKIPPY_MAX_LOGIT_BIAS);
        biases.assign(sampling->logit_bias, sampling->logit_bias + count);
    }
    const bool ignore_eos = sampling != nullptr &&
            (sampling->ignore_eos != 0 ||
                    (sampling->flags & SKIPPY_SAMPLING_FLAG_IGNORE_EOS) != 0);
    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    if (ignore_eos) {
        const int32_t token_count = llama_vocab_n_tokens(vocab);
        for (llama_token token = 0; token < token_count; ++token) {
            if (!llama_vocab_is_eog(vocab, token)) {
                continue;
            }
            const auto found = std::find_if(biases.begin(), biases.end(), [token](const llama_logit_bias & bias) {
                return bias.token == token;
            });
            if (found == biases.end()) {
                biases.push_back({token, -INFINITY});
            } else {
                found->bias = -INFINITY;
            }
        }
    }
    int32_t suppress_token_count = 0;
    const llama_token * suppress_tokens = llama_vocab_get_suppress_tokens(vocab, &suppress_token_count);
    return skippy_add_vocab_suppress_token_biases(
            std::move(biases), suppress_tokens, suppress_token_count);
}

static std::vector<std::string> skippy_dry_sequence_breaker_storage(
        const skippy_sampling_config * sampling) {
    const uint32_t count = std::min<uint32_t>(
            sampling->dry_sequence_breaker_count,
            SKIPPY_MAX_DRY_SEQUENCE_BREAKERS);
    std::vector<std::string> storage;
    storage.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const char * begin = sampling->dry_sequence_breakers[i];
        const char * end = std::find(
                begin,
                begin + SKIPPY_MAX_DRY_SEQUENCE_BREAKER_BYTES,
                '\0');
        storage.emplace_back(begin, end);
    }
    return storage;
}

static void skippy_add_configured_sampler(
        llama_sampler * chain,
        skippy_session * session,
        const skippy_sampling_config * sampling,
        float repeat_penalty,
        uint32_t sampler_type,
        uint32_t seed) {
    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    switch (sampler_type) {
        case SKIPPY_SAMPLER_PENALTIES:
        {
            const int32_t penalty_last_n = sampling->penalty_last_n == 0
                    ? -1
                    : sampling->penalty_last_n;
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    n_vocab,
                    penalty_last_n,
                    repeat_penalty,
                    sampling->frequency_penalty,
                    sampling->presence_penalty));
            break;
        }
        case SKIPPY_SAMPLER_DRY:
        {
            std::vector<std::string> breaker_storage =
                    skippy_dry_sequence_breaker_storage(sampling);
            std::vector<const char *> breakers;
            breakers.reserve(breaker_storage.size());
            for (const std::string & breaker : breaker_storage) {
                breakers.push_back(breaker.c_str());
            }
            llama_sampler_chain_add(chain, llama_sampler_init_dry(
                    vocab,
                    sampling->dry_multiplier,
                    sampling->dry_base,
                    sampling->dry_allowed_length,
                    sampling->dry_penalty_last_n,
                    breakers.data(),
                    breakers.size()));
            break;
        }
        case SKIPPY_SAMPLER_TOP_N_SIGMA:
            llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(sampling->top_nsigma));
            break;
        case SKIPPY_SAMPLER_TOP_K:
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(sampling->top_k));
            break;
        case SKIPPY_SAMPLER_TYPICAL_P:
            llama_sampler_chain_add(chain, llama_sampler_init_typical(sampling->typical_p, 0));
            break;
        case SKIPPY_SAMPLER_TOP_P:
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(sampling->top_p, 0));
            break;
        case SKIPPY_SAMPLER_MIN_P:
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(sampling->min_p, 0));
            break;
        case SKIPPY_SAMPLER_XTC:
            llama_sampler_chain_add(chain, llama_sampler_init_xtc(
                    sampling->xtc_probability,
                    sampling->xtc_threshold,
                    0,
                    seed));
            break;
        case SKIPPY_SAMPLER_TEMPERATURE:
            llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(
                    sampling->temperature,
                    sampling->dynatemp_range,
                    sampling->dynatemp_exponent));
            break;
        default:
            break;
    }
}

llama_sampler * skippy_build_sampling_chain(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);
    if (sampler == nullptr) {
        return nullptr;
    }

    const std::vector<llama_logit_bias> logit_biases =
            skippy_sampling_logit_biases(session, sampling);
    if (!logit_biases.empty()) {
        const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
        llama_sampler_chain_add(
                sampler,
                llama_sampler_init_logit_bias(
                        llama_vocab_n_tokens(vocab),
                        static_cast<int32_t>(logit_biases.size()),
                        logit_biases.data()));
    }
    if (skippy_sampling_is_greedy_equivalent(sampling)) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        ++session->sampling_rebuild_count;
        skippy_log_sampling_cache_metrics(session, "rebuild");
        return sampler;
    }
    const uint32_t seed = sampling->seed == 0 ? LLAMA_DEFAULT_SEED : sampling->seed;
    const float repeat_penalty = sampling->repeat_penalty == 0.0f
            ? 1.0f
            : sampling->repeat_penalty;
    static const uint32_t default_samplers[] = {
        SKIPPY_SAMPLER_PENALTIES,
        SKIPPY_SAMPLER_DRY,
        SKIPPY_SAMPLER_TOP_N_SIGMA,
        SKIPPY_SAMPLER_TOP_K,
        SKIPPY_SAMPLER_TYPICAL_P,
        SKIPPY_SAMPLER_TOP_P,
        SKIPPY_SAMPLER_MIN_P,
        SKIPPY_SAMPLER_XTC,
        SKIPPY_SAMPLER_TEMPERATURE,
    };
    const uint32_t configured_count =
            std::min<uint32_t>(sampling->sampler_count, SKIPPY_MAX_SAMPLERS);
    if (sampling->mirostat_mode == 1 || sampling->mirostat_mode == 2) {
        skippy_add_configured_sampler(
                sampler,
                session,
                sampling,
                repeat_penalty,
                SKIPPY_SAMPLER_TEMPERATURE,
                seed);
    } else if (configured_count == 0) {
        for (const uint32_t sampler_type : default_samplers) {
            if (sampler_type == SKIPPY_SAMPLER_TYPICAL_P &&
                    !(sampling->typical_p > 0.0f && sampling->typical_p < 1.0f)) {
                continue;
            }
            if (sampler_type == SKIPPY_SAMPLER_TOP_P &&
                    !(sampling->top_p > 0.0f && sampling->top_p < 1.0f)) {
                continue;
            }
            skippy_add_configured_sampler(
                    sampler,
                    session,
                    sampling,
                    repeat_penalty,
                    sampler_type,
                    seed);
        }
    } else {
        for (uint32_t i = 0; i < configured_count; ++i) {
            skippy_add_configured_sampler(
                    sampler,
                    session,
                    sampling,
                    repeat_penalty,
                    sampling->samplers[i],
                    seed);
        }
    }
    if (sampling->mirostat_mode == 1) {
        const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
        llama_sampler_chain_add(sampler, llama_sampler_init_mirostat(
                llama_vocab_n_tokens(vocab),
                seed,
                sampling->mirostat_entropy,
                sampling->mirostat_learning_rate,
                100));
    } else if (sampling->mirostat_mode == 2) {
        llama_sampler_chain_add(sampler, llama_sampler_init_mirostat_v2(
                seed,
                sampling->mirostat_entropy,
                sampling->mirostat_learning_rate));
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    }
    ++session->sampling_rebuild_count;
    skippy_log_sampling_cache_metrics(session, "rebuild");
    return sampler;
}

static bool skippy_ensure_plain_sampling_chain(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    if (session == nullptr || !skippy_sampling_enabled(sampling)) {
        return false;
    }

    if (session->sampling_chain != nullptr &&
            session->chat_sampling_metadata.empty() &&
            skippy_sampling_chain_is_reusable_for(session, sampling)) {
        skippy_store_sampling_config(session, sampling);
        return true;
    }

    skippy_clear_chat_sampling(session);
    session->sampling_chain = skippy_build_sampling_chain(session, sampling);
    if (session->sampling_chain == nullptr) {
        return false;
    }
    skippy_store_sampling_config(session, sampling);
    session->sampling_accepted_token_count = 0;
    skippy_refresh_backend_sampling(session, sampling);
    return true;
}

void skippy_refresh_backend_sampling(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    if (session == nullptr || session->sampling_backend_enabled || session->sampling_backend_retired) {
        // Backend initialization belongs to the sampler-chain lifetime. The
        // scheduler may reset and reuse the same chain on every iteration;
        // attaching it again violates llama_sampler_chain's one-shot init
        // contract. A retired chain was detached mid-lifetime (sampled verify
        // windows need every output row, which backend sampling forbids) and
        // must likewise never be re-attached.
        return;
    }
    session->sampling_backend_enabled = false;
    if (session->ctx != nullptr &&
            // Activation-export and embedding contexts require every output
            // row. Backend sampling accepts at most one row per sequence.
            !session->ctx->get_cparams().embeddings &&
            session->sampling_chain != nullptr &&
            session->grammar_sampler == nullptr &&
            session->reasoning_budget_sampler == nullptr &&
            skippy_sampling_is_greedy_equivalent(sampling)) {
        session->sampling_backend_enabled =
                llama_set_sampler(session->ctx, session->seq_id, session->sampling_chain);
    }
}

bool skippy_init_chat_grammar_sampler(
        skippy_session * session,
        const json & metadata,
        struct skippy_error ** out_error) {
    const std::string grammar = metadata.value("grammar", std::string());
    if (grammar.empty()) {
        return true;
    }

    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const bool grammar_lazy = metadata.value("grammar_lazy", false);
    session->grammar_lazy = grammar_lazy;
    std::vector<std::string> trigger_patterns_storage;
    std::vector<const char *> trigger_patterns;
    std::vector<llama_token> trigger_tokens;
    if (const auto triggers = metadata.find("grammar_triggers"); triggers != metadata.end() && triggers->is_array()) {
        for (const auto & trigger : *triggers) {
            const int type = trigger.value("type", -1);
            switch (type) {
                case 0:
                    if (trigger.contains("token")) {
                        trigger_tokens.push_back(static_cast<llama_token>(trigger.value("token", 0)));
                    }
                    break;
                case 1:
                    trigger_patterns_storage.push_back(skippy_regex_escape(trigger.value("value", std::string())));
                    break;
                case 2:
                    trigger_patterns_storage.push_back(trigger.value("value", std::string()));
                    break;
                case 3:
                {
                    std::string pattern = trigger.value("value", std::string());
                    if (pattern.empty()) {
                        pattern = "^$";
                    } else {
                        if (pattern.front() != '^') {
                            pattern.insert(pattern.begin(), '^');
                        }
                        if (pattern.back() != '$') {
                            pattern.push_back('$');
                        }
                    }
                    trigger_patterns_storage.push_back(std::move(pattern));
                    break;
                }
                default:
                    break;
            }
        }
    }
    trigger_patterns.reserve(trigger_patterns_storage.size());
    for (const auto & pattern : trigger_patterns_storage) {
        trigger_patterns.push_back(pattern.c_str());
    }

    session->grammar_sampler = grammar_lazy ?
            llama_sampler_init_grammar_lazy_patterns(
                    vocab,
                    grammar.c_str(),
                    "root",
                    trigger_patterns.data(),
                    trigger_patterns.size(),
                    trigger_tokens.data(),
                    trigger_tokens.size()) :
            llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
    if (session->grammar_sampler == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to initialize chat grammar sampler");
        return false;
    }

    if (!grammar_lazy) {
        const std::string generation_prompt = metadata.value("generation_prompt", std::string());
        try {
            for (const llama_token token : skippy_tokenize_text(vocab, generation_prompt, false, true)) {
                llama_sampler_accept(session->grammar_sampler, token);
            }
        } catch (const std::exception & e) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, e.what());
            return false;
        } catch (...) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to prefill chat grammar sampler");
            return false;
        }
    }
    return true;
}

bool skippy_init_chat_reasoning_budget_sampler(
        skippy_session * session,
        const skippy_sampling_config * sampling,
        const json & metadata,
        struct skippy_error ** out_error) {
    if (session == nullptr || sampling == nullptr) {
        return true;
    }
    const bool finite_budget = sampling->reasoning_budget_tokens >= 0;
    const bool supports_thinking = metadata.value("supports_thinking", false);
    const std::string start_tag = metadata.value("thinking_start_tag", std::string());
    const auto end_tags_json = metadata.find("thinking_end_tags");
    if (start_tag.empty() || end_tags_json == metadata.end() || !end_tags_json->is_array()) {
        if (finite_budget && supports_thinking) {
            skippy_set_error(
                    out_error,
                    SKIPPY_STATUS_INVALID_ARGUMENT,
                    "finite reasoning budget requires thinking start and end tags");
            return false;
        }
        return true;
    }
    const bool needs_sampler = finite_budget || session->grammar_lazy;
    if (!needs_sampler) {
        return true;
    }
    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const std::vector<llama_token> start_tokens = skippy_tokenize_text(vocab, start_tag, false, true);
    std::vector<std::vector<llama_token>> end_tokens;
    for (const auto & value : *end_tags_json) {
        if (!value.is_string()) {
            continue;
        }
        std::vector<llama_token> tokens = skippy_tokenize_text(vocab, value.get<std::string>(), false, true);
        if (!tokens.empty()) {
            end_tokens.push_back(std::move(tokens));
        }
    }
    if (start_tokens.empty() || end_tokens.empty()) {
        if (finite_budget && supports_thinking) {
            skippy_set_error(
                    out_error,
                    SKIPPY_STATUS_INVALID_ARGUMENT,
                    "finite reasoning budget requires non-empty tokenized thinking tags");
            return false;
        }
        return true;
    }
    const int32_t budget = sampling->reasoning_budget_tokens < 0
            ? std::numeric_limits<int32_t>::max()
            : sampling->reasoning_budget_tokens;
    session->reasoning_budget_sampler = common_reasoning_budget_init(
            vocab, {start_tokens}, end_tokens, end_tokens.front(), budget);
    if (session->reasoning_budget_sampler == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to initialize reasoning budget sampler");
        return false;
    }
    const std::string generation_prompt = metadata.value("generation_prompt", std::string());
    for (const llama_token token : skippy_tokenize_text(vocab, generation_prompt, false, true)) {
        llama_sampler_accept(session->reasoning_budget_sampler, token);
    }
    return true;
}

static bool skippy_chat_grammar_should_apply(const skippy_session * session) {
    if (session == nullptr || session->grammar_sampler == nullptr) {
        return false;
    }
    if (!session->grammar_lazy || session->reasoning_budget_sampler == nullptr) {
        return true;
    }
    const auto state = common_reasoning_budget_get_state(session->reasoning_budget_sampler);
    return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
}

static void skippy_sync_chat_sampling_history(skippy_session * session) {
    if (session == nullptr || session->sampling_chain == nullptr) {
        return;
    }
    while (session->sampling_accepted_token_count < session->token_history.size()) {
        const size_t index = session->sampling_accepted_token_count;
        const llama_token token = session->token_history[index];
        llama_sampler_accept(session->sampling_chain, token);
        const bool generated = index >= session->grammar_generated_start;
        const bool accept_grammar = generated && skippy_chat_grammar_should_apply(session);
        if (session->reasoning_budget_sampler != nullptr && generated) {
            llama_sampler_accept(session->reasoning_budget_sampler, token);
            if (!accept_grammar && session->grammar_sampler != nullptr &&
                    common_reasoning_budget_get_state(session->reasoning_budget_sampler) == REASONING_BUDGET_DONE) {
                const std::vector<llama_token> * end_match =
                        common_reasoning_budget_get_end_match(session->reasoning_budget_sampler);
                if (end_match != nullptr) {
                    for (const llama_token end_token : *end_match) {
                        llama_sampler_accept(session->grammar_sampler, end_token);
                    }
                }
            }
        }
        if (accept_grammar) {
            try {
                llama_sampler_accept(session->grammar_sampler, token);
            } catch (const std::exception & e) {
                skippy_disable_chat_grammar_after_exception(session, "accept", e.what());
            } catch (...) {
                skippy_disable_chat_grammar_after_exception(session, "accept", nullptr);
            }
        }
        session->sampling_accepted_token_count++;
    }
}

static llama_token skippy_chat_sample_token(skippy_session * session, int32_t logits_index) {
    skippy_sync_chat_sampling_history(session);
    if (session->sampling_backend_enabled) {
        const llama_token sampled = llama_get_sampled_token_ith(session->ctx, logits_index);
        skippy_mark_context_synchronized(session->stage_model);
        if (sampled != LLAMA_TOKEN_NULL) {
            llama_sampler_accept(session->sampling_chain, sampled);
            return sampled;
        }
        return skippy_greedy_sample_ith(session, logits_index);
    }
    llama_synchronize(session->ctx);
    skippy_mark_context_synchronized(session->stage_model);

    const llama_vocab * vocab = llama_model_get_vocab(session->stage_model->model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(session->ctx, logits_index);
    if (logits == nullptr) {
        return skippy_greedy_sample_ith(session, logits_index);
    }

    std::vector<llama_token_data> candidates;
    candidates.reserve(static_cast<size_t>(n_vocab));
    for (llama_token token = 0; token < n_vocab; ++token) {
        candidates.push_back({ token, logits[token], 0.0f });
    }
    llama_token_data_array cur = {
        candidates.data(),
        candidates.size(),
        -1,
        false,
    };
    if (session->reasoning_budget_sampler != nullptr) {
        llama_sampler_apply(session->reasoning_budget_sampler, &cur);
    }
    if (skippy_chat_grammar_should_apply(session)) {
        try {
            llama_sampler_apply(session->grammar_sampler, &cur);
        } catch (const std::exception & e) {
            skippy_disable_chat_grammar_after_exception(session, "apply", e.what());
        } catch (...) {
            skippy_disable_chat_grammar_after_exception(session, "apply", nullptr);
        }
    }
    llama_sampler_apply(session->sampling_chain, &cur);
    if (cur.selected < 0 || static_cast<size_t>(cur.selected) >= cur.size) {
        return skippy_greedy_sample_ith(session, logits_index);
    }
    return cur.data[cur.selected].id;
}

llama_token skippy_sample_token_ith(
        skippy_session * session,
        const skippy_sampling_config * sampling,
        int32_t logits_index) {
    if (session == nullptr || session->ctx == nullptr ||
            session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return 0;
    }
    if (skippy_mtp_greedy_sampling_fastpath_enabled() &&
            !session->sampling_backend_enabled &&
            session->grammar_sampler == nullptr &&
            session->reasoning_budget_sampler == nullptr &&
            skippy_sampling_is_greedy_equivalent(sampling)) {
        if (session->sampling_chain != nullptr) {
            skippy_clear_chat_sampling(session);
        }
        const bool ignore_eog =
                sampling != nullptr &&
                (sampling->flags & SKIPPY_SAMPLING_FLAG_IGNORE_EOS) != 0;
        return skippy_greedy_sample_context_with_eog_policy(
                session->stage_model->model,
                session->ctx,
                logits_index,
                ignore_eog);
    }
    if (session->sampling_chain != nullptr) {
        if (!session->chat_sampling_metadata.empty() ||
                (skippy_sampling_enabled(sampling) &&
                 session->sampling_config_valid &&
                 skippy_sampling_configs_equal(session->sampling_config, *sampling))) {
            return skippy_chat_sample_token(session, logits_index);
        }
        skippy_clear_chat_sampling(session);
    }
    if (!skippy_sampling_enabled(sampling)) {
        return skippy_greedy_sample_ith(session, logits_index);
    }
    if (skippy_ensure_plain_sampling_chain(session, sampling)) {
        return skippy_chat_sample_token(session, logits_index);
    }
    llama_synchronize(session->ctx);

    llama_sampler * sampler = skippy_build_sampling_chain(session, sampling);
    if (sampler == nullptr) {
        return skippy_greedy_sample_ith(session, logits_index);
    }

    for (const llama_token token : session->token_history) {
        llama_sampler_accept(sampler, token);
    }
    llama_token token = llama_sampler_sample(sampler, session->ctx, logits_index);
    llama_sampler_free(sampler);
    return token;
}

llama_token skippy_sample_token(
        skippy_session * session,
        const skippy_sampling_config * sampling) {
    return skippy_sample_token_ith(session, sampling, -1);
}
