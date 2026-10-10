#include "skippy.h"
#include "skippy/errors.h"
#include "skippy/environment.h"
#include "skippy/glm_dsa.h"
#include "skippy-verify-checkpoint.h"
#include "skippy/devices.h"

#include <mutex>
#include "gguf.h"
#include "llama-arch.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdio>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

#include "skippy/execution_internal.h"
#include "skippy/sampling_internal.h"
#include "skippy/speculative_internal.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"

void skippy_disable_chat_grammar_after_exception(
        skippy_session * session,
        const char * operation,
        const char * message) {
    if (session == nullptr || session->grammar_sampler == nullptr) {
        return;
    }
    fprintf(stderr, "skippy: disabling chat grammar after %s failed: %s\n",
            operation != nullptr ? operation : "grammar operation",
            message != nullptr ? message : "unknown native exception");
    llama_sampler_free(session->grammar_sampler);
    session->grammar_sampler = nullptr;
}

enum skippy_status skippy_require_usable_session(
        const skippy_session * session,
        skippy_error ** out_error) {
    if (session != nullptr && session->reset_required) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "session requires reset after failed execution or recovery");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    return SKIPPY_STATUS_OK;
}
int32_t skippy_stage_layer_count(const llama_model * model) {
    if (model == nullptr) {
        return 0;
    }

    // Stage ranges cover the executable trunk. Appended NextN/MTP blocks are
    // retained by the final stage's tensor filter and executed by its MTP
    // sidecar, but they are not additional pipeline stages.
    return llama_model_n_layer(model);
}

int32_t skippy_claim_execution_lane(skippy_model * model) {
    if (model == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(model->lane_mutex);
    for (uint32_t i = 0; i < model->lane_in_use.size(); ++i) {
        if (!model->lane_in_use[i]) {
            model->lane_in_use[i] = true;
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

void skippy_release_execution_lane(skippy_model * model, int32_t seq_id) {
    if (model == nullptr || seq_id < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(model->lane_mutex);
    const size_t lane = static_cast<size_t>(seq_id);
    if (lane < model->lane_in_use.size()) {
        model->lane_in_use[lane] = false;
    }
}

void skippy_mark_context_work_pending(skippy_model * model) {
    if (model != nullptr) {
        model->context_work_pending.store(true, std::memory_order_release);
    }
}

void skippy_mark_context_synchronized(skippy_model * model) {
    if (model != nullptr) {
        model->context_work_pending.store(false, std::memory_order_release);
    }
}

void skippy_synchronize_context_if_pending(skippy_model * model) {
    if (model != nullptr && model->ctx != nullptr &&
            model->context_work_pending.exchange(false, std::memory_order_acq_rel)) {
        model->ctx->synchronize();
    }
}

extern "C" {
enum skippy_status skippy_session_create(
        struct skippy_model * model,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || model->ctx == nullptr || out_session == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_session are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_session = nullptr;

    const int32_t seq_id = skippy_claim_execution_lane(model);
    if (seq_id < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "no skippy execution lane is available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_session * session = new skippy_session{};
    session->stage_model = model;
    session->ctx = model->ctx;
    session->seq_id = seq_id;
    session->borrowed_sequence = false;
    session->borrowed_prefix_tokens = 0;
    session->n_past = 0;
    session->mtp_next_pos = 0;
    if (skippy_mtp_available(session)) {
        skippy_mtp_clear_session_state(session);
    }
    if (static_cast<size_t>(seq_id) < model->lane_resident_prefix_tokens.size()) {
        model->lane_resident_prefix_tokens[static_cast<size_t>(seq_id)].clear();
    }
    if (llama_memory_t memory = model->ctx->get_memory()) {
        if (!llama_memory_seq_rm(memory, seq_id, -1, -1)) {
            skippy_release_execution_lane(model, seq_id);
            delete session;
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to initialize execution lane");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        skippy_emit_kv_event(
                SKIPPY_RUNTIME_EVENT_KIND_KV_INITIALIZED,
                static_cast<uint64_t>(seq_id),
                static_cast<uint64_t>(llama_n_ctx(model->ctx)),
                static_cast<uint64_t>(model->lane_count));
    }
    *out_session = session;
    return skippy_success(out_error);
}

struct llama_context * skippy_session_llama_context(
        struct skippy_session * session) {
    return session != nullptr ? session->ctx : nullptr;
}

int32_t skippy_session_position(
        const struct skippy_session * session) {
    return session != nullptr ? session->n_past : -1;
}

int32_t skippy_session_sequence_id(
        const struct skippy_session * session) {
    return session != nullptr ? session->seq_id : -1;
}

int32_t skippy_session_batch_size(
        const struct skippy_session * session) {
    return session != nullptr && session->ctx != nullptr ? llama_n_batch(session->ctx) : 0;
}

enum skippy_status skippy_session_begin_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (g_skippy_external_decode_session != nullptr && g_skippy_external_decode_session != session) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "another external decode session is already active on this thread");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->pending_activation_exports.clear();

    g_skippy_external_decode_session = session;

    return skippy_success(out_error);
}

enum skippy_status skippy_session_end_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (g_skippy_external_decode_session == session) {
        g_skippy_external_decode_session = nullptr;
    }
    skippy_mark_context_work_pending(session->stage_model);
    return skippy_success(out_error);
}

enum skippy_status skippy_session_set_position(
        struct skippy_session * session,
        int32_t n_past,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (n_past < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "n_past must be non-negative");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    session->n_past = n_past;
    if (session->token_history.size() > static_cast<size_t>(n_past)) {
        session->token_history.resize(static_cast<size_t>(n_past));
        skippy_clear_chat_sampling(session);
    }
    if (session->signal_history.size() > static_cast<size_t>(n_past)) {
        session->signal_history.resize(static_cast<size_t>(n_past));
    }
    skippy_mtp_clear_session_state(session);
    session->verify_checkpoints.clear();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_restore_token_history(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || session->stage_model == nullptr ||
            (token_count > 0 && token_ids == nullptr)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and token history are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session->n_past < 0 || token_count != static_cast<size_t>(session->n_past)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token history must match the imported session position");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const auto status = skippy_require_usable_session(session, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    try {
        // State payloads contain tensors, not sampler history. Absolute indices
        // keep grammar start and penalty windows identical to a cold prefill.
        std::vector<llama_token> history;
        if (token_count > 0) {
            history.assign(token_ids, token_ids + token_count);
        }
        const std::string metadata = session->chat_sampling_metadata;
        const skippy_sampling_config sampling = session->sampling_config;
        const bool sampling_valid = session->sampling_config_valid;
        const uint64_t generated_start = session->grammar_generated_start;
        session->token_history.swap(history);
        skippy_clear_chat_sampling(session);
        if (!metadata.empty()) {
            return skippy_session_configure_chat_sampling(session, sampling_valid ? &sampling : nullptr,
                    metadata.c_str(), generated_start, out_error);
        }
        return skippy_success(out_error);
    } catch (const std::exception & e) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, e.what());
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
}

enum skippy_status skippy_session_sample_current(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || out_predicted_token == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and out_predicted_token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    return skippy_store_sampled_token(
            skippy_sample_token(session, sampling), out_predicted_token, out_error);
}

enum skippy_status skippy_session_configure_chat_sampling(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        const char * metadata_json,
        uint64_t prompt_token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (metadata_json == nullptr || metadata_json[0] == '\0') {
        skippy_clear_chat_sampling(session);
        return skippy_success(out_error);
    }
    if (skippy_reuse_plain_chat_sampling(
            session, sampling, metadata_json, prompt_token_count)) {
        return skippy_success(out_error);
    }

    try {
        json metadata = json::parse(metadata_json);
        skippy_clear_chat_sampling(session);
        session->sampling_chain = skippy_build_sampling_chain(session, sampling);
        if (session->sampling_chain == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to initialize chat sampling chain");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        session->chat_sampling_metadata = metadata_json;
        session->sampling_config = sampling != nullptr ?
                *sampling : skippy_sampling_config{};
        session->sampling_config_valid = true;
        session->grammar_generated_start = prompt_token_count;
        session->sampling_accepted_token_count = 0;
        if (!skippy_init_chat_grammar_sampler(session, metadata, out_error)) {
            skippy_clear_chat_sampling(session);
            return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
        }
        if (!skippy_init_chat_reasoning_budget_sampler(session, sampling, metadata, out_error)) {
            skippy_clear_chat_sampling(session);
            return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
        }
        skippy_refresh_backend_sampling(session, sampling);
    } catch (const std::exception & e) {
        skippy_clear_chat_sampling(session);
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, e.what());
        return SKIPPY_STATUS_RUNTIME_ERROR;
    } catch (...) {
        skippy_clear_chat_sampling(session);
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to configure chat sampling");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_session_reset(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_synchronize_context_if_pending(session->stage_model);
    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    const llama_pos reset_from = session->borrowed_sequence
        ? static_cast<llama_pos>(session->borrowed_prefix_tokens)
        : -1;
    if (!llama_memory_seq_rm(memory, session->seq_id, reset_from, -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to reset execution lane");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->n_past = session->borrowed_sequence ? session->borrowed_prefix_tokens : 0;
    if (session->borrowed_sequence) {
        if (session->token_history.size() > static_cast<size_t>(session->borrowed_prefix_tokens)) {
            session->token_history.resize(static_cast<size_t>(session->borrowed_prefix_tokens));
        }
    } else {
        session->token_history.clear();
    }
    session->signal_history.clear();
    // Keep a reusable backend sampler attached across reset. Detaching it
    // makes the next sampled decode reattach the chain and mark the scheduler
    // for reserve, which rebuilds compute buffers at every request boundary.
    // Operators can restore detach-on-reset with
    // SKIPPY_DETACH_SAMPLER_ON_RESET=1. Single-session prefill requests no
    // output rows; batched prefill requests at most the final row, so the
    // reuse path remains within the one-output-per-sequence contract.
    const bool detach_sampler_on_reset =
            skippy_env_enabled("SKIPPY_DETACH_SAMPLER_ON_RESET");
    if (detach_sampler_on_reset || !skippy_reset_reusable_sampling(session)) {
        skippy_clear_chat_sampling(session);
    }
    skippy_mtp_clear_session_state(session);
    session->sampling_backend_verifying = false;
    session->reset_required = false;
    session->verify_checkpoints.clear();
    skippy_emit_kv_event(
            SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED,
            static_cast<uint64_t>(session->seq_id),
            static_cast<uint64_t>(std::max(session->n_past, 0)),
            0);
    return skippy_success(out_error);
}

enum skippy_status skippy_session_free(
        struct skippy_session * session,
        struct skippy_error ** out_error) {
    if (session != nullptr) {
        if (g_skippy_external_decode_session == session) {
            g_skippy_external_decode_session = nullptr;
        }
        if (session->ctx != nullptr) {
            if (session->preserve_prefix_on_free && !session->borrowed_sequence && session->stage_model != nullptr) {
                skippy_synchronize_context_if_pending(session->stage_model);
                if (llama_memory_t memory = session->ctx->get_memory()) {
                    llama_memory_seq_rm(
                            memory,
                            session->seq_id,
                            static_cast<llama_pos>(session->preserve_prefix_tokens),
                            -1);
                }
            } else {
                skippy_session_reset(session, nullptr);
                if (!session->borrowed_sequence && session->stage_model != nullptr && session->seq_id >= 0) {
                    const size_t lane = static_cast<size_t>(session->seq_id);
                    if (lane < session->stage_model->lane_resident_prefix_tokens.size()) {
                        session->stage_model->lane_resident_prefix_tokens[lane].clear();
                    }
                }
            }
        }
        skippy_clear_chat_sampling(session);
        if (!session->borrowed_sequence) {
            skippy_release_execution_lane(session->stage_model, session->seq_id);
        }
        skippy_mtp_clear_session_state(session);
        delete session;
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_session_last_token_signal(
        struct skippy_session * session,
        struct skippy_token_signal * out_signal,
        struct skippy_error ** out_error) {
    if (session == nullptr || out_signal == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and out_signal are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session->signal_history.size() < static_cast<size_t>(std::max(session->n_past, 0))) {
        skippy_record_signal(session, -1);
    }
    if (session->signal_history.empty()) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "no token signal is available");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_signal = session->signal_history.back();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_signal_window(
        struct skippy_session * session,
        uint32_t window_tokens,
        struct skippy_generation_signal_window * out_window,
        struct skippy_error ** out_error) {
    if (session == nullptr || out_window == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and out_window are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (window_tokens == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "window_tokens must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session->signal_history.empty()) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "no token signals are available");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const size_t count = std::min<size_t>(window_tokens, session->signal_history.size());
    const size_t start = session->signal_history.size() - count;
    double entropy_sum = 0.0;
    double margin_sum = 0.0;
    float max_entropy = 0.0f;
    float min_margin = std::numeric_limits<float>::infinity();
    uint32_t high_entropy_count = 0;
    for (size_t i = start; i < session->signal_history.size(); ++i) {
        const skippy_token_signal & signal = session->signal_history[i];
        entropy_sum += signal.entropy;
        margin_sum += signal.margin;
        max_entropy = std::max(max_entropy, signal.entropy);
        min_margin = std::min(min_margin, signal.margin);
        if (signal.entropy > 4.0f) {
            ++high_entropy_count;
        }
    }

    uint32_t repetition_count = 0;
    if (!session->token_history.empty()) {
        const size_t token_count = std::min<size_t>(window_tokens, session->token_history.size());
        const size_t token_start = session->token_history.size() - token_count;
        for (size_t i = token_start + 1; i < session->token_history.size(); ++i) {
            if (session->token_history[i] == session->token_history[i - 1]) {
                ++repetition_count;
            }
        }
    }

    *out_window = {};
    out_window->token_count = static_cast<uint32_t>(count);
    out_window->mean_entropy = static_cast<float>(entropy_sum / static_cast<double>(count));
    out_window->max_entropy = max_entropy;
    out_window->mean_margin = static_cast<float>(margin_sum / static_cast<double>(count));
    out_window->min_margin = min_margin;
    out_window->high_entropy_count = high_entropy_count;
    out_window->repetition_count = repetition_count;
    return skippy_success(out_error);
}

} // extern "C"
