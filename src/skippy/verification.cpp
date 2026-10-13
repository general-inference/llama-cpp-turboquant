#include "skippy.h"
#include "skippy/activation_internal.h"
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


extern "C" {
enum skippy_status skippy_verify_tokens_frame_sampled(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        size_t max_draft_tokens,
        struct skippy_native_mtp_draft * out_mtp_draft,
        struct skippy_error ** out_error) {
    if (out_mtp_draft != nullptr) {
        *out_mtp_draft = {
            1,
            false,
            0,
            {},
            0,
        };
    }
    if (out_token_count != nullptr) {
        *out_token_count = 0;
    }
    if (token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds activation descriptor range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    enum skippy_status status = skippy_validate_frame_input(
            session,
            input_desc,
            input_payload,
            token_count,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    status = skippy_prepare_output_activation_frame(
            session,
            token_count,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            input_desc,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (skippy_runtime_is_terminal_stage(&session->stage_model->config)) {
        if (out_token_count == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_token_count is required for output stages");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        *out_token_count = token_count;
        if (output_tokens == nullptr || output_token_capacity < token_count) {
            skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token output buffer is too small");
            return SKIPPY_STATUS_BUFFER_TOO_SMALL;
        }
    }

    if (skippy_runtime_is_terminal_stage(&session->stage_model->config) && session->sampling_backend_enabled) {
        // Backend (in-graph) sampling permits at most one output row per
        // sequence per decode, but sampled verification must output every row
        // of the window, so llama_decode would reject the batch outright.
        // Detach the backend sampler and fall back to CPU-chain sampling for
        // this session: the chain itself is untouched and the row sampling
        // below consumes it through skippy_sample_token_ith. This chain stays
        // retired; only building a fresh chain permits backend offload again.
        llama_set_sampler(session->ctx, session->seq_id, nullptr);
        session->sampling_backend_enabled = false;
        // The detached chain has already been backend-initialized and
        // llama_sampler_chain_backend_init asserts on reuse, so retire it
        // from offload for the rest of its lifetime; a rebuilt chain may
        // offload again.
        session->sampling_backend_retired = true;
    }

    const bool activation_input = skippy_is_filtered(session) && session->stage_model->config.layer_start > 0;
    status = skippy_checkpoint_verify_window(
            session,
            token_ids,
            token_count,
            activation_input ? input_desc : nullptr,
            activation_input ? input_payload : nullptr,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (activation_input) {
        status = skippy_runtime_is_terminal_stage(&session->stage_model->config) ?
                skippy_verify_activation_frame(session, input_desc, input_payload, token_ids, token_count, out_error) :
                skippy_decode_activation_frame(
                        session,
                        input_desc,
                        input_payload,
                        token_ids,
                        nullptr,
                        0,
                        token_count,
                        false,
                        SKIPPY_GLM_DSA_PHASE_HINT_VERIFY,
                        out_error);
    } else {
        status = skippy_runtime_is_terminal_stage(&session->stage_model->config) ?
                skippy_verify_token_batch(session, token_ids, token_count, out_error) :
                skippy_decode_tokens(
                        session,
                        token_ids,
                        token_count,
                        false,
                        SKIPPY_GLM_DSA_PHASE_HINT_VERIFY,
                        out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    status = skippy_copy_output_activation_frame(
            session, token_count, output_payload, input_desc, input_payload, out_output_payload_bytes, output_desc, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (skippy_runtime_is_terminal_stage(&session->stage_model->config)) {
        // Token-batch verification (no activation input) eagerly decodes and
        // records the complete batch before sampling; that eager append must
        // be removed so each input is exposed exactly once and in causal
        // order as its row is sampled. Activation-frame verification
        // (`skippy_verify_activation_frame`, used on stages fed by an
        // upstream stage) never records the batch, so its history already
        // ends at the pre-batch position and must not be truncated — doing
        // so silently drops real history rows and leaves the sampler ahead
        // of the rewritten history.
        if (!activation_input && session->token_history.size() < token_count) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "sampled verification token history underflow");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        const size_t history_size_before_verification = activation_input
                ? session->token_history.size()
                : session->token_history.size() - token_count;
        if (session->sampling_accepted_token_count > history_size_before_verification) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "sampled verification history is behind sampler state");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        session->token_history.resize(history_size_before_verification);
        // Any stateful sampler (grammar, penalties, or RNG) must not absorb
        // tokens from a rejected suffix.
        const bool stop_after_first_mismatch =
                session->grammar_sampler != nullptr || session->sampling_chain != nullptr;
        bool proposal_mismatched = false;
        for (size_t i = 0; i < token_count; ++i) {
            skippy_record_tokens(session, &token_ids[i], 1);
            const int32_t logits_index = static_cast<int32_t>(i);
            status = skippy_store_sampled_token(
                    skippy_sample_token_ith(session, sampling, logits_index), &output_tokens[i], out_error);
            if (status != SKIPPY_STATUS_OK) {
                *out_token_count = 0;
                return status;
            }
            if (stop_after_first_mismatch && i + 1 < token_count && output_tokens[i] != token_ids[i + 1]) {
                // Later rows are conditioned on a rejected token and are not
                // authoritative predictions. Stop advancing the sampler and
                // grammar state through that rejected suffix.
                proposal_mismatched = true;
                *out_token_count = i + 1;
                break;
            }
        }
        if (out_mtp_draft != nullptr && token_count > 0 && !proposal_mismatched) {
            status = skippy_mtp_propose_next(
                    session,
                    output_tokens[token_count - 1],
                    max_draft_tokens,
                    out_mtp_draft,
                    out_error);
            if (status != SKIPPY_STATUS_OK) {
                return status;
            }
        }
    } else {
        if (session->sampling_accepted_token_count > session->token_history.size()) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "sampled verification history is behind sampler state");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        skippy_record_tokens(session, token_ids, token_count);
    }

    return skippy_success(out_error);
}


} // extern "C"
