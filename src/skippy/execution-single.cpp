#include "skippy.h"
#include "skippy/activation_internal.h"
#include "skippy/graph_build_inputs.h"
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
enum skippy_status skippy_prefill_chunk(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const void * input_activations,
        size_t input_activation_bytes,
        void * output_activations,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        struct skippy_error ** out_error) {
    (void) input_activations;
    (void) input_activation_bytes;
    (void) output_activations;
    (void) output_activation_capacity;

    if (out_output_activation_bytes != nullptr) {
        *out_output_activation_bytes = 0;
    }

    return skippy_decode_tokens(
            session,
            token_ids,
            token_count,
            false,
            SKIPPY_GLM_DSA_PHASE_HINT_PREFILL,
            out_error);
}

enum skippy_status skippy_decode_step_sampled(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const void * input_activation,
        size_t input_activation_bytes,
        void * output_activation,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    (void) input_activation;
    (void) input_activation_bytes;
    (void) output_activation;
    (void) output_activation_capacity;

    if (out_output_activation_bytes != nullptr) {
        *out_output_activation_bytes = 0;
    }

    enum skippy_status status = skippy_decode_tokens(
            session,
            &token_id,
            1,
            true,
            SKIPPY_GLM_DSA_PHASE_HINT_DECODE,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        return skippy_store_sampled_token(
                skippy_sample_token(session, sampling), out_predicted_token, out_error);
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_decode_step_sampled_mtp(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        llama_token * out_predicted_token,
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

    enum skippy_status status = skippy_decode_step_sampled(
            session,
            token_id,
            sampling,
            nullptr,
            0,
            nullptr,
            0,
            nullptr,
            out_predicted_token,
            out_error);
    if (status != SKIPPY_STATUS_OK || out_predicted_token == nullptr || *out_predicted_token < 0) {
        return status;
    }

    const size_t bounded_max_draft_tokens =
            std::min<size_t>(max_draft_tokens, SKIPPY_NATIVE_MTP_MAX_DRAFT_TOKENS);
    return skippy_mtp_propose_next(session, *out_predicted_token, bounded_max_draft_tokens, out_mtp_draft, out_error);
}

enum skippy_status skippy_decode_batch_sampled(
        struct skippy_session * const * sessions,
        const llama_token * token_ids,
        const struct skippy_sampling_config * const * sampling,
        size_t request_count,
        llama_token * out_predicted_tokens,
        size_t predicted_token_capacity,
        struct skippy_error ** out_error) {
    if (sessions == nullptr || token_ids == nullptr || request_count == 0 || out_predicted_tokens == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sessions, token_ids, request_count, and out_predicted_tokens are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (predicted_token_capacity < request_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "predicted token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    if (request_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "request_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_session * first = sessions[0];
    if (first == nullptr || first->ctx == nullptr || first->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sessions must be active");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_filtered(first) && first->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "batched token decode requires the first runtime slice or a full model");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    const int32_t n_tokens = static_cast<int32_t>(request_count);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int32_t i = 0; i < n_tokens; ++i) {
        skippy_session * session = sessions[i];
        if (session == nullptr || session->ctx != first->ctx || session->stage_model != first->stage_model) {
            llama_batch_free(batch);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "all sessions must belong to the same stage model");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
            llama_batch_free(batch);
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "batched token decode requires the first runtime slice or a full model");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
        batch.token[i] = token_ids[i];
        batch.pos[i] = session->n_past;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = session->seq_id;
        batch.logits[i] = 1;
    }

    const skippy_graph_build_inputs graph_build_inputs = skippy_graph_build_inputs_from_config(
            &first->stage_model->config,
            SKIPPY_GLM_DSA_PHASE_HINT_DECODE);
    int32_t rc = 0;
    const bool decoded = skippy_contain_native_failure(
            out_error, [&] { rc = first->ctx->decode(batch, graph_build_inputs); });
    if (!decoded) {
        llama_batch_free(batch);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (rc != 0) {
        llama_batch_free(batch);
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    skippy_mark_context_work_pending(first->stage_model);
    llama_batch_free(batch);

    for (int32_t i = 0; i < n_tokens; ++i) {
        skippy_session * session = sessions[i];
        session->n_past += 1;
        skippy_record_tokens(session, &token_ids[i], 1);
        const skippy_sampling_config * request_sampling = sampling != nullptr ? sampling[i] : nullptr;
        const skippy_status sample_status = skippy_store_sampled_token(
                skippy_sample_token_ith(session, request_sampling, i), &out_predicted_tokens[i], out_error);
        if (sample_status != SKIPPY_STATUS_OK) {
            return sample_status;
        }
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_verify_tokens(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error) {
    if (out_token_count != nullptr) {
        *out_token_count = token_count;
    }
    if (session == nullptr || session->ctx == nullptr || token_ids == nullptr || token_count == 0 || out_token_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session, token_ids, token_count, and out_token_count are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (output_tokens == nullptr || output_token_capacity < token_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require activation input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    enum skippy_status status = skippy_checkpoint_verify_window(
            session,
            token_ids,
            token_count,
            nullptr,
            nullptr,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    status = skippy_verify_token_batch(session, token_ids, token_count, out_error);
    if (status == SKIPPY_STATUS_OK) {
        const int32_t n_tokens = static_cast<int32_t>(token_count);
        for (int32_t i = 0; i < n_tokens; ++i) {
            status = skippy_store_sampled_token(
                    skippy_greedy_sample_ith(session, i), &output_tokens[i], out_error);
            if (status != SKIPPY_STATUS_OK) {
                *out_token_count = 0;
                return status;
            }
        }
    }
    return status;
}

static enum skippy_status skippy_prefill_chunk_frame_impl(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const llama_pos * positions,
        size_t position_count,
        bool request_logits,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    if (out_predicted_token != nullptr) {
        *out_predicted_token = -1;
    }
    if (token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds activation descriptor range");
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

    const bool activation_input = skippy_is_filtered(session) && session->stage_model->config.layer_start > 0;
    if (activation_input) {
        status = skippy_decode_activation_frame(
                session,
                input_desc,
                input_payload,
                token_ids,
                positions,
                position_count,
                token_count,
                request_logits,
                SKIPPY_GLM_DSA_PHASE_HINT_PREFILL,
                out_error);
    } else {
        status = skippy_decode_tokens(
                session,
                token_ids,
                token_count,
                request_logits,
                SKIPPY_GLM_DSA_PHASE_HINT_PREFILL,
                out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = -1;
        if (skippy_runtime_is_terminal_stage(&session->stage_model->config)) {
            status = skippy_store_sampled_token(
                    skippy_sample_token(session, sampling), out_predicted_token, out_error);
            if (status != SKIPPY_STATUS_OK) {
                return status;
            }
        }
    }

    return skippy_copy_output_activation_frame(
            session, token_count, output_payload, input_desc, input_payload, out_output_payload_bytes, output_desc, out_error);
}

enum skippy_status skippy_prefill_chunk_frame(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            nullptr,
            0,
            false,
            nullptr,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            nullptr,
            out_error);
}

enum skippy_status skippy_prefill_chunk_frame_sampled(
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
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            nullptr,
            0,
            true,
            sampling,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            out_predicted_token,
            out_error);
}

enum skippy_status skippy_prefill_chunk_frame_with_positions(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const llama_pos * positions,
        size_t position_count,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            positions,
            position_count,
            false,
            nullptr,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            nullptr,
            out_error);
}

enum skippy_status skippy_prefill_chunk_frame_sampled_with_positions(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const llama_pos * positions,
        size_t position_count,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    return skippy_prefill_chunk_frame_impl(
            session,
            token_ids,
            token_count,
            positions,
            position_count,
            true,
            sampling,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            out_predicted_token,
            out_error);
}

enum skippy_status skippy_decode_step_frame_sampled(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error) {
    enum skippy_status status = skippy_validate_frame_input(
            session,
            input_desc,
            input_payload,
            1,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    status = skippy_prepare_output_activation_frame(
            session,
            1,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            input_desc,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (skippy_is_filtered(session) && session->stage_model->config.layer_start > 0) {
        status = skippy_decode_activation_frame(
                session,
                input_desc,
                input_payload,
                &token_id,
                nullptr,
                0,
                1,
                true,
                SKIPPY_GLM_DSA_PHASE_HINT_DECODE,
                out_error);
    } else {
        status = skippy_decode_tokens(
                session,
                &token_id,
                1,
                true,
                SKIPPY_GLM_DSA_PHASE_HINT_DECODE,
                out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    if (out_predicted_token != nullptr) {
        *out_predicted_token = -1;
        if (skippy_runtime_is_terminal_stage(&session->stage_model->config)) {
            status = skippy_store_sampled_token(
                    skippy_sample_token(session, sampling), out_predicted_token, out_error);
            if (status != SKIPPY_STATUS_OK) {
                return status;
            }
        }
    }

    return skippy_copy_output_activation_frame(
            session, 1, output_payload, input_desc, input_payload, out_output_payload_bytes, output_desc, out_error);
}

enum skippy_status skippy_decode_step_frame_sampled_mtp(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_token,
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

    enum skippy_status status = skippy_decode_step_frame_sampled(
            session,
            token_id,
            sampling,
            input_desc,
            input_payload,
            output_desc,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            out_predicted_token,
            out_error);
    if (status != SKIPPY_STATUS_OK || out_predicted_token == nullptr || *out_predicted_token < 0) {
        return status;
    }

    return skippy_mtp_propose_next(session, *out_predicted_token, max_draft_tokens, out_mtp_draft, out_error);
}


} // extern "C"
