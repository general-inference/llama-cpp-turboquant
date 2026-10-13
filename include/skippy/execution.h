#ifndef SKIPPY_EXECUTION_H
#define SKIPPY_EXECUTION_H

/**
 * @file skippy/execution.h
 * @brief Prefill, decode, verification, activation-frame, and batched execution.
 */

#include "activation.h"
#include "runtime.h"
#include "sampling.h"
#include "speculative_decoding.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Prefills tokens and emits a raw activation payload for the next stage. */
LLAMA_API enum skippy_status skippy_prefill_chunk(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const void * input_activations,
        size_t input_activation_bytes,
        void * output_activations,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        struct skippy_error ** out_error);

/** @brief Verifies a proposed token sequence and returns accepted output tokens. */
LLAMA_API enum skippy_status skippy_verify_tokens(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);

/** @brief Decodes and samples one token using raw activation buffers. */
LLAMA_API enum skippy_status skippy_decode_step_sampled(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        const void * input_activation,
        size_t input_activation_bytes,
        void * output_activation,
        size_t output_activation_capacity,
        size_t * out_output_activation_bytes,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error);

/** @brief Decodes one token and returns a native MTP draft proposal. */
LLAMA_API enum skippy_status skippy_decode_step_sampled_mtp(
        struct skippy_session * session,
        llama_token token_id,
        const struct skippy_sampling_config * sampling,
        llama_token * out_predicted_token,
        size_t max_draft_tokens,
        struct skippy_native_mtp_draft * out_mtp_draft,
        struct skippy_error ** out_error);

/**
 * @brief Describes one variable-length request in a mixed scheduler iteration.
 *
 * `positions` contains `token_count * n_pos_per_embd` entries when supplied.
 * A null positions pointer derives contiguous positions from the session. The
 * activation frame, when required by a downstream stage, must describe exactly
 * `token_count` tokens. `sample_last` requests logits and sampling only for the
 * final token in this request.
 */
struct skippy_iteration_request {
        struct skippy_session * session;
        const llama_token * token_ids;
        size_t token_count;
        const llama_pos * positions;
        size_t position_count;
        const struct skippy_sampling_config * sampling;
        const struct skippy_activation_desc * input_desc;
        const void * input_payload;
        bool sample_last;
};

/**
 * @brief Executes mixed prefill chunks and decode tokens in one model iteration.
 *
 * Requests may have different token counts. Activation outputs are request-major.
 * Sampled outputs are compact and explicitly map back to the request indexes
 * whose final rows requested logits. GLM-DSA inputs in one iteration must use
 * the same sideband width and stream count.
 */
LLAMA_API enum skippy_status skippy_iteration_batch_sampled(
        const struct skippy_iteration_request * requests,
        size_t request_count,
        struct skippy_activation_desc * output_descs,
        void * const * output_payloads,
        const size_t * output_payload_capacities,
        size_t * out_output_payload_bytes,
        size_t * out_sampled_request_indexes,
        llama_token * out_predicted_tokens,
        size_t sampled_output_capacity,
        size_t * out_sampled_output_count,
        struct skippy_error ** out_error);

/**
 * @brief Samples one predicted token for each session in a request batch.
 * @deprecated Use `skippy_iteration_batch_sampled`; this legacy serving entrypoint
 * cannot express mixed prefill and decode work.
 */
LLAMA_API SKIPPY_DEPRECATED("use skippy_iteration_batch_sampled")
enum skippy_status skippy_decode_batch_sampled(
        struct skippy_session * const * sessions,
        const llama_token * token_ids,
        const struct skippy_sampling_config * const * sampling,
        size_t request_count,
        llama_token * out_predicted_tokens,
        size_t predicted_token_capacity,
        struct skippy_error ** out_error);

/** @brief Prefills tokens using a described activation frame. */
LLAMA_API enum skippy_status skippy_prefill_chunk_frame(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const struct skippy_activation_desc * input_desc,
        const void * input_payload,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error);

/** @brief Prefills an activation frame and samples the final-stage token. */
LLAMA_API enum skippy_status skippy_prefill_chunk_frame_sampled(
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
        struct skippy_error ** out_error);

/** @brief Prefills a positioned token chunk using activation-frame I/O. */
LLAMA_API enum skippy_status skippy_prefill_chunk_frame_with_positions(
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
        struct skippy_error ** out_error);

/** @brief Prefills positioned tokens, then samples with activation-frame I/O. */
LLAMA_API enum skippy_status skippy_prefill_chunk_frame_sampled_with_positions(
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
        struct skippy_error ** out_error);

/** @brief Decodes and samples one token using activation-frame I/O. */
LLAMA_API enum skippy_status skippy_decode_step_frame_sampled(
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
        struct skippy_error ** out_error);

/** @brief Decodes one framed token and returns a native MTP draft proposal. */
LLAMA_API enum skippy_status skippy_decode_step_frame_sampled_mtp(
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
        struct skippy_error ** out_error);

/**
 * @brief Runs framed sampled decode for multiple sessions in one batch.
 * @deprecated Use `skippy_iteration_batch_sampled`; this legacy serving entrypoint
 * is restricted to one token per request.
 */
LLAMA_API SKIPPY_DEPRECATED("use skippy_iteration_batch_sampled")
enum skippy_status skippy_decode_step_frame_batch_sampled(
        struct skippy_session * const * sessions,
        const llama_token * token_ids,
        const struct skippy_sampling_config * const * sampling,
        const struct skippy_activation_desc * const * input_descs,
        const void * const * input_payloads,
        struct skippy_activation_desc * output_descs,
        void * const * output_payloads,
        const size_t * output_payload_capacities,
        size_t * out_output_payload_bytes,
        llama_token * out_predicted_tokens,
        size_t predicted_token_capacity,
        size_t request_count,
        struct skippy_error ** out_error);

/** @brief Verifies proposed tokens with framed activations and optional MTP drafting. */
LLAMA_API enum skippy_status skippy_verify_tokens_frame_sampled(
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
        struct skippy_error ** out_error);

/** @brief Copies the session's pending output activation into a caller buffer. */
LLAMA_API enum skippy_status skippy_session_copy_output_activation_frame(
        struct skippy_session * session,
        size_t token_count,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_EXECUTION_H
