#pragma once

#include "skippy/activation.h"
#include "skippy/common.h"

#include "llama.h"

#include <cstddef>
#include <cstdint>

struct skippy_session;
struct skippy_model;

enum class skippy_activation_frame_shape {
    reservation,
    live,
    live_within_reservation,
};

bool skippy_emits_activation_frame(const skippy_session * session);
int64_t skippy_resolve_activation_part_dimension(
        const skippy_session * session,
        const skippy_activation_part_desc & part,
        uint32_t dimension,
        bool live_graph);
bool skippy_has_activation_payload(const skippy_activation_desc * desc, const void * payload);
const skippy_activation_part_desc * skippy_activation_part_at(const skippy_activation_desc * desc, size_t index);
const skippy_activation_part_desc * skippy_activation_find_part(
        const skippy_activation_desc * desc,
        const uint8_t identity[SKIPPY_ACTIVATION_IDENTITY_BYTES]);
const void * skippy_activation_part_payload(
        const skippy_activation_desc * desc,
        const void * payload,
        const skippy_activation_part_desc * part);
void * skippy_activation_part_payload(
        const skippy_activation_desc * desc,
        void * payload,
        const skippy_activation_part_desc * part);
enum skippy_status skippy_checkpoint_verify_window(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        skippy_error ** out_error);
enum skippy_status skippy_validate_frame_input(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        size_t expected_token_count,
        skippy_error ** out_error);
enum skippy_status skippy_prepare_output_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        skippy_activation_desc * output_desc,
        const skippy_activation_desc * input_desc,
        skippy_error ** out_error,
        skippy_activation_frame_shape shape = skippy_activation_frame_shape::reservation);
enum skippy_status skippy_copy_output_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        size_t * out_output_payload_bytes,
        skippy_activation_desc * output_desc,
        skippy_error ** out_error);
enum skippy_status skippy_decode_activation_frame(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        const llama_token * token_ids,
        const llama_pos * positions,
        size_t position_count,
        size_t token_count,
        bool request_logits,
        int32_t glm_dsa_phase_hint,
        skippy_error ** out_error);
enum skippy_status skippy_verify_activation_frame(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        const llama_token * token_ids,
        size_t token_count,
        skippy_error ** out_error);
