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

bool skippy_memory_needs_verify_checkpoint(llama_memory_t memory) {
    return dynamic_cast<llama_memory_hybrid *>(memory) != nullptr ||
           dynamic_cast<llama_memory_hybrid_iswa *>(memory) != nullptr ||
           dynamic_cast<llama_memory_recurrent *>(memory) != nullptr;
}

// Recurrent memory can roll a bounded number of trailing positions back in
// place: the RS cache is widened to (1 + n_rs_seq) snapshot groups and
// llama_memory_recurrent::seq_rm rewinds via set_rs_idx when the rollback fits
// inside that bound. When it does, copying the whole sequence state to host
// memory before every verify is pure overhead, and the seq_rm path in
// skippy_trim_session restores the accepted prefix exactly - without the
// serial replay of accepted tokens that the checkpoint restore path performs.
//
// Staged sessions are excluded: they roll back across hosts, where an
// in-process RS index is not a usable restore point.
bool skippy_verify_window_can_roll_back_in_place(
        const skippy_session * session,
        size_t token_count) {
    if (session == nullptr || session->ctx == nullptr || skippy_is_filtered(session)) {
        return false;
    }
    const uint32_t n_rs_seq = session->ctx->get_cparams().n_rs_seq;
    if (n_rs_seq == 0 || token_count == 0 || token_count > static_cast<size_t>(n_rs_seq)) {
        return false;
    }
    return llm_arch_supports_rs_rollback(session->ctx->get_model().arch);
}

enum skippy_status skippy_checkpoint_verify_window(
        skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        skippy_error ** out_error) {
    enum skippy_status usable_status = skippy_require_usable_session(session, out_error);
    if (usable_status != SKIPPY_STATUS_OK) {
        return usable_status;
    }
    llama_memory_t memory = session->ctx->get_memory();
    if (!skippy_memory_needs_verify_checkpoint(memory)) {
        return skippy_success(out_error);
    }
    if (skippy_verify_window_can_roll_back_in_place(session, token_count)) {
        return skippy_success(out_error);
    }

    session->ctx->synchronize();
    const size_t state_size = llama_state_seq_get_size_ext(
            session->ctx,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (state_size == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to size verify-window state checkpoint");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_verify_checkpoint checkpoint;
    checkpoint.state.resize(state_size);
    const size_t written = llama_state_seq_get_data_ext(
            session->ctx,
            checkpoint.state.data(),
            checkpoint.state.size(),
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (written != checkpoint.state.size()) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to checkpoint verify-window sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    checkpoint.valid = true;
    checkpoint.token_start = session->n_past;
    checkpoint.token_count = token_count;
    if (token_ids != nullptr) {
        checkpoint.token_ids.assign(token_ids, token_ids + token_count);
    }
    checkpoint.has_activation_input = skippy_has_activation_payload(input_desc, input_payload);
    if (checkpoint.has_activation_input) {
        checkpoint.input_desc = *input_desc;
        const uint8_t * bytes = static_cast<const uint8_t *>(input_payload);
        checkpoint.input_payload.assign(bytes, bytes + input_desc->payload_bytes);
    }

    constexpr size_t max_verify_checkpoints = 64;
    if (session->verify_checkpoints.size() >= max_verify_checkpoints) {
        skippy_set_error(
                out_error,
                SKIPPY_STATUS_UNSUPPORTED,
                "verify-window pipeline depth exceeds supported checkpoint retention");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    session->verify_checkpoints.push_back(std::move(checkpoint));
    return skippy_success(out_error);
}

static enum skippy_status skippy_validate_frame_input_sequences(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        size_t expected_token_count,
        uint32_t expected_sequence_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const skippy_runtime_config & config = session->stage_model->config;
    if (skippy_runtime_is_source_stage(&config)) {
        if (skippy_has_activation_payload(input_desc, input_payload)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "this stage expects token input, not activation input");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        return SKIPPY_STATUS_OK;
    }

    if (input_desc == nullptr || input_payload == nullptr || input_desc->payload_bytes == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "non-first runtime slices require an activation frame input");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->version != SKIPPY_ACTIVATION_FRAME_VERSION || input_desc->reserved != 0 ||
            input_desc->part_count == 0 || input_desc->part_count > SKIPPY_ACTIVATION_MAX_PARTS) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame has an invalid multipart descriptor");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->sequence_count != expected_sequence_count || input_desc->token_count != expected_token_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame token or sequence count does not match request");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (input_desc->layer_end != config.layer_start) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame layer_end must match this stage layer_start");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const auto & boundary = session->stage_model->input_activation_boundary;
    if (!session->stage_model->has_input_activation_boundary || boundary.version != SKIPPY_ACTIVATION_BOUNDARY_DESC_VERSION ||
            std::memcmp(boundary.frontier_identity, input_desc->frontier_identity, SKIPPY_ACTIVATION_IDENTITY_BYTES) != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame frontier identity does not match this stage");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    for (size_t actual_index = 0; actual_index < input_desc->part_count; ++actual_index) {
        const auto & actual = input_desc->parts[actual_index];
        if ((actual.flags & ~SKIPPY_ACTIVATION_PART_OPTIONAL) != 0 || actual.rank == 0 ||
                actual.rank > SKIPPY_ACTIVATION_MAX_DIMS || actual.token_axis < 0 ||
                actual.token_axis >= static_cast<int32_t>(actual.rank) ||
                actual.ggml_type >= GGML_TYPE_COUNT) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part structure is invalid");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        for (size_t prior = 0; prior < actual_index; ++prior) {
            if (std::memcmp(actual.identity, input_desc->parts[prior].identity, SKIPPY_ACTIVATION_IDENTITY_BYTES) == 0) {
                skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame contains duplicate part identities");
                return SKIPPY_STATUS_INVALID_ARGUMENT;
            }
        }
        const auto type = static_cast<ggml_type>(actual.ggml_type);
        const uint64_t block_size = static_cast<uint64_t>(ggml_blck_size(type));
        uint64_t expected_stride = static_cast<uint64_t>(ggml_type_size(type));
        if (block_size == 0 || expected_stride == 0 ||
                actual.dimensions[0] <= 0 ||
                static_cast<uint64_t>(actual.dimensions[0]) % block_size != 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part type or packed width is invalid");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t dimension = 0; dimension < actual.rank; ++dimension) {
            if (actual.dimensions[dimension] <= 0 ||
                    (static_cast<int32_t>(dimension) == actual.token_axis &&
                     actual.dimensions[dimension] != static_cast<int64_t>(expected_token_count)) ||
                    actual.byte_strides[dimension] != expected_stride) {
                skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part dimensions or strides are invalid");
                return SKIPPY_STATUS_INVALID_ARGUMENT;
            }
            if (dimension == 0) {
                expected_stride = static_cast<uint64_t>(
                        ggml_row_size(type, actual.dimensions[0]));
            } else {
                const uint64_t extent = static_cast<uint64_t>(actual.dimensions[dimension]);
                if (expected_stride > std::numeric_limits<uint64_t>::max() / extent) {
                    skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part byte count overflows");
                    return SKIPPY_STATUS_INVALID_ARGUMENT;
                }
                expected_stride *= extent;
            }
        }
        if (actual.payload_bytes != expected_stride) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part payload size does not match its shape");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        bool planned = false;
        for (uint32_t expected_index = 0; expected_index < boundary.part_count; ++expected_index) {
            if (std::memcmp(boundary.parts[expected_index].identity, actual.identity,
                    SKIPPY_ACTIVATION_IDENTITY_BYTES) == 0) {
                planned = true;
                break;
            }
        }
        if (!planned && (actual.flags & SKIPPY_ACTIVATION_PART_OPTIONAL) == 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame contains an unplanned required part");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (skippy_activation_part_payload(input_desc, input_payload, &actual) == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part byte range exceeds the payload");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }
    for (size_t expected_index = 0; expected_index < boundary.part_count; ++expected_index) {
        const auto & expected = boundary.parts[expected_index];
        const auto * actual = skippy_activation_find_part(input_desc, expected.identity);
        if (actual == nullptr) {
            if ((expected.flags & SKIPPY_ACTIVATION_PART_OPTIONAL) != 0) continue;
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame is missing a required frontier part");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (actual->ggml_type != expected.ggml_type || actual->rank != expected.rank || actual->token_axis != expected.token_axis ||
                (actual->flags & ~SKIPPY_ACTIVATION_PART_OPTIONAL) != 0 || actual->token_axis < 0 ||
                actual->token_axis >= static_cast<int32_t>(actual->rank)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part structure does not match the planned frontier");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t dimension = 0; dimension < actual->rank; ++dimension) {
            const int64_t value = actual->dimensions[dimension];
            if (value <= 0 || (static_cast<int32_t>(dimension) == actual->token_axis && value != static_cast<int64_t>(expected_token_count)) ||
                    (expected.dimensions[dimension] > 0 && value != expected.dimensions[dimension])) {
                skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part dimensions do not match the planned frontier");
                return SKIPPY_STATUS_INVALID_ARGUMENT;
            }
        }
        if (skippy_activation_part_payload(input_desc, input_payload, actual) == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part byte range exceeds the payload");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (size_t index = 0; index < input_desc->part_count; ++index) {
        const auto & part = input_desc->parts[index];
        ranges.emplace_back(part.payload_offset, part.payload_offset + part.payload_bytes);
    }
    std::sort(ranges.begin(), ranges.end());
    uint64_t end = 0;
    for (const auto & range : ranges) {
        if (range.first != end || range.second < range.first) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part ranges must pack the payload without gaps or overlap");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        end = range.second;
    }
    if (end != input_desc->payload_bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "activation frame part ranges do not cover the payload");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    return SKIPPY_STATUS_OK;
}

enum skippy_status skippy_validate_frame_input(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        size_t expected_token_count,
        struct skippy_error ** out_error) {
    return skippy_validate_frame_input_sequences(
            session,
            input_desc,
            input_payload,
            expected_token_count,
            1,
            out_error);
}

enum skippy_status skippy_prepare_output_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        skippy_activation_desc * output_desc,
        const skippy_activation_desc * input_desc,
        struct skippy_error ** out_error,
        skippy_activation_frame_shape shape) {
    skippy_activation_desc descriptor = {};
    descriptor.version = SKIPPY_ACTIVATION_FRAME_VERSION;
    descriptor.producer_stage_index = session != nullptr ? session->stage_model->config.stage_index : -1;
    descriptor.layer_start = session != nullptr ? session->stage_model->config.layer_start : 0;
    descriptor.layer_end = session != nullptr ? session->stage_model->config.layer_end : 0;
    descriptor.token_count = static_cast<uint32_t>(std::min<size_t>(token_count, std::numeric_limits<uint32_t>::max()));
    descriptor.sequence_count = token_count > 0 ? 1 : 0;
    if (skippy_emits_activation_frame(session)) {
        const auto & boundary = session->stage_model->output_activation_boundary;
        descriptor.part_count = boundary.part_count;
        std::memcpy(descriptor.frontier_identity, boundary.frontier_identity, SKIPPY_ACTIVATION_IDENTITY_BYTES);
        uint64_t offset = 0;
        for (size_t index = 0; index < boundary.part_count; ++index) {
            auto & part = descriptor.parts[index];
            part = boundary.parts[index];
            if (part.token_axis < 0 || part.token_axis >= static_cast<int32_t>(part.rank)) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "output frontier has an invalid token axis");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            part.dimensions[part.token_axis] = static_cast<int64_t>(token_count);
            for (uint32_t dimension = 0; dimension < part.rank; ++dimension) {
                if (part.dimensions[dimension] < 0) {
                    const auto * forwarded = input_desc == nullptr ? nullptr : skippy_activation_find_part(input_desc, part.identity);
                    if (forwarded != nullptr && forwarded->dimensions[dimension] > 0) part.dimensions[dimension] = forwarded->dimensions[dimension];
                    else {
                        part.dimensions[dimension] = skippy_resolve_activation_part_dimension(
                                session, part, dimension, shape != skippy_activation_frame_shape::reservation);
                        if (part.dimensions[dimension] > 0) continue;
                        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "output frontier has an unresolved dynamic dimension");
                        return SKIPPY_STATUS_RUNTIME_ERROR;
                    }
                }
            }
            const auto type = static_cast<ggml_type>(part.ggml_type);
            if (part.dimensions[0] <= 0 || part.dimensions[0] % ggml_blck_size(type) != 0) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "output frontier part has an invalid packed width");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            part.byte_strides[0] = ggml_type_size(type);
            part.byte_strides[1] = ggml_row_size(type, part.dimensions[0]);
            for (uint32_t dimension = 2; dimension < SKIPPY_ACTIVATION_MAX_DIMS; ++dimension)
                part.byte_strides[dimension] = part.byte_strides[dimension - 1] * static_cast<uint64_t>(part.dimensions[dimension - 1]);
            part.payload_offset = offset;
            part.payload_bytes = part.byte_strides[part.rank - 1] * static_cast<uint64_t>(part.dimensions[part.rank - 1]);
            offset += part.payload_bytes;
        }
        auto append_optional = [&](const skippy_activation_part_desc & source) -> bool {
            if ((source.flags & SKIPPY_ACTIVATION_PART_OPTIONAL) == 0) return true;
            if (skippy_activation_find_part(&descriptor, source.identity) != nullptr) return true;
            if (descriptor.part_count >= SKIPPY_ACTIVATION_MAX_PARTS ||
                    source.payload_bytes > std::numeric_limits<uint64_t>::max() - offset) return false;
            auto & part = descriptor.parts[descriptor.part_count++];
            part = source;
            part.payload_offset = offset;
            offset += part.payload_bytes;
            return true;
        };
        if (input_desc != nullptr) {
            for (uint32_t index = 0; index < input_desc->part_count; ++index) {
                if (!append_optional(input_desc->parts[index])) {
                    skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "optional activation directory exceeds frame limits");
                    return SKIPPY_STATUS_RUNTIME_ERROR;
                }
            }
        }
        for (const auto & pending : session->pending_activation_exports) {
            if (!append_optional(pending.desc)) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "optional activation directory exceeds frame limits");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        }
        descriptor.payload_bytes = offset;
    }
    const size_t payload_bytes = static_cast<size_t>(descriptor.payload_bytes);

    if (out_output_payload_bytes != nullptr) {
        *out_output_payload_bytes = payload_bytes;
    }

    if (payload_bytes > 0) {
        if (output_payload == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_payload is required for runtime-slice activation output");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (output_payload_capacity < payload_bytes) {
            if (shape == skippy_activation_frame_shape::live_within_reservation) {
                session->reset_required = true;
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "live activation output exceeds its graph reservation; session requires reset");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "output activation buffer is too small");
            return SKIPPY_STATUS_BUFFER_TOO_SMALL;
        }
    } else if (output_payload_capacity > 0 && output_payload == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_payload is required when output capacity is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (output_desc != nullptr) {
        *output_desc = descriptor;
    }

    return SKIPPY_STATUS_OK;
}

enum skippy_status skippy_decode_batch(
        skippy_session * session,
        llama_batch batch,
        size_t token_count,
        int32_t glm_dsa_phase_hint,
        struct skippy_error ** out_error,
        const skippy_graph_build_inputs * graph_build_inputs) {
    if (session == nullptr || session->ctx == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and at least one token are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    enum skippy_status usable_status = skippy_require_usable_session(session, out_error);
    if (usable_status != SKIPPY_STATUS_OK) {
        return usable_status;
    }
    skippy_graph_build_inputs inputs = graph_build_inputs != nullptr ? *graph_build_inputs : skippy_graph_build_inputs{};
    const skippy_graph_build_inputs configured_inputs = skippy_graph_build_inputs_from_config(
            &session->stage_model->config, glm_dsa_phase_hint);
    inputs.selection = configured_inputs.selection;
    inputs.activation_import_identities = configured_inputs.activation_import_identities;
    inputs.activation_import_bindings = configured_inputs.activation_import_bindings;
    inputs.activation_export_identities = configured_inputs.activation_export_identities;
    inputs.activation_export_bindings = configured_inputs.activation_export_bindings;
    inputs.resident_tensor_names = configured_inputs.resident_tensor_names;
    inputs.execution_contract = configured_inputs.execution_contract;
    const char * phase_log = getenv("SKIPPY_GLM_DSA_LOG_DIRECT_SPARSE_DECISIONS");
    if (phase_log != nullptr && atoi(phase_log) != 0) {
        LLAMA_LOG_INFO(
                "skippy: glm_dsa_phase_hint requested=%d active=%d tokens=%zu\n",
                glm_dsa_phase_hint,
                inputs.selection.glm_dsa_phase_hint,
                token_count);
    }
    skippy_glm_dsa_op_timing_begin(session, token_count);
    int32_t rc = 0;
    const bool decoded = skippy_contain_native_failure(
            out_error, [&] { rc = session->ctx->decode(batch, inputs); });
    skippy_glm_dsa_op_timing_end(session);
    if (!decoded) {
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (rc != 0) {
        // llama.cpp's own diagnostics go to its log callback, which stage
        // hosts may not surface; carry the batch and session placement in the
        // error so a remote failure names the violated invariant.
        llama_memory_t memory = session->ctx->get_memory();
        const llama_pos mem_pos_max = memory != nullptr ? memory->seq_pos_max(session->seq_id) : -1;
        const llama_pos batch_pos0 = batch.pos != nullptr && batch.n_tokens > 0 ? batch.pos[0] : -1;
        char detail[256];
        std::snprintf(
                detail,
                sizeof(detail),
                "llama_decode failed rc=%d n_tokens=%d embd=%d pos0=%d n_past=%d seq=%d mem_pos_max=%d",
                rc,
                batch.n_tokens,
                batch.embd != nullptr ? 1 : 0,
                batch_pos0,
                session->n_past,
                session->seq_id,
                mem_pos_max);
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, detail);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    skippy_mark_context_work_pending(session->stage_model);
    session->n_past += static_cast<int32_t>(token_count);
    return skippy_success(out_error);
}

static enum skippy_status skippy_prepare_empty_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        skippy_activation_desc * output_desc,
        struct skippy_error ** out_error) {
    if (output_payload_capacity > 0 && output_payload == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_payload is required when output capacity is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (out_output_payload_bytes != nullptr) {
        *out_output_payload_bytes = 0;
    }

    if (output_desc != nullptr) {
        *output_desc = {};
        output_desc->version = SKIPPY_ACTIVATION_FRAME_VERSION;
        output_desc->producer_stage_index = session != nullptr ? session->stage_model->config.stage_index : -1;
        output_desc->layer_start = session != nullptr ? session->stage_model->config.layer_start : 0;
        output_desc->layer_end = session != nullptr ? session->stage_model->config.layer_end : 0;
        output_desc->token_count = static_cast<uint32_t>(std::min<size_t>(token_count, std::numeric_limits<uint32_t>::max()));
        output_desc->sequence_count = token_count > 0 ? 1 : 0;
        output_desc->payload_bytes = 0;
    }

    return SKIPPY_STATUS_OK;
}

enum skippy_status skippy_copy_output_activation_frame(
        skippy_session * session,
        size_t token_count,
        void * output_payload,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        size_t * out_output_payload_bytes,
        skippy_activation_desc * output_desc,
        struct skippy_error ** out_error) {
    struct pending_activation_cleanup {
        skippy_session * session;
        ~pending_activation_cleanup() {
            if (session != nullptr) session->pending_activation_exports.clear();
        }
    } cleanup{session};
    if (!skippy_emits_activation_frame(session)) {
        return skippy_success(out_error);
    }

    skippy_activation_desc local_desc = {};
    skippy_activation_desc * desc = output_desc != nullptr ? output_desc : &local_desc;
    if (desc->version != SKIPPY_ACTIVATION_FRAME_VERSION || desc->part_count == 0) {
        const enum skippy_status status = skippy_prepare_output_activation_frame(
                session, token_count, output_payload, std::numeric_limits<size_t>::max(), nullptr,
                desc, input_desc, out_error);
        if (status != SKIPPY_STATUS_OK) return status;
    }
    // Preflight uses the scheduler's reserved graph to bound buffer size before
    // decode mutates state. Emit dimensions from the actual completed graph.
    const size_t prepared_bytes = desc->payload_bytes;
    const enum skippy_status shape_status = skippy_prepare_output_activation_frame(
            session, token_count, output_payload, prepared_bytes, out_output_payload_bytes,
            desc, input_desc, out_error, skippy_activation_frame_shape::live_within_reservation);
    if (shape_status != SKIPPY_STATUS_OK) return shape_status;
    skippy_synchronize_context_if_pending(session->stage_model);

    llm_graph_result * result = session->ctx->get_gf_res_prev_active();
    const auto * ports = result == nullptr ? nullptr : &result->get_skippy_activation_exports();
    const size_t graph_part_count = session->stage_model->output_activation_boundary.part_count;
    if (ports == nullptr || ports->size() != graph_part_count || desc->part_count < graph_part_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "live graph exports do not match the prepared activation directory");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    for (size_t index = 0; index < graph_part_count; ++index) {
        const auto & part = desc->parts[index];
        ggml_tensor * tensor = (*ports)[index].tensor;
        if (tensor == nullptr || static_cast<uint32_t>(tensor->type) != part.ggml_type || (*ports)[index].token_axis != part.token_axis) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "live graph export does not match its activation part descriptor");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        for (uint32_t dimension = 0; dimension < part.rank; ++dimension) {
            if (tensor->ne[dimension] < part.dimensions[dimension] || (dimension != 0 && tensor->ne[dimension] != part.dimensions[dimension])) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "live graph export shape does not match its activation part descriptor");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        }
        uint8_t * destination = static_cast<uint8_t *>(output_payload) + part.payload_offset;
        const size_t row_bytes = ggml_row_size(tensor->type, part.dimensions[0]);
        size_t destination_offset = 0;
        for (int64_t i3 = 0; i3 < part.dimensions[3]; ++i3) {
            for (int64_t i2 = 0; i2 < part.dimensions[2]; ++i2) {
                for (int64_t i1 = 0; i1 < part.dimensions[1]; ++i1) {
                    const size_t source_offset = static_cast<size_t>(i1) * tensor->nb[1] +
                            static_cast<size_t>(i2) * tensor->nb[2] + static_cast<size_t>(i3) * tensor->nb[3];
                    ggml_backend_tensor_get(tensor, destination + destination_offset, source_offset, row_bytes);
                    destination_offset += row_bytes;
                }
            }
        }
        if (destination_offset != part.payload_bytes) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "packed graph export size does not match its activation part descriptor");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
    }
    for (size_t index = graph_part_count; index < desc->part_count; ++index) {
        const auto & part = desc->parts[index];
        const void * source = nullptr;
        size_t source_bytes = 0;
        if (input_desc != nullptr && input_payload != nullptr) {
            const auto * forwarded = skippy_activation_find_part(input_desc, part.identity);
            if (forwarded != nullptr) {
                source = skippy_activation_part_payload(input_desc, input_payload, forwarded);
                source_bytes = forwarded->payload_bytes;
            }
        }
        if (source == nullptr) {
            for (const auto & pending : session->pending_activation_exports) {
                if (std::memcmp(pending.desc.identity, part.identity, SKIPPY_ACTIVATION_IDENTITY_BYTES) == 0) {
                    source = pending.payload.data();
                    source_bytes = pending.payload.size();
                    break;
                }
            }
        }
        if (source == nullptr || source_bytes != part.payload_bytes) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "optional activation part payload is unavailable");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        std::memcpy(static_cast<uint8_t *>(output_payload) + part.payload_offset, source, source_bytes);
    }
    if (out_output_payload_bytes != nullptr) *out_output_payload_bytes = desc->payload_bytes;
    return skippy_success(out_error);

}

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
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || input_desc == nullptr || input_payload == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and activation frame input are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    if (token_ids == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "Graph Filter V2 activation input requires the token-id sideband");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const int32_t n_pos_per_embd = session->stage_model->model->hparams.n_pos_per_embd();
    if (n_pos_per_embd <= 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model position sideband width must be positive");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const size_t n_pos_per_token = static_cast<size_t>(n_pos_per_embd);
    if (token_count > std::numeric_limits<size_t>::max() / n_pos_per_token) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "position sideband expected count exceeds size_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const size_t expected_position_count = token_count*n_pos_per_token;
    if (positions != nullptr && position_count != expected_position_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "position sideband count must equal token_count * n_pos_per_embd");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    std::vector<llama_pos> pos_storage(expected_position_count);
    std::vector<int32_t> n_seq_id_storage(n_tokens);
    std::vector<llama_seq_id> seq_id_0(1, session->seq_id);
    std::vector<llama_seq_id *> seq_id_storage(n_tokens + 1, nullptr);
    std::vector<int8_t> logits_storage(n_tokens);

    for (int32_t i = 0; i < n_tokens; ++i) {
        n_seq_id_storage[i] = 1;
        seq_id_storage[i] = seq_id_0.data();
        logits_storage[i] = request_logits && i == n_tokens - 1 ? 1 : 0;
    }
    if (positions != nullptr) {
        std::memcpy(pos_storage.data(), positions, expected_position_count*sizeof(llama_pos));
    } else if (n_pos_per_embd == 4) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            const llama_pos position = session->n_past + i;
            pos_storage[               i] = position;
            pos_storage[    n_tokens + i] = position;
            pos_storage[2 * n_tokens + i] = position;
            pos_storage[3 * n_tokens + i] = 0;
        }
    } else {
        for (int32_t i = 0; i < n_tokens; ++i) {
            pos_storage[i] = session->n_past + i;
        }
    }

    llama_batch batch = {
        /*n_tokens =*/ n_tokens,
        /*token    =*/ const_cast<llama_token *>(token_ids),
        /*embd     =*/ nullptr,
        /*pos      =*/ pos_storage.data(),
        /*n_seq_id =*/ n_seq_id_storage.data(),
        /*seq_id   =*/ seq_id_storage.data(),
        /*logits   =*/ logits_storage.data(),
    };

    skippy_graph_build_inputs graph_build_inputs;
    skippy_graph_build_inputs_set_activation_tokens(graph_build_inputs, token_ids, token_count);
    const llama_pos token_start = pos_storage.empty() ? session->n_past : pos_storage[0];
    skippy_graph_build_inputs_set_activation_frontier(
            graph_build_inputs,
            session->stage_model->input_activation_boundary,
            input_desc,
            input_payload,
            token_start);
    enum skippy_status status = skippy_decode_batch(
            session,
            batch,
            token_count,
            glm_dsa_phase_hint,
            out_error,
            &graph_build_inputs);
    const float * mtp_input_embeddings = skippy_mtp_input_embeddings_from_activation_frame(
            input_desc, input_payload, token_count);
    if (status == SKIPPY_STATUS_OK && (token_ids != nullptr || mtp_input_embeddings != nullptr)) {
        status = skippy_mtp_sync_target_inputs(
                session, token_ids, mtp_input_embeddings, token_count, token_start, out_error);
    }
    if (status == SKIPPY_STATUS_OK && token_ids != nullptr) {
        skippy_record_tokens(session, token_ids, token_count);
    }
    return status;
}

enum skippy_status skippy_verify_activation_frame(
        skippy_session * session,
        const skippy_activation_desc * input_desc,
        const void * input_payload,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || input_desc == nullptr || input_payload == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session and activation frame input are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t n_tokens = static_cast<int32_t>(token_count);
    if (token_ids == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "Graph Filter V2 activation verification requires the token-id sideband");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const int32_t n_pos_per_embd = session->stage_model->model->hparams.n_pos_per_embd();
    if (n_pos_per_embd <= 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model position sideband width must be positive");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const size_t n_pos_per_token = static_cast<size_t>(n_pos_per_embd);
    if (token_count > std::numeric_limits<size_t>::max() / n_pos_per_token) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "position sideband expected count exceeds size_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const size_t expected_position_count = token_count*n_pos_per_token;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    llama_token * owned_batch_tokens = batch.token;
    llama_pos * owned_batch_positions = batch.pos;
    std::vector<llama_pos> pos_storage(expected_position_count);
    batch.pos = pos_storage.data();
    batch.token = const_cast<llama_token *>(token_ids);
    batch.n_tokens = n_tokens;

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = session->seq_id;
        batch.logits[i] = 1;
    }
    if (n_pos_per_embd == 4) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            const llama_pos position = session->n_past + i;
            batch.pos[               i] = position;
            batch.pos[    n_tokens + i] = position;
            batch.pos[2 * n_tokens + i] = position;
            batch.pos[3 * n_tokens + i] = 0;
        }
    } else {
        for (int32_t i = 0; i < n_tokens; ++i) {
            batch.pos[i] = session->n_past + i;
        }
    }

    skippy_graph_build_inputs graph_build_inputs;
    skippy_graph_build_inputs_set_activation_tokens(graph_build_inputs, token_ids, token_count);
    const llama_pos token_start = session->n_past;
    skippy_graph_build_inputs_set_activation_frontier(
            graph_build_inputs,
            session->stage_model->input_activation_boundary,
            input_desc,
            input_payload,
            token_start);
    enum skippy_status status = skippy_decode_batch(
            session,
            batch,
            token_count,
            SKIPPY_GLM_DSA_PHASE_HINT_VERIFY,
            out_error,
            &graph_build_inputs);
    batch.token = owned_batch_tokens;
    batch.pos = owned_batch_positions;
    llama_batch_free(batch);
    const float * mtp_input_embeddings = skippy_mtp_input_embeddings_from_activation_frame(
            input_desc, input_payload, token_count);
    if (status == SKIPPY_STATUS_OK && (token_ids != nullptr || mtp_input_embeddings != nullptr)) {
        status = skippy_mtp_sync_target_inputs(
                session, token_ids, mtp_input_embeddings, token_count, token_start, out_error);
    }
    // Deliberately does NOT record the batch into token_history. Sampled
    // verification exposes each input exactly once and in causal order as its
    // row is sampled (see skippy_verify_tokens_frame_sampled), and for
    // activation-input stages it does not rewind first — the token-batch path
    // rewinds only because its decode appends eagerly. Recording here as well
    // would feed every input to the sampler chain twice.
    return status;
}

static std::vector<uint8_t> skippy_verify_prefix_activation_payload(
        const skippy_verify_checkpoint & checkpoint,
        size_t token_count,
        skippy_activation_desc & prefix_desc) {
    prefix_desc = checkpoint.input_desc;
    prefix_desc.token_count = static_cast<uint32_t>(token_count);
    prefix_desc.payload_bytes = 0;
    std::vector<uint8_t> payload;
    for (size_t index = 0; index < prefix_desc.part_count; ++index) {
        const auto & source_part = checkpoint.input_desc.parts[index];
        auto & part = prefix_desc.parts[index];
        if (part.token_axis < 0 || part.token_axis >= static_cast<int32_t>(part.rank)) return {};
        part.dimensions[part.token_axis] = static_cast<int64_t>(token_count);
        const auto type = static_cast<ggml_type>(part.ggml_type);
        part.byte_strides[0] = ggml_type_size(type);
        part.byte_strides[1] = ggml_row_size(type, part.dimensions[0]);
        for (uint32_t dimension = 2; dimension < SKIPPY_ACTIVATION_MAX_DIMS; ++dimension)
            part.byte_strides[dimension] = part.byte_strides[dimension - 1] * static_cast<uint64_t>(part.dimensions[dimension - 1]);
        part.payload_offset = payload.size();
        const size_t row_bytes = static_cast<size_t>(part.byte_strides[1]);
        const auto * source = static_cast<const uint8_t *>(skippy_activation_part_payload(
                &checkpoint.input_desc, checkpoint.input_payload.data(), &source_part));
        if (source == nullptr) return {};
        for (int64_t i3 = 0; i3 < part.dimensions[3]; ++i3) {
            for (int64_t i2 = 0; i2 < part.dimensions[2]; ++i2) {
                for (int64_t i1 = 0; i1 < part.dimensions[1]; ++i1) {
                    const size_t source_offset = static_cast<size_t>(i1) * source_part.byte_strides[1] +
                            static_cast<size_t>(i2) * source_part.byte_strides[2] + static_cast<size_t>(i3) * source_part.byte_strides[3];
                    payload.insert(payload.end(), source + source_offset, source + source_offset + row_bytes);
                }
            }
        }
        part.payload_bytes = payload.size() - static_cast<size_t>(part.payload_offset);
        prefix_desc.payload_bytes += part.payload_bytes;
    }
    return payload;
}


static bool skippy_trim_verify_attention(
        llama_memory_t memory,
        llama_seq_id seq_id,
        llama_pos token_start) {
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return iswa->seq_rm(seq_id, token_start, -1);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        return hybrid->get_mem_attn()->seq_rm(seq_id, token_start, -1);
    }
    if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        return hybrid_iswa->get_mem_attn()->seq_rm(seq_id, token_start, -1);
    }
    return true;
}

static void skippy_mark_verify_restore_failed(skippy_session * session) {
    session->reset_required = true;
    session->verify_checkpoints.clear();
    skippy_mtp_clear_session_state(session);
}

static enum skippy_status skippy_fail_verify_restore(
        skippy_session * session,
        enum skippy_status status,
        const char * message,
        skippy_error ** out_error) {
    skippy_mark_verify_restore_failed(session);
    skippy_set_error(out_error, status, message);
    return status;
}

enum skippy_status skippy_restore_verify_prefix(
        skippy_session * session,
        uint64_t token_count,
        bool * out_restored,
        skippy_error ** out_error) {
    *out_restored = false;
    const skippy_verify_checkpoint * checkpoint = nullptr;
    for (auto it = session->verify_checkpoints.rbegin(); it != session->verify_checkpoints.rend(); ++it) {
        const uint64_t token_start = static_cast<uint64_t>(it->token_start);
        const uint64_t token_end = token_start + it->token_count;
        if (it->valid && token_count >= token_start && token_count < token_end) {
            checkpoint = &*it;
            break;
        }
    }
    if (checkpoint == nullptr) {
        return skippy_success(out_error);
    }

    const size_t read = llama_state_seq_set_data_ext(
            session->ctx,
            checkpoint->state.data(),
            checkpoint->state.size(),
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (read != checkpoint->state.size()) {
        return skippy_fail_verify_restore(
                session,
                SKIPPY_STATUS_RUNTIME_ERROR,
                "failed to restore verify-window sequence state",
                out_error);
    }

    llama_memory_t memory = session->ctx->get_memory();
    if (!skippy_trim_verify_attention(memory, session->seq_id, checkpoint->token_start)) {
        return skippy_fail_verify_restore(
                session,
                SKIPPY_STATUS_RUNTIME_ERROR,
                "failed to trim verify-window attention suffix",
                out_error);
    }

    const uint64_t token_start = static_cast<uint64_t>(checkpoint->token_start);
    session->n_past = checkpoint->token_start;
    const size_t accepted_count = static_cast<size_t>(token_count - token_start);
    if (accepted_count > 0 && checkpoint->token_ids.empty()) {
        return skippy_fail_verify_restore(
                session,
                SKIPPY_STATUS_RUNTIME_ERROR,
                "verify-window checkpoint is missing token ids",
                out_error);
    }

    enum skippy_status status = SKIPPY_STATUS_OK;
    if (accepted_count > 0 && checkpoint->has_activation_input) {
        skippy_activation_desc input_desc = {};
        std::vector<uint8_t> input_payload = skippy_verify_prefix_activation_payload(
                *checkpoint,
                accepted_count,
                input_desc);
        if (input_payload.empty()) {
            return skippy_fail_verify_restore(
                    session,
                    SKIPPY_STATUS_RUNTIME_ERROR,
                    "verify-window checkpoint has an invalid activation prefix payload",
                    out_error);
        }
        status = skippy_decode_activation_frame(
                session,
                &input_desc,
                input_payload.data(),
                checkpoint->token_ids.data(),
                nullptr,
                0,
                accepted_count,
                false,
                SKIPPY_GLM_DSA_PHASE_HINT_VERIFY,
                out_error);
    } else if (accepted_count > 0) {
        status = skippy_decode_tokens(
                session,
                checkpoint->token_ids.data(),
                accepted_count,
                false,
                SKIPPY_GLM_DSA_PHASE_HINT_VERIFY,
                out_error);
    }
    if (status != SKIPPY_STATUS_OK) {
        skippy_mark_verify_restore_failed(session);
        return status;
    }
    *out_restored = true;
    return SKIPPY_STATUS_OK;
}
