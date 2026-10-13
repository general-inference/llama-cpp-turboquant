#include "skippy.h"
#include "skippy/activation_internal.h"
#include "skippy/errors.h"
#include "skippy/graph_build_inputs.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"
#include "skippy/sampling_internal.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <vector>

static bool skippy_batch_part_compatible(
        const skippy_activation_part_desc & left,
        const skippy_activation_part_desc & right) {
    if (std::memcmp(left.identity, right.identity, SKIPPY_ACTIVATION_IDENTITY_BYTES) != 0 ||
            left.ggml_type != right.ggml_type || left.rank != right.rank ||
            left.token_axis != right.token_axis || left.flags != right.flags) return false;
    for (uint32_t dimension = 0; dimension < left.rank; ++dimension) {
        if (static_cast<int32_t>(dimension) != left.token_axis &&
                left.dimensions[dimension] != right.dimensions[dimension]) return false;
    }
    return true;
}

static enum skippy_status skippy_build_batched_frontier(
        const skippy_iteration_request * requests,
        size_t request_count,
        const std::vector<size_t> & offsets,
        skippy_graph_build_inputs & graph_inputs,
        std::vector<std::vector<uint8_t>> & storage,
        skippy_error ** out_error) {
    if (requests[0].input_desc == nullptr) return SKIPPY_STATUS_OK;
    const auto & boundary = requests[0].session->stage_model->input_activation_boundary;
    if (requests[0].input_desc->part_count != boundary.part_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "iteration batching accepts planned frontier parts only");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    storage.reserve(boundary.part_count);
    graph_inputs.activation_frontier.reserve(boundary.part_count);
    for (size_t part_index = 0; part_index < boundary.part_count; ++part_index) {
        const auto * first = skippy_activation_find_part(
                requests[0].input_desc, boundary.parts[part_index].identity);
        if (first == nullptr || first->token_axis < 0 ||
                first->token_axis >= static_cast<int32_t>(first->rank)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "batched activation frontier is missing a planned part");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        const size_t token_axis = static_cast<size_t>(first->token_axis);
        const size_t inner_bytes = static_cast<size_t>(first->byte_strides[token_axis]);
        const size_t first_plane_bytes = inner_bytes*requests[0].token_count;
        if (inner_bytes == 0 || first_plane_bytes == 0 ||
                first->payload_bytes % first_plane_bytes != 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "activation part cannot be concatenated along its token axis");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        const size_t outer_count = static_cast<size_t>(first->payload_bytes)/first_plane_bytes;
        const size_t batch_plane_bytes = inner_bytes*offsets.back();
        storage.emplace_back(batch_plane_bytes*outer_count);
        auto & combined = storage.back();
        for (size_t request_index = 0; request_index < request_count; ++request_index) {
            const auto * desc = requests[request_index].input_desc;
            if (desc == nullptr || desc->part_count != boundary.part_count) {
                skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "iteration batching requires the same planned frontier for every request");
                return SKIPPY_STATUS_INVALID_ARGUMENT;
            }
            const auto * part = skippy_activation_find_part(desc, first->identity);
            const size_t request_plane_bytes = inner_bytes*requests[request_index].token_count;
            if (part == nullptr || !skippy_batch_part_compatible(*first, *part) ||
                    part->payload_bytes != request_plane_bytes*outer_count) {
                skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "iteration activation shapes differ outside the token axis");
                return SKIPPY_STATUS_INVALID_ARGUMENT;
            }
            const auto * source = static_cast<const uint8_t *>(
                    skippy_activation_part_payload(desc, requests[request_index].input_payload, part));
            for (size_t outer = 0; outer < outer_count; ++outer) {
                std::memcpy(
                        combined.data() + outer*batch_plane_bytes + offsets[request_index]*inner_bytes,
                        source + outer*request_plane_bytes,
                        request_plane_bytes);
            }
        }
        skippy_activation_frontier_part_input binding;
        binding.values = combined.data();
        binding.payload_bytes = combined.size();
        binding.ggml_type = first->ggml_type;
        binding.rank = first->rank;
        binding.token_axis = first->token_axis;
        binding.pos_start = 0;
        for (size_t dimension = 0; dimension < SKIPPY_ACTIVATION_MAX_DIMS; ++dimension) {
            binding.dimensions[dimension] = first->dimensions[dimension];
            binding.byte_strides[dimension] = first->byte_strides[dimension];
        }
        binding.dimensions[token_axis] = static_cast<int64_t>(offsets.back());
        for (size_t request_index = 0; request_index < request_count; ++request_index) {
            binding.sequence_ids.push_back(requests[request_index].session->seq_id);
            binding.sequence_offsets.push_back(offsets[request_index]);
            binding.sequence_pos_starts.push_back(requests[request_index].positions != nullptr ?
                    requests[request_index].positions[0] : requests[request_index].session->n_past);
        }
        graph_inputs.activation_frontier.push_back(std::move(binding));
    }
    return SKIPPY_STATUS_OK;
}

static enum skippy_status skippy_copy_batched_exports(
        skippy_session * session,
        const std::vector<size_t> & offsets,
        size_t request_index,
        skippy_activation_desc * desc,
        void * payload,
        skippy_error ** out_error) {
    if (!skippy_emits_activation_frame(session)) return SKIPPY_STATUS_OK;
    const auto & boundary = session->stage_model->output_activation_boundary;
    if (desc == nullptr || desc->part_count != boundary.part_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "iteration batching accepts planned frontier parts only");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    llm_graph_result * result = session->ctx->get_gf_res_prev_active();
    const auto * ports = result == nullptr ? nullptr : &result->get_skippy_activation_exports();
    if (ports == nullptr || ports->size() != boundary.part_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                "live graph exports do not match the batched activation directory");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    const size_t token_count = offsets[request_index + 1] - offsets[request_index];
    const enum skippy_status shape_status = skippy_prepare_output_activation_frame(
            session, token_count, payload, desc->payload_bytes, nullptr, desc, nullptr, out_error,
            skippy_activation_frame_shape::live_within_reservation);
    if (shape_status != SKIPPY_STATUS_OK) return shape_status;
    for (size_t part_index = 0; part_index < boundary.part_count; ++part_index) {
        const auto & part = desc->parts[part_index];
        ggml_tensor * tensor = (*ports)[part_index].tensor;
        if (part.rank == 0 || part.rank > SKIPPY_ACTIVATION_MAX_DIMS ||
                part.token_axis < 0 || part.token_axis >= static_cast<int32_t>(part.rank)) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                    "batched graph export has an invalid token axis");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        const size_t token_axis = static_cast<size_t>(part.token_axis);
        const size_t inner_bytes = static_cast<size_t>(part.byte_strides[token_axis]);
        const size_t destination_plane_bytes = inner_bytes*token_count;
        if (tensor == nullptr || tensor->type != static_cast<ggml_type>(part.ggml_type) ||
                tensor->nb[token_axis] != inner_bytes || destination_plane_bytes == 0 ||
                part.payload_bytes % destination_plane_bytes != 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                    "batched graph export does not match its activation descriptor");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        // Exports come from the last microbatch, so slicing them by request
        // offset is only defined while that microbatch covers the whole batch.
        if (tensor->ne[token_axis] != static_cast<int64_t>(offsets.back())) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                    "exported activation token axis does not cover the batched iteration");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        const size_t outer_count = static_cast<size_t>(part.payload_bytes)/destination_plane_bytes;
        auto * destination = static_cast<uint8_t *>(payload) + part.payload_offset;
        for (size_t outer = 0; outer < outer_count; ++outer) {
            size_t remainder = outer;
            size_t source_offset = offsets[request_index]*inner_bytes;
            for (uint32_t dimension = static_cast<uint32_t>(token_axis + 1);
                    dimension < part.rank; ++dimension) {
                const size_t coordinate =
                        remainder % static_cast<size_t>(part.dimensions[dimension]);
                remainder /= static_cast<size_t>(part.dimensions[dimension]);
                source_offset += coordinate*tensor->nb[dimension];
            }
            ggml_backend_tensor_get(tensor, destination + outer*destination_plane_bytes,
                    source_offset, destination_plane_bytes);
        }
    }
    return SKIPPY_STATUS_OK;
}

extern "C" {

enum skippy_status skippy_iteration_batch_sampled(
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
        struct skippy_error ** out_error) {
    if (requests == nullptr || request_count == 0 || out_sampled_output_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "requests, request_count, and out_sampled_output_count are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    skippy_session * first = requests[0].session;
    if (first == nullptr || first->ctx == nullptr || first->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "requests must use active sessions");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const int32_t n_pos_per_embd = first->stage_model->model->hparams.n_pos_per_embd();
    const bool request_logits = skippy_runtime_is_terminal_stage(&first->stage_model->config);
    const bool request_exports = skippy_emits_activation_frame(first);
    if (n_pos_per_embd <= 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "stage input dimensions cannot be batched");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    std::unordered_set<skippy_session *> unique_sessions;
    std::vector<size_t> offsets(request_count + 1, 0);
    size_t sampled_count = 0;
    for (size_t index = 0; index < request_count; ++index) {
        const auto & request = requests[index];
        if (request.session == nullptr || request.session->ctx != first->ctx ||
                request.session->stage_model != first->stage_model || request.token_ids == nullptr ||
                request.token_count == 0 || !unique_sessions.insert(request.session).second) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "iteration requests must use unique sessions on one stage model");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (offsets[index] > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - request.token_count) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "iteration token count exceeds int32 range");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        const size_t expected_positions = request.token_count*static_cast<size_t>(n_pos_per_embd);
        if ((request.positions == nullptr && request.position_count != 0) ||
                (request.positions != nullptr && request.position_count != expected_positions)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "iteration position sideband shape is invalid");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        enum skippy_status status = skippy_validate_frame_input(
                request.session, request.input_desc, request.input_payload,
                request.token_count, out_error);
        if (status != SKIPPY_STATUS_OK) return status;
        if (request.input_desc != nullptr &&
                request.input_desc->part_count != request.session->stage_model->input_activation_boundary.part_count) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "iteration batching accepts planned frontier parts only");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        status = skippy_prepare_output_activation_frame(
                request.session, request.token_count,
                output_payloads == nullptr ? nullptr : output_payloads[index],
                output_payload_capacities == nullptr ? 0 : output_payload_capacities[index],
                out_output_payload_bytes == nullptr ? nullptr : &out_output_payload_bytes[index],
                output_descs == nullptr ? nullptr : &output_descs[index],
                request.input_desc, out_error);
        if (status != SKIPPY_STATUS_OK) return status;
        if (output_descs != nullptr && request_exports &&
                output_descs[index].part_count != request.session->stage_model->output_activation_boundary.part_count) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "iteration batching accepts planned frontier parts only");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        offsets[index + 1] = offsets[index] + request.token_count;
        if (request.sample_last && request_logits) ++sampled_count;
    }
    // Batched exports are sliced by request offset, which is the microbatch row
    // order only when llama decodes the batch with split_simple, so a stage that
    // exports activations needs a unified KV cache. Multi-lane exporting stages
    // cannot turn that off (model_loading.cpp rejects the combination), and a
    // terminal stage never reads exports.
    if (request_exports && request_count > 1 && !first->ctx->kv_unified()) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "batched activation exports require a unified KV cache");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    // Recurrent and hybrid memory splits an all-output batch by sequence
    // (split_seq), so the last microbatch only carries one request's rows and
    // its exports cannot be sliced by request offset. The runtime routes those
    // stages one request at a time; reaching this means that routing is wrong.
    if (request_exports && request_count > 1 &&
            (llm_arch_is_recurrent(first->stage_model->model->arch) ||
             llm_arch_is_hybrid(first->stage_model->model->arch))) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "recurrent and hybrid stages cannot batch activation exports");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    // Exports are read from the last microbatch's graph, so an exporting stage
    // must see the whole batch in one microbatch. The scheduler caps batched
    // tokens at n_ubatch; exceeding it here is a scheduler bug.
    if (request_exports && offsets.back() > static_cast<size_t>(first->ctx->n_ubatch())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "iteration batch exceeds the native microbatch on an exporting stage");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (sampled_count > sampled_output_capacity ||
            (sampled_count > 0 &&
             (out_sampled_request_indexes == nullptr || out_predicted_tokens == nullptr))) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL,
                "sampled iteration output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    *out_sampled_output_count = sampled_count;

    skippy_graph_build_inputs graph_inputs = skippy_graph_build_inputs_from_config(
            &first->stage_model->config, SKIPPY_GLM_DSA_PHASE_HINT_AUTO);
    std::vector<std::vector<uint8_t>> frontier_storage;
    enum skippy_status status = skippy_build_batched_frontier(
            requests, request_count, offsets, graph_inputs, frontier_storage, out_error);
    if (status != SKIPPY_STATUS_OK) return status;

    const size_t total_tokens = offsets.back();
    std::vector<llama_token> tokens(total_tokens);
    std::vector<llama_pos> positions(total_tokens*static_cast<size_t>(n_pos_per_embd));
    std::vector<int32_t> n_seq_id(total_tokens, 1);
    std::vector<llama_seq_id> seq_values(total_tokens);
    std::vector<llama_seq_id *> seq_ids(total_tokens);
    std::vector<int8_t> logits(total_tokens, 0);
    for (size_t request_index = 0; request_index < request_count; ++request_index) {
        const auto & request = requests[request_index];
        std::copy(request.token_ids, request.token_ids + request.token_count,
                tokens.begin() + offsets[request_index]);
        for (size_t token = 0; token < request.token_count; ++token) {
            const size_t flat = offsets[request_index] + token;
            seq_values[flat] = request.session->seq_id;
            seq_ids[flat] = &seq_values[flat];
            logits[flat] = request_exports ||
                    (request_logits && request.sample_last && token + 1 == request.token_count);
            for (int32_t axis = 0; axis < n_pos_per_embd; ++axis) {
                const size_t destination = static_cast<size_t>(axis)*total_tokens + flat;
                if (request.positions != nullptr) {
                    positions[destination] =
                            request.positions[static_cast<size_t>(axis)*request.token_count + token];
                } else {
                    const llama_pos value =
                            request.session->n_past + static_cast<llama_pos>(token);
                    positions[destination] = axis == 3 ? 0 : value;
                }
            }
        }
    }
    skippy_graph_build_inputs_set_activation_tokens(graph_inputs, tokens.data(), total_tokens);
    llama_batch batch = {
        static_cast<int32_t>(total_tokens),
        tokens.data(),
        nullptr,
        positions.data(),
        n_seq_id.data(),
        seq_ids.data(),
        logits.data(),
    };
    int32_t decode_rc = 0;
    if (!skippy_contain_native_failure(
                out_error, [&] { decode_rc = first->ctx->decode(batch, graph_inputs); })) {
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (decode_rc != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                "llama_decode failed for scheduler iteration");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    skippy_mark_context_work_pending(first->stage_model);
    skippy_synchronize_context_if_pending(first->stage_model);

    size_t sampled_index = 0;
    for (size_t request_index = 0; request_index < request_count; ++request_index) {
        const auto & request = requests[request_index];
        status = skippy_copy_batched_exports(
                request.session, offsets, request_index,
                output_descs == nullptr ? nullptr : &output_descs[request_index],
                output_payloads == nullptr ? nullptr : output_payloads[request_index],
                out_error);
        if (status != SKIPPY_STATUS_OK) return status;
        if (out_output_payload_bytes != nullptr && output_descs != nullptr) {
            out_output_payload_bytes[request_index] = output_descs[request_index].payload_bytes;
        }
        request.session->n_past += static_cast<int32_t>(request.token_count);
        skippy_record_tokens(request.session, request.token_ids, request.token_count);
        if (request.sample_last && request_logits) {
            // Logits are indexed by the original batch row; llama.cpp handles
            // sparse output-row translation internally.
            const skippy_status sample_status = skippy_store_sampled_token(
                    skippy_sample_token_ith(
                            request.session, request.sampling,
                            static_cast<int32_t>(offsets[request_index + 1] - 1)),
                    &out_predicted_tokens[sampled_index], out_error);
            if (sample_status != SKIPPY_STATUS_OK) return sample_status;
            out_sampled_request_indexes[sampled_index++] = request_index;
        }
    }
    *out_sampled_output_count = sampled_index;
    return skippy_success(out_error);
}

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
        struct skippy_error ** out_error) {
    if (sessions == nullptr || token_ids == nullptr ||
            out_predicted_tokens == nullptr || request_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "sessions, token_ids, request_count, and predicted outputs are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (predicted_token_capacity < request_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL,
                "predicted token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    std::vector<skippy_iteration_request> requests(request_count);
    for (size_t index = 0; index < request_count; ++index) {
        requests[index].session = sessions[index];
        requests[index].token_ids = &token_ids[index];
        requests[index].token_count = 1;
        requests[index].sampling = sampling == nullptr ? nullptr : sampling[index];
        requests[index].input_desc = input_descs == nullptr ? nullptr : input_descs[index];
        requests[index].input_payload =
                input_payloads == nullptr ? nullptr : input_payloads[index];
        requests[index].sample_last = true;
    }
    std::vector<size_t> sampled_indexes(request_count);
    size_t sampled_count = 0;
    const enum skippy_status status = skippy_iteration_batch_sampled(
            requests.data(), request_count, output_descs, output_payloads,
            output_payload_capacities, out_output_payload_bytes, sampled_indexes.data(),
            out_predicted_tokens, predicted_token_capacity, &sampled_count, out_error);
    if (status != SKIPPY_STATUS_OK) return status;
    if (sampled_count != request_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                "batched decode did not sample every request");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    return skippy_success(out_error);
}

} // extern "C"
