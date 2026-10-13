#pragma once

#include "skippy/activation.h"
#include "skippy/runtime.h"
#include "skippy/activation_internal.h"
#include "skippy/runtime_support.h"

#include "llama-graph.h"

#include <cstddef>
#include <cstdint>
#include <limits>

inline skippy_graph_build_inputs skippy_graph_build_inputs_from_config(
        const skippy_runtime_config * config,
        int32_t glm_dsa_phase_hint = SKIPPY_GLM_DSA_PHASE_HINT_AUTO) {
    skippy_graph_build_inputs inputs;
    const bool has_phase_hint = glm_dsa_phase_hint != SKIPPY_GLM_DSA_PHASE_HINT_AUTO;
    if ((config != nullptr && (skippy_runtime_has_stage_plan(config) || config->glm_dsa_policy_profile != 0)) ||
            has_phase_hint) {
        inputs.selection.enabled = skippy_runtime_has_stage_plan(config);
        if (config != nullptr) {
            if (config->execution_contract != nullptr) inputs.execution_contract = config->execution_contract;
            inputs.selection.layer_start = config->layer_start;
            inputs.selection.layer_end = config->layer_end;
            inputs.selection.glm_dsa_policy_profile = config->glm_dsa_policy_profile;
            inputs.selection.glm_dsa_policy_flags = config->glm_dsa_policy_flags;
            inputs.selection.glm_dsa_short_prefill_max_tokens = config->glm_dsa_short_prefill_max_tokens;
            inputs.selection.glm_dsa_direct_sparse_decode_max_top_k = config->glm_dsa_direct_sparse_decode_max_top_k;
            inputs.selection.glm_dsa_dense_sparse_mask_max_bytes = config->glm_dsa_dense_sparse_mask_max_bytes;
            inputs.selection.glm_dsa_compact_flash_min_kv = config->glm_dsa_compact_flash_min_kv;
            if (config->activation_import_bindings != nullptr) {
                for (size_t index = 0; index < config->activation_import_identity_count; ++index) {
                    inputs.activation_import_bindings.emplace_back(config->activation_import_bindings[index]);
                }
            }
            if (config->activation_import_identities != nullptr) {
                for (size_t index = 0; index < config->activation_import_identity_count; ++index) {
                    inputs.activation_import_identities.emplace_back(config->activation_import_identities[index]);
                }
            }
            if (config->activation_export_bindings != nullptr) {
                for (size_t index = 0; index < config->activation_export_identity_count; ++index) {
                    inputs.activation_export_bindings.emplace_back(config->activation_export_bindings[index]);
                }
            }
            if (config->activation_export_identities != nullptr) {
                for (size_t index = 0; index < config->activation_export_identity_count; ++index) {
                    inputs.activation_export_identities.emplace_back(config->activation_export_identities[index]);
                }
            }
            if (config->resident_tensor_names != nullptr) {
                for (size_t index = 0; index < config->resident_tensor_name_count; ++index) {
                    inputs.resident_tensor_names.emplace_back(config->resident_tensor_names[index]);
                }
            }
        }
        inputs.selection.glm_dsa_phase_hint = glm_dsa_phase_hint;
    }
    return inputs;
}

inline void skippy_graph_build_inputs_set_activation_tokens(
        skippy_graph_build_inputs & inputs,
        const llama_token * tokens,
        size_t token_count) {
    if (tokens != nullptr && token_count > 0 && token_count <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        inputs.activation_tokens.tokens = tokens;
        inputs.activation_tokens.token_count = static_cast<uint32_t>(token_count);
    }
}

inline void skippy_graph_build_inputs_set_activation_frontier(
        skippy_graph_build_inputs & inputs,
        const skippy_activation_boundary_desc & boundary,
        const skippy_activation_desc * desc,
        const void * payload,
        llama_pos pos_start) {
    inputs.activation_frontier.clear();
    inputs.activation_frontier.reserve(boundary.part_count);
    for (size_t index = 0; index < boundary.part_count; ++index) {
        const auto * part = skippy_activation_find_part(desc, boundary.parts[index].identity);
        skippy_activation_frontier_part_input binding;
        if (part != nullptr) {
            binding.values = skippy_activation_part_payload(desc, payload, part);
            binding.payload_bytes = part->payload_bytes;
            binding.ggml_type = part->ggml_type;
            binding.rank = part->rank;
            binding.token_axis = part->token_axis;
            for (size_t dimension = 0; dimension < SKIPPY_ACTIVATION_MAX_DIMS; ++dimension) {
                binding.dimensions[dimension] = part->dimensions[dimension];
                binding.byte_strides[dimension] = part->byte_strides[dimension];
            }
            binding.pos_start = pos_start;
        }
        inputs.activation_frontier.push_back(binding);
    }
}

inline void skippy_graph_build_inputs_set_mtp_embeddings(
        skippy_graph_build_inputs & inputs,
        const float * values,
        uint32_t token_count,
        uint32_t n_embd,
        llama_pos pos_start) {
    if (values != nullptr) {
        inputs.mtp_embeddings = { values, token_count, n_embd, pos_start };
    }
}
