#pragma once

#include "block_boundaries.h"
#include "graph_identity.h"
#include "state_effects.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

struct skippy_stage_slice_selection {
    int32_t layer_start = 0;
    int32_t layer_end = 0;
    const ggml_cgraph * source_graph = nullptr;
    std::vector<int32_t> node_indices;
    std::vector<const ggml_tensor *> activation_imports;
    std::vector<const ggml_tensor *> activation_exports;
    std::vector<const ggml_tensor *> request_inputs;
    std::vector<std::string> parameter_identities;
    std::vector<skippy_state_effect> state_effects;
    std::string graph_identity;
    std::string slice_identity;
    std::vector<std::string> node_identities;
    std::vector<std::string> activation_import_identities;
    std::vector<std::string> activation_export_identities;
    std::vector<std::string> request_input_identities;
    std::vector<std::string> state_effect_identities;
    std::string source_snapshot_identity;
    std::unordered_map<const ggml_tensor *, std::string> source_value_identities;
    std::unordered_map<const ggml_tensor *, std::string> source_value_metadata_identities;
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> source_value_references;
};

struct skippy_execution_profile_guard {
    uint32_t n_tokens = 0;
    uint32_t n_sequences = 0;
    uint32_t n_outputs = 0;
    uint32_t n_recurrent_rollback_sequences = 0;
    std::string graph_configuration_id;
    std::string backend_id;
};

struct skippy_stage_profile_input {
    std::string profile_id;
    ggml_cgraph * graph = nullptr;
    const std::vector<skippy_block_boundary> * boundaries = nullptr;
    const std::vector<skippy_state_effect> * state_effects = nullptr;
    const std::vector<skippy_parameter_binding> * parameters = nullptr;
    const std::vector<skippy_request_input_binding> * request_inputs = nullptr;
    int32_t layer_count = 0;
    skippy_execution_profile_guard guard;
    const ggml_tensor * explicit_output = nullptr;
};

struct skippy_stage_profile_plan {
    std::string profile_id;
    const ggml_cgraph * graph = nullptr;
    skippy_execution_profile_guard guard;
    std::string graph_identity;
    std::string profile_identity;
    skippy_stage_slice_selection slice;
};

struct skippy_auxiliary_graph_plan {
    std::string graph_identity;
    std::vector<std::string> parameter_identities;
};

struct skippy_auxiliary_profile_plan {
    std::string profile_id;
    std::string graph_identity;
    std::vector<std::string> parameter_identities;
    int32_t nextn_layer_offset = 0;
};

struct skippy_guarded_stage_slice_plan {
    int32_t layer_count = 0;
    int32_t layer_start = 0;
    int32_t layer_end = 0;
    std::vector<skippy_stage_profile_plan> profiles;
    std::vector<skippy_auxiliary_profile_plan> auxiliary_profiles;
    std::vector<std::string> resident_parameter_identities;
};

bool skippy_build_auxiliary_graph_plan(
        ggml_cgraph * graph,
        const std::vector<skippy_state_effect> & state_effects,
        const std::vector<skippy_parameter_binding> & parameters,
        const std::vector<skippy_request_input_binding> & request_inputs,
        skippy_auxiliary_graph_plan & plan,
        std::string & error);

bool skippy_select_stage_slice(
        ggml_cgraph * graph,
        const std::vector<skippy_block_boundary> & boundaries,
        const std::vector<skippy_state_effect> & state_effects,
        const std::vector<skippy_parameter_binding> & parameters,
        const std::vector<skippy_request_input_binding> & request_inputs,
        int32_t layer_count,
        int32_t layer_start,
        int32_t layer_end,
        skippy_stage_slice_selection & selection,
        std::string & error,
        const ggml_tensor * explicit_output = nullptr);

bool skippy_build_guarded_stage_slice_plan(
        const std::vector<skippy_stage_profile_input> & profiles,
        int32_t layer_start,
        int32_t layer_end,
        skippy_guarded_stage_slice_plan & plan,
        std::string & error);

bool skippy_validate_guarded_stage_chain(
        const std::vector<const skippy_guarded_stage_slice_plan *> & stages,
        std::string & error);
