#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;
struct skippy_block_boundary;
struct skippy_activation_part_desc;
struct skippy_execution_profile_guard;
struct skippy_guarded_stage_slice_plan;
struct skippy_stage_slice_selection;
struct skippy_state_effect;

bool skippy_build_activation_frontier_identity(
        const skippy_activation_part_desc * parts,
        size_t part_count,
        uint8_t output[32]);

bool skippy_build_activation_port_identity(
        const std::string & binding,
        std::string & identity);

struct skippy_parameter_binding {
    const ggml_tensor * tensor = nullptr;
    std::string identity;
};

struct skippy_request_input_binding {
    const ggml_tensor * tensor = nullptr;
    std::string identity;
};

struct skippy_graph_semantics {
    std::string identity;
    std::unordered_map<const ggml_tensor *, std::string> value_identities;
};

bool skippy_build_graph_semantics(
        const ggml_cgraph * graph,
        const std::vector<skippy_block_boundary> & boundaries,
        const std::vector<skippy_state_effect> & state_effects,
        const std::vector<skippy_parameter_binding> & parameters,
        const std::vector<skippy_request_input_binding> & request_inputs,
        skippy_graph_semantics & result,
        std::string & error);

bool skippy_finalize_stage_slice_identity(
        const skippy_graph_semantics & graph,
        skippy_stage_slice_selection & selection,
        std::string & error);

bool skippy_validate_stage_slice_identity(
        const skippy_stage_slice_selection & selection,
        std::string & error);

bool skippy_build_profile_identity(
        const std::string & profile_id,
        const std::string & graph_identity,
        const skippy_execution_profile_guard & guard,
        std::string & identity,
        std::string & error);

bool skippy_build_stage_plan_identity(
        const std::string & package_id,
        const skippy_guarded_stage_slice_plan & plan,
        std::string & identity,
        std::string & error);
