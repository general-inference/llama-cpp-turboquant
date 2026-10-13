#pragma once

// An initial raw-embedding boundary is also a request leaf. Its reshaped
// checkpoint must cross the split, unlike ordinary request-only auxiliaries.
static int check_raw_embedding_checkpoint() {
    ggml_init_params params = {
        ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true,
    };
    ggml_context_ptr context(ggml_init(params));
    auto * input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * positions = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * weight = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    ggml_set_input(input);
    ggml_set_input(positions);
    auto * checkpoint = ggml_reshape_3d(context.get(), input, 4, 1, 1);
    auto * auxiliary = ggml_scale(context.get(), positions, 2.0f);
    auto * first = ggml_add(context.get(), ggml_mul(context.get(), input, weight), checkpoint);
    auto * output = ggml_add(context.get(), ggml_add(context.get(), first, checkpoint), auxiliary);
    auto * graph = ggml_new_graph_custom(context.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    const std::vector<skippy_block_boundary> boundaries = {
        {skippy_block_boundary_kind::entry, 0, input},
        {skippy_block_boundary_kind::exit, 0, first},
        {skippy_block_boundary_kind::entry, 1, first},
        {skippy_block_boundary_kind::exit, 1, output},
    };
    const std::vector<skippy_parameter_binding> parameters = {{weight, "weight"}};
    const std::vector<skippy_request_input_binding> requests = {
        {input, "embeddings"}, {positions, "positions"},
    };
    const std::vector<skippy_state_effect> effects;
    const skippy_execution_profile_guard guard = {1, 1, 1, 0, "checkpoint", "cpu"};
    const std::vector<skippy_stage_profile_input> profiles = {{
        "embeddings:decode", graph, &boundaries, &effects, &parameters, &requests, 2, guard, output,
    }};
    skippy_guarded_stage_slice_plan source, final;
    std::string error;
    if (!skippy_build_guarded_stage_slice_plan(profiles, 0, 1, source, error) ||
        !skippy_build_guarded_stage_slice_plan(profiles, 1, 2, final, error) ||
        !skippy_validate_guarded_stage_chain({&source, &final}, error)) {
        std::fprintf(stderr, "raw embedding checkpoint chain failed: %s\n", error.c_str());
        return 90;
    }
    const auto & exports = source.profiles.front().slice.activation_exports;
    const auto & imports = final.profiles.front().slice.activation_imports;
    if (!contains_tensor(exports, checkpoint) || !contains_tensor(imports, checkpoint) ||
        contains_tensor(exports, auxiliary) || contains_tensor(imports, auxiliary)) {
        std::fprintf(stderr, "raw embedding checkpoint or request-only auxiliary misclassified\n");
        return 91;
    }
    return 0;
}
