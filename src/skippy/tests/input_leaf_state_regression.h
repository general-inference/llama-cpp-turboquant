#pragma once

static int check_input_leaf_state_inference() {
    ggml_init_params params = {ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * absent = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    auto * state = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(input);
    ggml_set_input(absent);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(state)), ggml_backend_buffer_free);
    if (!buffer) { return 92; }
    state->buffer = buffer.get();
    auto * first = ggml_scale(ctx.get(), input, 2.0f);
    auto * write = ggml_cpy(ctx.get(), first, state);
    auto * output = ggml_add(ctx.get(), first, state);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, write);
    ggml_build_forward_expand(graph, output);
    std::vector<skippy_block_boundary> boundaries = {
        {skippy_block_boundary_kind::entry, 0, input},
        {skippy_block_boundary_kind::exit, 0, output},
    };
    std::vector<skippy_state_effect> effects;
    std::string error;
    if (!skippy_infer_state_effects({}, graph, boundaries, effects, error) || effects.size() != 2 ||
        effects[0].access != skippy_state_access::read || effects[1].access != skippy_state_access::write ||
        effects[1].layer != 0 || effects[1].storage != state || effects[1].effect_node != write ||
        effects[1].write_ordinal != 0 || effects[1].kind != skippy_state_kind::derived_persistent) {
        std::fprintf(stderr, "input-leaf state inference failed: %s\n", error.c_str());
        return 93;
    }
    // A leaf outside this graph and a missing exit remain invalid boundaries.
    boundaries[0].tensor = absent;
    if (skippy_infer_state_effects({}, graph, boundaries, effects, error)) { return 94; }
    boundaries[0].tensor = input;
    boundaries[1].tensor = absent;
    if (skippy_infer_state_effects({}, graph, boundaries, effects, error)) { return 95; }
    // Only the initial block can start before executable graph nodes.
    boundaries = {
        {skippy_block_boundary_kind::entry, 0, input},
        {skippy_block_boundary_kind::exit, 0, first},
        {skippy_block_boundary_kind::entry, 1, input},
        {skippy_block_boundary_kind::exit, 1, output},
    };
    if (skippy_infer_state_effects({}, graph, boundaries, effects, error)) { return 96; }
    return 0;
}
