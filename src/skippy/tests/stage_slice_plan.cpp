#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-model.h"
#include "skippy/executable_graph.h"
#include "../../../ggml/src/ggml-impl.h"
#include "skippy/runtime_support.h"
#include "skippy/stage_slice_plan.h"
#include "skippy/state_effects.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static bool contains_tensor(
        const std::vector<const ggml_tensor *> & tensors,
        const ggml_tensor * expected) {
    return std::find(tensors.begin(), tensors.end(), expected) != tensors.end();
}

struct semantic_fixture {
    explicit semantic_fixture(const char * prefix) {
        ggml_init_params params = {
            ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true,
        };
        context.reset(ggml_init(params));
        request = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        parameter = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        state = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
        ggml_set_input(request);
        ggml_set_name(request, (std::string(prefix) + ".request").c_str());
        ggml_set_name(parameter, (std::string(prefix) + ".parameter").c_str());
        ggml_set_name(state, (std::string(prefix) + ".state").c_str());
        entry = ggml_mul(context.get(), request, parameter);
        side = ggml_scale(context.get(), entry, 2.0f);
        exit_0 = ggml_add(context.get(), entry, side);
        prepared = ggml_add(context.get(), exit_0, state);
        write_0 = ggml_cpy(context.get(), prepared, state);
        exit_1 = ggml_mul(context.get(), prepared, parameter);
        write_1 = ggml_cpy(context.get(), exit_1, state);
        ggml_set_name(entry, (std::string(prefix) + ".entry").c_str());
        ggml_set_name(exit_0, (std::string(prefix) + ".exit0").c_str());
        ggml_set_name(exit_1, (std::string(prefix) + ".exit1").c_str());
        graph = ggml_new_graph_custom(context.get(), 64, false);
        ggml_build_forward_expand(graph, write_0);
        ggml_build_forward_expand(graph, write_1);
        ggml_build_forward_expand(graph, exit_1);
        boundaries = {
            {skippy_block_boundary_kind::entry, 0, entry},
            {skippy_block_boundary_kind::exit, 0, exit_0},
            {skippy_block_boundary_kind::entry, 1, exit_0},
            {skippy_block_boundary_kind::exit, 1, exit_1},
        };
        effects = {
            {skippy_state_kind::recurrent_conv, skippy_state_access::read, 1, state, nullptr},
            {skippy_state_kind::recurrent_conv, skippy_state_access::write, 1, state, write_0, 0},
            {skippy_state_kind::recurrent_conv, skippy_state_access::write, 1, state, write_1, 1},
        };
        parameters = {{parameter, "weight.shared"}};
        requests = {{request, "tokens"}};
    }

    ggml_context_ptr context;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * request = nullptr;
    ggml_tensor * parameter = nullptr;
    ggml_tensor * state = nullptr;
    ggml_tensor * entry = nullptr;
    ggml_tensor * side = nullptr;
    ggml_tensor * exit_0 = nullptr;
    ggml_tensor * prepared = nullptr;
    ggml_tensor * write_0 = nullptr;
    ggml_tensor * exit_1 = nullptr;
    ggml_tensor * write_1 = nullptr;
    std::vector<skippy_block_boundary> boundaries;
    std::vector<skippy_state_effect> effects;
    std::vector<skippy_parameter_binding> parameters;
    std::vector<skippy_request_input_binding> requests;
};

static int check_semantic_identities() {
    semantic_fixture left("left");
    semantic_fixture right("renamed");
    skippy_graph_semantics left_semantics;
    skippy_graph_semantics right_semantics;
    std::string error;
    if (!skippy_build_graph_semantics(left.graph, left.boundaries, left.effects,
                left.parameters, left.requests, left_semantics, error) ||
            !skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, right_semantics, error) ||
            left_semantics.identity != right_semantics.identity) {
        std::fprintf(stderr, "pointer-different renamed semantic graphs disagree: %s\n", error.c_str());
        return 60;
    }

    const skippy_execution_profile_guard guard = {4, 1, 4, 0, "semantic-config", "cpu"};
    const std::vector<skippy_stage_profile_input> left_profiles = {{
        "semantic", left.graph, &left.boundaries, &left.effects,
        &left.parameters, &left.requests, 2, guard, left.exit_1,
    }};
    const std::vector<skippy_stage_profile_input> right_profiles = {{
        "semantic", right.graph, &right.boundaries, &right.effects,
        &right.parameters, &right.requests, 2, guard, right.exit_1,
    }};
    skippy_guarded_stage_slice_plan source;
    skippy_guarded_stage_slice_plan final;
    if (!skippy_build_guarded_stage_slice_plan(left_profiles, 0, 1, source, error) ||
            !skippy_build_guarded_stage_slice_plan(right_profiles, 1, 2, final, error)) {
        std::fprintf(stderr, "semantic stage plans failed: %s\n", error.c_str());
        return 61;
    }
    const std::vector<const skippy_guarded_stage_slice_plan *> chain = {&source, &final};
    if (!skippy_validate_guarded_stage_chain(chain, error) ||
            source.profiles[0].graph == final.profiles[0].graph ||
            source.profiles[0].graph_identity != final.profiles[0].graph_identity ||
            source.profiles[0].profile_identity != final.profiles[0].profile_identity ||
            source.profiles[0].slice.activation_export_identities !=
                    final.profiles[0].slice.activation_import_identities) {
        std::fprintf(stderr, "semantic frontier chain failed: %s\n", error.c_str());
        return 62;
    }
    skippy_guarded_stage_slice_plan whole;
    if (!skippy_build_guarded_stage_slice_plan(left_profiles, 0, 2, whole, error)) {
        std::fprintf(stderr, "single-stage semantic plan failed: %s\n", error.c_str());
        return 80;
    }
    const std::vector<const skippy_guarded_stage_slice_plan *> single_stage = {&whole};
    if (!skippy_validate_guarded_stage_chain(single_stage, error)) {
        std::fprintf(stderr, "single-stage semantic chain failed: %s\n", error.c_str());
        return 81;
    }
    auto changed_profile = whole;
    changed_profile.profiles[0].profile_identity[0] ^= 1;
    const std::vector<const skippy_guarded_stage_slice_plan *> changed_profile_chain = {&changed_profile};
    if (skippy_validate_guarded_stage_chain(changed_profile_chain, error)) {
        std::fprintf(stderr, "single-stage profile identity tamper was accepted\n");
        return 82;
    }
    auto changed_whole_slice = whole;
    changed_whole_slice.profiles[0].slice.slice_identity[0] ^= 1;
    const std::vector<const skippy_guarded_stage_slice_plan *> changed_slice_chain = {&changed_whole_slice};
    if (skippy_validate_guarded_stage_chain(changed_slice_chain, error)) {
        std::fprintf(stderr, "single-stage slice identity tamper was accepted\n");
        return 83;
    }
    auto changed_range = whole;
    changed_range.profiles[0].slice = source.profiles[0].slice;
    const std::vector<const skippy_guarded_stage_slice_plan *> changed_range_chain = {&changed_range};
    if (skippy_validate_guarded_stage_chain(changed_range_chain, error)) {
        std::fprintf(stderr, "single-stage slice range substitution was accepted\n");
        return 88;
    }
    skippy_guarded_stage_slice_plan right_whole;
    if (!skippy_build_guarded_stage_slice_plan(right_profiles, 0, 2, right_whole, error)) {
        std::fprintf(stderr, "right full-range semantic plan failed: %s\n", error.c_str());
        return 89;
    }
    auto changed_adjacent = final;
    changed_adjacent.profiles[0].slice = right_whole.profiles[0].slice;
    const std::vector<const skippy_guarded_stage_slice_plan *> changed_adjacent_chain = {
        &source, &changed_adjacent,
    };
    if (skippy_validate_guarded_stage_chain(changed_adjacent_chain, error)) {
        std::fprintf(stderr, "adjacent-stage slice range substitution was accepted\n");
        return 90;
    }

    skippy_executable_graph executable;
    if (!skippy_build_executable_graph(left.graph, source.profiles[0].slice, executable, error) ||
            executable.slice_identity != source.profiles[0].slice.slice_identity ||
            executable.node_identities != source.profiles[0].slice.node_identities) {
        std::fprintf(stderr, "semantic executable clone failed: %s\n", error.c_str());
        return 63;
    }
    auto changed_slice = source.profiles[0].slice;
    changed_slice.node_identities[0][0] ^= 1;
    if (skippy_build_executable_graph(left.graph, changed_slice, executable, error)) {
        std::fprintf(stderr, "tampered executable semantic mapping was accepted\n");
        return 64;
    }
    auto changed_import = final.profiles[0].slice;
    changed_import.activation_imports[0] = right.request;
    if (skippy_build_executable_graph(right.graph, changed_import, executable, error)) {
        std::fprintf(stderr, "swapped executable import pointer was accepted\n");
        return 78;
    }
    auto changed_export = source.profiles[0].slice;
    changed_export.activation_exports[0] = left.entry;
    if (skippy_build_executable_graph(left.graph, changed_export, executable, error)) {
        std::fprintf(stderr, "swapped executable export pointer was accepted\n");
        return 79;
    }
    const int32_t stale_op_param = left.exit_0->op_params[0];
    left.exit_0->op_params[0] ^= 1;
    if (skippy_build_executable_graph(left.graph, source.profiles[0].slice, executable, error)) {
        std::fprintf(stderr, "stale executable op parameter was accepted\n");
        return 84;
    }
    left.exit_0->op_params[0] = stale_op_param;
    const int32_t stale_flags = left.exit_0->flags;
    left.exit_0->flags ^= GGML_TENSOR_FLAG_OUTPUT;
    if (skippy_build_executable_graph(left.graph, source.profiles[0].slice, executable, error)) {
        std::fprintf(stderr, "stale executable flags were accepted\n");
        return 85;
    }
    left.exit_0->flags = stale_flags;
    const size_t stale_stride = left.exit_0->nb[1];
    left.exit_0->nb[1] += sizeof(float);
    if (skippy_build_executable_graph(left.graph, source.profiles[0].slice, executable, error)) {
        std::fprintf(stderr, "stale executable stride was accepted\n");
        return 86;
    }
    left.exit_0->nb[1] = stale_stride;
    ggml_tensor * stale_source = left.exit_0->src[0];
    left.exit_0->src[0] = left.request;
    if (skippy_build_executable_graph(left.graph, source.profiles[0].slice, executable, error)) {
        std::fprintf(stderr, "stale executable source reference was accepted\n");
        return 87;
    }
    left.exit_0->src[0] = stale_source;
    const size_t stale_parameter_stride = left.parameter->nb[1];
    left.parameter->nb[1] += sizeof(float);
    if (skippy_build_executable_graph(left.graph, source.profiles[0].slice, executable, error)) {
        std::fprintf(stderr, "stale executable parameter metadata was accepted\n");
        return 91;
    }
    left.parameter->nb[1] = stale_parameter_stride;
    auto changed_snapshot = source.profiles[0].slice;
    changed_snapshot.source_value_identities.begin()->second[0] ^= 1;
    if (skippy_build_executable_graph(left.graph, changed_snapshot, executable, error)) {
        std::fprintf(stderr, "tampered executable source snapshot was accepted\n");
        return 92;
    }

    const int32_t op_param = right.exit_1->op_params[0];
    right.exit_1->op_params[0] ^= 1;
    skippy_graph_semantics changed_semantics;
    if (!skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, changed_semantics, error) ||
            changed_semantics.identity == left_semantics.identity) {
        std::fprintf(stderr, "operation parameter perturbation did not change graph identity\n");
        return 65;
    }
    right.exit_1->op_params[0] = op_param;
    const size_t stride = right.exit_1->nb[1];
    right.exit_1->nb[1] += sizeof(float);
    if (!skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, changed_semantics, error) ||
            changed_semantics.identity == left_semantics.identity) {
        std::fprintf(stderr, "stride perturbation did not change graph identity\n");
        return 66;
    }
    right.exit_1->nb[1] = stride;
    const int32_t flags = right.exit_1->flags;
    right.exit_1->flags |= GGML_TENSOR_FLAG_OUTPUT;
    if (!skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, changed_semantics, error) ||
            changed_semantics.identity == left_semantics.identity) {
        std::fprintf(stderr, "stable flag perturbation did not change graph identity\n");
        return 67;
    }
    right.exit_1->flags = flags;

    auto reordered_effects = right.effects;
    std::reverse(reordered_effects.begin(), reordered_effects.end());
    if (!skippy_build_graph_semantics(right.graph, right.boundaries, reordered_effects,
                right.parameters, right.requests, changed_semantics, error) ||
            changed_semantics.identity != left_semantics.identity) {
        std::fprintf(stderr, "state effect enumeration changed ordered semantics: %s\n", error.c_str());
        return 68;
    }
    auto invalid_effects = right.effects;
    std::swap(invalid_effects[1].write_ordinal, invalid_effects[2].write_ordinal);
    if (skippy_build_graph_semantics(right.graph, right.boundaries, invalid_effects,
                right.parameters, right.requests, changed_semantics, error)) {
        std::fprintf(stderr, "reversed state-write order was accepted\n");
        return 69;
    }

    const int32_t unknown_flags = right.entry->flags;
    right.entry->flags |= 1 << 20;
    if (skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, changed_semantics, error)) {
        std::fprintf(stderr, "unknown semantic flags were accepted\n");
        return 70;
    }
    right.entry->flags = unknown_flags;
    ggml_tensor * original_source = right.entry->src[0];
    right.entry->src[0] = right.exit_1;
    if (skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, changed_semantics, error)) {
        std::fprintf(stderr, "forward semantic reference was accepted\n");
        return 71;
    }
    ggml_tensor * foreign = ggml_new_tensor_1d(right.context.get(), GGML_TYPE_F32, 4);
    right.entry->src[0] = foreign;
    if (skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, changed_semantics, error)) {
        std::fprintf(stderr, "unresolved semantic reference was accepted\n");
        return 72;
    }
    right.entry->src[0] = original_source;

    if (skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, {}, changed_semantics, error)) {
        std::fprintf(stderr, "missing request-input binding was accepted\n");
        return 73;
    }
    auto duplicate_requests = right.requests;
    duplicate_requests.push_back(right.requests.front());
    if (skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, duplicate_requests, changed_semantics, error)) {
        std::fprintf(stderr, "duplicate request-input binding was accepted\n");
        return 74;
    }
    ggml_tensor * other_request = ggml_new_tensor_1d(right.context.get(), GGML_TYPE_F32, 4);
    ggml_set_input(other_request);
    duplicate_requests = right.requests;
    duplicate_requests.push_back({other_request, "tokens"});
    if (skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, duplicate_requests, changed_semantics, error)) {
        std::fprintf(stderr, "duplicate request-input identity was accepted\n");
        return 75;
    }
    auto conflicting_requests = right.requests;
    conflicting_requests.push_back({right.parameter, "conflict"});
    if (skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, conflicting_requests, changed_semantics, error)) {
        std::fprintf(stderr, "conflicting structural binding was accepted\n");
        return 76;
    }

    right.state->flags |= GGML_TENSOR_FLAG_INPUT;
    if (!skippy_build_graph_semantics(right.graph, right.boundaries, right.effects,
                right.parameters, right.requests, changed_semantics, error)) {
        std::fprintf(stderr, "registered state owner was treated as a request input: %s\n", error.c_str());
        return 77;
    }
    right.state->flags &= ~GGML_TENSOR_FLAG_INPUT;
    return 0;
}

static int check_tied_parameter_identity() {
    ggml_init_params params = {
        ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * request = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * input_weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * output_weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(request);
    ggml_tensor * hidden = ggml_mul(ctx.get(), request, input_weight);
    ggml_tensor * output = ggml_add(ctx.get(), hidden, output_weight);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);

    const std::vector<skippy_parameter_binding> parameters = {
        {input_weight, "weight.tied"},
        {output_weight, "weight.tied"},
    };
    const std::vector<skippy_request_input_binding> requests = {{request, "tokens"}};
    skippy_graph_semantics semantics;
    std::string error;
    if (!skippy_build_graph_semantics(graph, {}, {}, parameters, requests, semantics, error) ||
            semantics.value_identities.at(input_weight) != semantics.value_identities.at(output_weight)) {
        std::fprintf(stderr, "tied parameter instances did not share semantic identity: %s\n", error.c_str());
        return 92;
    }
    return 0;
}

static int check_middle_stage_selection() {
    std::vector<uint8_t> buffer(ggml_tensor_overhead() * 96 + ggml_graph_overhead_custom(96, false));
    ggml_init_params params = { buffer.size(), buffer.data(), true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * request = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(tokens);
    ggml_set_input(request);
    ggml_tensor * parameter_0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * parameter_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * parameter_2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);

    ggml_tensor * entry_0 = ggml_mul(ctx, tokens, parameter_0);
    ggml_tensor * pass_through = ggml_mul(ctx, entry_0, parameter_0);
    ggml_tensor * request_only = ggml_view_1d(ctx, request, 4, 0);
    ggml_tensor * exit_0 = ggml_add(ctx, ggml_add(ctx, entry_0, parameter_0), request_only);
    ggml_tensor * local_state_read = ggml_add(ctx, state, request);
    ggml_tensor * exit_1 = ggml_add(ctx, exit_0, ggml_mul(ctx, local_state_read, parameter_1));
    ggml_tensor * state_write = ggml_cpy(ctx, exit_1, state);
    ggml_tensor * exit_2 = ggml_add(
            ctx,
            ggml_add(ctx, ggml_mul(ctx, exit_1, parameter_2), pass_through),
            request_only);
    ggml_tensor * logits = ggml_mul(ctx, exit_2, parameter_2);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 96, false);
    ggml_build_forward_expand(graph, pass_through);
    ggml_build_forward_expand(graph, state_write);
    ggml_build_forward_expand(graph, logits);

    const std::vector<skippy_block_boundary> boundaries = {
        {skippy_block_boundary_kind::entry, 0, entry_0},
        {skippy_block_boundary_kind::exit, 0, exit_0},
        {skippy_block_boundary_kind::entry, 1, exit_0},
        {skippy_block_boundary_kind::exit, 1, exit_1},
        {skippy_block_boundary_kind::entry, 2, exit_1},
        {skippy_block_boundary_kind::exit, 2, exit_2},
    };
    const std::vector<skippy_state_effect> effects = {
        {skippy_state_kind::recurrent_conv, skippy_state_access::read, 1, state, nullptr},
        {skippy_state_kind::recurrent_conv, skippy_state_access::write, 1, state, state_write, 0},
    };
    const std::vector<skippy_parameter_binding> parameter_bindings = {
        {parameter_0, "block.0.weight"},
        {parameter_1, "block.1.weight"},
        {parameter_2, "block.2.weight"},
    };
    const std::vector<skippy_request_input_binding> request_bindings = {
        {tokens, "tokens"},
        {request, "request.sideband"},
    };

    skippy_stage_slice_selection selection;
    std::string error;
    if (!skippy_select_stage_slice(
                graph,
                boundaries,
                effects,
                parameter_bindings,
                request_bindings,
                3,
                1,
                2,
                selection,
                error)) {
        std::fprintf(stderr, "middle stage selection failed: %s\n", error.c_str());
        ggml_free(ctx);
        return 1;
    }
    if (selection.parameter_identities != std::vector<std::string>{"block.1.weight"}) {
        std::fprintf(stderr, "middle stage selected the wrong parameters:");
        for (const std::string & identity : selection.parameter_identities) {
            std::fprintf(stderr, " %s", identity.c_str());
        }
        std::fprintf(stderr, "\n");
        ggml_free(ctx);
        return 2;
    }
    if (!contains_tensor(selection.activation_imports, exit_0) ||
            !contains_tensor(selection.activation_imports, pass_through) ||
            !contains_tensor(selection.activation_exports, exit_1) ||
            !contains_tensor(selection.activation_exports, pass_through) ||
            contains_tensor(selection.activation_imports, request_only) ||
            contains_tensor(selection.activation_exports, request_only)) {
        std::fprintf(stderr, "middle stage lost an activation or pass-through frontier\n");
        ggml_free(ctx);
        return 3;
    }
    if (!contains_tensor(selection.request_inputs, request) ||
            contains_tensor(selection.request_inputs, tokens)) {
        std::fprintf(stderr, "middle stage request-input classification is incorrect\n");
        ggml_free(ctx);
        return 4;
    }
    if (selection.state_effects.size() != 2) {
        std::fprintf(stderr, "middle stage did not retain its state owner and write effect\n");
        ggml_free(ctx);
        return 5;
    }
    if (selection.node_indices.size() < 4) {
        std::fprintf(stderr, "middle stage omitted locally resolvable state preparation\n");
        ggml_free(ctx);
        return 6;
    }

    if (!skippy_select_stage_slice(
                graph,
                boundaries,
                effects,
                parameter_bindings,
                request_bindings,
                3,
                0,
                1,
                selection,
                error) ||
            !contains_tensor(selection.activation_exports, pass_through) ||
            contains_tensor(selection.activation_exports, request_only)) {
        std::fprintf(stderr, "source stage did not export its detached pass-through value: %s\n", error.c_str());
        ggml_free(ctx);
        return 7;
    }

    const skippy_stage_slice_selection normalized = selection;
    std::vector<skippy_parameter_binding> reversed_bindings = parameter_bindings;
    std::reverse(reversed_bindings.begin(), reversed_bindings.end());
    if (!skippy_select_stage_slice(
                graph,
                boundaries,
                effects,
                reversed_bindings,
                request_bindings,
                3,
                0,
                1,
                selection,
                error) ||
            selection.node_indices != normalized.node_indices ||
            selection.activation_imports != normalized.activation_imports ||
            selection.activation_exports != normalized.activation_exports ||
            selection.request_inputs != normalized.request_inputs ||
            selection.parameter_identities != normalized.parameter_identities) {
        std::fprintf(stderr, "stage selection changed with parameter enumeration order: %s\n", error.c_str());
        ggml_free(ctx);
        return 8;
    }

    const skippy_execution_profile_guard prefill_guard = {
        8,
        1,
        8,
        0,
        "config-a",
        "cpu",
    };
    const skippy_execution_profile_guard decode_guard = {
        1,
        1,
        1,
        0,
        "config-a",
        "cpu",
    };
    std::vector<skippy_parameter_binding> decode_parameter_bindings = parameter_bindings;
    decode_parameter_bindings[1].identity = "block.1.decode_weight";
    std::vector<skippy_stage_profile_input> profiles = {
        {"prefill", graph, &boundaries, &effects, &parameter_bindings, &request_bindings, 3, prefill_guard, logits},
        {"decode", graph, &boundaries, &effects, &decode_parameter_bindings, &request_bindings, 3, decode_guard, logits},
    };
    skippy_guarded_stage_slice_plan guarded_plan;
    if (!skippy_build_guarded_stage_slice_plan(profiles, 1, 2, guarded_plan, error) ||
            guarded_plan.layer_count != 3 ||
            guarded_plan.layer_start != 1 ||
            guarded_plan.layer_end != 2 ||
            guarded_plan.profiles.size() != 2 ||
            guarded_plan.profiles[0].profile_id != "decode" ||
            guarded_plan.profiles[1].profile_id != "prefill" ||
            guarded_plan.profiles[0].guard.n_tokens != 1 ||
            guarded_plan.profiles[1].guard.n_tokens != 8 ||
            guarded_plan.profiles[0].slice.parameter_identities != std::vector<std::string>{"block.1.decode_weight"} ||
            guarded_plan.profiles[1].slice.parameter_identities != std::vector<std::string>{"block.1.weight"} ||
            guarded_plan.resident_parameter_identities !=
                    std::vector<std::string>{"block.1.decode_weight", "block.1.weight"}) {
        std::fprintf(stderr, "guarded profile plan was not normalized: %s\n", error.c_str());
        ggml_free(ctx);
        return 9;
    }
    const skippy_guarded_stage_slice_plan normalized_plan = guarded_plan;
    std::reverse(profiles.begin(), profiles.end());
    if (!skippy_build_guarded_stage_slice_plan(profiles, 1, 2, guarded_plan, error) ||
            guarded_plan.profiles[0].profile_id != normalized_plan.profiles[0].profile_id ||
            guarded_plan.profiles[1].profile_id != normalized_plan.profiles[1].profile_id ||
            guarded_plan.profiles[0].slice.node_indices != normalized_plan.profiles[0].slice.node_indices ||
            guarded_plan.profiles[1].slice.node_indices != normalized_plan.profiles[1].slice.node_indices ||
            guarded_plan.resident_parameter_identities != normalized_plan.resident_parameter_identities) {
        std::fprintf(stderr, "guarded profile plan changed with input order: %s\n", error.c_str());
        ggml_free(ctx);
        return 10;
    }
    profiles.push_back(profiles.front());
    if (skippy_build_guarded_stage_slice_plan(profiles, 1, 2, guarded_plan, error)) {
        std::fprintf(stderr, "duplicate execution profile identity was accepted\n");
        ggml_free(ctx);
        return 11;
    }
    profiles.pop_back();
    profiles[0].guard.n_sequences = 0;
    if (skippy_build_guarded_stage_slice_plan(profiles, 1, 2, guarded_plan, error)) {
        std::fprintf(stderr, "invalid execution profile guard was accepted\n");
        ggml_free(ctx);
        return 12;
    }
    profiles[0].guard = decode_guard;
    profiles[0].guard.n_tokens = 5;
    profiles[0].guard.n_sequences = 2;
    if (skippy_build_guarded_stage_slice_plan(profiles, 1, 2, guarded_plan, error)) {
        std::fprintf(stderr, "uneven multi-sequence execution profile was accepted\n");
        ggml_free(ctx);
        return 13;
    }
    profiles[0].guard = decode_guard;

    std::vector<skippy_stage_profile_input> ordered_profiles = {
        {"decode", graph, &boundaries, &effects, &decode_parameter_bindings, &request_bindings, 3, decode_guard, logits},
        {"prefill", graph, &boundaries, &effects, &parameter_bindings, &request_bindings, 3, prefill_guard, logits},
    };
    skippy_guarded_stage_slice_plan source_plan;
    skippy_guarded_stage_slice_plan middle_plan;
    skippy_guarded_stage_slice_plan final_plan;
    if (!skippy_build_guarded_stage_slice_plan(ordered_profiles, 0, 1, source_plan, error) ||
            !skippy_build_guarded_stage_slice_plan(ordered_profiles, 1, 2, middle_plan, error) ||
            !skippy_build_guarded_stage_slice_plan(ordered_profiles, 2, 3, final_plan, error)) {
        std::fprintf(stderr, "complete guarded stage chain could not be built: %s\n", error.c_str());
        ggml_free(ctx);
        return 14;
    }
    const std::vector<const skippy_guarded_stage_slice_plan *> stage_chain = {
        &source_plan,
        &middle_plan,
        &final_plan,
    };
    if (!skippy_validate_guarded_stage_chain(stage_chain, error)) {
        std::fprintf(stderr, "complete guarded stage chain was rejected: %s\n", error.c_str());
        ggml_free(ctx);
        return 15;
    }
    const auto removed_export = source_plan.profiles[0].slice.activation_exports.back();
    source_plan.profiles[0].slice.activation_exports.pop_back();
    if (skippy_validate_guarded_stage_chain(stage_chain, error)) {
        std::fprintf(stderr, "incomplete activation frontier was accepted\n");
        ggml_free(ctx);
        return 16;
    }
    source_plan.profiles[0].slice.activation_exports.push_back(removed_export);
    middle_plan.profiles[0].guard.n_tokens += 1;
    if (skippy_validate_guarded_stage_chain(stage_chain, error)) {
        std::fprintf(stderr, "mismatched adjacent profile guards were accepted\n");
        ggml_free(ctx);
        return 17;
    }
    middle_plan.profiles[0].guard.n_tokens -= 1;

    const std::vector<skippy_block_boundary> foreign_state_boundaries = {
        {skippy_block_boundary_kind::entry, 0, entry_0},
        {skippy_block_boundary_kind::exit, 0, exit_1},
        {skippy_block_boundary_kind::entry, 1, exit_1},
        {skippy_block_boundary_kind::exit, 1, exit_2},
        {skippy_block_boundary_kind::entry, 2, exit_2},
        {skippy_block_boundary_kind::exit, 2, logits},
    };
    if (skippy_select_stage_slice(
                graph,
                foreign_state_boundaries,
                effects,
                parameter_bindings,
                request_bindings,
                3,
                0,
                1,
                selection,
                error)) {
        std::fprintf(stderr, "slice accepted persistent state owned by another layer\n");
        ggml_free(ctx);
        return 18;
    }

    ggml_set_input(parameter_1);
    if (skippy_select_stage_slice(
                graph,
                boundaries,
                effects,
                parameter_bindings,
                request_bindings,
                3,
                1,
                2,
                selection,
                error)) {
        std::fprintf(stderr, "multiply classified parameter was accepted\n");
        ggml_free(ctx);
        return 19;
    }

    ggml_free(ctx);
    return 0;
}

static int check_inferred_registered_state_write() {
    std::vector<uint8_t> buffer(ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false));
    ggml_init_params params = { buffer.size(), buffer.data(), true };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * state = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * entry = ggml_add(ctx.get(), input, state);
    ggml_tensor * write = ggml_cpy(ctx.get(), entry, state);
    ggml_tensor * output = ggml_mul(ctx.get(), entry, input);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, write);
    ggml_build_forward_expand(graph, output);
    const std::vector<skippy_block_boundary> boundaries = {
        {skippy_block_boundary_kind::entry, 0, entry},
        {skippy_block_boundary_kind::exit, 0, output},
    };
    const std::vector<skippy_state_effect> explicit_effects = {
        {skippy_state_kind::recurrent_conv, skippy_state_access::read, 0, state, nullptr},
    };
    std::vector<skippy_state_effect> effects;
    std::string error;
    if (!skippy_infer_state_effects(explicit_effects, graph, boundaries, effects, error) ||
            effects.size() != 2 || effects[1].kind != skippy_state_kind::recurrent_conv ||
            effects[1].access != skippy_state_access::write || effects[1].layer != 0 ||
            effects[1].storage != state || effects[1].effect_node != write ||
            effects[1].write_ordinal != 0) {
        std::fprintf(stderr, "registered-state write inference failed: %s\n", error.c_str());
        return 80;
    }
    return 0;
}

static bool count_compute(void * data) {
    ++*static_cast<int *>(data);
    return false;
}

static int check_executable_slice(bool export_norm) {
    ggml_init_params params = {ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * state = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(input);
    ggml_set_input(weight);
    ggml_set_input(state);
    ggml_tensor * imported = ggml_scale(ctx.get(), input, 2.0f);
    ggml_tensor * passthrough = ggml_scale(ctx.get(), input, 7.0f);
    ggml_tensor * norm = ggml_rms_norm(ctx.get(), imported, 1e-5f);
    ggml_tensor * omitted = ggml_scale(ctx.get(), norm, 3.0f);
    ggml_tensor * product = ggml_mul(ctx.get(), norm, weight);
    ggml_tensor * write_1 = ggml_cpy(ctx.get(), product, state);
    ggml_tensor * view = ggml_view_1d(ctx.get(), product, 2, sizeof(float));
    ggml_tensor * output = ggml_cont(ctx.get(), view);
    ggml_tensor * write_2 = ggml_cpy(ctx.get(), imported, state);
    ggml_tensor * ordered_only = ggml_scale(ctx.get(), weight, 100.0f);
    ggml_cgraph * source = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(source, passthrough);
    ggml_build_forward_expand(source, omitted);
    ggml_build_forward_expand(source, write_1);
    ggml_build_forward_expand(source, output);
    ggml_build_forward_expand(source, write_2);
    ggml_build_forward_order(source, ordered_only);
    const std::vector<ggml_tensor *> selected = {norm, product, write_1, view, output, write_2, ordered_only};
    std::vector<int32_t> indices;
    int norm_index = -1;
    for (int i = 0; i < source->n_nodes; ++i) {
        if (source->nodes[i] == norm) {
            norm_index = i;
        }
        if (std::find(selected.begin(), selected.end(), source->nodes[i]) != selected.end()) {
            indices.push_back(i);
        }
    }
    if (indices.size() != selected.size() || indices[1] != indices[0] + 2 ||
            ggml_node_get_use_count(source, norm_index) != 2) {
        return 20;
    }
    std::vector<ggml_tensor> source_snapshots;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx.get()); tensor; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
        source_snapshots.push_back(*tensor);
    }
    const std::vector<const ggml_tensor *> imports = {imported, passthrough};
    std::vector<const ggml_tensor *> exports = {output, passthrough, view};
    if (export_norm) {
        exports.push_back(norm);
    }
    skippy_executable_graph slice;
    std::string error;
    if (!skippy_build_executable_graph(source, indices, imports, exports, slice, error)) {
        std::fprintf(stderr, "executable graph failed: %s\n", error.c_str());
        return 21;
    }
    if (slice.graph->n_nodes != int(selected.size()) || slice.graph->n_leafs != 4 ||
            slice.tensors.count(input) || slice.tensors.count(omitted) ||
            slice.graph->visited_hash_set.keys == source->visited_hash_set.keys ||
            ggml_node_get_use_count(slice.graph, 0) != 1) {
        return 22;
    }
    for (size_t i = 0; i < selected.size(); ++i) {
        if (slice.graph->nodes[i] != slice.tensors.at(selected[i])) {
            return 23;
        }
    }
    if (slice.tensors.at(imported)->op != GGML_OP_NONE ||
            slice.tensors.at(imported)->src[0] || slice.tensors.at(imported)->view_src ||
            (slice.tensors.at(imported)->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (slice.tensors.at(ordered_only)->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            slice.tensors.at(view)->view_src != slice.tensors.at(product) ||
            slice.tensors.at(view)->view_offs != sizeof(float) ||
            !(slice.tensors.at(product)->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            slice.tensors.at(write_1)->view_src != slice.tensors.at(state) ||
            slice.tensors.at(write_2)->view_src != slice.tensors.at(state)) {
        return 24;
    }
    if (ggml_can_fuse(slice.graph, 0, {GGML_OP_RMS_NORM, GGML_OP_MUL}) == export_norm) {
        return 25;
    }
    // Every tensor has its own visited slot and exact operand-edge count, not a copy of the full graph's counts.
    for (const auto & entry : slice.tensors) {
        const ggml_tensor * tensor = entry.second;
        size_t slot = ggml_hash_find(&slice.graph->visited_hash_set, tensor);
        if (slot == GGML_HASHSET_FULL || !ggml_bitset_get(slice.graph->visited_hash_set.used, slot)) {
            return 26;
        }
        int uses = 0;
        for (int i = 0; i < slice.graph->n_nodes; ++i) {
            for (const ggml_tensor * src : slice.graph->nodes[i]->src) {
                uses += src == tensor;
            }
        }
        if (slice.graph->use_counts[slot] != uses || tensor->data || tensor->buffer || tensor->extra) {
            return 27;
        }
    }
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    ggml_backend_t backends[] = {cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 64, false, true));
    if (ggml_backend_sched_get_buffer_size(sched.get(), cpu.get()) != 0 ||
            !ggml_backend_sched_alloc_graph(sched.get(), slice.graph) ||
            ggml_backend_sched_get_buffer_size(sched.get(), cpu.get()) == 0) {
        return 28;
    }
    const float x[] = {2, 4, 6, 8};
    const float w[] = {1, 2, 3, 4};
    const float side[] = {7, 14, 21, 28};
    ggml_backend_tensor_set(slice.tensors.at(imported), x, 0, sizeof(x));
    ggml_backend_tensor_set(slice.tensors.at(weight), w, 0, sizeof(w));
    ggml_backend_tensor_set(slice.tensors.at(passthrough), side, 0, sizeof(side));
    int compute_calls = 0;
    ggml_backend_cpu_set_abort_callback(cpu.get(), count_compute, &compute_calls);
    if (ggml_backend_sched_graph_compute(sched.get(), slice.graph) != GGML_STATUS_SUCCESS) {
        return 29;
    }
    float actual[4];
    ggml_backend_tensor_get(slice.tensors.at(output), actual, 0, 2 * sizeof(float));
    for (int i = 0; i < 2; ++i) {
        float expected = x[i + 1] * w[i + 1] / std::sqrt(30.0f + 1e-5f);
        if (std::fabs(actual[i] - expected) > 1e-5f) {
            return 30;
        }
    }
    ggml_backend_tensor_get(slice.tensors.at(state), actual, 0, sizeof(actual));
    if (std::memcmp(actual, x, sizeof(x)) != 0) {
        return 31;
    }
    ggml_backend_tensor_get(slice.tensors.at(passthrough), actual, 0, sizeof(actual));
    if (std::memcmp(actual, side, sizeof(side)) != 0) {
        return 32;
    }
    if (export_norm) {
        ggml_backend_tensor_get(slice.tensors.at(norm), actual, 0, sizeof(actual));
        for (int i = 0; i < 4; ++i) {
            if (std::fabs(actual[i] - x[i] / std::sqrt(30.0f + 1e-5f)) > 1e-5f) {
                return 33;
            }
        }
    }
    // The CPU abort hook runs once per computed (possibly fused) kernel, not once per graph node.
    if (compute_calls != (export_norm ? 5 : 4)) {
        std::fprintf(stderr, "unexpected kernel count: %d (export_norm=%d)\n", compute_calls, export_norm);
        return 34;
    }
    size_t snapshot = 0;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx.get()); tensor; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
        if (std::memcmp(tensor, &source_snapshots[snapshot++], sizeof(*tensor)) != 0) {
            return 35;
        }
    }
    std::fprintf(stderr, "executable slice: nodes=%d leaves=%d kernels=%d exported_norm=%d\n",
                 slice.graph->n_nodes, slice.graph->n_leafs, compute_calls, export_norm);
    return 0;
}

static int check_invalid_executable_slices() {
    ggml_init_params params = {ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * a = ggml_scale(ctx.get(), x, 2.0f);
    ggml_tensor * b = ggml_scale(ctx.get(), a, 3.0f);
    ggml_tensor * foreign = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, b);
    skippy_executable_graph result;
    std::string error;
    auto rejects = [&](std::vector<int32_t> indices, std::vector<const ggml_tensor *> imports,
                       std::vector<const ggml_tensor *> exports) {
        if (skippy_build_executable_graph(graph, indices, imports, exports, result, error)) {
            return false;
        }
        return !error.empty() && !result.graph && !result.context && result.tensors.empty();
    };
    if (!rejects({1}, {}, {b}) || !rejects({0, 0}, {}, {a}) ||
            !rejects({1, 0}, {}, {b}) || !rejects({2}, {}, {}) ||
            !rejects({-1}, {}, {}) || !rejects({1}, {a, a}, {b}) ||
            !rejects({0, 1}, {a}, {b}) || !rejects({1}, {foreign}, {b}) ||
            !rejects({1}, {a}, {foreign}) || !rejects({1}, {a}, {b, b})) {
        return 40;
    }
    // An import-only pass-through is a real allocated leaf, not an empty graph view.
    if (!skippy_build_executable_graph(graph, {}, {a}, {a}, result, error) ||
            result.graph->n_nodes != 0 || result.graph->n_leafs != 1 ||
            !(result.tensors.at(a)->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 41;
    }
    b->src[0] = b;
    if (!rejects({1}, {}, {b})) {
        return 42;
    }
    b->src[0] = a;
    b->op = GGML_OP_CUSTOM;
    if (!rejects({1}, {a}, {b})) {
        return 43;
    }
    b->op = GGML_OP_SCALE;
    ggml_tensor * imported_view = ggml_view_1d(ctx.get(), x, 3, sizeof(float));
    ggml_tensor * nested_view = ggml_view_1d(ctx.get(), imported_view, 2, sizeof(float));
    ggml_build_forward_expand(graph, nested_view);
    // Both views have flattened x as their base. Never bind nested_view to an unrelated copy of x.
    if (!rejects({3}, {imported_view}, {nested_view})) {
        return 44;
    }
    ggml_tensor * storage_only_view = ggml_view_1d(ctx.get(), foreign, 2, 0);
    storage_only_view->src[0] = nullptr;
    ggml_build_forward_expand(graph, storage_only_view);
    if (!skippy_build_executable_graph(graph, {4}, {}, {storage_only_view}, result, error) ||
            result.graph->n_nodes != 1 || result.graph->n_leafs != 1 ||
            result.tensors.at(storage_only_view)->view_src != result.tensors.at(foreign) ||
            !(result.tensors.at(foreign)->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 45;
    }
    const size_t slot = ggml_hash_find(&result.graph->visited_hash_set, result.tensors.at(foreign));
    if (slot == GGML_HASHSET_FULL || !ggml_bitset_get(result.graph->visited_hash_set.used, slot)) {
        return 46;
    }
    return 0;
}

static int check_state_preparation_execution(bool empty_zero, bool empty_extra) {
    ggml_init_params params = {ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 4);
    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * main_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * extra_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, empty_extra ? 0 : 1);
    for (auto * request : {input, main_ids, extra_ids}) { ggml_set_input(request); }
    ggml_tensor * zero = ggml_scale_inplace(ctx.get(), ggml_view_1d(ctx.get(), state, empty_zero ? 0 : 2, 0), 0);
    ggml_tensor * main = ggml_get_rows(ctx.get(), state, main_ids);
    ggml_tensor * extra = ggml_cpy(ctx.get(), ggml_get_rows(ctx.get(), state, extra_ids),
            ggml_view_2d(ctx.get(), state, 2, empty_extra ? 0 : 1, state->nb[1], 4 * sizeof(float)));
    ggml_tensor * entry = ggml_scale(ctx.get(), input, 1);
    ggml_tensor * output = ggml_add(ctx.get(), entry, main);
    ggml_tensor * final = ggml_cpy(ctx.get(), output, ggml_view_1d(ctx.get(), state, 2, 0));
    ggml_tensor * rollback = ggml_cpy(ctx.get(), main, ggml_view_1d(ctx.get(), state, 2, 2 * sizeof(float)));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    for (auto * root : {zero, main, extra, final, rollback, output}) { ggml_build_forward_expand(graph, root); }
    const auto kind = skippy_state_kind::recurrent_conv;
    const auto write = skippy_state_access::write;
    const std::vector<skippy_state_effect> effects = {
        {kind, write, 0, state, zero, 0},
        {kind, write, 0, state, extra, 1},
        {kind, write, 0, state, final, 2},
        {kind, write, 0, state, rollback, 3},
        {kind, skippy_state_access::read, 0, state, nullptr},
    };
    const std::vector<skippy_block_boundary> boundaries = {
        {skippy_block_boundary_kind::entry, 0, entry},
        {skippy_block_boundary_kind::exit, 0, output},
    };
    const std::vector<skippy_request_input_binding> request_bindings = {
        {input, "input"},
        {main_ids, "state.main_indices"},
        {extra_ids, "state.extra_indices"},
    };
    skippy_stage_slice_selection selection;
    skippy_executable_graph slice;
    std::string error;
    if (!skippy_select_stage_slice(graph, boundaries, effects, {}, request_bindings, 1, 0, 1,
                selection, error, output) ||
            !selection.activation_imports.empty() || selection.state_effects.size() != effects.size() ||
            !skippy_build_executable_graph(graph, selection, slice, error)) {
        std::fprintf(stderr, "state preparation slice failed: %s\n", error.c_str());
        return 50;
    }
    for (auto * root : {zero, main, extra, final, rollback}) {
        if (!slice.tensors.count(root)) { return 51; }
    }
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    ggml_backend_t backends[] = {cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 64, false, true));
    if (!ggml_backend_sched_alloc_graph(sched.get(), slice.graph)) { return 52; }
    const float initial[] = {9, 10, 2, 3, 7, 8, 11, 12};
    const float delta[] = {5, 6};
    const int32_t main_id = 0;
    const int32_t extra_id = 1;
    ggml_backend_tensor_set(slice.tensors.at(state), initial, 0, sizeof(initial));
    ggml_backend_tensor_set(slice.tensors.at(input), delta, 0, sizeof(delta));
    ggml_backend_tensor_set(slice.tensors.at(main_ids), &main_id, 0, sizeof(main_id));
    if (!empty_extra) { ggml_backend_tensor_set(slice.tensors.at(extra_ids), &extra_id, 0, sizeof(extra_id)); }
    if (ggml_backend_sched_graph_compute(sched.get(), slice.graph) != GGML_STATUS_SUCCESS) { return 53; }
    float actual[8];
    ggml_backend_tensor_get(slice.tensors.at(state), actual, 0, sizeof(actual));
    const float expected[] = {
        empty_zero ? 14.0f : 5.0f, empty_zero ? 16.0f : 6.0f,
        empty_zero ? 9.0f : 0.0f, empty_zero ? 10.0f : 0.0f,
        empty_extra ? 7.0f : 2.0f, empty_extra ? 8.0f : 3.0f, 11, 12,
    };
    if (std::memcmp(actual, expected, sizeof(expected)) != 0) {
        std::fprintf(stderr, "state preparation values changed (empty_zero=%d):", empty_zero);
        for (float value : actual) { std::fprintf(stderr, " %g", value); }
        std::fprintf(stderr, "\n");
        return 54;
    }
    std::fprintf(stderr, "state preparation execution: empty_zero=%d empty_extra=%d ordered_writes=4 untouched_tail=2\n", empty_zero, empty_extra);
    return 0;
}

static int check_explicit_final_output() {
    std::vector<uint8_t> buffer(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
    ggml_init_params params = { buffer.size(), buffer.data(), true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * request = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * parameter_0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * parameter_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * parameter_2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(request);
    ggml_tensor * entry_0 = ggml_mul(ctx, request, parameter_0);
    ggml_tensor * exit_0 = ggml_add(ctx, entry_0, parameter_0);
    ggml_tensor * exit_1 = ggml_add(ctx, exit_0, parameter_1);
    ggml_tensor * exit_2 = ggml_add(ctx, exit_1, parameter_2);
    ggml_tensor * logits = ggml_mul(ctx, exit_2, parameter_2);
    ggml_tensor * trailing_unrelated = ggml_mul(ctx, request, parameter_0);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, logits);
    ggml_build_forward_order(graph, trailing_unrelated);
    int32_t trailing_index = -1;
    for (int32_t index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        if (ggml_graph_node(graph, index) == trailing_unrelated) {
            trailing_index = index;
            break;
        }
    }
    const std::vector<skippy_block_boundary> boundaries = {
        {skippy_block_boundary_kind::entry, 0, entry_0},
        {skippy_block_boundary_kind::exit, 0, exit_0},
        {skippy_block_boundary_kind::entry, 1, exit_0},
        {skippy_block_boundary_kind::exit, 1, exit_1},
        {skippy_block_boundary_kind::entry, 2, exit_1},
        {skippy_block_boundary_kind::exit, 2, exit_2},
    };
    const std::vector<skippy_parameter_binding> parameters = {
        {parameter_0, "block.0.weight"},
        {parameter_1, "block.1.weight"},
        {parameter_2, "block.2.weight"},
    };
    const std::vector<skippy_request_input_binding> requests = {
        {request, "tokens"},
    };
    skippy_stage_slice_selection selection;
    std::string error;
    if (!skippy_select_stage_slice(graph, boundaries, {}, parameters, requests, 3, 2, 3,
                selection, error, logits) ||
            selection.parameter_identities != std::vector<std::string>{"block.2.weight"} ||
            !contains_tensor(selection.activation_imports, exit_1) ||
            !selection.activation_exports.empty() || trailing_index < 0 ||
            std::find(selection.node_indices.begin(), selection.node_indices.end(),
                    trailing_index) != selection.node_indices.end()) {
        std::fprintf(stderr, "explicit final output cone was not selected: %s\n", error.c_str());
        ggml_free(ctx);
        return 78;
    }
    if (skippy_select_stage_slice(graph, boundaries, {}, parameters, requests, 3, 2, 3,
                selection, error)) {
        std::fprintf(stderr, "final slice accepted without an explicit output root\n");
        ggml_free(ctx);
        return 79;
    }
    ggml_free(ctx);
    return 0;
}

static int check_mtp_stage_layer_count() {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_NEMOTRON_H_MOE, llama_model_default_params()));
    model->hparams.n_layer_all = 53;
    model->hparams.n_layer_nextn = 1;

    if (skippy_stage_layer_count(model.get()) != 52) {
        std::fprintf(stderr, "appended NextN/MTP block was counted as a pipeline stage\n");
        return 80;
    }
    return 0;
}

static int check_auxiliary_graph_parameter_closure() {
    ggml_init_params params = {
        ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor * request = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * used = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * unused = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_set_input(request);
    ggml_tensor * output = ggml_mul(ctx.get(), request, used);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);

    const std::vector<skippy_parameter_binding> parameters = {
        {used, "mtp.used"},
        {unused, "mtp.unused"},
    };
    const std::vector<skippy_request_input_binding> requests = {{request, "mtp.input"}};
    skippy_auxiliary_graph_plan plan;
    std::string error;
    if (!skippy_build_auxiliary_graph_plan(graph, {}, parameters, requests, plan, error) ||
            plan.graph_identity.empty() ||
            plan.parameter_identities != std::vector<std::string>{"mtp.used"}) {
        std::fprintf(stderr, "auxiliary graph parameter closure was incorrect: %s\n", error.c_str());
        return 91;
    }
    return 0;
}

#include "residual_checkpoint_regression.h"
#include "input_leaf_state_regression.h"

int main() {
    int result = check_input_leaf_state_inference();
    if (!result) { result = check_raw_embedding_checkpoint(); }
    if (!result) { result = check_semantic_identities(); }
    if (!result) { result = check_tied_parameter_identity(); }
    if (!result) { result = check_middle_stage_selection(); }
    if (!result) { result = check_inferred_registered_state_write(); }
    if (!result) { result = check_explicit_final_output(); }
    if (!result) { result = check_state_preparation_execution(false, false); }
    if (!result) { result = check_state_preparation_execution(true, false); }
    if (!result) { result = check_state_preparation_execution(true, true); }
    if (!result) { result = check_executable_slice(false); }
    if (!result) { result = check_executable_slice(true); }
    if (!result) { result = check_invalid_executable_slices(); }
    if (!result) { result = check_mtp_stage_layer_count(); }
    if (!result) { result = check_auxiliary_graph_parameter_closure(); }
    if (result) { std::fprintf(stderr, "stage slice test failed: %d\n", result); }
    return result;
}
