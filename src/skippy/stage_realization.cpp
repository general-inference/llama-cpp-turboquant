#include "stage_realization.h"

#include "stage_slice_plan.h"
#include "stage_execution_contract.h"

#include "llama-graph.h"
#include "llama-model.h"

#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

std::vector<skippy_parameter_binding> parameter_bindings(const llama_model & model) {
    std::vector<skippy_parameter_binding> result;
    result.reserve(model.tensors_by_name.size());
    for (const auto & entry : model.tensors_by_name) {
        if (entry.second != nullptr) {
            result.push_back({entry.second, entry.first});
        }
    }
    return result;
}

void require_equal(
        const char * kind,
        const std::vector<std::string> & planned,
        const std::vector<std::string> & realized) {
    if (planned != realized) {
        auto render = [](const std::vector<std::string> & values) {
            std::string output;
            for (const auto & value : values) {
                if (!output.empty()) output += ",";
                output += value;
            }
            return output;
        };
        throw std::runtime_error(std::string("admitted stage ") + kind +
                " do not match the realized graph slice: planned=[" + render(planned) +
                "] realized=[" + render(realized) + "]");
    }
}

int32_t infer_token_axis(const ggml_tensor * tensor, uint32_t n_tokens) {
    // Canonical hidden-state tensors are normally [width, tokens]. Prefer that
    // axis when dimensions are ambiguous, then cover token-major auxiliaries.
    for (int32_t axis : {1, 0, 2, 3}) {
        if (axis < GGML_MAX_DIMS && tensor->ne[axis] == static_cast<int64_t>(n_tokens)) {
            return axis;
        }
    }
    throw std::runtime_error("stage frontier tensor has no token dimension");
}

ggml_tensor * detach_import(ggml_context * context, const ggml_tensor * source) {
    ggml_tensor * result = ggml_dup_tensor(context, source);
    std::memcpy(result->nb, source->nb, sizeof(result->nb));
    ggml_set_name(result, source->name);
    result->op = GGML_OP_NONE;
    std::memset(result->op_params, 0, sizeof(result->op_params));
    result->flags = GGML_TENSOR_FLAG_INPUT;
    result->view_src = nullptr;
    result->view_offs = 0;
    result->buffer = nullptr;
    result->data = nullptr;
    for (auto & operand : result->src) {
        operand = nullptr;
    }
    return result;
}

void replace_references(
        const skippy_stage_slice_selection & selection,
        const std::unordered_map<const ggml_tensor *, ggml_tensor *> & replacements) {
    for (int32_t node_index : selection.node_indices) {
        ggml_tensor * node = selection.source_graph->nodes[node_index];
        const auto view = replacements.find(node->view_src);
        if (view != replacements.end()) {
            node->view_src = view->second;
        }
        for (auto & operand : node->src) {
            const auto found = replacements.find(operand);
            if (found != replacements.end()) {
                operand = found->second;
            }
        }
    }
}

const ggml_tensor * primary_output(const llm_graph_result & result) {
    if (result.get_logits() != nullptr) return result.get_logits();
    if (result.get_embd_pooled() != nullptr) return result.get_embd_pooled();
    return result.get_embd();
}

void describe_frontier(
        const std::vector<const ggml_tensor *> & tensors,
        std::vector<std::string> & identities,
        std::vector<std::string> & bindings) {
    for (const ggml_tensor * tensor : tensors) {
        const char * binding = tensor == nullptr ? nullptr : ggml_get_name(tensor);
        std::string identity;
        if (binding == nullptr || binding[0] == '\0' ||
                !skippy_build_activation_port_identity(binding, identity)) {
            throw std::runtime_error("realized stage frontier requires stable named ports");
        }
        identities.push_back(std::move(identity));
        bindings.emplace_back(binding);
    }
}

}

int32_t skippy_activation_token_axis(const ggml_tensor * tensor, uint32_t n_tokens, int32_t prepared_axis) {
    if (prepared_axis < 0) return infer_token_axis(tensor, n_tokens);
    return skippy_admitted_token_axis(tensor, n_tokens, prepared_axis);
}

int32_t skippy_admitted_token_axis(const ggml_tensor * tensor, uint32_t n_tokens, int32_t admitted_axis) {
    if (admitted_axis < 0) {
        throw std::runtime_error(
                "live stage requires an explicit admitted token axis for " +
                std::string(tensor == nullptr ? "<null>" : ggml_get_name(tensor)) +
                (tensor == nullptr ? std::string() :
                    " dims=" + std::to_string(tensor->ne[0]) + "x" + std::to_string(tensor->ne[1]) + "x" +
                    std::to_string(tensor->ne[2]) + "x" + std::to_string(tensor->ne[3]) +
                    " op=" + ggml_op_name(tensor->op) + " source0=" +
                    (tensor->src[0] == nullptr ? std::string("<null>") :
                        std::string(ggml_get_name(tensor->src[0])) + "/" + ggml_op_name(tensor->src[0]->op))));
    }
    if (admitted_axis >= GGML_MAX_DIMS || tensor->ne[admitted_axis] != static_cast<int64_t>(n_tokens)) {
        throw std::runtime_error("live stage frontier does not match its prepared token axis");
    }
    return admitted_axis;
}

void skippy_realize_stage_graph(
        const llama_model & model,
        const llm_graph_params & params,
        llm_graph_result & result) {
    if (!params.build_inputs.selection.enabled) {
        return;
    }

    const auto parameters = parameter_bindings(model);
    std::string error;
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        skippy_auxiliary_graph_plan auxiliary;
        if (!skippy_build_auxiliary_graph_plan(result.get_gf(), result.get_state_effects(),
                    parameters, result.get_request_input_bindings(), auxiliary, error)) {
            throw std::runtime_error("cannot realize admitted MTP graph: " + error);
        }
        skippy_validate_execution_dependencies(params.build_inputs, true,
                params.cparams.nextn_layer_offset, auxiliary.parameter_identities);
        return;
    }
    skippy_stage_slice_selection selection;
    if (!skippy_select_stage_slice(
                result.get_gf(),
                result.get_block_boundaries(),
                result.get_state_effects(),
                parameters,
                result.get_request_input_bindings(),
                static_cast<int32_t>(params.hparams.n_layer()),
                params.build_inputs.selection.layer_start,
                params.build_inputs.selection.layer_end,
                selection,
                error,
                primary_output(result))) {
        throw std::runtime_error("cannot realize admitted stage graph: " + error);
    }

    skippy_validate_execution_dependencies(params.build_inputs, false, 0, selection.parameter_identities,
                                           params.ubatch.token == nullptr && params.ubatch.embd != nullptr);
    std::vector<std::string> realized_import_identities;
    std::vector<std::string> realized_import_bindings;
    std::vector<std::string> realized_export_identities;
    std::vector<std::string> realized_export_bindings;
    describe_frontier(selection.activation_imports,
            realized_import_identities, realized_import_bindings);
    describe_frontier(selection.activation_exports,
            realized_export_identities, realized_export_bindings);
    require_equal("activation import identities",
            params.build_inputs.activation_import_identities,
            realized_import_identities);
    require_equal("activation import bindings",
            params.build_inputs.activation_import_bindings,
            realized_import_bindings);
    require_equal("activation export identities",
            params.build_inputs.activation_export_identities,
            realized_export_identities);
    require_equal("activation export bindings",
            params.build_inputs.activation_export_bindings,
            realized_export_bindings);
    if (params.build_inputs.activation_import_bindings.size() != selection.activation_imports.size() ||
            params.build_inputs.activation_export_bindings.size() != selection.activation_exports.size()) {
        throw std::runtime_error("admitted stage frontier bindings do not match the realized graph slice");
    }

    std::vector<int32_t> import_token_axes;
    import_token_axes.reserve(selection.activation_imports.size());
    std::vector<bool> explicit_import_token_axes;
    explicit_import_token_axes.reserve(selection.activation_imports.size());
    for (size_t index = 0; index < selection.activation_imports.size(); ++index) {
        const ggml_tensor * source = selection.activation_imports[index];
        if (index < params.build_inputs.activation_frontier.size()) {
            const int32_t admitted_axis = params.build_inputs.activation_frontier[index].token_axis;
            import_token_axes.push_back(skippy_activation_token_axis(source, params.ubatch.n_tokens, admitted_axis));
            explicit_import_token_axes.push_back(admitted_axis >= 0);
        } else {
            import_token_axes.push_back(infer_token_axis(source, params.ubatch.n_tokens));
            explicit_import_token_axes.push_back(false);
        }
    }

    std::unordered_map<const ggml_tensor *, ggml_tensor *> replacements;
    std::vector<skippy_activation_port> imports;
    imports.reserve(selection.activation_imports.size());
    for (size_t index = 0; index < selection.activation_imports.size(); ++index) {
        const ggml_tensor * source = selection.activation_imports[index];
        ggml_tensor * detached = detach_import(result.get_ctx(), source);
        replacements.emplace(source, detached);
        imports.push_back({detached, import_token_axes[index], false,
                params.build_inputs.activation_import_bindings[index], explicit_import_token_axes[index]});
    }
    replace_references(selection, replacements);

    ggml_cgraph * graph = ggml_new_graph_custom(
            result.get_ctx(), result.get_max_nodes(), false);
    graph->order = selection.source_graph->order;
    for (const auto & request : result.get_request_input_bindings()) {
        if (request.tensor != nullptr) {
            ggml_build_forward_order(graph, const_cast<ggml_tensor *>(request.tensor));
        }
    }
    for (const auto & port : imports) {
        ggml_build_forward_order(graph, port.tensor);
    }
    for (int32_t node_index : selection.node_indices) {
        ggml_tensor * node = selection.source_graph->nodes[node_index];
        if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
            ggml_build_forward_expand(graph, node);
        } else {
            ggml_build_forward_order(graph, node);
        }
    }

    const auto graph_contains = [graph](const ggml_tensor * tensor) {
        if (tensor == nullptr) return false;
        for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            if (ggml_graph_node(graph, index) == tensor) return true;
        }
        return false;
    };
    const auto add_sampling_outputs = [graph](const std::vector<ggml_tensor *> & tensors) {
        for (ggml_tensor * tensor : tensors) {
            if (tensor != nullptr) ggml_build_forward_expand(graph, tensor);
        }
    };
    // Backend samplers are downstream consumers of the terminal logits. They
    // are not part of the dependency slice selected for the logits themselves,
    // but decode reads their outputs asynchronously after graph execution.
    if (graph_contains(result.get_logits())) {
        add_sampling_outputs(result.t_sampled);
        add_sampling_outputs(result.t_sampled_probs);
        add_sampling_outputs(result.t_sampled_logits);
        add_sampling_outputs(result.t_candidates);
    }

    std::vector<skippy_activation_port> exports;
    exports.reserve(selection.activation_exports.size());
    for (size_t index = 0; index < selection.activation_exports.size(); ++index) {
        ggml_tensor * tensor = const_cast<ggml_tensor *>(selection.activation_exports[index]);
        const auto replacement = replacements.find(tensor);
        if (replacement != replacements.end()) {
            tensor = replacement->second;
        }
        ggml_set_output(tensor);
        const int32_t prepared_axis = index < params.activation_export_token_axes.size()
                ? params.activation_export_token_axes[index] : -1;
        const int32_t axis = skippy_activation_token_axis(tensor, params.ubatch.n_tokens, prepared_axis);
        exports.push_back({tensor, axis, false,
                params.build_inputs.activation_export_bindings[index], prepared_axis >= 0});
    }
    result.replace_with_stage_graph(graph, std::move(imports), std::move(exports),
            std::move(selection.state_effects));
}
