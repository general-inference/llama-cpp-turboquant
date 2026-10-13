#include "stage_slice_plan.h"

#include "ggml.h"
#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace {

const ggml_tensor * canonical_storage(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

template <typename T>
void append_once(std::vector<T> & values, std::unordered_set<T> & seen, T value) {
    if (seen.insert(value).second) {
        values.push_back(value);
    }
}

struct selector {
    selector(
            ggml_cgraph * graph,
            const std::vector<skippy_block_boundary> & boundaries,
            const std::vector<skippy_state_effect> & effects,
            int32_t layer_count,
            int32_t layer_start,
            int32_t layer_end,
            skippy_stage_slice_selection & result,
            std::string & error,
            const ggml_tensor * explicit_output) :
        graph(graph),
        boundaries(boundaries),
        effects(effects),
        layer_count(layer_count),
        layer_start(layer_start),
        layer_end(layer_end),
        result(result),
        error(error),
        explicit_output(explicit_output) {
    }

    ggml_cgraph * graph;
    const std::vector<skippy_block_boundary> & boundaries;
    const std::vector<skippy_state_effect> & effects;
    int32_t layer_count;
    int32_t layer_start;
    int32_t layer_end;
    skippy_stage_slice_selection & result;
    std::string & error;
    const ggml_tensor * explicit_output;

    std::unordered_map<const ggml_tensor *, int32_t> node_index;
    std::unordered_set<const ggml_tensor *> graph_leaves;
    std::unordered_set<int32_t> core_selected_nodes;
    std::unordered_map<const ggml_tensor *, int8_t> core_dependency_cache;
    std::unordered_set<const ggml_tensor *> core_dependency_visiting;
    std::unordered_map<const ggml_tensor *, std::vector<std::string>> parameter_names;
    std::unordered_map<const ggml_tensor *, std::string> request_names;
    std::unordered_set<const ggml_tensor *> state_storage;
    std::unordered_map<const ggml_tensor *, std::unordered_set<int32_t>> state_layers;
    std::unordered_map<const ggml_tensor *, std::unordered_set<skippy_state_residency>> state_residencies;
    std::unordered_set<const ggml_tensor *> block_boundaries;
    std::unordered_set<const ggml_tensor *> prior_boundaries;
    std::unordered_set<const ggml_tensor *> live_start;
    std::unordered_set<const ggml_tensor *> live_end;
    std::unordered_set<int32_t> selected_nodes;
    std::unordered_set<const ggml_tensor *> imported;
    std::unordered_set<const ggml_tensor *> exported;
    std::unordered_set<const ggml_tensor *> request_inputs;
    std::set<std::string> parameters;
    std::unordered_map<const ggml_tensor *, int8_t> local_cache;
    std::unordered_set<const ggml_tensor *> local_visiting;
    std::unordered_map<const ggml_tensor *, int8_t> foreign_state_cache;
    std::unordered_set<const ggml_tensor *> foreign_state_visiting;
    std::unordered_map<const ggml_tensor *, int8_t> request_derived_cache;
    std::unordered_set<const ggml_tensor *> request_derived_visiting;

    int structural_class_count(const ggml_tensor * tensor) const {
        if (node_index.find(tensor) != node_index.end()) {
            return 0;
        }
        const ggml_tensor * canonical = canonical_storage(tensor);
        const bool is_parameter = parameter_names.find(tensor) != parameter_names.end() ||
                                  parameter_names.find(canonical) != parameter_names.end();
        const bool is_request = request_names.find(tensor) != request_names.end() ||
                                request_names.find(canonical) != request_names.end();
        const bool is_state = state_storage.find(canonical) != state_storage.end();
        const bool is_constant = !is_parameter && !is_request && !is_state && canonical != nullptr &&
                                 graph_leaves.find(canonical) != graph_leaves.end() &&
                                 canonical->op == GGML_OP_NONE && canonical->buffer == nullptr &&
                                 (canonical->flags & GGML_TENSOR_FLAG_INPUT) == 0;
        const int count = static_cast<int>(is_parameter) + static_cast<int>(is_request) +
                          static_cast<int>(is_state) + static_cast<int>(is_constant);
        return count;
    }

    bool state_owner_belongs_to_slice(const ggml_tensor * storage, int32_t owner) const {
        const auto residency = state_residencies.find(canonical_storage(storage));
        return (residency != state_residencies.end() &&
                    residency->second.count(skippy_state_residency::per_stage) != 0) ||
                (owner >= layer_start && owner < layer_end);
    }

    bool state_owner_belongs_to_slice(const skippy_state_effect & effect) const {
        return state_owner_belongs_to_slice(effect.storage, effect.layer);
    }

    bool state_owned_by_slice(const ggml_tensor * tensor) {
        const auto owners = state_layers.find(canonical_storage(tensor));
        if (owners == state_layers.end() || owners->second.empty()) {
            error = "persistent state has no registered owner";
            return false;
        }
        const auto residencies = state_residencies.find(canonical_storage(tensor));
        if (residencies == state_residencies.end()) {
            error = "persistent state has no residency semantics";
            return false;
        }
        for (int32_t owner : owners->second) {
            const bool local = owner >= layer_start && owner < layer_end;
            const bool replicated = residencies->second.count(skippy_state_residency::per_stage) != 0;
            if (!local && !replicated) {
                error = "persistent state ";
                error += ggml_get_name(tensor);
                error += " is owned outside the requested slice";
                return false;
            }
        }
        return true;
    }

    bool record_structural(const ggml_tensor * tensor) {
        if (node_index.find(tensor) != node_index.end()) {
            return false;
        }
        const ggml_tensor * canonical = canonical_storage(tensor);
        const auto exact_parameter = parameter_names.find(tensor);
        const auto canonical_parameter = parameter_names.find(canonical);
        const auto exact_request = request_names.find(tensor);
        const auto canonical_request = request_names.find(canonical);
        const bool is_parameter = exact_parameter != parameter_names.end() ||
                                  canonical_parameter != parameter_names.end();
        const bool is_request = exact_request != request_names.end() ||
                                canonical_request != request_names.end();
        const bool is_state = state_storage.find(canonical) != state_storage.end();
        const bool is_constant = !is_parameter && !is_request && !is_state && canonical != nullptr &&
                                 graph_leaves.find(canonical) != graph_leaves.end() &&
                                 canonical->op == GGML_OP_NONE && canonical->buffer == nullptr &&
                                 (canonical->flags & GGML_TENSOR_FLAG_INPUT) == 0;
        const int count = static_cast<int>(is_parameter) + static_cast<int>(is_request) +
                          static_cast<int>(is_state) + static_cast<int>(is_constant);
        if (count > 1) {
            error = "graph dependency has multiple structural leaf classes";
            return false;
        }
        if (is_parameter) {
            const auto & names = exact_parameter != parameter_names.end()
                    ? exact_parameter->second
                    : canonical_parameter->second;
            parameters.insert(names.begin(), names.end());
            return true;
        }
        if (is_request) {
            append_once(result.request_inputs, request_inputs, canonical);
            return true;
        }
        if (is_constant) {
            return true;
        }
        return is_state && state_owned_by_slice(canonical);
    }

    bool locally_resolvable(const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return true;
        }
        const auto cached = local_cache.find(tensor);
        if (cached != local_cache.end()) {
            return cached->second > 0;
        }
        if (prior_boundaries.find(tensor) != prior_boundaries.end()) {
            local_cache[tensor] = -1;
            return false;
        }
        const int structural_classes = structural_class_count(tensor);
        if (structural_classes > 1) {
            error = "graph dependency has multiple structural leaf classes";
            return false;
        }
        if (structural_classes == 1) {
            local_cache[tensor] = 1;
            return true;
        }
        if (tensor->op == GGML_OP_NONE || node_index.find(tensor) == node_index.end()) {
            local_cache[tensor] = -1;
            return false;
        }
        if (!local_visiting.insert(tensor).second) {
            error = "graph dependency cycle encountered while classifying a frontier";
            return false;
        }
        bool local = tensor->view_src == tensor ? true : locally_resolvable(tensor->view_src);
        for (const ggml_tensor * source : tensor->src) {
            if (source == tensor) {
                continue;
            }
            local = locally_resolvable(source) && local;
        }
        local_visiting.erase(tensor);
        local_cache[tensor] = local ? 1 : -1;
        return local;
    }

    bool depends_on_foreign_state(const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return false;
        }
        const auto cached = foreign_state_cache.find(tensor);
        if (cached != foreign_state_cache.end()) {
            return cached->second > 0;
        }
        if (node_index.find(tensor) == node_index.end()) {
            const auto owners = state_layers.find(canonical_storage(tensor));
            if (owners == state_layers.end()) {
                foreign_state_cache[tensor] = -1;
                return false;
            }
            for (int32_t owner : owners->second) {
                if (!state_owner_belongs_to_slice(tensor, owner)) {
                    foreign_state_cache[tensor] = 1;
                    return true;
                }
            }
            foreign_state_cache[tensor] = -1;
            return false;
        }
        if (!foreign_state_visiting.insert(tensor).second) {
            error = "graph dependency cycle encountered while resolving state ownership";
            return false;
        }
        bool foreign = tensor->view_src == tensor ? false : depends_on_foreign_state(tensor->view_src);
        for (const ggml_tensor * source : tensor->src) {
            if (source == tensor) {
                continue;
            }
            foreign = depends_on_foreign_state(source) || foreign;
        }
        foreign_state_visiting.erase(tensor);
        foreign_state_cache[tensor] = foreign ? 1 : -1;
        return foreign;
    }

    bool depends_on_core_selection(const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return false;
        }
        const auto found = node_index.find(tensor);
        if (found == node_index.end()) {
            return false;
        }
        if (core_selected_nodes.find(found->second) != core_selected_nodes.end()) {
            return true;
        }
        const auto cached = core_dependency_cache.find(tensor);
        if (cached != core_dependency_cache.end()) {
            return cached->second > 0;
        }
        if (!core_dependency_visiting.insert(tensor).second) {
            error = "graph dependency cycle encountered while resolving block ownership";
            return false;
        }
        bool depends = tensor->view_src == tensor ? false : depends_on_core_selection(tensor->view_src);
        for (const ggml_tensor * source : tensor->src) {
            if (source == tensor) {
                continue;
            }
            depends = depends_on_core_selection(source) || depends;
        }
        core_dependency_visiting.erase(tensor);
        core_dependency_cache[tensor] = depends ? 1 : -1;
        return depends;
    }

    bool request_derived(const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return true;
        }
        const auto cached = request_derived_cache.find(tensor);
        if (cached != request_derived_cache.end()) {
            return cached->second > 0;
        }
        // Layer boundaries carry activations even when the boundary itself
        // is a raw-embedding request leaf. Only auxiliary cones outside the blocks
        // may be rebuilt independently by the adjacent stage.
        if (block_boundaries.find(tensor) != block_boundaries.end()) {
            request_derived_cache[tensor] = -1;
            return false;
        }
        if (node_index.find(tensor) == node_index.end()) {
            const int structural_classes = structural_class_count(tensor);
            if (structural_classes > 1) {
                error = "graph dependency has multiple structural leaf classes";
                return false;
            }
            if (structural_classes == 0) {
                request_derived_cache[tensor] = -1;
                return false;
            }
            const ggml_tensor * canonical = canonical_storage(tensor);
            const auto owners = state_layers.find(canonical);
            if (owners != state_layers.end()) {
                for (int32_t owner : owners->second) {
                    if (!state_owner_belongs_to_slice(tensor, owner)) {
                        request_derived_cache[tensor] = -1;
                        return false;
                    }
                }
            }
            request_derived_cache[tensor] = 1;
            return true;
        }
        if (!request_derived_visiting.insert(tensor).second) {
            error = "graph dependency cycle encountered while resolving request ownership";
            return false;
        }
        bool has_dependency = false;
        bool derived = true;
        if (tensor->view_src != nullptr && tensor->view_src != tensor) {
            has_dependency = true;
            derived = request_derived(tensor->view_src);
        }
        for (const ggml_tensor * source : tensor->src) {
            if (source == nullptr || source == tensor) {
                continue;
            }
            has_dependency = true;
            derived = request_derived(source) && derived;
        }
        request_derived_visiting.erase(tensor);
        derived = has_dependency && derived;
        request_derived_cache[tensor] = derived ? 1 : -1;
        return derived;
    }

    void add_live_edges(int32_t frontier, std::unordered_set<const ggml_tensor *> & live) const {
        for (int32_t consumer_index = frontier + 1; consumer_index < ggml_graph_n_nodes(graph); ++consumer_index) {
            const ggml_tensor * consumer = ggml_graph_node(graph, consumer_index);
            auto consider = [&](const ggml_tensor * source) {
                if (source == nullptr) {
                    return;
                }
                const auto producer = node_index.find(source);
                if (producer == node_index.end() || producer->second <= frontier) {
                    live.insert(source);
                }
            };
            consider(consumer->view_src);
            for (const ggml_tensor * source : consumer->src) {
                consider(source);
            }
        }
    }

    bool normalize_graph_tensors(std::vector<const ggml_tensor *> & tensors) {
        for (const ggml_tensor * tensor : tensors) {
            if (node_index.find(tensor) == node_index.end()) {
                error = "selected tensor is absent from the normalized graph order";
                return false;
            }
        }
        std::sort(tensors.begin(), tensors.end(), [&](const ggml_tensor * left, const ggml_tensor * right) {
            return node_index.at(left) < node_index.at(right);
        });
        return true;
    }

    bool normalize_request_tensors(std::vector<const ggml_tensor *> & tensors) {
        std::unordered_map<const ggml_tensor *, std::string> keys;
        std::set<std::string> unique_keys;
        for (const ggml_tensor * tensor : tensors) {
            const auto found = request_names.find(canonical_storage(tensor));
            if (found == request_names.end()) {
                error = "selected request input has no explicit identity";
                return false;
            }
            const std::string & key = found->second;
            if (!unique_keys.insert(key).second) {
                error = "request inputs do not have unique explicit identities";
                return false;
            }
            keys.emplace(tensor, key);
        }
        std::sort(tensors.begin(), tensors.end(), [&](const ggml_tensor * left, const ggml_tensor * right) {
            return keys.at(left) < keys.at(right);
        });
        return true;
    }

    bool walk(const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return true;
        }
        if (record_structural(tensor)) {
            return true;
        }
        if (!error.empty()) {
            return false;
        }
        if (prior_boundaries.find(tensor) != prior_boundaries.end()) {
            append_once(result.activation_imports, imported, tensor);
            return true;
        }
        if (live_start.find(tensor) != live_start.end()) {
            const bool local = locally_resolvable(tensor);
            if (!error.empty()) {
                return false;
            }
            if (!local) {
                append_once(result.activation_imports, imported, tensor);
                return true;
            }
        }
        const auto found = node_index.find(tensor);
        if (found == node_index.end()) {
            error = "graph dependency is neither structural nor a live activation import";
            return false;
        }
        if (!selected_nodes.insert(found->second).second) {
            return true;
        }
        if (tensor->view_src != tensor && !walk(tensor->view_src)) {
            return false;
        }
        for (const ggml_tensor * source : tensor->src) {
            if (source != tensor && !walk(source)) {
                return false;
            }
        }
        return true;
    }

    bool select() {
        if (graph == nullptr || layer_count <= 0 || layer_start < 0 ||
                layer_start >= layer_end || layer_end > layer_count) {
            error = "invalid stage slice request";
            return false;
        }
        if (!skippy_validate_block_boundaries(boundaries, 0, layer_count, error) ||
                !skippy_validate_state_effects(effects, graph, error)) {
            return false;
        }
        for (int32_t index = 0; index < ggml_graph_n_nodes(graph); ++index) {
            const ggml_tensor * tensor = ggml_graph_node(graph, index);
            node_index.emplace(tensor, index);
        }
        for (int32_t index = 0; index < graph->n_leafs; ++index) {
            const ggml_tensor * leaf = canonical_storage(graph->leafs[index]);
            if (leaf == nullptr) {
                error = "graph has a cyclic leaf view";
                return false;
            }
            graph_leaves.insert(leaf);
        }
        for (const auto & effect : effects) {
            const ggml_tensor * storage = canonical_storage(effect.storage);
            state_storage.insert(storage);
            state_layers[storage].insert(effect.layer);
            state_residencies[storage].insert(effect.residency);
        }
        for (const auto & boundary : boundaries) {
            block_boundaries.insert(boundary.tensor);
        }
        for (int32_t layer = 0; layer < layer_start; ++layer) {
            prior_boundaries.insert(boundaries[2 * layer].tensor);
            prior_boundaries.insert(boundaries[2 * layer + 1].tensor);
        }
        if (layer_start > 0) {
            prior_boundaries.insert(boundaries[2 * layer_start].tensor);
            const auto start = node_index.find(boundaries[2 * layer_start].tensor);
            if (start == node_index.end()) {
                error = "stage start boundary is not a graph node";
                return false;
            }
            add_live_edges(start->second, live_start);
            live_start.insert(boundaries[2 * layer_start].tensor);
        }
        if (layer_end < layer_count) {
            const auto finish = node_index.find(boundaries[2 * (layer_end - 1) + 1].tensor);
            if (finish == node_index.end()) {
                error = "stage end boundary is not a graph node";
                return false;
            }
            add_live_edges(finish->second, live_end);
        }

        const ggml_tensor * primary_output = explicit_output;
        if (layer_end != layer_count) {
            primary_output = boundaries[2 * (layer_end - 1) + 1].tensor;
        } else if (primary_output == nullptr) {
            error = "final stage has no explicit logits or embedding output";
            return false;
        }
        if (!walk(primary_output)) {
            return false;
        }
        for (const auto & effect : effects) {
            if (!state_owner_belongs_to_slice(effect)) {
                continue;
            }
            result.state_effects.push_back(effect);
            if (effect.access == skippy_state_access::write && !walk(effect.effect_node)) {
                return false;
            }
        }
        core_selected_nodes = selected_nodes;

        for (const ggml_tensor * tensor : live_end) {
            const int structural_classes = structural_class_count(tensor);
            if (structural_classes > 1) {
                error = "graph dependency has multiple structural leaf classes";
                return false;
            }
            if (structural_classes == 1) {
                continue;
            }
            const auto producer = node_index.find(tensor);
            if (producer != node_index.end() && selected_nodes.find(producer->second) == selected_nodes.end()) {
                const bool local = locally_resolvable(tensor);
                if (!error.empty()) {
                    return false;
                }
                if (!local) {
                    append_once(result.activation_imports, imported, tensor);
                } else {
                    const bool block_owned = depends_on_core_selection(tensor);
                    if (!error.empty()) {
                        return false;
                    }
                    if (block_owned && !depends_on_foreign_state(tensor) && !walk(tensor)) {
                        return false;
                    }
                    if (!error.empty()) {
                        return false;
                    }
                }
            }
            if (producer != node_index.end() && selected_nodes.find(producer->second) != selected_nodes.end()) {
                const bool request_only = request_derived(tensor);
                if (!error.empty()) {
                    return false;
                }
                if (!request_only) {
                    append_once(result.activation_exports, exported, tensor);
                }
            } else if (imported.find(tensor) != imported.end()) {
                append_once(result.activation_exports, exported, tensor);
            }
        }

        if (layer_start > 0 && imported.find(boundaries[2 * layer_start].tensor) == imported.end()) {
            error = "stage start boundary is not an activation import";
            return false;
        }
        if (layer_end < layer_count && exported.find(primary_output) == exported.end()) {
            error = "stage end boundary is not an activation export";
            return false;
        }

        result.layer_start = layer_start;
        result.layer_end = layer_end;
        result.source_graph = graph;
        result.node_indices.assign(selected_nodes.begin(), selected_nodes.end());
        std::sort(result.node_indices.begin(), result.node_indices.end());
        result.parameter_identities.assign(parameters.begin(), parameters.end());
        std::sort(result.state_effects.begin(), result.state_effects.end(), [&](const auto & left, const auto & right) {
            const int32_t left_position = left.effect_node == nullptr ? -1 : node_index.at(left.effect_node);
            const int32_t right_position = right.effect_node == nullptr ? -1 : node_index.at(right.effect_node);
            return std::tie(left.layer, left.kind, left.access, left.write_ordinal, left_position) <
                    std::tie(right.layer, right.kind, right.access, right.write_ordinal, right_position);
        });
        return normalize_graph_tensors(result.activation_imports) &&
               normalize_graph_tensors(result.activation_exports) &&
               normalize_request_tensors(result.request_inputs);
    }
};

}

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
        const ggml_tensor * explicit_output) {
    selection = {};
    error.clear();
    std::vector<skippy_state_effect> effective_state_effects;
    if (!skippy_infer_state_effects(
                state_effects, graph, boundaries, effective_state_effects, error)) {
        return false;
    }
    skippy_graph_semantics semantics;
    if (!skippy_build_graph_semantics(
                graph, boundaries, effective_state_effects, parameters, request_inputs, semantics, error)) {
        return false;
    }
    selector state(
        graph,
        boundaries,
        effective_state_effects,
        layer_count,
        layer_start,
        layer_end,
        selection,
        error,
        explicit_output);
    for (const auto & parameter : parameters) {
        state.parameter_names[parameter.tensor].push_back(parameter.identity);
    }
    for (const auto & request : request_inputs) {
        state.request_names.emplace(canonical_storage(request.tensor), request.identity);
    }
    return state.select() && skippy_finalize_stage_slice_identity(semantics, selection, error);
}

bool skippy_build_guarded_stage_slice_plan(
        const std::vector<skippy_stage_profile_input> & profiles,
        int32_t layer_start,
        int32_t layer_end,
        skippy_guarded_stage_slice_plan & plan,
        std::string & error) {
    plan = {};
    error.clear();
    if (profiles.empty()) {
        error = "guarded stage plan requires at least one execution profile";
        return false;
    }

    std::vector<const skippy_stage_profile_input *> ordered;
    ordered.reserve(profiles.size());
    for (const auto & profile : profiles) {
        ordered.push_back(&profile);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto * left, const auto * right) {
        return left->profile_id < right->profile_id;
    });

    std::set<std::string> resident_parameters;
    std::string previous_profile_id;
    for (const auto * profile : ordered) {
        if (profile->profile_id.empty()) {
            error = "execution profile identity is empty";
            return false;
        }
        if (profile->profile_id == previous_profile_id) {
            error = "duplicate execution profile identity";
            return false;
        }
        previous_profile_id = profile->profile_id;
        if (profile->graph == nullptr || profile->boundaries == nullptr ||
                profile->state_effects == nullptr || profile->parameters == nullptr ||
                profile->request_inputs == nullptr) {
            error = "execution profile graph contract is incomplete";
            return false;
        }
        if (profile->layer_count <= 0 ||
                (plan.layer_count != 0 && plan.layer_count != profile->layer_count)) {
            error = "execution profiles disagree on model layer count";
            return false;
        }
        if (profile->guard.n_tokens == 0 || profile->guard.n_sequences == 0 ||
                profile->guard.n_outputs == 0 ||
                profile->guard.n_tokens % profile->guard.n_sequences != 0 ||
                profile->guard.n_outputs > profile->guard.n_tokens ||
                profile->guard.graph_configuration_id.empty() ||
                profile->guard.backend_id.empty()) {
            error = "execution profile guard is invalid";
            return false;
        }

        skippy_stage_profile_plan profile_plan;
        profile_plan.profile_id = profile->profile_id;
        profile_plan.graph = profile->graph;
        profile_plan.guard = profile->guard;
        if (!skippy_select_stage_slice(
                    profile->graph,
                    *profile->boundaries,
                    *profile->state_effects,
                    *profile->parameters,
                    *profile->request_inputs,
                    profile->layer_count,
                    layer_start,
                    layer_end,
                    profile_plan.slice,
                    error,
                    profile->explicit_output)) {
            error = "execution profile " + profile->profile_id + " rejected: " + error;
            return false;
        }
        if (profile_plan.slice.node_indices.empty()) {
            error = "execution profile " + profile->profile_id + " produced an empty stage slice";
            return false;
        }
        profile_plan.graph_identity = profile_plan.slice.graph_identity;
        if (!skippy_build_profile_identity(
                    profile_plan.profile_id,
                    profile_plan.graph_identity,
                    profile_plan.guard,
                    profile_plan.profile_identity,
                    error)) {
            error = "execution profile " + profile->profile_id + " rejected: " + error;
            return false;
        }
        resident_parameters.insert(
                profile_plan.slice.parameter_identities.begin(),
                profile_plan.slice.parameter_identities.end());
        plan.layer_count = profile->layer_count;
        plan.profiles.push_back(std::move(profile_plan));
    }

    plan.layer_start = layer_start;
    plan.layer_end = layer_end;
    plan.resident_parameter_identities.assign(
            resident_parameters.begin(), resident_parameters.end());
    return true;
}

bool skippy_build_auxiliary_graph_plan(
        ggml_cgraph * graph,
        const std::vector<skippy_state_effect> & state_effects,
        const std::vector<skippy_parameter_binding> & parameters,
        const std::vector<skippy_request_input_binding> & request_inputs,
        skippy_auxiliary_graph_plan & plan,
        std::string & error) {
    plan = {};
    error.clear();

    skippy_graph_semantics semantics;
    if (!skippy_build_graph_semantics(
                graph, {}, state_effects, parameters, request_inputs, semantics, error)) {
        return false;
    }

    std::unordered_set<const ggml_tensor *> graph_leaves;
    graph_leaves.reserve(static_cast<size_t>(graph->n_leafs));
    for (int32_t index = 0; index < graph->n_leafs; ++index) {
        const ggml_tensor * storage = canonical_storage(graph->leafs[index]);
        if (storage == nullptr) {
            error = "auxiliary graph has a cyclic leaf view";
            return false;
        }
        graph_leaves.insert(storage);
    }

    std::set<std::string> resident_parameters;
    for (const auto & parameter : parameters) {
        const ggml_tensor * storage = canonical_storage(parameter.tensor);
        if (storage == nullptr) {
            error = "auxiliary graph parameter has a cyclic view";
            return false;
        }
        if (graph_leaves.find(storage) != graph_leaves.end()) {
            resident_parameters.insert(parameter.identity);
        }
    }
    if (resident_parameters.empty()) {
        error = "auxiliary graph has no resident parameters";
        return false;
    }

    plan.graph_identity = semantics.identity;
    plan.parameter_identities.assign(resident_parameters.begin(), resident_parameters.end());
    return true;
}

bool skippy_validate_guarded_stage_chain(
        const std::vector<const skippy_guarded_stage_slice_plan *> & stages,
        std::string & error) {
    error.clear();
    if (stages.empty()) {
        error = "stage chain is empty";
        return false;
    }
    for (size_t index = 0; index < stages.size(); ++index) {
        const auto * stage = stages[index];
        if (stage == nullptr || stage->layer_count <= 0 || stage->profiles.empty() ||
                stage->layer_start < 0 || stage->layer_start >= stage->layer_end ||
                stage->layer_end > stage->layer_count) {
            error = "stage chain contains an invalid plan";
            return false;
        }
        if ((index == 0 && stage->layer_start != 0) ||
                (index > 0 && stage->layer_start != stages[index - 1]->layer_end) ||
                (index + 1 == stages.size() && stage->layer_end != stage->layer_count)) {
            error = "stage chain ranges are not a complete contiguous partition";
            return false;
        }
        for (const auto & profile : stage->profiles) {
            std::string identity;
            if (profile.slice.layer_start != stage->layer_start ||
                    profile.slice.layer_end != stage->layer_end ||
                    profile.graph == nullptr || profile.graph != profile.slice.source_graph ||
                    profile.graph_identity != profile.slice.graph_identity ||
                    !skippy_build_profile_identity(profile.profile_id, profile.graph_identity,
                            profile.guard, identity, error) ||
                    profile.profile_identity != identity ||
                    !skippy_validate_stage_slice_identity(profile.slice, error)) {
                error = "stage chain contains an invalid execution profile identity";
                return false;
            }
        }
        if (index == 0) {
            continue;
        }
        const auto * previous = stages[index - 1];
        if (stage->layer_count != previous->layer_count ||
                stage->profiles.size() != previous->profiles.size()) {
            error = "adjacent stages disagree on model or profile count";
            return false;
        }
        for (size_t profile_index = 0; profile_index < stage->profiles.size(); ++profile_index) {
            const auto & left = previous->profiles[profile_index];
            const auto & right = stage->profiles[profile_index];
            if (left.profile_id != right.profile_id || left.graph_identity != right.graph_identity ||
                    left.profile_identity != right.profile_identity) {
                error = "adjacent stages disagree on execution profile guard";
                return false;
            }
            if (left.slice.activation_export_identities != right.slice.activation_import_identities) {
                const auto append_identities = [&](const char * label,
                        const std::vector<std::string> & identities,
                        const std::vector<const ggml_tensor *> & tensors) {
                    error += label;
                    error += "=[";
                    for (size_t identity_index = 0; identity_index < identities.size(); ++identity_index) {
                        if (identity_index > 0) {
                            error += ",";
                        }
                        error += identities[identity_index];
                        if (identity_index < tensors.size()) {
                            error += "(";
                            error += ggml_get_name(tensors[identity_index]);
                            error += ")";
                        }
                    }
                    error += "]";
                };
                error = "adjacent stages have an incomplete activation frontier: ";
                append_identities("exports", left.slice.activation_export_identities, left.slice.activation_exports);
                error += " ";
                append_identities("imports", right.slice.activation_import_identities, right.slice.activation_imports);
                return false;
            }
        }
    }
    return true;
}
