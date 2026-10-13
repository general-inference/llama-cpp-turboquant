#include "executable_graph.h"

#include "graph_identity.h"
#include "stage_slice_plan.h"

#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace {

bool is_storage_leaf(const ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_NONE || tensor->view_src != nullptr) {
        return false;
    }
    for (const ggml_tensor * src : tensor->src) {
        if (src != nullptr) {
            return false;
        }
    }
    return true;
}

bool is_custom_op(ggml_op op) {
    return op == GGML_OP_MAP_CUSTOM1 || op == GGML_OP_MAP_CUSTOM2 ||
           op == GGML_OP_MAP_CUSTOM3 || op == GGML_OP_CUSTOM;
}

bool same_layout(const ggml_tensor * source, const ggml_tensor * clone) {
    return source->type == clone->type &&
            std::memcmp(source->ne, clone->ne, sizeof(source->ne)) == 0 &&
            std::memcmp(source->nb, clone->nb, sizeof(source->nb)) == 0;
}

bool verify_clone(
        const ggml_cgraph * source,
        const std::vector<int32_t> & node_indices,
        const std::vector<const ggml_tensor *> & imports,
        const skippy_executable_graph & clone,
        std::string & error) {
    std::unordered_set<const ggml_tensor *> imported(imports.begin(), imports.end());
    std::unordered_set<const ggml_tensor *> clone_tensors;
    for (const auto & entry : clone.tensors) {
        if (entry.first == nullptr || entry.second == nullptr || !clone_tensors.insert(entry.second).second ||
                !same_layout(entry.first, entry.second)) {
            error = "executable clone has an invalid or conflicting semantic tensor mapping";
            return false;
        }
        if (imported.count(entry.first)) {
            if (entry.second->op != GGML_OP_NONE || entry.second->view_src != nullptr ||
                    (entry.second->flags & GGML_TENSOR_FLAG_INPUT) == 0 ||
                    (entry.second->flags & ~(GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) != 0) {
                error = "executable clone did not detach an imported semantic value";
                return false;
            }
            for (const ggml_tensor * dependency : entry.second->src) {
                if (dependency != nullptr) {
                    error = "executable clone retained an imported semantic dependency";
                    return false;
                }
            }
            continue;
        }
        if (entry.first->op != entry.second->op || entry.first->view_offs != entry.second->view_offs ||
                std::memcmp(entry.first->op_params, entry.second->op_params, sizeof(entry.first->op_params)) != 0 ||
                (entry.first->flags & ~GGML_TENSOR_FLAG_OUTPUT) !=
                        (entry.second->flags & ~GGML_TENSOR_FLAG_OUTPUT)) {
            error = "executable clone changed selected semantic tensor metadata";
            return false;
        }
        const ggml_tensor * expected_view = entry.first->view_src == nullptr
                ? nullptr : clone.tensors.at(entry.first->view_src);
        if (entry.second->view_src != expected_view) {
            error = "executable clone changed a selected semantic view reference";
            return false;
        }
        for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            const ggml_tensor * expected = entry.first->src[source_index] == nullptr
                    ? nullptr : clone.tensors.at(entry.first->src[source_index]);
            if (entry.second->src[source_index] != expected) {
                error = "executable clone changed a selected semantic operand reference";
                return false;
            }
        }
    }
    std::vector<const ggml_tensor *> expected_nodes;
    for (int32_t index : node_indices) {
        const ggml_tensor * tensor = source->nodes[index];
        if (!is_storage_leaf(tensor)) {
            expected_nodes.push_back(clone.tensors.at(tensor));
        }
    }
    if (expected_nodes.size() != static_cast<size_t>(clone.graph->n_nodes)) {
        error = "executable clone node count does not match the selected semantics";
        return false;
    }
    for (size_t index = 0; index < expected_nodes.size(); ++index) {
        if (clone.graph->nodes[index] != expected_nodes[index]) {
            error = "executable clone node order does not match the selected semantics";
            return false;
        }
    }
    return true;
}

struct graph_builder {
    skippy_executable_graph result;
    std::vector<const ggml_tensor *> tensors;
    std::unordered_set<const ggml_tensor *> available;
    std::unordered_set<const ggml_tensor *> imported;
    std::string & error;

    bool dependency(const ggml_tensor * tensor) {
        if (tensor == nullptr || available.count(tensor)) {
            return true;
        }
        if (!is_storage_leaf(tensor)) {
            error = "slice dependency is not an earlier selected node, import, or storage leaf";
            return false;
        }
        available.insert(tensor);
        tensors.push_back(tensor);
        return true;
    }

    bool collect(
            const ggml_cgraph * source,
            const std::vector<int32_t> & indices,
            const std::vector<const ggml_tensor *> & imports,
            const std::vector<const ggml_tensor *> & exports) {
        std::unordered_set<const ggml_tensor *> members;
        std::unordered_set<const ggml_tensor *> selected;
        for (int i = 0; i < source->n_nodes; ++i) {
            if (source->nodes[i] == nullptr || !members.insert(source->nodes[i]).second) {
                error = "source graph has null or duplicate nodes";
                return false;
            }
        }
        for (int i = 0; i < source->n_leafs; ++i) {
            members.insert(source->leafs[i]);
        }
        int32_t previous = -1;
        for (int32_t index : indices) {
            if (index <= previous || index < 0 || index >= source->n_nodes) {
                error = "slice node indices must be unique and in source order";
                return false;
            }
            previous = index;
            selected.insert(source->nodes[index]);
        }
        for (const ggml_tensor * tensor : imports) {
            if (tensor == nullptr || !members.count(tensor) || selected.count(tensor) ||
                    !imported.insert(tensor).second) {
                error = "slice imports must be unique source tensors outside the selection";
                return false;
            }
            available.insert(tensor);
            tensors.push_back(tensor);
        }
        for (int32_t index : indices) {
            const ggml_tensor * tensor = source->nodes[index];
            if (is_custom_op(tensor->op)) {
                error = "slice cannot relocate opaque custom-op state";
                return false;
            }
            // ggml flattens view_src. A detached imported view cannot silently reuse its old base.
            for (const ggml_tensor * input : imports) {
                if (input->view_src && tensor->view_src == input->view_src) {
                    error = "slice view requires alias rebasing across an imported view";
                    return false;
                }
            }
            // Storage leaves can be graph nodes when marked PARAM.
            if (!is_storage_leaf(tensor)) {
                if (!dependency(tensor->view_src)) {
                    return false;
                }
                for (const ggml_tensor * src : tensor->src) {
                    if (!dependency(src)) {
                        return false;
                    }
                }
            }
            if (available.insert(tensor).second) {
                tensors.push_back(tensor);
            }
        }
        std::unordered_set<const ggml_tensor *> exported;
        for (const ggml_tensor * tensor : exports) {
            if (tensor == nullptr || (!selected.count(tensor) && !imported.count(tensor)) ||
                    !exported.insert(tensor).second) {
                error = "slice exports must be unique selected or imported tensors";
                return false;
            }
        }
        return true;
    }

    void clone_tensors() {
        for (const ggml_tensor * source : tensors) {
            ggml_tensor * tensor = ggml_dup_tensor(result.context.get(), source);
            std::memcpy(tensor->nb, source->nb, sizeof(tensor->nb));
            ggml_set_name(tensor, source->name);
            tensor->flags = source->flags & ~GGML_TENSOR_FLAG_COMPUTE;
            if (imported.count(source)) {
                tensor->flags = GGML_TENSOR_FLAG_INPUT;
            } else {
                tensor->op = source->op;
                std::memcpy(tensor->op_params, source->op_params, sizeof(tensor->op_params));
                tensor->view_offs = source->view_offs;
            }
            result.tensors.emplace(source, tensor);
        }
        for (const ggml_tensor * source : tensors) {
            if (imported.count(source)) {
                continue;
            }
            ggml_tensor * tensor = result.tensors.at(source);
            if (source->view_src != nullptr) {
                tensor->view_src = result.tensors.at(source->view_src);
            }
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (source->src[i] != nullptr) {
                    tensor->src[i] = result.tensors.at(source->src[i]);
                }
            }
        }
    }
};

}

bool skippy_build_executable_graph(
        const ggml_cgraph * source,
        const std::vector<int32_t> & node_indices,
        const std::vector<const ggml_tensor *> & imports,
        const std::vector<const ggml_tensor *> & exports,
        skippy_executable_graph & result,
        std::string & error) {
    result = {};
    error.clear();
    if (source == nullptr || source->n_nodes < 0 || source->n_leafs < 0 ||
            (node_indices.empty() && imports.empty())) {
        error = "invalid executable slice request";
        return false;
    }
    graph_builder builder{{}, {}, {}, {}, error};
    if (!builder.collect(source, node_indices, imports, exports)) {
        return false;
    }
    const size_t count = std::max(size_t(1), builder.tensors.size());
    if (count > size_t(std::numeric_limits<int32_t>::max()) / GGML_MAX_SRC) {
        error = "executable slice exceeds graph capacity";
        return false;
    }
    ggml_init_params params = {
        ggml_tensor_overhead() * count + ggml_graph_overhead_custom(count, false), nullptr, true,
    };
    builder.result.context.reset(ggml_init(params));
    if (!builder.result.context) {
        error = "cannot allocate executable slice metadata";
        return false;
    }
    builder.clone_tensors();
    ggml_cgraph * graph = ggml_new_graph_custom(builder.result.context.get(), count, false);
    graph->order = source->order;
    // Reuse ggml's traversal to rebuild leaves, visited slots, use counts, and compute flags.
    // Include storage reachable only through view_src as well as ordinary operand leaves.
    for (const ggml_tensor * tensor : builder.tensors) {
        if (is_storage_leaf(tensor) || builder.imported.count(tensor)) {
            ggml_build_forward_order(graph, builder.result.tensors.at(tensor));
        }
    }
    for (const ggml_tensor * tensor : exports) {
        ggml_set_output(builder.result.tensors.at(tensor));
    }
    // An exported view must also keep its backing allocation alive.
    for (const auto & entry : builder.result.tensors) {
        if (entry.second->flags & GGML_TENSOR_FLAG_OUTPUT) {
            for (ggml_tensor * output = entry.second->view_src; output; output = output->view_src) {
                ggml_set_output(output);
            }
        }
    }
    for (int32_t index : node_indices) {
        const ggml_tensor * tensor = source->nodes[index];
        if (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) {
            ggml_build_forward_expand(graph, builder.result.tensors.at(tensor));
        } else {
            ggml_build_forward_order(graph, builder.result.tensors.at(tensor));
        }
    }
    builder.result.graph = graph;
    if (!verify_clone(source, node_indices, imports, builder.result, error)) {
        return false;
    }
    result = std::move(builder.result);
    return true;
}

bool skippy_build_executable_graph(
        const ggml_cgraph * source,
        const skippy_stage_slice_selection & selection,
        skippy_executable_graph & result,
        std::string & error) {
    result = {};
    if (source == nullptr || selection.source_graph != source ||
            !skippy_validate_stage_slice_identity(selection, error)) {
        if (error.empty()) {
            error = "executable source does not match the planned semantic graph";
        }
        return false;
    }
    if (!skippy_build_executable_graph(
                source,
                selection.node_indices,
                selection.activation_imports,
                selection.activation_exports,
                result,
                error)) {
        return false;
    }
    result.slice_identity = selection.slice_identity;
    result.node_identities = selection.node_identities;
    return true;
}
