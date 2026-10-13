#include "graph_dependencies.h"

#include "ggml.h"
#include "llama-model.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

std::vector<std::string> skippy_graph_parameter_dependencies(
        ggml_cgraph * graph,
        const llama_model * model) {
    if (graph == nullptr || model == nullptr) {
        return {};
    }

    std::unordered_map<const ggml_tensor *, std::vector<const std::string *>> names_by_tensor;
    for (const auto & [name, tensor] : model->tensors_by_name) {
        if (tensor != nullptr) {
            names_by_tensor[tensor].push_back(&name);
        }
    }

    std::vector<const ggml_tensor *> pending;
    pending.reserve(static_cast<size_t>(ggml_graph_n_nodes(graph)));
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        pending.push_back(ggml_graph_node(graph, index));
    }

    std::unordered_set<const ggml_tensor *> visited;
    std::set<std::string> dependencies;
    while (!pending.empty()) {
        const ggml_tensor * tensor = pending.back();
        pending.pop_back();
        if (tensor == nullptr || !visited.insert(tensor).second) {
            continue;
        }

        const auto names = names_by_tensor.find(tensor);
        if (names != names_by_tensor.end()) {
            for (const std::string * name : names->second) {
                dependencies.insert(*name);
            }
        }

        if (tensor->view_src != nullptr) {
            pending.push_back(tensor->view_src);
        }
        for (const ggml_tensor * source : tensor->src) {
            if (source != nullptr) {
                pending.push_back(source);
            }
        }
    }

    return { dependencies.begin(), dependencies.end() };
}
