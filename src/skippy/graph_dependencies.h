#pragma once

#include <string>
#include <vector>

struct ggml_cgraph;
struct llama_model;

std::vector<std::string> skippy_graph_parameter_dependencies(
        ggml_cgraph * graph,
        const llama_model * model);
