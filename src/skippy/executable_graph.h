#pragma once

#include "ggml-cpp.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct skippy_executable_graph {
    ggml_context_ptr context;
    ggml_cgraph * graph = nullptr;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> tensors;
    std::string slice_identity;
    std::vector<std::string> node_identities;
};

struct skippy_stage_slice_selection;

// Clone tensor metadata only. The caller binds leaf storage and supplies import payloads before execution.
// Imports become independent leaves with the original layout; local aliases keep their canonical storage.
// No tensor, buffer, graph metadata, or compute flag in source is modified or borrowed.
// Ordered-only nodes retain that status. Opaque custom ops and unresolved alias bases fail closed.
// Keep the context alive until its scheduler is released; source tensor keys are identity handles only.
bool skippy_build_executable_graph(
        const ggml_cgraph * source,
        const std::vector<int32_t> & node_indices,
        const std::vector<const ggml_tensor *> & imports,
        const std::vector<const ggml_tensor *> & exports,
        skippy_executable_graph & result,
        std::string & error);

bool skippy_build_executable_graph(
        const ggml_cgraph * source,
        const skippy_stage_slice_selection & selection,
        skippy_executable_graph & result,
        std::string & error);
