#pragma once

#include "graph_identity.h"

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

struct skippy_package_tensor_binding_input {
    std::string tensor_id;
    std::string native_name;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> dimensions;
    uint16_t split_no = 0;
    uint64_t data_offset = 0;
    uint64_t stored_length = 0;
};

bool skippy_bind_package_tensors(
        const llama_model * model,
        const std::vector<skippy_package_tensor_binding_input> & inputs,
        std::vector<skippy_parameter_binding> & bindings,
        std::string & error);
