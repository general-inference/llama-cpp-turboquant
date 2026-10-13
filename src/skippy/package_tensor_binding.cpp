#include "package_tensor_binding.h"

#include "llama-model.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace {

bool valid_dimensions(const std::vector<int64_t> & dimensions) {
    return !dimensions.empty() &&
           dimensions.size() <= GGML_MAX_DIMS &&
           std::all_of(dimensions.begin(), dimensions.end(), [](int64_t dimension) {
               return dimension > 0;
           });
}

bool tensor_dimensions_match(
        const std::vector<int64_t> & source,
        const std::vector<int64_t> & dimensions) {
    // GGUF permits equivalent ranks padded with trailing ones.
    for (size_t index = 0; index < GGML_MAX_DIMS; ++index) {
        const int64_t expected = index < dimensions.size() ? dimensions[index] : 1;
        if (source[index] != expected) {
            return false;
        }
    }
    return true;
}

}

bool skippy_bind_package_tensors(
        const llama_model * model,
        const std::vector<skippy_package_tensor_binding_input> & inputs,
        std::vector<skippy_parameter_binding> & bindings,
        std::string & error) {
    bindings.clear();
    error.clear();
    if (model == nullptr || inputs.empty()) {
        error = "package tensor binding requires a model and a non-empty catalog";
        return false;
    }

    std::map<std::string, std::vector<const ggml_tensor *>> native_tensors;
    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        if (name.empty() || tensor == nullptr) {
            error = "native model tensor inventory is incomplete";
            return false;
        }
        auto & instances = native_tensors[name];
        if (std::find(instances.begin(), instances.end(), tensor) == instances.end()) {
            instances.push_back(tensor);
        }
    }
    if (native_tensors.size() != inputs.size()) {
        error = "package and native tensor inventory sizes differ";
        return false;
    }

    std::set<std::string> native_names;
    std::string previous_tensor_id;
    bindings.reserve(inputs.size());
    for (const auto & input : inputs) {
        if (input.tensor_id.empty() ||
                (!previous_tensor_id.empty() && input.tensor_id <= previous_tensor_id)) {
            error = "package tensor IDs are empty or not strictly increasing";
            return false;
        }
        previous_tensor_id = input.tensor_id;
        if (input.native_name.empty() || !native_names.insert(input.native_name).second) {
            error = "package native tensor locators are empty or duplicated";
            return false;
        }
        if (input.type < 0 || input.type >= GGML_TYPE_COUNT ||
                !valid_dimensions(input.dimensions) || input.stored_length == 0) {
            error = "package tensor metadata is incomplete";
            return false;
        }
        const auto found = native_tensors.find(input.native_name);
        if (found == native_tensors.end()) {
            error = "package tensor locator is absent from the native model: " + input.native_name;
            return false;
        }
        // Compare the package against the original physical GGUF descriptor,
        // not the logical shape produced by TENSOR_ALLOW_RESHAPE. Equal element
        // counts alone would also accept forged package layouts.
        ggml_type source_type = GGML_TYPE_COUNT;
        std::vector<int64_t> source_dimensions;
        if (!model->tensor_source_metadata(input.native_name.c_str(), source_type, source_dimensions) ||
                source_type != input.type || !tensor_dimensions_match(source_dimensions, input.dimensions)) {
            error = "package tensor metadata differs from its GGUF source: " + input.native_name;
            return false;
        }
        for (const ggml_tensor * tensor : found->second) {
            if (tensor->type != input.type || !ggml_is_contiguous(tensor) ||
                    ggml_nbytes(tensor) != input.stored_length) {
                error = "package tensor metadata differs from the native model: " + input.native_name;
                return false;
            }
        }
        uint16_t split_no = 0;
        size_t data_offset = 0;
        size_t stored_length = 0;
        if (!model->tensor_source_location(
                    input.native_name.c_str(), split_no, data_offset, stored_length) ||
                split_no != input.split_no ||
                data_offset != input.data_offset ||
                stored_length != input.stored_length) {
            error = "package tensor storage locator differs from the native model: " + input.native_name;
            return false;
        }
        for (const ggml_tensor * tensor : found->second) {
            bindings.push_back({tensor, input.tensor_id});
        }
    }
    return true;
}
