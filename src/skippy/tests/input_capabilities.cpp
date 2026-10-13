#include "ggml.h"
#include "gguf.h"
#include "skippy/stage_plan.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// Realize a stage from a synthetic GGUF without reading any weight payloads.
// The expected profile count distinguishes token-only from embedding-capable graphs.
int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s model.gguf expected-profiles\n", argv[0]);
        return 1;
    }
    ggml_context * raw_context = nullptr;
    gguf_init_params params{true, &raw_context};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(argv[1], params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(raw_context, ggml_free);
    if (!metadata || !context) {
        return 1;
    }
    const auto * arch = gguf_get_val_str(metadata.get(), gguf_find_key(metadata.get(), "general.architecture"));
    const std::string layers_key = std::string(arch) + ".block_count";
    const int32_t layers = gguf_get_val_u32(metadata.get(), gguf_find_key(metadata.get(), layers_key.c_str()));
    std::vector<skippy_stage_planner_tensor_v1> tensors;
    for (int64_t index = 0; index < gguf_get_n_tensors(metadata.get()); ++index) {
        const char * name = gguf_get_tensor_name(metadata.get(), index);
        const auto * source = ggml_get_tensor(context.get(), name);
        skippy_stage_planner_tensor_v1 tensor{};
        tensor.abi_version = 1;
        tensor.struct_size = sizeof(tensor);
        tensor.tensor_id = name;
        tensor.native_name = name;
        tensor.ggml_type = source->type;
        tensor.dimension_count = ggml_n_dims(source);
        std::copy(source->ne, source->ne + 4, tensor.dimensions);
        tensor.data_offset = gguf_get_data_offset(metadata.get()) + gguf_get_tensor_offset(metadata.get(), index);
        tensor.stored_length = ggml_nbytes(source);
        tensors.push_back(tensor);
    }
    std::sort(tensors.begin(), tensors.end(), [](const auto & left, const auto & right) {
        return std::string(left.tensor_id) < std::string(right.tensor_id);
    });
    const skippy_stage_planner_profile_v1 profile{1, sizeof(profile), "prefill", 8, 1, 8, 0};
    const char * paths[] = {argv[1]};
    const std::string package_id = "sha256:" + std::string(64, 'a');
    skippy_stage_planner_config_v1 config{1, sizeof(config), package_id.c_str(),
        paths, 1, tensors.data(), tensors.size(), &profile, 1, "test.graph", "test.cpu"};
    skippy_error * error = nullptr;
    skippy_stage_planner * planner = nullptr;
    auto status = skippy_stage_planner_create_v1(&config, &planner, &error);
    std::unique_ptr<skippy_stage_planner, decltype(&skippy_stage_planner_free)> owner(planner, skippy_stage_planner_free);
    skippy_stage_plan * plan = nullptr;
    if (status == SKIPPY_STATUS_OK) {
        status = skippy_stage_planner_realize_v1(planner, 0, layers, &plan, &error);
    }
    std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> plan_owner(plan, skippy_stage_plan_free);
    skippy_stage_plan_desc_v1 descriptor{};
    if (status == SKIPPY_STATUS_OK) {
        status = skippy_stage_plan_describe_v1(plan, &descriptor, &error);
    }
    if (status != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "planning failed: %s\n", error ? error->message : "unknown");
        skippy_error_free(error);
        return 1;
    }
    const size_t expected = std::strtoul(argv[2], nullptr, 10);
    if (descriptor.profile_count != expected) {
        std::fprintf(stderr, "expected %zu profiles, got %zu\n", expected, descriptor.profile_count);
        return 1;
    }
    return 0;
}
