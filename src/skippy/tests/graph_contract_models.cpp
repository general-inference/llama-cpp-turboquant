#include "ggml.h"
#include "gguf.h"
#include "skippy/stage_plan.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "graph_contract_checks.h"
#include "../../vendor/nlohmann/json.hpp"

// Validate real architecture graphs using sparse synthetic GGUFs, without inference.
static int run_contract(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s model.gguf [causal|stateless|shared-kv-18|encoder-decoder-rejection]\n", argv[0]);
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
    const std::string mtp_key = std::string(arch) + ".nextn_predict_layers";
    const int64_t mtp_index = gguf_find_key(metadata.get(), mtp_key.c_str());
    const int32_t mtp_layers = mtp_index < 0 ? 0 : gguf_get_val_u32(metadata.get(), mtp_index);
    const int32_t layers = gguf_get_val_u32(metadata.get(), gguf_find_key(metadata.get(), layers_key.c_str())) - mtp_layers;
    const std::string mode = argc == 3 ? argv[2] : "causal";
    const int32_t shared_tail = mode == "shared-kv-18" ? 18 : mode == "shared-kv-22" ? 22 : layers;
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
    const skippy_stage_planner_profile_v1 profiles[] = {
        {1, sizeof(skippy_stage_planner_profile_v1), "batched-decode", 2, 2, 2, 0},
        {1, sizeof(skippy_stage_planner_profile_v1), "decode", 1, 1, 1, 0},
        {1, sizeof(skippy_stage_planner_profile_v1), "prefill", 8, 1, 8, 0},
        {1, sizeof(skippy_stage_planner_profile_v1), "prefill-last", 8, 1, 1, 0},
    };
    const char * paths[] = {argv[1]};
    const std::string package_id = "sha256:" + std::string(64, 'a');
    skippy_stage_planner_config_v1 config{1, sizeof(config), package_id.c_str(),
        paths, 1, tensors.data(), tensors.size(), profiles, 4, "test.graph", "test.cpu"};
    skippy_error * error = nullptr;
    skippy_stage_planner * planner = nullptr;
    auto status = skippy_stage_planner_create_v1(&config, &planner, &error);
    std::unique_ptr<skippy_stage_planner, decltype(&skippy_stage_planner_free)> owner(planner, skippy_stage_planner_free);
    if (status != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "planner creation failed: %s\n", error ? error->message : "unknown");
        skippy_error_free(error);
        return 1;
    }
    check_invalid_contract_requests(planner, config, layers);
    if (mode == "encoder-decoder-rejection") {
        for (int32_t start = 0; start < layers; ++start) {
            skippy_stage_plan * rejected = nullptr;
            status = skippy_stage_planner_realize_v1(planner, start, layers, &rejected, &error);
            if (status != SKIPPY_STATUS_UNSUPPORTED || rejected != nullptr || error == nullptr ||
                    std::string(error->message) != "stage planner does not support encoder-decoder cross-attention") {
                skippy_stage_plan_free(rejected);
                skippy_error_free(error);
                throw std::runtime_error("encoder-decoder staging must reject before graph construction");
            }
            skippy_error_free(error);
            error = nullptr;
        }
        return 0;
    }
    skippy_stage_plan * whole_raw = nullptr;
    if (skippy_stage_planner_realize_v1(planner, 0, layers, &whole_raw, &error) != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "whole-model plan failed: %s\n", error ? error->message : "unknown");
        skippy_error_free(error);
        return 1;
    }
    std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> whole(whole_raw, skippy_stage_plan_free);
    const auto all_parameters = contract_parameters(whole.get());
    const auto all_states = contract_states(whole.get());
    if (mode == "stateless" && !all_states.empty()) {
        throw std::runtime_error("stateless graph contract unexpectedly owns mutable state");
    }
    check_state_ownership(whole.get(), mode != "stateless");
    check_mtp_contract(whole.get(), mtp_layers);
    // Balanced canary cuts, edge cuts, and a multi-stage chain exercise forwarding.
    std::vector<std::vector<int32_t>> chains = {
        {0, 1, layers}, {0, layers / 2, layers}, {0, layers - 1, layers},
        {0, layers / 3, (2 * layers + 1) / 3, layers},
    };
    for (int32_t cut = 2; cut < layers - 1; ++cut) { chains.push_back({0, cut, layers}); }
    for (const auto & cuts : chains) {
        const bool forbidden_cut = std::any_of(cuts.begin() + 1, cuts.end() - 1,
                [shared_tail](int32_t cut) { return cut > shared_tail; });
        std::vector<std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)>> owned;
        std::vector<const skippy_stage_plan *> chain;
        std::set<std::string> resident_union, state_union;
        for (size_t i = 1; i < cuts.size(); ++i) {
            skippy_stage_plan * plan = nullptr;
            status = skippy_stage_planner_realize_v1(planner, cuts[i-1], cuts[i], &plan, &error);
            owned.emplace_back(plan, skippy_stage_plan_free);
            if (status != SKIPPY_STATUS_OK) { break; }
            skippy_stage_plan_desc_v1 descriptor{};
            status = skippy_stage_plan_describe_v1(plan, &descriptor, &error);
            if (status != SKIPPY_STATUS_OK) { break; }
            if (descriptor.profile_count != (mode == "causal-tokens" ? 4u : 8u)) {
                std::fprintf(stderr, "expected token and embedding decode/prefill profiles, got %zu\n", size_t(descriptor.profile_count));
                return 1;
            }
            std::vector<std::string> names;
            for (size_t profile_index = 0; profile_index < descriptor.profile_count; ++profile_index) {
                skippy_stage_plan_profile_desc_v1 profile{};
                const char * name = nullptr;
                size_t length = 0;
                if (skippy_stage_plan_profile_at_v1(plan, profile_index, &profile, &error) != SKIPPY_STATUS_OK ||
                    skippy_stage_plan_string_v1(plan, profile.profile_id, &name, &length, &error) != SKIPPY_STATUS_OK) {
                    std::fprintf(stderr, "cannot inspect planned profile\n");
                    skippy_error_free(error);
                    return 1;
                }
                names.emplace_back(name, length);
            }
            std::sort(names.begin(), names.end());
            std::vector<std::string> expected_names{"batched-decode", "decode", "prefill", "prefill-last"};
            if (mode != "causal-tokens") {
                expected_names.insert(expected_names.end(), {"embeddings:batched-decode", "embeddings:decode", "embeddings:prefill", "embeddings:prefill-last"});
                std::sort(expected_names.begin(), expected_names.end());
            }
            if (names != expected_names) {
                std::fprintf(stderr, "planned profile identities differ from requested input modes\n");
                return 1;
            }
            auto resident = contract_parameters(plan);
            resident_union.insert(resident.begin(), resident.end());
            check_state_ownership(plan, false);
            const auto states = contract_states(plan);
            state_union.insert(states.begin(), states.end());
            check_mtp_contract(plan, cuts[i] == layers ? mtp_layers : 0);
            chain.push_back(plan);
        }
        if (forbidden_cut) {
            if (status != SKIPPY_STATUS_UNSUPPORTED || !error ||
                std::string(error->message).find("owned outside the requested slice") == std::string::npos) {
                std::fprintf(stderr, "shared-KV cut must reject foreign state ownership\n");
                skippy_error_free(error);
                return 1;
            }
            skippy_error_free(error);
            error = nullptr;
            continue;
        }
        if (status == SKIPPY_STATUS_OK) {
            status = skippy_stage_plan_validate_chain_v1(chain.data(), chain.size(), &error);
        }
        if (status != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "%s split %d: %s\n", arch, cuts[1], error ? error->message : "unknown");
            skippy_error_free(error);
            return 1;
        }
        if (state_union != all_states) {
            throw std::runtime_error("split state effects differ from whole-model plan");
        }
        if (resident_union != all_parameters) {
            std::fprintf(stderr, "split parameter closure differs from whole-model plan\n");
            return 1;
        }
        // A missing middle stage must not be accepted as a complete contract.
        if (chain.size() == 3) {
            const skippy_stage_plan * gap[] = {chain.front(), chain.back()};
            if (skippy_stage_plan_validate_chain_v1(gap, 2, &error) == SKIPPY_STATUS_OK) {
                std::fprintf(stderr, "accepted a chain with a missing stage\n");
                return 1;
            }
            skippy_error_free(error);
            error = nullptr;
        }
    }
    config.backend_id = "test.other-backend";
    skippy_stage_planner * other_raw = nullptr;
    if (skippy_stage_planner_create_v1(&config, &other_raw, &error) != SKIPPY_STATUS_OK) { return 1; }
    std::unique_ptr<skippy_stage_planner, decltype(&skippy_stage_planner_free)> other(other_raw, skippy_stage_planner_free);
    skippy_stage_plan * left_raw = nullptr, * right_raw = nullptr;
    if (skippy_stage_planner_realize_v1(planner, 0, layers / 2, &left_raw, &error) != SKIPPY_STATUS_OK ||
        skippy_stage_planner_realize_v1(other.get(), layers / 2, layers, &right_raw, &error) != SKIPPY_STATUS_OK) { return 1; }
    std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> left(left_raw, skippy_stage_plan_free), right(right_raw, skippy_stage_plan_free);
    const skippy_stage_plan * incompatible[] = {left.get(), right.get()};
    if (skippy_stage_plan_validate_chain_v1(incompatible, 2, &error) == SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "accepted incompatible backend contracts\n");
        return 1;
    }
    skippy_error_free(error);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run_contract(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "graph contract failed for %s: %s\n", argc > 1 ? argv[1] : "unknown model", error.what());
        return 1;
    }
}
