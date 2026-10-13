#pragma once
#include <set>
#include <stdexcept>
#include "../../vendor/nlohmann/json.hpp"

static std::string contract_string(const skippy_stage_plan * plan, skippy_stage_plan_string_ref_v1 ref) {
    const char * data = nullptr;
    size_t length = 0;
    if (skippy_stage_plan_string_v1(plan, ref, &data, &length, nullptr) != SKIPPY_STATUS_OK) {
        throw std::runtime_error("invalid contract string reference");
    }
    return std::string(data, length);
}

static std::set<std::string> contract_parameters(const skippy_stage_plan * plan) {
    skippy_stage_plan_desc_v1 desc{};
    if (skippy_stage_plan_describe_v1(plan, &desc, nullptr) != SKIPPY_STATUS_OK) {
        throw std::runtime_error("invalid plan descriptor");
    }
    std::set<std::string> result;
    for (size_t i = 0; i < desc.resident_tensor_count; ++i) {
        skippy_stage_plan_value_desc_v1 value{};
        if (skippy_stage_plan_resident_tensor_at_v1(plan, i, &value, nullptr) != SKIPPY_STATUS_OK) {
            throw std::runtime_error("invalid parameter descriptor");
        }
        if (!result.insert(contract_string(plan, value.identity) + "=" + contract_string(plan, value.binding)).second) {
            throw std::runtime_error("duplicate resident parameter");
        }
    }
    if (result.empty()) { throw std::runtime_error("stage has no resident parameters"); }
    return result;
}

static void check_state_ownership(const skippy_stage_plan * plan, bool causal = true) {
    skippy_stage_plan_desc_v1 desc{};
    if (skippy_stage_plan_describe_v1(plan, &desc, nullptr) != SKIPPY_STATUS_OK) {
        throw std::runtime_error("invalid state plan descriptor");
    }
    for (size_t p = 0; p < desc.profile_count; ++p) {
        skippy_stage_plan_profile_desc_v1 profile{};
        if (skippy_stage_plan_profile_at_v1(plan, p, &profile, nullptr) != SKIPPY_STATUS_OK) {
            throw std::runtime_error("invalid state profile descriptor");
        }
        if (causal && profile.state_effect_count == 0) { throw std::runtime_error("causal profile lost all state effects"); }
        std::set<std::string> reads, writes;
        for (size_t i = 0; i < profile.state_effect_count; ++i) {
            skippy_stage_plan_state_desc_v1 state{};
            if (skippy_stage_plan_state_at_v1(plan, p, i, &state, nullptr) != SKIPPY_STATUS_OK) {
                throw std::runtime_error("invalid state descriptor");
            }
            if (state.residency == SKIPPY_STAGE_PLAN_STATE_LAYER_LOCAL &&
                (state.layer < desc.layer_start || state.layer >= desc.layer_end)) {
                throw std::runtime_error("stage accesses foreign layer-local state");
            }
            auto identity = contract_string(plan, state.identity);
            if (identity.empty()) { throw std::runtime_error("state identity is empty"); }
            (state.access == SKIPPY_STAGE_PLAN_STATE_READ ? reads : writes).insert(identity);
        }
        if (causal && (reads.empty() || writes.empty())) { throw std::runtime_error("causal profile lost state reads or writes"); }
    }
}

static void check_mtp_contract(const skippy_stage_plan * plan, int32_t expected_heads) {
    skippy_stage_plan_desc_v1 desc{};
    if (skippy_stage_plan_describe_v1(plan, &desc, nullptr) != SKIPPY_STATUS_OK) {
        throw std::runtime_error("invalid MTP plan descriptor");
    }
    const auto contract = nlohmann::json::parse(contract_string(plan, desc.execution_contract));
    const auto & heads = contract.at("mtp");
    if (heads.size() != size_t(expected_heads)) {
        throw std::runtime_error("execution contract omitted or misplaced MTP heads");
    }
    std::set<std::string> residents;
    for (size_t i = 0; i < desc.resident_tensor_count; ++i) {
        skippy_stage_plan_value_desc_v1 value{};
        if (skippy_stage_plan_resident_tensor_at_v1(plan, i, &value, nullptr) != SKIPPY_STATUS_OK) {
            throw std::runtime_error("invalid MTP resident descriptor");
        }
        residents.insert(contract_string(plan, value.identity));
    }
    for (int32_t i = 0; i < expected_heads; ++i) {
        if (heads.at(i).at("offset") != i || heads.at(i).at("parameters").empty()) {
            throw std::runtime_error("MTP head lacks offset or parameters");
        }
        for (const auto & parameter : heads.at(i).at("parameters")) {
            if (!residents.count(parameter.get<std::string>())) {
                throw std::runtime_error("MTP parameter missing from terminal stage");
            }
        }
    }
}

static std::set<std::string> contract_states(const skippy_stage_plan * plan) {
    skippy_stage_plan_desc_v1 desc{};
    if (skippy_stage_plan_describe_v1(plan, &desc, nullptr) != SKIPPY_STATUS_OK) {
        throw std::runtime_error("invalid state plan descriptor");
    }
    std::set<std::string> result;
    for (size_t p = 0; p < desc.profile_count; ++p) {
        skippy_stage_plan_profile_desc_v1 profile{};
        if (skippy_stage_plan_profile_at_v1(plan, p, &profile, nullptr) != SKIPPY_STATUS_OK) {
            throw std::runtime_error("invalid state profile");
        }
        for (size_t i = 0; i < profile.state_effect_count; ++i) {
            skippy_stage_plan_state_desc_v1 state{};
            if (skippy_stage_plan_state_at_v1(plan, p, i, &state, nullptr) != SKIPPY_STATUS_OK) {
                throw std::runtime_error("invalid state effect");
            }
            result.insert(contract_string(plan, profile.profile_id) + ":" +
                    contract_string(plan, state.identity) + ":" + std::to_string(state.access));
        }
    }
    return result;
}

static void check_invalid_contract_requests(skippy_stage_planner * planner,
        skippy_stage_planner_config_v1 config, int32_t layers) {
    const int32_t ranges[][2] = {{-1, layers}, {0, layers + 1}, {1, 1}, {2, 1}};
    for (const auto & range : ranges) {
        skippy_stage_plan * invalid = nullptr;
        skippy_error * error = nullptr;
        const auto status = skippy_stage_planner_realize_v1(planner, range[0], range[1], &invalid, &error);
        const bool rejected = status == SKIPPY_STATUS_INVALID_ARGUMENT && invalid == nullptr;
        skippy_stage_plan_free(invalid);
        skippy_error_free(error);
        if (!rejected) { throw std::runtime_error("invalid stage range was admitted"); }
    }
    skippy_stage_planner_profile_v1 invalid_profile{1, sizeof(invalid_profile), "invalid", 1, 1, 2, 0};
    config.profiles = &invalid_profile;
    config.profile_count = 1;
    skippy_stage_planner * invalid = nullptr;
    skippy_error * error = nullptr;
    const auto status = skippy_stage_planner_create_v1(&config, &invalid, &error);
    const bool rejected = status == SKIPPY_STATUS_INVALID_ARGUMENT && invalid == nullptr;
    skippy_stage_planner_free(invalid);
    skippy_error_free(error);
    if (!rejected) { throw std::runtime_error("impossible output profile was admitted"); }
}
