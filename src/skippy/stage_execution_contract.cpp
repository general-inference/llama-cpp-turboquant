#include "stage_execution_contract.h"
#include "llama-graph.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace {
std::vector<std::string> canonical(std::vector<std::string> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::vector<std::string> read_parameters(const nlohmann::json & value) {
    const auto parameters = value.get<std::vector<std::string>>();
    if (parameters.empty() || canonical(parameters) != parameters ||
            std::any_of(parameters.begin(), parameters.end(), [](const std::string & name) { return name.empty(); })) {
        throw std::runtime_error("admitted profile parameters must be nonempty, sorted and unique");
    }
    return parameters;
}
}

void skippy_validate_execution_dependencies(const skippy_graph_build_inputs & inputs,
                                            bool                              auxiliary,
                                            int32_t                           offset,
                                            const std::vector<std::string> &  parameters,
                                            bool                              embedding_input) {
    const auto residents = canonical(inputs.resident_tensor_names);
    const auto realized = canonical(parameters);
    if (inputs.execution_contract.empty()) {
        throw std::runtime_error("staged execution requires an admitted execution contract");
    }
    const auto contract = nlohmann::json::parse(inputs.execution_contract);
    if (contract.at("version") != 1 ||
            contract.at("layer_start") != inputs.selection.layer_start ||
            contract.at("layer_end") != inputs.selection.layer_end) {
        throw std::runtime_error("admitted execution contract version or stage range mismatch");
    }
    const auto decoder = read_parameters(contract.at("decoder"));
    std::set<std::string> union_parameters(decoder.begin(), decoder.end());
    std::vector<std::string> expected = auxiliary ? std::vector<std::string>() : decoder;
    if (contract.contains("decoder_embeddings")) {
        const auto embeddings = read_parameters(contract.at("decoder_embeddings"));
        union_parameters.insert(embeddings.begin(), embeddings.end());
        if (!auxiliary && embedding_input) {
            expected = embeddings;
        }
    } else if (!auxiliary && embedding_input) {
        throw std::runtime_error("raw embedding input has no admitted execution profile");
    }
    int32_t expected_offset = 0;
    const auto & mtp = contract.at("mtp");
    if (!mtp.is_array()) throw std::runtime_error("admitted auxiliary profiles must be an array");
    for (const auto & profile : mtp) {
        if (profile.at("offset") != expected_offset++) {
            throw std::runtime_error("admitted auxiliary profile offsets must be contiguous and unique");
        }
        const auto required = read_parameters(profile.at("parameters"));
        union_parameters.insert(required.begin(), required.end());
        if (auxiliary && profile.at("offset") == offset) expected = required;
    }
    if (residents != std::vector<std::string>(union_parameters.begin(), union_parameters.end())) {
        throw std::runtime_error("admitted stage residents do not match the execution profile union");
    }
    if (expected.empty() || expected != realized) {
        throw std::runtime_error("realized graph parameters do not match the admitted execution profile: expected=" +
                                 nlohmann::json(expected).dump() + " realized=" + nlohmann::json(realized).dump());
    }
}
