#include "skippy/stage_plan.h"

#include "errors.h"
#include "graph_build_inputs.h"
#include "graph_identity.h"
#include "model_load_config_internal.h"
#include "package_tensor_binding.h"
#include "stage_slice_plan.h"
#include "../vendor/nlohmann/json.hpp"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama-context.h"
#include "llama-memory.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using model_owner = std::shared_ptr<llama_model>;
using context_owner = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct owned_profile {
    std::string profile_id;
    uint32_t n_tokens = 0;
    uint32_t n_sequences = 0;
    uint32_t n_outputs = 0;
    uint32_t n_recurrent_rollback_sequences = 0;
};

struct profile_output {
    skippy_stage_plan_profile_desc_v1 descriptor{};
    std::vector<skippy_stage_plan_value_desc_v1> imports;
    std::vector<skippy_stage_plan_value_desc_v1> exports;
    std::vector<skippy_stage_plan_value_desc_v1> requests;
    std::vector<skippy_stage_plan_state_desc_v1> states;
};

bool read_string(const char * value, std::string & output) {
    if (value == nullptr) {
        return false;
    }
    output = value;
    return !output.empty();
}

bool valid_package_id(const std::string & value) {
    if (value.size() != 71 || value.compare(0, 7, "sha256:") != 0) {
        return false;
    }
    return std::all_of(value.begin() + 7, value.end(), [](unsigned char byte) {
        return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
    });
}

bool valid_versioned(uint32_t version, uint32_t size, uint32_t expected, size_t minimum) {
    return version == expected && size >= minimum;
}

bool read_u16(const gguf_context * metadata, const char * key, uint16_t & value) {
    const int index = gguf_find_key(metadata, key);
    if (index < 0 || gguf_get_kv_type(metadata, index) != GGUF_TYPE_UINT16) {
        return false;
    }
    value = gguf_get_val_u16(metadata, index);
    return true;
}

bool validate_ordered_shards(const std::vector<std::string> & paths, std::string & error) {
    std::set<std::string> unique;
    for (size_t index = 0; index < paths.size(); ++index) {
        if (paths[index].empty() || !unique.insert(paths[index]).second) {
            error = "stage planner shard paths are empty or duplicated";
            return false;
        }
        gguf_init_params params = {true, nullptr};
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
                gguf_init_from_file(paths[index].c_str(), params), gguf_free);
        if (!metadata) {
            error = "stage planner cannot inspect explicit shard";
            return false;
        }
        if (paths.size() > 1) {
            uint16_t split_no = 0;
            uint16_t split_count = 0;
            if (!read_u16(metadata.get(), "split.no", split_no) ||
                    !read_u16(metadata.get(), "split.count", split_count) ||
                    split_no != index || split_count != paths.size()) {
                error = "stage planner shards are not in exact split.no order";
                return false;
            }
        }
    }
    return true;
}

skippy_status fail(skippy_error ** out_error, skippy_status status, const std::string & message) {
    skippy_set_error(out_error, status, message.c_str());
    return status;
}

}

struct skippy_stage_planner {
    std::string package_id;
    std::string graph_configuration_id;
    std::string backend_id;
    model_owner model;
    std::vector<skippy_parameter_binding> bindings;
    std::vector<owned_profile> profiles;
};

struct skippy_stage_plan {
    std::string package_id;
    std::string plan_id;
    model_owner model;
    std::vector<context_owner> contexts;
    skippy_guarded_stage_slice_plan guarded;
    std::vector<char> strings;
    skippy_stage_plan_desc_v1 descriptor{};
    std::vector<profile_output> profiles;
    std::set<std::string>                        embedding_profiles;
    std::vector<skippy_stage_plan_value_desc_v1> residents;

    skippy_stage_plan_string_ref_v1 add_string(const std::string & value) {
        const skippy_stage_plan_string_ref_v1 result = {
            static_cast<uint64_t>(strings.size()), static_cast<uint64_t>(value.size())};
        strings.insert(strings.end(), value.begin(), value.end());
        return result;
    }
};

namespace {

void append_values(
        skippy_stage_plan & plan,
        const std::vector<std::string> & identities,
        std::vector<skippy_stage_plan_value_desc_v1> & values) {
    for (const auto & identity : identities) {
        values.push_back({SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION,
            sizeof(skippy_stage_plan_value_desc_v1), plan.add_string(identity), {}});
    }
}

bool append_frontier_values(
        skippy_stage_plan & plan,
        const std::vector<std::string> & identities,
        const std::vector<const ggml_tensor *> & tensors,
        std::vector<skippy_stage_plan_value_desc_v1> & values,
        std::string & error) {
    if (identities.size() != tensors.size()) {
        error = "stage planner frontier identities do not match frontier tensors";
        return false;
    }
    std::set<std::string> bindings;
    for (size_t index = 0; index < identities.size(); ++index) {
        const char * binding = tensors[index] == nullptr ? nullptr : ggml_get_name(tensors[index]);
        std::string port_identity;
        if (identities[index].empty() || binding == nullptr || binding[0] == '\0' ||
                !bindings.insert(binding).second ||
                !skippy_build_activation_port_identity(binding, port_identity)) {
            error = "stage planner frontier tensors require unique stable names";
            return false;
        }
        values.push_back({SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION,
            sizeof(skippy_stage_plan_value_desc_v1), plan.add_string(port_identity),
            plan.add_string(binding)});
    }
    return true;
}

bool build_execution_contract(skippy_stage_plan & plan, const skippy_stage_planner & planner,
        std::string & error) {
    std::map<std::string, std::string> names;
    for (const auto & binding : planner.bindings) names.emplace(binding.identity, ggml_get_name(binding.tensor));
    const auto native_names = [&](const std::vector<std::string> & identities) {
        std::vector<std::string> result;
        for (const auto & identity : identities) result.push_back(names.at(identity));
        std::sort(result.begin(), result.end());
        return result;
    };
    std::vector<std::string> decoder, embeddings;
    for (const auto & profile : plan.guarded.profiles) {
        auto &     expected   = plan.embedding_profiles.count(profile.profile_id) ? embeddings : decoder;
        const auto parameters = native_names(profile.slice.parameter_identities);
        if (expected.empty()) {
            expected = parameters;
        }
        if (parameters != expected) {
            error = "decoder input profiles disagree on parameter dependencies";
            return false;
        }
    }
    std::map<int32_t, std::vector<std::string>> mtp;
    for (const auto & profile : plan.guarded.auxiliary_profiles) {
        const auto parameters = native_names(profile.parameter_identities);
        const auto inserted = mtp.emplace(profile.nextn_layer_offset, parameters);
        if (!inserted.second && inserted.first->second != parameters) {
            error = "MTP execution profiles disagree on parameter dependencies";
            return false;
        }
    }
    nlohmann::ordered_json auxiliary = nlohmann::ordered_json::array();
    for (const auto & entry : mtp) auxiliary.push_back({{"offset", entry.first}, {"parameters", entry.second}});
    nlohmann::ordered_json contract = {{"version", 1},
        {"layer_start", plan.guarded.layer_start}, {"layer_end", plan.guarded.layer_end},
        {"decoder", decoder}, {"mtp", auxiliary}};
    if (!embeddings.empty()) {
        contract["decoder_embeddings"] = embeddings;
    }
    plan.descriptor.execution_contract = plan.add_string(contract.dump());
    return true;
}

bool build_descriptors(skippy_stage_plan & plan, std::string & error) {
    if (!skippy_build_stage_plan_identity(plan.package_id, plan.guarded, plan.plan_id, error)) {
        return false;
    }
    plan.descriptor.abi_version = SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION;
    plan.descriptor.struct_size = sizeof(skippy_stage_plan_desc_v1);
    plan.descriptor.package_id = plan.add_string(plan.package_id);
    plan.descriptor.plan_id = plan.add_string(plan.plan_id);
    plan.descriptor.layer_count = plan.guarded.layer_count;
    plan.descriptor.layer_start = plan.guarded.layer_start;
    plan.descriptor.layer_end = plan.guarded.layer_end;
    plan.descriptor.profile_count = plan.guarded.profiles.size();
    plan.descriptor.resident_tensor_count = plan.guarded.resident_parameter_identities.size();
    append_values(plan, plan.guarded.resident_parameter_identities, plan.residents);
    for (const auto & profile : plan.guarded.profiles) {
        profile_output output;
        output.descriptor.abi_version = SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION;
        output.descriptor.struct_size = sizeof(skippy_stage_plan_profile_desc_v1);
        output.descriptor.profile_id = plan.add_string(profile.profile_id);
        output.descriptor.graph_identity = plan.add_string(profile.graph_identity);
        output.descriptor.profile_identity = plan.add_string(profile.profile_identity);
        output.descriptor.slice_identity = plan.add_string(profile.slice.slice_identity);
        output.descriptor.source_snapshot_identity = plan.add_string(profile.slice.source_snapshot_identity);
        output.descriptor.graph_configuration_id = plan.add_string(profile.guard.graph_configuration_id);
        output.descriptor.backend_id = plan.add_string(profile.guard.backend_id);
        output.descriptor.n_tokens = profile.guard.n_tokens;
        output.descriptor.n_sequences = profile.guard.n_sequences;
        output.descriptor.n_outputs = profile.guard.n_outputs;
        output.descriptor.n_recurrent_rollback_sequences = profile.guard.n_recurrent_rollback_sequences;
        output.descriptor.activation_import_count = profile.slice.activation_import_identities.size();
        output.descriptor.activation_export_count = profile.slice.activation_export_identities.size();
        output.descriptor.request_input_count = profile.slice.request_input_identities.size();
        output.descriptor.state_effect_count = profile.slice.state_effect_identities.size();
        if (!append_frontier_values(plan, profile.slice.activation_import_identities,
                    profile.slice.activation_imports, output.imports, error) ||
                !append_frontier_values(plan, profile.slice.activation_export_identities,
                    profile.slice.activation_exports, output.exports, error)) {
            return false;
        }
        append_values(plan, profile.slice.request_input_identities, output.requests);
        for (size_t index = 0; index < profile.slice.state_effects.size(); ++index) {
            const auto & effect = profile.slice.state_effects[index];
            output.states.push_back({SKIPPY_STAGE_PLAN_STATE_DESC_V1_ABI_VERSION,
                sizeof(skippy_stage_plan_state_desc_v1),
                plan.add_string(profile.slice.state_effect_identities[index]),
                static_cast<int32_t>(effect.kind), static_cast<int32_t>(effect.access),
                effect.layer, static_cast<int32_t>(effect.residency), effect.write_ordinal});
        }
        plan.profiles.push_back(std::move(output));
    }
    return true;
}

bool build_profile(const skippy_stage_planner & planner,
                   const owned_profile &        profile,
                   context_owner &              context,
                   skippy_stage_profile_input & input,
                   std::string &                error,
                   bool                         embedding_input = false) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(32, profile.n_tokens);
    params.n_batch = profile.n_tokens;
    params.n_ubatch = profile.n_tokens;
    params.n_seq_max = profile.n_sequences;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    // Resolve graph capabilities through the ordinary context, as execution does.
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    // A classifier or pooled embedding context runs without the LM head, so the
    // profile must observe the same model-derived context mode as the runtime.
    const auto capabilities = skippy_context_capabilities_of(planner.model.get());
    if (capabilities.classifier) {
        params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }
    params.embeddings = capabilities.pooled_embeddings;
    // Semantic planning must observe the model graph before backend rewrites.
    params.op_offload = false;
    const skippy_graph_build_inputs build_inputs{};
    context.reset(llama_init_from_model_with_graph_inputs(planner.model.get(), params, build_inputs));
    if (!context) {
        error = "stage planner cannot create metadata-only graph context";
        return false;
    }
    llama_memory_context_ptr memory_context;
    if (llama_memory_t memory = llama_get_memory(context.get())) {
        memory_context = memory->init_full();
    }
    ggml_cgraph * graph = context->graph_reserve(profile.n_tokens, profile.n_sequences, profile.n_outputs,
                                                 memory_context.get(), true, nullptr, build_inputs, embedding_input);
    const llm_graph_result * result = context->get_gf_res_reserve();
    if (graph == nullptr || result == nullptr || result->get_build_inputs().selection.enabled) {
        error = "stage planner cannot build an unfiltered whole-model graph";
        return false;
    }
    const ggml_tensor * explicit_output = result->get_logits();
    if (explicit_output == nullptr) {
        explicit_output = result->get_embd_pooled();
    }
    if (explicit_output == nullptr) {
        explicit_output = result->get_embd();
    }
    if (explicit_output == nullptr) {
        error = "stage planner graph has no logits or embedding output";
        return false;
    }
    input = {profile.profile_id, graph, &result->get_block_boundaries(),
        &result->get_state_effects(), &planner.bindings, &result->get_request_input_bindings(),
        llama_model_n_layer(planner.model.get()),
        {profile.n_tokens, profile.n_sequences, profile.n_outputs,
            profile.n_recurrent_rollback_sequences, planner.graph_configuration_id,
            planner.backend_id},
        explicit_output};
    return true;
}

bool build_mtp_profile(
        const skippy_stage_planner & planner,
        const owned_profile & profile,
        int32_t nextn_layer_offset,
        context_owner & context,
        skippy_auxiliary_graph_plan & output,
        std::string & error) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = std::max<uint32_t>(32, profile.n_tokens);
    params.n_batch = profile.n_tokens;
    params.n_ubatch = profile.n_tokens;
    params.n_seq_max = profile.n_sequences;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    params.op_offload = false;
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.embeddings = false;
    const skippy_graph_build_inputs build_inputs{};
    context.reset(llama_init_from_model_with_graph_inputs(planner.model.get(), params, build_inputs));
    if (!context) {
        error = "stage planner cannot create metadata-only MTP graph context";
        return false;
    }
    context->set_nextn_layer_offset(nextn_layer_offset);
    llama_memory_context_ptr memory_context;
    if (llama_memory_t memory = llama_get_memory(context.get())) {
        memory_context = memory->init_full();
    }
    ggml_cgraph * graph = context->graph_reserve(profile.n_tokens, profile.n_sequences,
            profile.n_outputs, memory_context.get(), true, nullptr, build_inputs);
    const llm_graph_result * result = context->get_gf_res_reserve();
    if (graph == nullptr || result == nullptr || result->get_build_inputs().selection.enabled) {
        error = "stage planner cannot build an unfiltered MTP graph";
        return false;
    }
    return skippy_build_auxiliary_graph_plan(graph, result->get_state_effects(),
            planner.bindings, result->get_request_input_bindings(), output, error);
}

bool add_mtp_profiles(
        const skippy_stage_planner & planner,
        skippy_stage_plan & plan,
        std::string & error) {
    // A router cache slot can reuse nextn storage without being an MTP head.
    if (planner.model->hparams.router_layer >= 0 ||
            plan.guarded.layer_end != plan.guarded.layer_count ||
            llama_model_n_layer_nextn(planner.model.get()) <= 0) {
        return true;
    }

    std::set<std::string> residents(plan.guarded.resident_parameter_identities.begin(),
            plan.guarded.resident_parameter_identities.end());
    const int32_t nextn_layer_count = llama_model_n_layer_nextn(planner.model.get());
    for (const auto & profile : planner.profiles) {
        for (int32_t nextn_layer_offset = 0; nextn_layer_offset < nextn_layer_count; ++nextn_layer_offset) {
            context_owner context(nullptr, llama_free);
            skippy_auxiliary_graph_plan auxiliary;
            if (!build_mtp_profile(planner, profile, nextn_layer_offset, context, auxiliary, error)) {
                error = "execution profile " + profile.profile_id + " MTP graph offset " +
                        std::to_string(nextn_layer_offset) + " rejected: " + error;
                return false;
            }
            const std::string auxiliary_profile_id = profile.profile_id + ":mtp-" +
                    std::to_string(nextn_layer_offset);
            plan.guarded.auxiliary_profiles.push_back({auxiliary_profile_id, auxiliary.graph_identity, auxiliary.parameter_identities, nextn_layer_offset});
            residents.insert(auxiliary.parameter_identities.begin(), auxiliary.parameter_identities.end());
            plan.contexts.push_back(std::move(context));
        }
    }
    plan.guarded.resident_parameter_identities.assign(residents.begin(), residents.end());
    return true;
}

const std::vector<skippy_stage_plan_value_desc_v1> * select_values(
        const profile_output & profile, skippy_stage_plan_value_kind kind) {
    switch (kind) {
        case SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_IMPORT: return &profile.imports;
        case SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT: return &profile.exports;
        case SKIPPY_STAGE_PLAN_VALUE_REQUEST_INPUT: return &profile.requests;
    }
    return nullptr;
}

}

skippy_status skippy_stage_planner_create_v1(
        const skippy_stage_planner_config_v1 * config,
        skippy_stage_planner ** out_planner,
        skippy_error ** out_error) {
    if (out_planner != nullptr) {
        *out_planner = nullptr;
    }
    if (out_planner == nullptr || config == nullptr || !valid_versioned(config->abi_version,
            config->struct_size, SKIPPY_STAGE_PLANNER_CONFIG_V1_ABI_VERSION, sizeof(*config))) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage planner config and output are required at ABI version 1");
    }
    try {
        auto planner = std::make_unique<skippy_stage_planner>();
        if (!read_string(config->package_id, planner->package_id) ||
                !valid_package_id(planner->package_id) ||
                !read_string(config->graph_configuration_id, planner->graph_configuration_id) ||
                !read_string(config->backend_id, planner->backend_id) ||
                config->shard_paths == nullptr ||
                config->shard_count == 0 || config->tensors == nullptr || config->tensor_count == 0 ||
                config->profiles == nullptr || config->profile_count == 0) {
            return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "stage planner config is incomplete");
        }
        std::vector<std::string> paths;
        paths.reserve(config->shard_count);
        for (size_t index = 0; index < config->shard_count; ++index) {
            std::string path;
            if (!read_string(config->shard_paths[index], path)) {
                return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "stage planner shard path is empty");
            }
            paths.push_back(std::move(path));
        }
        std::string error;
        if (!validate_ordered_shards(paths, error)) {
            return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, error);
        }
        std::vector<const char *> native_paths;
        for (const auto & path : paths) {
            native_paths.push_back(path.c_str());
        }
        llama_model_params params = llama_model_default_params();
        ggml_backend_dev_t planner_devices[] = {
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr,
        };
        if (planner_devices[0] == nullptr) {
            return fail(out_error, SKIPPY_STATUS_MODEL_ERROR,
                    "stage planner cannot locate the CPU backend");
        }
        params.devices = planner_devices;
        params.n_gpu_layers = 0;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.use_extra_bufts = false;
        params.no_alloc = true;
        params.load_mtp = true;
        planner->model = model_owner(llama_model_load_from_splits(
                native_paths.data(), native_paths.size(), params), llama_model_free);
        if (!planner->model) {
            return fail(out_error, SKIPPY_STATUS_MODEL_ERROR,
                    "stage planner cannot load explicit metadata shards");
        }
        std::vector<skippy_package_tensor_binding_input> tensors;
        tensors.reserve(config->tensor_count);
        for (size_t index = 0; index < config->tensor_count; ++index) {
            const auto & source = config->tensors[index];
            if (!valid_versioned(source.abi_version, source.struct_size,
                    SKIPPY_STAGE_PLANNER_TENSOR_V1_ABI_VERSION, sizeof(source)) ||
                    source.dimension_count == 0 || source.dimension_count > SKIPPY_STAGE_PLAN_MAX_DIMS ||
                    source.split_no > std::numeric_limits<uint16_t>::max()) {
                return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "stage planner tensor descriptor is malformed");
            }
            skippy_package_tensor_binding_input tensor;
            if (!read_string(source.tensor_id, tensor.tensor_id) ||
                    !read_string(source.native_name, tensor.native_name)) {
                return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "stage planner tensor identity is empty");
            }
            tensor.type = static_cast<ggml_type>(source.ggml_type);
            tensor.dimensions.assign(source.dimensions, source.dimensions + source.dimension_count);
            tensor.split_no = static_cast<uint16_t>(source.split_no);
            tensor.data_offset = source.data_offset;
            tensor.stored_length = source.stored_length;
            tensors.push_back(std::move(tensor));
        }
        if (!skippy_bind_package_tensors(planner->model.get(), tensors, planner->bindings, error)) {
            return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, error);
        }
        std::string previous_profile;
        for (size_t index = 0; index < config->profile_count; ++index) {
            const auto & source = config->profiles[index];
            owned_profile profile;
            if (!valid_versioned(source.abi_version, source.struct_size,
                    SKIPPY_STAGE_PLANNER_PROFILE_V1_ABI_VERSION, sizeof(source)) ||
                    !read_string(source.profile_id, profile.profile_id) ||
                    (!previous_profile.empty() && profile.profile_id <= previous_profile) ||
                    source.n_tokens == 0 || source.n_sequences == 0 || source.n_outputs == 0 ||
                    source.n_outputs > source.n_tokens ||
                    source.n_recurrent_rollback_sequences > source.n_sequences) {
                return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "stage planner profiles are malformed or not strictly sorted");
            }
            previous_profile = profile.profile_id;
            profile.n_tokens = source.n_tokens;
            profile.n_sequences = source.n_sequences;
            profile.n_outputs = source.n_outputs;
            profile.n_recurrent_rollback_sequences = source.n_recurrent_rollback_sequences;
            planner->profiles.push_back(std::move(profile));
        }
        *out_planner = planner.release();
        return skippy_success(out_error);
    } catch (const std::exception & exception) {
        return fail(out_error, SKIPPY_STATUS_MODEL_ERROR, exception.what());
    } catch (...) {
        return fail(out_error, SKIPPY_STATUS_MODEL_ERROR, "stage planner creation failed");
    }
}

void skippy_stage_planner_free(skippy_stage_planner * planner) {
    delete planner;
}

skippy_status skippy_stage_planner_realize_v1(
        const skippy_stage_planner * planner, int32_t layer_start, int32_t layer_end,
        skippy_stage_plan ** out_plan, skippy_error ** out_error) {
    if (out_plan != nullptr) {
        *out_plan = nullptr;
    }
    if (planner == nullptr || out_plan == nullptr || layer_start < 0 || layer_start >= layer_end ||
            layer_end > llama_model_n_layer(planner == nullptr ? nullptr : planner->model.get())) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage planner range and output are invalid");
    }
    if (llama_model_has_encoder(planner->model.get()) && llama_model_has_decoder(planner->model.get())) {
        return fail(out_error, SKIPPY_STATUS_UNSUPPORTED,
                "stage planner does not support encoder-decoder cross-attention");
    }
    try {
        auto plan = std::make_unique<skippy_stage_plan>();
        plan->package_id = planner->package_id;
        plan->model = planner->model;
        std::vector<skippy_stage_profile_input> inputs;
        inputs.reserve(planner->profiles.size() * 2);
        std::string error;
        for (size_t index = 0; index < planner->profiles.size(); ++index) {
            context_owner context(nullptr, llama_free);
            skippy_stage_profile_input input;
            if (!build_profile(*planner, planner->profiles[index], context, input, error)) {
                return fail(out_error, SKIPPY_STATUS_UNSUPPORTED, error);
            }
            const bool supports_embeddings =
                std::any_of(input.request_inputs->begin(), input.request_inputs->end(),
                            [](const auto & request) { return request.identity.compare(0, 11, "embeddings.") == 0; });
            inputs.push_back(input);
            plan->contexts.push_back(std::move(context));
            if (supports_embeddings) {
                owned_profile profile = planner->profiles[index];
                profile.profile_id    = "embeddings:" + profile.profile_id;
                if (!build_profile(*planner, profile, context, input, error, true)) {
                    return fail(out_error, SKIPPY_STATUS_UNSUPPORTED, error);
                }
                plan->embedding_profiles.insert(profile.profile_id);
                inputs.push_back(input);
                plan->contexts.push_back(std::move(context));
            }
        }
        if (!skippy_build_guarded_stage_slice_plan(inputs, layer_start, layer_end,
                    plan->guarded, error) ||
                !add_mtp_profiles(*planner, *plan, error) ||
                !build_execution_contract(*plan, *planner, error) || !build_descriptors(*plan, error)) {
            return fail(out_error, SKIPPY_STATUS_UNSUPPORTED, error);
        }
        *out_plan = plan.release();
        return skippy_success(out_error);
    } catch (const std::exception & exception) {
        return fail(out_error, SKIPPY_STATUS_UNSUPPORTED, exception.what());
    } catch (...) {
        return fail(out_error, SKIPPY_STATUS_UNSUPPORTED, "stage plan realization failed");
    }
}

void skippy_stage_plan_free(skippy_stage_plan * plan) {
    delete plan;
}

skippy_status skippy_stage_plan_describe_v1(
        const skippy_stage_plan * plan, skippy_stage_plan_desc_v1 * out_desc,
        skippy_error ** out_error) {
    if (plan == nullptr || out_desc == nullptr) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage plan and descriptor output are required");
    }
    *out_desc = plan->descriptor;
    return skippy_success(out_error);
}

skippy_status skippy_stage_plan_profile_at_v1(
        const skippy_stage_plan * plan, size_t index,
        skippy_stage_plan_profile_desc_v1 * out_desc, skippy_error ** out_error) {
    if (plan == nullptr || out_desc == nullptr || index >= plan->profiles.size()) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "stage plan profile index is invalid");
    }
    *out_desc = plan->profiles[index].descriptor;
    return skippy_success(out_error);
}

skippy_status skippy_stage_plan_resident_tensor_at_v1(
        const skippy_stage_plan * plan, size_t index,
        skippy_stage_plan_value_desc_v1 * out_desc, skippy_error ** out_error) {
    if (plan == nullptr || out_desc == nullptr || index >= plan->residents.size()) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage plan resident tensor index is invalid");
    }
    *out_desc = plan->residents[index];
    return skippy_success(out_error);
}

skippy_status skippy_stage_plan_value_at_v1(
        const skippy_stage_plan * plan, size_t profile_index,
        skippy_stage_plan_value_kind kind, size_t index,
        skippy_stage_plan_value_desc_v1 * out_desc, skippy_error ** out_error) {
    if (plan == nullptr || out_desc == nullptr || profile_index >= plan->profiles.size()) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage plan value profile index is invalid");
    }
    const auto * values = select_values(plan->profiles[profile_index], kind);
    if (values == nullptr || index >= values->size()) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage plan value kind or index is invalid");
    }
    *out_desc = (*values)[index];
    return skippy_success(out_error);
}

skippy_status skippy_stage_plan_state_at_v1(
        const skippy_stage_plan * plan, size_t profile_index, size_t index,
        skippy_stage_plan_state_desc_v1 * out_desc, skippy_error ** out_error) {
    if (plan == nullptr || out_desc == nullptr || profile_index >= plan->profiles.size() ||
            index >= plan->profiles[profile_index].states.size()) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage plan state profile or index is invalid");
    }
    *out_desc = plan->profiles[profile_index].states[index];
    return skippy_success(out_error);
}

skippy_status skippy_stage_plan_string_v1(
        const skippy_stage_plan * plan, skippy_stage_plan_string_ref_v1 reference,
        const char ** out_data, size_t * out_length, skippy_error ** out_error) {
    if (out_data != nullptr) {
        *out_data = nullptr;
    }
    if (out_length != nullptr) {
        *out_length = 0;
    }
    if (plan == nullptr || out_data == nullptr || out_length == nullptr ||
            reference.offset > plan->strings.size() ||
            reference.length > plan->strings.size() - reference.offset) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "stage plan string reference or output is invalid");
    }
    *out_data = plan->strings.data() + reference.offset;
    *out_length = static_cast<size_t>(reference.length);
    return skippy_success(out_error);
}

skippy_status skippy_stage_plan_validate_chain_v1(
        const skippy_stage_plan * const * plans, size_t plan_count,
        skippy_error ** out_error) {
    if (plans == nullptr || plan_count == 0) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "stage plan chain is empty");
    }
    std::vector<const skippy_guarded_stage_slice_plan *> guarded;
    const std::string package_id = plans[0] == nullptr ? std::string() : plans[0]->package_id;
    for (size_t index = 0; index < plan_count; ++index) {
        if (plans[index] == nullptr || plans[index]->package_id != package_id) {
            return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "stage plan chain package identities differ");
        }
        guarded.push_back(&plans[index]->guarded);
    }
    std::string error;
    if (!skippy_validate_guarded_stage_chain(guarded, error)) {
        return fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, error);
    }
    return skippy_success(out_error);
}
