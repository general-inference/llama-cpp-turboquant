#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama.h"
#include "skippy/block_boundaries.h"
#include "skippy/executable_graph.h"
#include "skippy/graph_dependencies.h"
#include "skippy/stage_slice_plan.h"
#include "skippy/state_effects.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

static bool is_dummy_buffer(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           tensor->buffer != nullptr &&
           ggml_backend_buffer_get_size(tensor->buffer) == 0 &&
           ggml_backend_buffer_get_base(tensor->buffer) == nullptr;
}

static bool check_model_buffers(const llama_model * model) {
    const auto & tensors = llama_internal_get_tensor_map(model);
    if (tensors.empty()) {
        std::fprintf(stderr, "metadata-only model has no tensors\n");
        return false;
    }
    for (const auto & [name, tensor] : tensors) {
        if (!is_dummy_buffer(tensor)) {
            std::fprintf(stderr, "model tensor %s has a payload allocation\n", name.c_str());
            return false;
        }
    }
    return true;
}

static bool check_kv_buffers(const llama_kv_cache * memory) {
    const auto layer_ids = memory->get_layer_ids();
    if (layer_ids.empty()) {
        std::fprintf(stderr, "metadata-only attention memory has no layers\n");
        return false;
    }
    for (uint32_t layer_id : layer_ids) {
        if (!is_dummy_buffer(memory->get_k_storage(static_cast<int32_t>(layer_id)))) {
            std::fprintf(stderr, "attention memory layer %u has a payload allocation\n", layer_id);
            return false;
        }
    }
    return true;
}

static bool check_recurrent_buffers(const llama_memory_recurrent * memory) {
    size_t tensor_count = 0;
    for (const auto * tensor : memory->r_l) {
        if (tensor != nullptr) {
            ++tensor_count;
            if (!is_dummy_buffer(tensor)) {
                std::fprintf(stderr, "recurrent R state has a payload allocation\n");
                return false;
            }
        }
    }
    for (const auto * tensor : memory->s_l) {
        if (tensor != nullptr) {
            ++tensor_count;
            if (!is_dummy_buffer(tensor)) {
                std::fprintf(stderr, "recurrent S state has a payload allocation\n");
                return false;
            }
        }
    }
    for (const auto * tensor : memory->p_l) {
        if (tensor != nullptr) {
            ++tensor_count;
            if (!is_dummy_buffer(tensor)) {
                std::fprintf(stderr, "recurrent P state has a payload allocation\n");
                return false;
            }
        }
    }
    if (tensor_count == 0) {
        std::fprintf(stderr, "metadata-only recurrent memory has no state tensors\n");
        return false;
    }
    return true;
}

static bool check_memory_buffers(llama_context * context, bool expect_recurrent) {
    llama_memory_t memory = llama_get_memory(context);
    if (memory == nullptr) {
        std::fprintf(stderr, "metadata-only context has no memory module\n");
        return false;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        return check_kv_buffers(hybrid->get_mem_attn()) &&
               check_recurrent_buffers(hybrid->get_mem_recr());
    }
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory)) {
        return check_recurrent_buffers(recurrent);
    }
    if (auto * dsv4 = dynamic_cast<llama_kv_cache_dsv4 *>(memory)) {
        auto * raw = dsv4->get_raw();
        return raw != nullptr &&
               check_kv_buffers(raw->get_swa()) &&
               check_kv_buffers(dsv4->get_csa()) &&
               check_kv_buffers(dsv4->get_hca()) &&
               check_kv_buffers(dsv4->get_lid());
    }
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return check_kv_buffers(iswa->get_base()) &&
               check_kv_buffers(iswa->get_swa());
    }
    if (auto * msa = dynamic_cast<llama_kv_cache_msa *>(memory)) {
        return check_kv_buffers(msa->get_base()) &&
               check_kv_buffers(msa->get_idx());
    }
    if (expect_recurrent) {
        std::fprintf(stderr, "fixture did not construct recurrent memory\n");
        return false;
    }
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (kv == nullptr) {
        std::fprintf(stderr, "fixture did not construct attention memory\n");
        return false;
    }
    return check_kv_buffers(kv);
}

static bool check_compute_buffers(llama_context * context, const char * profile_name) {
    ggml_backend_sched_t scheduler = context->get_sched();
    for (int index = 0; index < ggml_backend_sched_get_n_backends(scheduler); ++index) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(scheduler, index);
        const size_t buffer_size = ggml_backend_sched_get_buffer_size(scheduler, backend);
        if (buffer_size != 0) {
            std::fprintf(stderr, "metadata-only %s graph allocated %zu compute bytes on %s\n",
                    profile_name,
                    buffer_size,
                    ggml_backend_name(backend));
            return false;
        }
    }
    return true;
}

static void count_payload_read(ggml_tensor * tensor, void * user_data) {
    GGML_UNUSED(tensor);
    auto * payload_read_count = static_cast<size_t *>(user_data);
    ++*payload_read_count;
}

static llama_model * load_metadata_only_model(const char * model_path) {
    ggml_context * tensor_context = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &tensor_context,
    };
    gguf_context * metadata = gguf_init_from_file(model_path, params);
    if (metadata == nullptr || tensor_context == nullptr) {
        std::fprintf(stderr, "failed to inspect fixture GGUF\n");
        if (tensor_context != nullptr) {
            ggml_free(tensor_context);
        }
        if (metadata != nullptr) {
            gguf_free(metadata);
        }
        return nullptr;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.no_alloc = true;
    size_t payload_read_count = 0;
    llama_model * model = llama_model_init_from_user(
            metadata,
            count_payload_read,
            &payload_read_count,
            model_params);
    ggml_free(tensor_context);
    gguf_free(metadata);
    if (payload_read_count != 0) {
        std::fprintf(stderr, "metadata-only load requested %zu weight payloads\n", payload_read_count);
        llama_model_free(model);
        return nullptr;
    }
    return model;
}

struct graph_profile_result {
    int node_count = 0;
    size_t boundary_pairs = 0;
    size_t state_effects = 0;
    std::vector<std::string> dependencies;
};

enum class request_contract {
    unchecked,
    token_kv,
};

static bool check_request_contract(
        const skippy_stage_slice_selection & selection,
        const std::vector<skippy_request_input_binding> & bindings,
        request_contract contract,
        const char * profile_name,
        int32_t layer_start,
        int32_t layer_end,
        int32_t layer_count) {
    if (contract == request_contract::unchecked) {
        return true;
    }

    std::vector<std::string> expected = {
        "attention.key_indices.0",
        "attention.mask.0",
        "attention.value_indices.0",
        "positions.0",
    };
    if (layer_start == 0) {
        expected.push_back("tokens.0");
    }
    if (layer_end == layer_count) {
        expected.push_back("output.indices.0");
    }
    std::sort(expected.begin(), expected.end());
    std::vector<std::string> actual;
    for (const ggml_tensor * input : selection.request_inputs) {
        const auto found = std::find_if(bindings.begin(), bindings.end(), [input](const auto & binding) {
            return binding.tensor == input;
        });
        if (found == bindings.end()) {
            std::fprintf(stderr,
                    "metadata-only %s layers %d..%d contain an unbound request input\n",
                    profile_name, layer_start, layer_end);
            return false;
        }
        actual.push_back(found->identity);
    }
    std::sort(actual.begin(), actual.end());
    if (actual != expected) {
        std::fprintf(stderr,
                "metadata-only %s layers %d..%d request-input identities differ from the token/KV contract\n",
                profile_name, layer_start, layer_end);
        return false;
    }

    const size_t expected_imports = layer_start == 0 ? 0 : 1;
    const size_t expected_exports = layer_end == layer_count ? 0 : 1;
    if (selection.activation_imports.size() != expected_imports ||
            selection.activation_exports.size() != expected_exports) {
        std::fprintf(stderr,
                "metadata-only %s layers %d..%d do not expose exactly one hidden-state frontier per cut\n",
                profile_name, layer_start, layer_end);
        return false;
    }
    return true;
}

static size_t layer_state_count(
        const std::vector<skippy_state_effect> & effects,
        int32_t layer,
        skippy_state_kind kind,
        skippy_state_access access) {
    size_t count = 0;
    for (const auto & effect : effects) {
        if (effect.layer == layer && effect.kind == kind && effect.access == access) {
            ++count;
        }
    }
    return count;
}

static bool check_stage_slices(
        const llm_graph_result * graph_result,
        ggml_cgraph * graph,
        const llama_model * model,
        const char * profile_name,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs,
        uint32_t n_rs_seq,
        request_contract contract,
        uint32_t expected_activation_width) {
    std::vector<skippy_parameter_binding> parameters;
    std::map<std::string, uint32_t> parameter_ordinals;
    for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
        const uint32_t ordinal = parameter_ordinals[name]++;
        parameters.push_back({tensor, name + "." + std::to_string(ordinal)});
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const skippy_execution_profile_guard guard = {
        n_tokens,
        n_seqs,
        n_outputs,
        n_rs_seq,
        "noalloc-fixture-v1",
        "cpu",
    };
    const std::vector<skippy_stage_profile_input> profiles = {{
        profile_name,
        graph,
        &graph_result->get_block_boundaries(),
        &graph_result->get_state_effects(),
        &parameters,
        &graph_result->get_request_input_bindings(),
        n_layer,
        guard,
        graph_result->get_logits() != nullptr
                ? graph_result->get_logits()
                : graph_result->get_embd(),
    }};
    skippy_guarded_stage_slice_plan full_plan;
    std::string full_plan_error;
    if (!skippy_build_guarded_stage_slice_plan(
                profiles,
                0,
                n_layer,
                full_plan,
                full_plan_error)) {
        std::fprintf(stderr, "metadata-only %s full slice invalid: %s\n",
                profile_name, full_plan_error.c_str());
        return false;
    }
    if (!check_request_contract(
                full_plan.profiles[0].slice,
                graph_result->get_request_input_bindings(),
                contract,
                profile_name,
                0,
                n_layer,
                n_layer)) {
        return false;
    }
    const auto & full_parameter_identities = full_plan.profiles[0].slice.parameter_identities;
    std::vector<std::string> selected_dependencies;
    std::vector<skippy_guarded_stage_slice_plan> stage_plans;
    bool has_downstream_recurrent_layer = false;
    bool proved_state_preparation_retained = false;
    bool proved_self_contained_recurrent_source = false;
    const bool recurrent_layer0 = model->hparams.is_recr(0);
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        skippy_guarded_stage_slice_plan stage_plan;
        std::string error;
        if (!skippy_build_guarded_stage_slice_plan(
                    profiles,
                    layer,
                    layer + 1,
                    stage_plan,
                    error)) {
            std::fprintf(stderr, "metadata-only %s layer %d slice invalid: %s\n",
                    profile_name, layer, error.c_str());
            return false;
        }
        const auto & selection = stage_plan.profiles[0].slice;
        if (!check_request_contract(
                    selection,
                    graph_result->get_request_input_bindings(),
                    contract,
                    profile_name,
                    layer,
                    layer + 1,
                    n_layer)) {
            return false;
        }
        if (selection.node_indices.empty() || selection.parameter_identities.empty()) {
            std::fprintf(stderr, "metadata-only %s layer %d slice is incomplete\n",
                    profile_name, layer);
            return false;
        }
        skippy_executable_graph executable;
        if (!skippy_build_executable_graph(graph, selection, executable, error) ||
                executable.slice_identity != selection.slice_identity ||
                executable.node_identities != selection.node_identities) {
            std::fprintf(stderr, "metadata-only %s layer %d executable semantic clone invalid: %s\n",
                    profile_name, layer, error.c_str());
            return false;
        }
        // Rebuild at the actual profile shape. The model builder is the sole
        // source of runtime geometry; the partition must be deterministic over
        // separately owned graphs, without extracting any executable recipe.
        llm_graph_result fresh(ggml_graph_size(graph));
        llm_graph_params fresh_params = graph_result->get_params();
        fresh_params.res = &fresh;
        model->build_graph(fresh_params);
        skippy_stage_slice_selection fresh_slice;
        if (!skippy_select_stage_slice(fresh.get_gf(), fresh.get_block_boundaries(),
                    fresh.get_state_effects(), parameters, fresh.get_request_input_bindings(),
                    n_layer, layer, layer + 1, fresh_slice, error,
                    fresh.get_logits() ? fresh.get_logits() : fresh.get_embd()) ||
                fresh_slice.slice_identity != selection.slice_identity ||
                fresh_slice.node_identities != selection.node_identities ||
                fresh_slice.parameter_identities != selection.parameter_identities) {
            std::fprintf(stderr, "fresh %s layer %d graph slice differs: %s\n",
                    profile_name, layer, error.c_str());
            return false;
        }
        if (!std::is_sorted(selection.node_indices.begin(), selection.node_indices.end()) ||
                std::adjacent_find(selection.node_indices.begin(), selection.node_indices.end()) != selection.node_indices.end()) {
            std::fprintf(stderr, "metadata-only %s layer %d slice nodes are not normalized\n",
                    profile_name, layer);
            return false;
        }
        for (int32_t index : selection.node_indices) {
            if (index < 0 || index >= ggml_graph_n_nodes(graph)) {
                std::fprintf(stderr, "metadata-only %s layer %d slice has an invalid node index\n",
                        profile_name, layer);
                return false;
            }
        }
        if (layer > 0 && selection.activation_imports.empty()) {
            std::fprintf(stderr, "metadata-only %s layer %d slice has no activation import\n",
                    profile_name, layer);
            return false;
        }
        if (layer + 1 < n_layer && selection.activation_exports.empty()) {
            std::fprintf(stderr, "metadata-only %s layer %d slice has no activation export\n",
                    profile_name, layer);
            return false;
        }
        if (expected_activation_width > 0) {
            for (const auto * boundary : selection.activation_imports) {
                if (ggml_nelements(boundary) % n_tokens != 0 ||
                        ggml_nelements(boundary) / n_tokens != expected_activation_width) {
                    std::fprintf(stderr,
                            "metadata-only %s layer %d activation import width is %lld, expected %u\n",
                            profile_name, layer,
                            static_cast<long long>(ggml_nelements(boundary) / n_tokens),
                            expected_activation_width);
                    return false;
                }
            }
            for (const auto * boundary : selection.activation_exports) {
                if (ggml_nelements(boundary) % n_tokens != 0 ||
                        ggml_nelements(boundary) / n_tokens != expected_activation_width) {
                    std::fprintf(stderr,
                            "metadata-only %s layer %d activation export width is %lld, expected %u\n",
                            profile_name, layer,
                            static_cast<long long>(ggml_nelements(boundary) / n_tokens),
                            expected_activation_width);
                    return false;
                }
            }
        }
        for (const auto & effect : graph_result->get_state_effects()) {
            if (effect.layer != layer || effect.access != skippy_state_access::write) {
                continue;
            }
            bool retained = false;
            for (int32_t index : selection.node_indices) {
                retained |= ggml_graph_node(graph, index) == effect.effect_node;
            }
            if (!retained) {
                std::fprintf(stderr, "metadata-only %s layer %d omitted owning write ordinal %lld\n",
                        profile_name, layer, static_cast<long long>(effect.write_ordinal));
                return false;
            }
        }
        if (model->hparams.is_recr(layer)) {
            if (model->arch == LLM_ARCH_KIMI_LINEAR || model->arch == LLM_ARCH_KIMI_K3) {
                // KDA constructs its state preparation inside the block body.
            } else if (layer > 0) {
                has_downstream_recurrent_layer = true;
                // The retained-owning-write loop above already proves the
                // recurrent state preparation cone stayed inside the slice.
                proved_state_preparation_retained = true;
            } else {
                if (!selection.activation_imports.empty()) {
                    std::fprintf(stderr,
                            "metadata-only %s recurrent layer 0 slice imports an activation\n",
                            profile_name);
                    return false;
                }
                if (selection.request_inputs.empty()) {
                    std::fprintf(stderr,
                            "metadata-only %s recurrent layer 0 slice has no request input\n",
                            profile_name);
                    return false;
                }
                std::unordered_set<int32_t> selected(
                        selection.node_indices.begin(), selection.node_indices.end());
                size_t write_effects = 0;
                for (const auto & effect : selection.state_effects) {
                    if (effect.layer != 0 || effect.access != skippy_state_access::write) {
                        continue;
                    }
                    ++write_effects;
                    int32_t effect_index = -1;
                    for (int32_t index = 0; index < ggml_graph_n_nodes(graph); ++index) {
                        if (ggml_graph_node(graph, index) == effect.effect_node) {
                            effect_index = index;
                            break;
                        }
                    }
                    if (effect_index < 0 || selected.find(effect_index) == selected.end()) {
                        std::fprintf(stderr,
                                "metadata-only %s recurrent layer 0 slice does not own a state write node\n",
                                profile_name);
                        return false;
                    }
                }
                if (write_effects == 0) {
                    std::fprintf(stderr,
                            "metadata-only %s recurrent layer 0 slice carries no state write effects\n",
                            profile_name);
                    return false;
                }
                proved_self_contained_recurrent_source = true;
            }
        }
        selected_dependencies.insert(
                selected_dependencies.end(),
                selection.parameter_identities.begin(),
                selection.parameter_identities.end());
        stage_plans.push_back(std::move(stage_plan));
    }
    std::vector<const skippy_guarded_stage_slice_plan *> stage_chain;
    stage_chain.reserve(stage_plans.size());
    for (const auto & stage_plan : stage_plans) {
        stage_chain.push_back(&stage_plan);
    }
    std::string chain_error;
    if (!skippy_validate_guarded_stage_chain(stage_chain, chain_error)) {
        std::fprintf(stderr, "metadata-only %s stage chain invalid: %s\n",
                profile_name, chain_error.c_str());
        for (size_t stage_index = 1; stage_index < stage_plans.size(); ++stage_index) {
            const auto & exports = stage_plans[stage_index - 1].profiles[0].slice.activation_exports;
            const auto & imports = stage_plans[stage_index].profiles[0].slice.activation_imports;
            std::fprintf(stderr, "boundary %zu exports=%zu imports=%zu\n",
                    stage_index, exports.size(), imports.size());
            for (const auto * tensor : exports) {
                std::fprintf(stderr, "  export %p %s\n", static_cast<const void *>(tensor), ggml_get_name(tensor));
            }
            for (const auto * tensor : imports) {
                std::fprintf(stderr, "  import %p %s\n", static_cast<const void *>(tensor), ggml_get_name(tensor));
            }
        }
        return false;
    }
    std::sort(selected_dependencies.begin(), selected_dependencies.end());
    selected_dependencies.erase(
            std::unique(selected_dependencies.begin(), selected_dependencies.end()),
            selected_dependencies.end());
    if (selected_dependencies != full_parameter_identities) {
        std::fprintf(stderr, "metadata-only %s slices do not cover the exact graph dependency set\n",
                profile_name);
        return false;
    }
    if (has_downstream_recurrent_layer && !proved_state_preparation_retained) {
        std::fprintf(stderr, "metadata-only %s did not prove retained recurrent state preparation\n",
                profile_name);
        return false;
    }
    if (recurrent_layer0 && !proved_self_contained_recurrent_source) {
        std::fprintf(stderr, "metadata-only %s did not prove a self-contained recurrent source stage\n",
                profile_name);
        return false;
    }
    return true;
}

static bool layer_has_state(
        const std::vector<skippy_state_effect> & effects,
        int32_t layer,
        skippy_state_kind kind,
        skippy_state_access access) {
    return layer_state_count(effects, layer, kind, access) > 0;
}

enum class layer_state_class {
    kv,
    k,
    recurrent,
    kda,
    none,
};

static bool parse_layer_classes(const char * spec, std::vector<layer_state_class> & classes) {
    std::string token;
    for (const char * c = spec;; ++c) {
        if (*c == ',' || *c == '\0') {
            if (token == "kv") {
                classes.push_back(layer_state_class::kv);
            } else if (token == "k") {
                classes.push_back(layer_state_class::k);
            } else if (token == "recurrent") {
                classes.push_back(layer_state_class::recurrent);
            } else if (token == "kda") {
                classes.push_back(layer_state_class::kda);
            } else if (token == "none") {
                classes.push_back(layer_state_class::none);
            } else {
                return false;
            }
            token.clear();
            if (*c == '\0') {
                return !classes.empty();
            }
        } else {
            token += *c;
        }
    }
}

static bool check_state_effects(
        const llm_graph_result * graph_result,
        ggml_cgraph * graph,
        const llama_model * model,
        const char * profile_name,
        const std::vector<layer_state_class> & layer_classes,
        size_t expected_conv_writes) {
    std::vector<skippy_state_effect> effects;
    std::string effect_error;
    if (!skippy_infer_state_effects(graph_result->get_state_effects(), graph,
                graph_result->get_block_boundaries(), effects, effect_error)) {
        std::fprintf(stderr, "metadata-only %s state effects invalid: %s\n",
                profile_name, effect_error.c_str());
        return false;
    }

    const int32_t n_layer = llama_model_n_layer(model);
    if (layer_classes.size() != static_cast<size_t>(n_layer)) {
        std::fprintf(stderr, "fixture declares %d layers but %zu layer classes were specified\n",
                n_layer, layer_classes.size());
        return false;
    }

    // every declared layer carries exactly its expected local state class
    for (int32_t il = 0; il < n_layer; ++il) {
        const layer_state_class expected = layer_classes[il];

        const size_t kv_key_reads    = layer_state_count(effects, il, skippy_state_kind::kv_key, skippy_state_access::read);
        const size_t kv_key_writes   = layer_state_count(effects, il, skippy_state_kind::kv_key, skippy_state_access::write);
        const size_t kv_value_reads  = layer_state_count(effects, il, skippy_state_kind::kv_value, skippy_state_access::read);
        const size_t kv_value_writes = layer_state_count(effects, il, skippy_state_kind::kv_value, skippy_state_access::write);
        const size_t conv_reads      = layer_state_count(effects, il, skippy_state_kind::recurrent_conv, skippy_state_access::read);
        const size_t conv_writes     = layer_state_count(effects, il, skippy_state_kind::recurrent_conv, skippy_state_access::write);
        const size_t ssm_reads       = layer_state_count(effects, il, skippy_state_kind::recurrent_ssm, skippy_state_access::read);
        const size_t ssm_writes      = layer_state_count(effects, il, skippy_state_kind::recurrent_ssm, skippy_state_access::write);

        const size_t kv_total   = kv_key_reads + kv_key_writes + kv_value_reads + kv_value_writes;
        const size_t recr_total = conv_reads + conv_writes + ssm_reads + ssm_writes;

        bool ok = true;
        switch (expected) {
            case layer_state_class::kv:
                ok = kv_key_reads == 1 && kv_key_writes == 1 &&
                     kv_value_reads == 1 && kv_value_writes == 1 &&
                     recr_total == 0;
                break;
            case layer_state_class::k:
                ok = kv_key_reads == 1 && kv_key_writes == 1 &&
                     kv_value_reads == 0 && kv_value_writes == 0 &&
                     recr_total == 0;
                break;
            case layer_state_class::recurrent:
                ok = conv_reads == 1 && ssm_reads == 1 && ssm_writes == 3 &&
                     conv_writes == expected_conv_writes + 2 &&
                     kv_total == 0;
                if (conv_writes != expected_conv_writes + 2) {
                    std::fprintf(stderr,
                            "metadata-only %s layer %d registered %zu conv writes, expected %zu\n",
                            profile_name, il, conv_writes, expected_conv_writes + 2);
                    return false;
                }
                break;
            case layer_state_class::kda:
                ok = conv_reads == 1 && conv_writes == 5 &&
                     ssm_reads == 1 && ssm_writes == 3 && kv_total == 0;
                break;
            case layer_state_class::none:
                ok = kv_total == 0 && recr_total == 0;
                break;
        }
        if (!ok) {
            std::fprintf(stderr,
                    "metadata-only %s layer %d does not declare exactly its expected state class "
                    "(kv_key=%zu/%zu kv_value=%zu/%zu conv=%zu/%zu ssm=%zu/%zu)\n",
                    profile_name, il, kv_key_reads, kv_key_writes,
                    kv_value_reads, kv_value_writes, conv_reads, conv_writes,
                    ssm_reads, ssm_writes);
            return false;
        }
    }

    for (const auto & owner : effects) {
        if (owner.access != skippy_state_access::read ||
                (owner.kind != skippy_state_kind::recurrent_conv && owner.kind != skippy_state_kind::recurrent_ssm)) {
            continue;
        }
        int zero_position = -1;
        int copy_position = -1;
        int output_position = -1;
        const ggml_tensor * extra_read = nullptr;
        for (const auto & effect : effects) {
            if (effect.storage != owner.storage || effect.access != skippy_state_access::write) { continue; }
            const ggml_tensor * node = effect.effect_node;
            if (node->op != (effect.write_ordinal == 0 ? GGML_OP_SCALE : GGML_OP_CPY)) { return false; }
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                if (ggml_graph_node(graph, i) != node) { continue; }
                if (effect.write_ordinal == 0) { zero_position = i; }
                if (effect.write_ordinal == 1) { copy_position = i; extra_read = node->src[0]; }
            }
        }
        const bool is_kda = layer_classes[owner.layer] == layer_state_class::kda;
        const ggml_op output_op = owner.kind == skippy_state_kind::recurrent_conv || is_kda
                ? GGML_OP_GET_ROWS
                : GGML_OP_SSM_SCAN;
        for (int i = zero_position + 1; i < copy_position; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node != extra_read && node->op == output_op &&
                    skippy_state_canonical_storage(node->src[0]) == owner.storage) {
                output_position = i;
                break;
            }
        }
        if (!(zero_position >= 0 && zero_position < output_position && output_position < copy_position)) {
            std::fprintf(stderr, "metadata-only %s layer %d lost zero/output/extra-copy order\n", profile_name, owner.layer);
            return false;
        }
    }

    // persistent state stays local: no state storage is an activation boundary
    const ggml_tensor * activation_in  = skippy_state_canonical_storage(graph_result->get_skippy_activation_input());
    const ggml_tensor * activation_out = skippy_state_canonical_storage(graph_result->get_skippy_activation_output());
    for (const auto & effect : effects) {
        if (effect.storage == activation_in || effect.storage == activation_out) {
            std::fprintf(stderr,
                    "metadata-only %s state storage leaked into an activation boundary\n",
                    profile_name);
            return false;
        }
    }

    return true;
}

static bool build_graph_profile(
        llama_model * model,
        bool expect_recurrent,
        const char * profile_name,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs,
        const std::vector<layer_state_class> & layer_classes,
        uint32_t n_rs_seq,
        request_contract contract,
        uint32_t expected_activation_width,
        graph_profile_result & result) {
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 32;
    context_params.n_batch = 8;
    context_params.n_ubatch = 8;
    context_params.n_seq_max = n_seqs;
    context_params.n_rs_seq = n_rs_seq;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create metadata-only %s context\n", profile_name);
        return false;
    }

    llama_memory_context_ptr memory_context;
    if (llama_memory_t memory = llama_get_memory(context)) {
        memory_context = memory->init_full();
        if (!memory_context) {
            std::fprintf(stderr, "failed to initialize metadata-only %s memory context\n", profile_name);
            llama_free(context);
            return false;
        }
    }

    ggml_cgraph * graph = context->graph_reserve(
            n_tokens,
            n_seqs,
            n_outputs,
            memory_context.get(),
            true);
    if (graph == nullptr || ggml_graph_n_nodes(graph) == 0) {
        std::fprintf(stderr, "metadata-only %s graph is empty\n", profile_name);
        llama_free(context);
        return false;
    }
    if (!check_memory_buffers(context, expect_recurrent) || !check_compute_buffers(context, profile_name)) {
        llama_free(context);
        return false;
    }
    result.node_count = ggml_graph_n_nodes(graph);

    const llm_graph_result * graph_result = context->get_gf_res_reserve();
    const auto & boundaries = graph_result->get_block_boundaries();
    std::string boundary_error;
    if (!skippy_validate_block_boundaries(boundaries, 0, llama_model_n_layer(model), boundary_error)) {
        std::fprintf(stderr, "metadata-only %s block boundaries invalid: %s\n",
                profile_name, boundary_error.c_str());
        llama_free(context);
        return false;
    }
    result.boundary_pairs = boundaries.size() / 2;

    // mirror the mamba2 slot loop: writes = min(n_seq_tokens, 1 + effective n_rs_seq)
    const uint32_t effective_rs_seq = llama_n_rs_seq(context);
    if (n_rs_seq > 0 && effective_rs_seq != n_rs_seq) {
        std::fprintf(stderr, "metadata-only %s requested n_rs_seq=%u but context clamped it to %u\n",
                profile_name, n_rs_seq, effective_rs_seq);
        llama_free(context);
        return false;
    }
    const size_t expected_conv_writes = std::min<size_t>(n_tokens / n_seqs, 1 + effective_rs_seq);

    if (!check_state_effects(graph_result, graph, model, profile_name, layer_classes, expected_conv_writes)) {
        llama_free(context);
        return false;
    }
    result.state_effects = graph_result->get_state_effects().size();

    result.dependencies = skippy_graph_parameter_dependencies(graph, model);
    if (result.dependencies.empty()) {
        std::fprintf(stderr, "metadata-only %s graph has no parameter dependencies\n", profile_name);
        llama_free(context);
        return false;
    }
    if (!std::is_sorted(result.dependencies.begin(), result.dependencies.end()) ||
            std::adjacent_find(result.dependencies.begin(), result.dependencies.end()) != result.dependencies.end()) {
        std::fprintf(stderr, "metadata-only %s parameter dependencies are not normalized\n", profile_name);
        llama_free(context);
        return false;
    }
    if (!check_stage_slices(
                graph_result,
                graph,
                model,
                profile_name,
                n_tokens,
                n_seqs,
                n_outputs,
                n_rs_seq,
                contract,
                expected_activation_width)) {
        llama_free(context);
        return false;
    }
    llama_free(context);
    return true;
}

static int check_noalloc_graphs(
        const char * model_path,
        bool expect_recurrent,
        const std::vector<layer_state_class> & layer_classes,
        uint32_t n_rs_seq,
        request_contract contract,
        uint32_t expected_activation_width) {
    llama_model * model = load_metadata_only_model(model_path);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load metadata-only model\n");
        return 1;
    }
    if (!check_model_buffers(model)) {
        llama_model_free(model);
        return 2;
    }

    graph_profile_result prefill;
    graph_profile_result batched;
    graph_profile_result decode;
    if (!build_graph_profile(model, expect_recurrent, "prefill", 8, 1, 8, layer_classes, n_rs_seq, contract, expected_activation_width, prefill) ||
            !build_graph_profile(model, expect_recurrent, "batched", 8, 2, 8, layer_classes, n_rs_seq, contract, expected_activation_width, batched) ||
            !build_graph_profile(model, expect_recurrent, "decode", 1, 1, 1, layer_classes, n_rs_seq, contract, expected_activation_width, decode)) {
        llama_model_free(model);
        return 3;
    }
    if (prefill.dependencies != batched.dependencies) {
        std::fprintf(stderr,
                "metadata-only parameter closure differs between prefill (%zu) and batch (%zu)\n",
                prefill.dependencies.size(),
                batched.dependencies.size());
        llama_model_free(model);
        return 4;
    }
    if (prefill.dependencies != decode.dependencies) {
        std::fprintf(stderr,
                "metadata-only parameter closure differs between prefill (%zu) and decode (%zu)\n",
                prefill.dependencies.size(),
                decode.dependencies.size());
        llama_model_free(model);
        return 5;
    }
    const size_t expected_pairs = static_cast<size_t>(llama_model_n_layer(model));
    if (prefill.boundary_pairs != expected_pairs ||
            batched.boundary_pairs != expected_pairs ||
            decode.boundary_pairs != expected_pairs) {
        std::fprintf(stderr,
                "metadata-only block boundary pairs differ from declared layers: prefill=%zu batched=%zu decode=%zu expected=%zu\n",
                prefill.boundary_pairs,
                batched.boundary_pairs,
                decode.boundary_pairs,
                expected_pairs);
        llama_model_free(model);
        return 6;
    }

    std::printf("metadata-only planning: prefill_nodes=%d batched_nodes=%d decode_nodes=%d parameters=%zu block_pairs=%zu state_effects=%zu recurrent=%d\n",
            prefill.node_count,
            batched.node_count,
            decode.node_count,
            prefill.dependencies.size(),
            prefill.boundary_pairs,
            prefill.state_effects,
            expect_recurrent ? 1 : 0);
    llama_model_free(model);
    return 0;
}

static bool check_boundary_validation_rejects() {
    ggml_tensor a = {};
    ggml_tensor b = {};
    ggml_tensor c = {};
    std::string error;

    const std::vector<skippy_block_boundary> valid = {
        {skippy_block_boundary_kind::entry, 0, &a},
        {skippy_block_boundary_kind::exit,  0, &b},
        {skippy_block_boundary_kind::entry, 1, &b},
        {skippy_block_boundary_kind::exit,  1, &c},
    };
    if (!skippy_validate_block_boundaries(valid, 0, 2, error)) {
        std::fprintf(stderr, "valid boundary chain rejected: %s\n", error.c_str());
        return false;
    }

    const std::vector<std::vector<skippy_block_boundary>> rejected = {
        // missing layer 1 records
        {{skippy_block_boundary_kind::entry, 0, &a},
         {skippy_block_boundary_kind::exit,  0, &b}},
        // duplicated entry instead of exit
        {{skippy_block_boundary_kind::entry, 0, &a},
         {skippy_block_boundary_kind::entry, 0, &a},
         {skippy_block_boundary_kind::entry, 1, &b},
         {skippy_block_boundary_kind::exit,  1, &c}},
        // unordered layers
        {{skippy_block_boundary_kind::entry, 1, &b},
         {skippy_block_boundary_kind::exit,  1, &c},
         {skippy_block_boundary_kind::entry, 0, &a},
         {skippy_block_boundary_kind::exit,  0, &b}},
        // entry does not chain from previous exit
        {{skippy_block_boundary_kind::entry, 0, &a},
         {skippy_block_boundary_kind::exit,  0, &b},
         {skippy_block_boundary_kind::entry, 1, &a},
         {skippy_block_boundary_kind::exit,  1, &c}},
        // null canonical tensor
        {{skippy_block_boundary_kind::entry, 0, &a},
         {skippy_block_boundary_kind::exit,  0, nullptr},
         {skippy_block_boundary_kind::entry, 1, &b},
         {skippy_block_boundary_kind::exit,  1, &c}},
    };
    for (size_t i = 0; i < rejected.size(); ++i) {
        if (skippy_validate_block_boundaries(rejected[i], 0, 2, error)) {
            std::fprintf(stderr, "invalid boundary case %zu was accepted\n", i);
            return false;
        }
    }
    return true;
}

static bool check_state_validation_rejects() {
    std::vector<uint8_t> buf(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
    ggml_init_params init = { buf.size(), buf.data(), true };
    ggml_context * ctx = ggml_init(init);
    ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * other = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * zero = ggml_scale_inplace(ctx, ggml_view_1d(ctx, storage, 0, 0), 0);
    ggml_tensor * copy = ggml_cpy(ctx, value, storage);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    ggml_tensor * rows = ggml_set_rows(ctx, storage, value, ids);
    ggml_tensor * stray = ggml_cpy(ctx, value, storage);
    ggml_tensor * add = ggml_add(ctx, storage, value);
    ggml_tensor * sub = ggml_sub(ctx, storage, value);
    ggml_tensor * scale = ggml_scale(ctx, storage, 0);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (auto * node : {zero, copy, rows, add, sub, scale}) {
        ggml_build_forward_expand(graph, node);
    }

    std::string error;
    const auto kv = skippy_state_kind::kv_key;
    const auto rd = skippy_state_access::read;
    const auto wr = skippy_state_access::write;
    const std::vector<skippy_state_effect> valid = {
        {kv, wr, 0, storage, zero, 0},
        {kv, wr, 0, storage, copy, 1},
        {kv, wr, 0, storage, rows, 2},
        {kv, rd, 0, storage, nullptr},
    };
    bool ok = skippy_validate_state_effects(valid, graph, error);
    if (!ok) {
        std::fprintf(stderr, "valid zero-sized/multi-write/late-owner effects rejected: %s\n", error.c_str());
    }
    auto reject = [&](const char * name, const std::vector<skippy_state_effect> & effects) {
        error.clear();
        if (skippy_validate_state_effects(effects, graph, error) || error.empty()) {
            std::fprintf(stderr, "invalid state case accepted: %s\n", name);
            ok = false;
        }
    };
    auto changed = valid;
    std::reverse(changed.begin(), changed.end());
    if (!skippy_validate_state_effects(changed, graph, error)) { ok = false; }
    changed = valid;
    changed.erase(changed.begin());
    reject("omitted preparation", changed);
    changed[0].write_ordinal = 0;
    changed[1].write_ordinal = 1;
    reject("omitted preparation after renumbering", changed);
    changed = valid; changed[0].write_ordinal = -1; reject("missing ordinal", changed);
    changed = valid; changed[1].write_ordinal = 0; reject("duplicate ordinal", changed);
    changed = valid; changed[2].write_ordinal = 3; reject("ordinal gap", changed);
    changed = valid; std::swap(changed[0].write_ordinal, changed[1].write_ordinal); reject("reversed ordinals", changed);
    const auto flags = zero->flags;
    zero->flags &= ~GGML_TENSOR_FLAG_COMPUTE; reject("non-executing write", valid); zero->flags = flags;
    changed = valid; changed[0].effect_node = stray; reject("stray write", changed);
    changed = valid; changed[0].effect_node = nullptr; reject("missing node", changed);
    changed = valid; changed[0].effect_node = copy; reject("duplicate node", changed);
    for (auto * node : {add, sub, scale}) {
        changed = valid; changed[0].effect_node = node; reject("non-mutation or allocating SCALE", changed);
    }
    changed = valid; changed.pop_back(); reject("missing owner", changed);
    changed = valid; changed.push_back(valid.back()); reject("duplicate owner", changed);
    changed = valid; changed.push_back({skippy_state_kind::kv_value, rd, 0, storage, nullptr}); reject("conflicting owner", changed);
    changed = valid; changed[0].kind = static_cast<skippy_state_kind>(99); reject("unknown kind", changed);
    changed = valid; changed[0].access = static_cast<skippy_state_access>(99); reject("unknown access", changed);
    changed = valid; changed[0].layer = -1; reject("invalid layer", changed);
    changed = valid; changed[0].storage = nullptr; reject("null storage", changed);
    changed = valid; changed[0].storage = zero; reject("noncanonical storage", changed);
    changed = valid; changed.back().effect_node = zero; reject("read event instead of owner", changed);
    changed = valid; changed.back().write_ordinal = 0; reject("read has write ordinal", changed);
    changed = valid;
    changed.push_back({kv, rd, 0, other, nullptr});
    if (!skippy_validate_state_effects(changed, graph, error)) {
        std::fprintf(stderr, "distinct same-kind state owners rejected: %s\n", error.c_str());
        ok = false;
    }
    changed = valid;
    for (auto & effect : changed) { effect.storage = other; }
    reject("wrong declared storage", changed);
    auto * destination = copy->src[1];
    copy->src[1] = other; reject("wrong CPY destination", valid); copy->src[1] = destination;
    destination = rows->src[2];
    rows->src[2] = value; reject("wrong SET_ROWS destination", valid); rows->src[2] = destination;
    const size_t offset = zero->view_offs;
    zero->view_offs = ggml_nbytes(storage) + 1; reject("invalid empty view", valid); zero->view_offs = offset;
    const size_t stride = copy->nb[1];
    copy->nb[1] = SIZE_MAX; reject("overflowing view span", valid); copy->nb[1] = stride;
    const int64_t extent = copy->ne[1];
    copy->ne[1] = -1; reject("negative view extent", valid); copy->ne[1] = extent;
    const size_t nonempty_offset = copy->view_offs;
    copy->view_offs = ggml_nbytes(storage); reject("nonempty view out of bounds", valid); copy->view_offs = nonempty_offset;
    auto * base = zero->view_src;
    zero->view_src = zero; reject("cyclic view", valid); zero->view_src = base;
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 7) {
        std::fprintf(stderr, "usage: %s MODEL_PATH EXPECT_RECURRENT LAYER_CLASSES [N_RS_SEQ] [REQUEST_CONTRACT] [ACTIVATION_WIDTH]\n", argv[0]);
        return 64;
    }
    if (!check_boundary_validation_rejects()) {
        return 65;
    }
    if (!check_state_validation_rejects()) {
        return 66;
    }
    std::vector<layer_state_class> layer_classes;
    if (!parse_layer_classes(argv[3], layer_classes)) {
        std::fprintf(stderr, "invalid layer class list: %s\n", argv[3]);
        return 64;
    }
    const uint32_t n_rs_seq = argc >= 5 ? static_cast<uint32_t>(std::atoi(argv[4])) : 0;
    request_contract contract = request_contract::unchecked;
    if (argc >= 6) {
        if (std::string(argv[5]) != "token-kv" && std::string(argv[5]) != "unchecked") {
            std::fprintf(stderr, "invalid request contract: %s\n", argv[5]);
            return 64;
        }
        contract = std::string(argv[5]) == "token-kv" ? request_contract::token_kv : request_contract::unchecked;
    }
    const uint32_t expected_activation_width = argc == 7 ? static_cast<uint32_t>(std::atoi(argv[6])) : 0;
    llama_backend_init();
    const int result = check_noalloc_graphs(argv[1], std::string(argv[2]) == "1", layer_classes, n_rs_seq, contract, expected_activation_width);
    llama_backend_free();
    return result;
}
