#include "llama-context.h"
#include "llama-graph.h"
#include "llama-kv-cache-msa.h"
#include "ggml-cpu.h"
#include "skippy/stage_execution_contract.h"
#include "../../vendor/nlohmann/json.hpp"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama.h"
#include "skippy/stage_slice_plan.h"
#include "skippy/stage_realization.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class recording_input final : public llm_graph_input_i {
public:
    void set_build_inputs(const skippy_graph_build_inputs & inputs) override {
        observed = inputs;
        set_count++;
    }

    void set_input(const llama_ubatch *) override {
        input_count++;
    }

    skippy_graph_build_inputs observed;
    uint32_t set_count = 0;
    uint32_t input_count = 0;
};

static skippy_graph_build_inputs make_inputs(uint32_t id) {
    skippy_graph_build_inputs inputs;
    inputs.selection.enabled = true;
    inputs.selection.layer_start = static_cast<int32_t>(id);
    inputs.selection.layer_end = static_cast<int32_t>(id + 3);
    inputs.selection.glm_dsa_policy_profile = static_cast<int32_t>(id + 10);
    inputs.selection.glm_dsa_policy_flags = id + 20;
    inputs.selection.glm_dsa_short_prefill_max_tokens = static_cast<int32_t>(id + 30);
    inputs.selection.glm_dsa_direct_sparse_decode_max_top_k = static_cast<int32_t>(id + 40);
    inputs.selection.glm_dsa_dense_sparse_mask_max_bytes = id + 50;
    inputs.selection.glm_dsa_compact_flash_min_kv = static_cast<int32_t>(id + 60);
    inputs.selection.glm_dsa_phase_hint = static_cast<int32_t>(id % 4);
    inputs.activation_tokens.tokens = reinterpret_cast<const llama_token *>(static_cast<uintptr_t>(id + 1));
    inputs.activation_tokens.token_count = id + 2;
    inputs.mtp_embeddings.values = reinterpret_cast<const float *>(static_cast<uintptr_t>(id + 19));
    inputs.mtp_embeddings.token_count = id + 20;
    inputs.mtp_embeddings.n_embd = id + 21;
    inputs.mtp_embeddings.pos_start = static_cast<llama_pos>(id + 22);
    return inputs;
}

static bool matches(const skippy_graph_build_inputs & inputs, uint32_t id) {
    const auto expected = make_inputs(id);
    return inputs.selection.enabled == expected.selection.enabled &&
           inputs.selection.layer_start == expected.selection.layer_start &&
           inputs.selection.layer_end == expected.selection.layer_end &&
           inputs.activation_tokens.tokens == expected.activation_tokens.tokens &&
           inputs.activation_tokens.token_count == expected.activation_tokens.token_count &&
           inputs.mtp_embeddings.values == expected.mtp_embeddings.values;
}

static int check_embedding_scale_predicate() {
    if (!skippy_scale_embedding_input_for_test(0, 1)) return 107;
    if (skippy_scale_embedding_input_for_test(1, 1)) return 108;
    if (!skippy_scale_embedding_input_for_test(1, 0)) return 109;
    return 0;
}

static int check_sequential_and_nested_lifetimes() {
    llm_graph_result outer(64);
    auto outer_input = std::make_unique<recording_input>();
    auto * outer_observer = outer_input.get();
    outer.add_input(std::move(outer_input));
    outer.set_inputs(nullptr, make_inputs(7));
    if (!matches(outer.get_build_inputs(), 7) || !matches(outer_observer->observed, 7)) {
        return 1;
    }

    {
        llm_graph_result inner(64);
        inner.set_inputs(nullptr, make_inputs(23));
        if (!matches(inner.get_build_inputs(), 23) || !matches(outer.get_build_inputs(), 7)) {
            return 2;
        }
    }

    outer.set_inputs(nullptr, make_inputs(11));
    if (!matches(outer.get_build_inputs(), 11) || !matches(outer_observer->observed, 11) ||
            outer_observer->set_count != 2 || outer_observer->input_count != 2) {
        return 3;
    }

    outer.reset();
    if (outer.get_build_inputs().selection.enabled || outer.get_build_inputs().activation_tokens.tokens != nullptr) {
        return 4;
    }
    return 0;
}

static int check_prepared_export_token_axis() {
    ggml_tensor tensor = {};
    tensor.ne[0] = 16;
    tensor.ne[1] = 1; // residual checkpoint count
    tensor.ne[2] = 8; // token count
    tensor.ne[3] = 1;
    const int32_t prepared = skippy_activation_token_axis(&tensor, 8, -1);
    if (prepared != 2) return 101;
    tensor.ne[2] = 1;
    if (skippy_activation_token_axis(&tensor, 1, prepared) != 2) return 102;
    tensor.ne[1] = 2;
    tensor.ne[2] = 2;
    if (skippy_activation_token_axis(&tensor, 2, prepared) != 2) return 103;
    tensor.ne[2] = 3;
    try {
        skippy_activation_token_axis(&tensor, 2, prepared);
        return 104;
    } catch (const std::runtime_error &) {}
    tensor.ne[0] = 8;
    tensor.ne[1] = 8;
    tensor.ne[2] = 1;
    bool rejected_missing_axis = false;
    try {
        skippy_admitted_token_axis(&tensor, 8, -1);
    } catch (const std::runtime_error &) {
        rejected_missing_axis = true;
    }
    if (!rejected_missing_axis) return 105;
    if (skippy_admitted_token_axis(&tensor, 8, 1) != 1) return 106;
    return 0;
}

static int check_topology_reuse_guards() {
    if (const int result = check_prepared_export_token_axis()) return result;
    auto base = make_inputs(31);
    skippy_activation_frontier_part_input frontier;
    frontier.values = reinterpret_cast<const void *>(static_cast<uintptr_t>(34));
    frontier.payload_bytes = 35;
    frontier.ggml_type = GGML_TYPE_F32;
    frontier.rank = 2;
    frontier.token_axis = 1;
    frontier.dimensions[0] = 8;
    frontier.dimensions[1] = 1;
    frontier.byte_strides[0] = 4;
    frontier.byte_strides[1] = 32;
    base.activation_frontier.push_back(frontier);
    base.activation_import_bindings.push_back("l_out-3");
    base.activation_export_bindings.push_back("l_out-7");
    auto changed_pointer = base;
    changed_pointer.activation_tokens.tokens = reinterpret_cast<const llama_token *>(static_cast<uintptr_t>(999));
    if (!base.topology_equal(changed_pointer)) {
        return 5;
    }

    auto changed_filter = base;
    changed_filter.selection.layer_end++;
    if (base.topology_equal(changed_filter)) {
        return 6;
    }

    auto removed_frontier = base;
    removed_frontier.activation_frontier.clear();
    if (base.topology_equal(removed_frontier)) {
        return 7;
    }

    auto changed_shape = base;
    changed_shape.mtp_embeddings.n_embd++;
    if (base.topology_equal(changed_shape)) {
        return 8;
    }

    auto changed_frontier_shape = base;
    changed_frontier_shape.activation_frontier[0].dimensions[0]++;
    if (base.topology_equal(changed_frontier_shape)) {
        return 9;
    }

    auto changed_import_binding = base;
    changed_import_binding.activation_import_bindings[0] = "l_out-4";
    if (base.topology_equal(changed_import_binding)) {
        return 10;
    }

    auto changed_export_binding = base;
    changed_export_binding.activation_export_bindings[0] = "l_out-8";
    if (base.topology_equal(changed_export_binding)) {
        return 11;
    }
    return 0;
}

static int check_concurrent_builds() {
    constexpr uint32_t thread_count = 12;
    constexpr uint32_t iterations = 200;
    std::atomic<uint32_t> ready { 0 };
    std::atomic<bool> start { false };
    std::atomic<bool> failed { false };
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (uint32_t thread_id = 0; thread_id < thread_count; ++thread_id) {
        threads.emplace_back([&, thread_id] {
            llm_graph_result result(64);
            auto input = std::make_unique<recording_input>();
            auto * observer = input.get();
            result.add_input(std::move(input));
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
                const uint32_t id = thread_id*iterations + iteration + 1;
                result.set_inputs(nullptr, make_inputs(id));
                if (!matches(result.get_build_inputs(), id) || !matches(observer->observed, id)) {
                    failed.store(true, std::memory_order_release);
                    return;
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != thread_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto & thread : threads) {
        thread.join();
    }
    return failed.load(std::memory_order_acquire) ? 9 : 0;
}

static int check_mtp_request_input_registration() {
    llm_graph_result result(64);
    auto input = std::make_unique<llm_graph_input_embd_h>(4, 4);
    input->tokens = ggml_new_tensor_1d(result.get_ctx(), GGML_TYPE_I32, 2);
    input->embd = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 4, 2);
    input->h = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 4, 2);
    const ggml_tensor * tokens = input->tokens;
    const ggml_tensor * embeddings = input->embd;
    const ggml_tensor * hidden = input->h;
    result.add_input(std::move(input));

    const auto & bindings = result.get_request_input_bindings();
    if (bindings.size() != 3 ||
            bindings[0].tensor != tokens || bindings[0].identity != "tokens.0" ||
            bindings[1].tensor != embeddings || bindings[1].identity != "embeddings.0" ||
            bindings[2].tensor != hidden || bindings[2].identity != "mtp.hidden.0") {
        return 20;
    }
    return 0;
}

static int check_activation_port_binding() {
    llm_graph_result result(64);
    auto * first = ggml_new_tensor_1d(result.get_ctx(), GGML_TYPE_F32, 4);
    auto * second = ggml_new_tensor_1d(result.get_ctx(), GGML_TYPE_F32, 4);
    auto * third = ggml_new_tensor_1d(result.get_ctx(), GGML_TYPE_F32, 4);
    ggml_set_name(first, "stable_second");
    ggml_set_name(second, "stable_first");
    ggml_set_name(third, "local_third");
    result.add_skippy_activation_import(first, 0);
    result.add_skippy_activation_import(second, 0);
    result.add_skippy_activation_import(third, 0);

    llm_graph_params params{};
    params.build_inputs.activation_import_bindings = {
        "stable_first", "stable_second", "stable_third",
    };
    result.set_outputs(params);

    const auto & ports = result.get_skippy_activation_imports();
    if (ports.size() != 3 ||
            ports[0].binding != "stable_first" ||
            ports[1].binding != "stable_second" ||
            ports[2].binding != "stable_third") {
        return 21;
    }
    return 0;
}

static int check_activation_port_alias_is_not_duplicated() {
    llm_graph_result result(64);
    auto * storage = ggml_new_tensor_2d(result.get_ctx(), GGML_TYPE_F32, 4, 2);
    auto * view = ggml_view_2d(result.get_ctx(), storage, 4, 2, storage->nb[1], 0);
    result.add_skippy_activation_export(storage, 1);
    result.t_embd = view;

    llm_graph_params params{};
    params.build_inputs.selection.enabled = true;
    params.build_inputs.activation_export_bindings = {"stable_export"};
    result.set_outputs(params);

    const auto & ports = result.get_skippy_activation_exports();
    if (ports.size() != 1 || ports[0].tensor != storage ||
            ports[0].binding != "stable_export") {
        return 22;
    }
    bool embedding_view_is_scheduled = false;
    for (int32_t index = 0; index < ggml_graph_n_nodes(result.get_gf()); ++index) {
        if (ggml_graph_node(result.get_gf(), index) == view) {
            embedding_view_is_scheduled = true;
            break;
        }
    }
    if (!embedding_view_is_scheduled) {
        return 23;
    }

    return 0;
}

static llama_context_params make_context_params(uint32_t n_seq_max) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 8;
    params.n_ubatch = 8;
    params.n_seq_max = n_seq_max;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return params;
}

static llama_model * load_model(const char * path) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    return llama_model_load_from_file(path, params);
}

static const ggml_tensor * primary_output(const llm_graph_result & result) {
    if (result.get_logits() != nullptr) return result.get_logits();
    if (result.get_embd_pooled() != nullptr) return result.get_embd_pooled();
    return result.get_embd();
}

static bool append_frontier_contract(
        const std::vector<const ggml_tensor *> & tensors,
        std::vector<std::string> & identities,
        std::vector<std::string> & bindings) {
    for (const ggml_tensor * tensor : tensors) {
        const char * name = tensor == nullptr ? nullptr : ggml_get_name(tensor);
        std::string identity;
        if (name == nullptr || name[0] == '\0' ||
                !skippy_build_activation_port_identity(name, identity)) {
            return false;
        }
        identities.push_back(std::move(identity));
        bindings.emplace_back(name);
    }
    return true;
}

static void report_topology_difference(
        const skippy_graph_build_inputs & actual,
        const skippy_graph_build_inputs & expected) {
    std::fprintf(stderr,
            "topology fields: imports=%d import-bindings=%d exports=%d export-bindings=%d "
            "residents=%d frontiers=%zu/%zu selection=%d/%d range=%d..%d/%d..%d "
            "tokens=%d:%u/%d:%u mtp=%d:%u:%u/%d:%u:%u\n",
            actual.activation_import_identities == expected.activation_import_identities,
            actual.activation_import_bindings == expected.activation_import_bindings,
            actual.activation_export_identities == expected.activation_export_identities,
            actual.activation_export_bindings == expected.activation_export_bindings,
            actual.resident_tensor_names == expected.resident_tensor_names,
            actual.activation_frontier.size(), expected.activation_frontier.size(),
            actual.selection.enabled, expected.selection.enabled,
            actual.selection.layer_start, actual.selection.layer_end,
            expected.selection.layer_start, expected.selection.layer_end,
            actual.activation_tokens.tokens != nullptr, actual.activation_tokens.token_count,
            expected.activation_tokens.tokens != nullptr, expected.activation_tokens.token_count,
            actual.mtp_embeddings.values != nullptr, actual.mtp_embeddings.token_count,
            actual.mtp_embeddings.n_embd,
            expected.mtp_embeddings.values != nullptr, expected.mtp_embeddings.token_count,
            expected.mtp_embeddings.n_embd);
}

static bool make_stage_inputs(
        llama_model * model,
        uint32_t n_seq_max,
        int32_t layer_start,
        int32_t layer_end,
        skippy_graph_build_inputs & inputs) {
    const skippy_graph_build_inputs whole_model_inputs;
    std::unique_ptr<llama_context, decltype(&llama_free)> planning(
            llama_init_from_model_with_graph_inputs(
                model, make_context_params(n_seq_max), whole_model_inputs),
            llama_free);
    if (!planning) {
        return false;
    }

    llama_memory_context_ptr memory_context;
    if (llama_memory_t memory = llama_get_memory(planning.get())) {
        memory_context = memory->init_full();
    }
    ggml_cgraph * graph = planning->graph_reserve(
            8, n_seq_max, n_seq_max, memory_context.get(), true, nullptr, whole_model_inputs);
    const llm_graph_result * result = planning->get_gf_res_reserve();
    if (graph == nullptr || result == nullptr) {
        return false;
    }

    std::vector<skippy_parameter_binding> parameters;
    parameters.reserve(model->tensors_by_name.size());
    for (const auto & [name, tensor] : model->tensors_by_name) {
        if (tensor != nullptr) {
            parameters.push_back({tensor, name});
        }
    }

    skippy_stage_slice_selection selection;
    std::string error;
    if (!skippy_select_stage_slice(
                graph,
                result->get_block_boundaries(),
                result->get_state_effects(),
                parameters,
                result->get_request_input_bindings(),
                llama_model_n_layer(model),
                layer_start,
                layer_end,
                selection,
                error,
                primary_output(*result))) {
        std::fprintf(stderr, "cannot admit graph input test stage: %s\n", error.c_str());
        return false;
    }

    inputs.selection.enabled = true;
    inputs.selection.layer_start = layer_start;
    inputs.selection.layer_end = layer_end;
    inputs.resident_tensor_names = selection.parameter_identities;
    std::sort(inputs.resident_tensor_names.begin(), inputs.resident_tensor_names.end());
    inputs.execution_contract = nlohmann::json({{"version", 1},
        {"layer_start", layer_start}, {"layer_end", layer_end},
        {"decoder", inputs.resident_tensor_names}, {"mtp", nlohmann::json::array()}}).dump();
    return append_frontier_contract(selection.activation_imports,
                   inputs.activation_import_identities, inputs.activation_import_bindings) &&
           append_frontier_contract(selection.activation_exports,
                   inputs.activation_export_identities, inputs.activation_export_bindings);
}

static int check_sched_reserve_and_memory_update(const char * model_path) {
    llama_model * model = load_model(model_path);
    if (model == nullptr || llama_model_n_layer(model) < 2) {
        llama_model_free(model);
        return 10;
    }

    skippy_graph_build_inputs startup;
    skippy_graph_build_inputs reserve_inputs;
    skippy_graph_build_inputs update_inputs;
    if (!make_stage_inputs(model, 2, 0, llama_model_n_layer(model), startup) ||
            !make_stage_inputs(model, 2, 0, 1, reserve_inputs) ||
            !make_stage_inputs(model, 2, 1, 2, update_inputs)) {
        llama_model_free(model);
        return 11;
    }
    if (startup.topology_equal(reserve_inputs) || startup.topology_equal(update_inputs)) {
        llama_model_free(model);
        return 11;
    }

    llama_context * context = llama_init_from_model_with_graph_inputs(
            model, make_context_params(2), startup);
    if (context == nullptr) {
        llama_model_free(model);
        return 12;
    }
    if (!context->get_gf_res_reserve()->get_build_inputs().topology_equal(startup)) {
        llama_free(context);
        llama_model_free(model);
        return 13;
    }
    llama_free(context);

    context = llama_init_from_model_with_graph_inputs(
            model, make_context_params(2), reserve_inputs);
    if (context == nullptr) {
        llama_model_free(model);
        return 14;
    }
    const auto * reserve_result = context->get_gf_res_reserve();
    const bool reserve_topology_matches =
            reserve_result->get_build_inputs().topology_equal(reserve_inputs);
    const size_t expected_boundary_count =
            static_cast<size_t>(2*llama_model_n_layer(model));
    if (!reserve_topology_matches ||
            reserve_result->get_block_boundaries().size() != expected_boundary_count) {
        std::fprintf(stderr,
                "reserve contract mismatch: topology=%d boundaries=%zu expected=%zu\n",
                reserve_topology_matches,
                reserve_result->get_block_boundaries().size(),
                expected_boundary_count);
        report_topology_difference(reserve_result->get_build_inputs(), reserve_inputs);
        llama_free(context);
        llama_model_free(model);
        return 15;
    }

    llama_memory_t memory = llama_get_memory(context);
    if (memory == nullptr) {
        llama_free(context);
        llama_model_free(model);
        return 16;
    }
    memory->seq_cp(0, 1, -1, -1);
    if (!context->memory_update(false, reserve_inputs)) {
        llama_free(context);
        llama_model_free(model);
        return 17;
    }
    const auto * update_result = context->get_gf_res_reserve();
    const bool update_topology_matches =
            update_result->get_build_inputs().topology_equal(reserve_inputs);
    if (!update_topology_matches ||
            update_result->get_block_boundaries().size() != expected_boundary_count) {
        std::fprintf(stderr,
                "memory-update contract mismatch: topology=%d boundaries=%zu expected=%zu\n",
                update_topology_matches,
                update_result->get_block_boundaries().size(),
                expected_boundary_count);
        llama_free(context);
        llama_model_free(model);
        return 18;
    }

    context->set_causal_attn(false);
    try {
        context->sched_reserve(reserve_inputs);
    } catch (const std::runtime_error &) {
        llama_free(context);
        llama_model_free(model);
        return 19;
    }
    if (!context->get_gf_res_reserve()->get_build_inputs().topology_equal(reserve_inputs)) {
        llama_free(context);
        llama_model_free(model);
        return 20;
    }

    llama_free(context);
    context = llama_init_from_model_with_graph_inputs(
            model, make_context_params(2), update_inputs);
    if (context == nullptr ||
            !context->get_gf_res_reserve()->get_build_inputs().topology_equal(update_inputs)) {
        llama_free(context);
        llama_model_free(model);
        return 21;
    }
    llama_free(context);
    llama_model_free(model);
    return 0;
}

static const char * fused_op_name(llm_fused_op op) {
    switch (op) {
        case LLM_FUSED_OP_FLASH_ATTN:         return "flash attention";
        case LLM_FUSED_OP_GDN_AR:             return "Gated Delta Net (autoregressive)";
        case LLM_FUSED_OP_GDN_CH:             return "Gated Delta Net (chunked)";
        case LLM_FUSED_OP_LIGHTNING_INDEXER:  return "Lightning Indexer";
        case LLM_FUSED_OP_DSV4_HC_PRE:        return "DeepSeek V4 HC pre";
        case LLM_FUSED_OP_DSV4_HC_COMB:       return "DeepSeek V4 HC comb";
        case LLM_FUSED_OP_DSV4_HC_POST:       return "DeepSeek V4 HC post";
    }
    return "unknown fused op";
}

// A stage reserves only its own layers. Every fused op resolved on auto (flash
// attention and the DeepSeek V4 HC fusions) must resolve the same way it does
// for the whole model instead of tripping on the fused nodes of layers the stage
// does not run, and for every fused op of the whole model the stage must keep
// the nodes of exactly the layers it runs.
static int check_stage_fused_ops_auto(const char * model_path) {
    llama_model * model = load_model(model_path);
    const int32_t n_layer = model == nullptr ? 0 : llama_model_n_layer(model);
    if (n_layer < 2) {
        llama_model_free(model);
        return 40;
    }

    struct auto_fused_op {
        llm_fused_op op;
        bool llama_cparams::* enabled;
    };
    const auto_fused_op auto_ops[] = {
        {LLM_FUSED_OP_FLASH_ATTN, &llama_cparams::flash_attn},
        {LLM_FUSED_OP_DSV4_HC_PRE, &llama_cparams::fused_dsv4_hc_pre},
        {LLM_FUSED_OP_DSV4_HC_COMB, &llama_cparams::fused_dsv4_hc_comb},
        {LLM_FUSED_OP_DSV4_HC_POST, &llama_cparams::fused_dsv4_hc_post},
    };
    // Per (op, layer) node counts, not a set of layers: DeepSeek V4 builds two
    // HC nodes of the same op inside one layer (attention and FFN), so a stage
    // that dropped one of them would still keep the layer in a set comparison.
    using fused_node_counts = std::map<llm_fused_op, std::map<int32_t, size_t>>;

    const auto resolve_auto = [model, &auto_ops](const skippy_graph_build_inputs & inputs,
            std::vector<llm_fused_op> & disabled, fused_node_counts & counts) {
        llama_context_params params = make_context_params(1);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        llama_context * context = llama_init_from_model_with_graph_inputs(model, params, inputs);
        if (context == nullptr) {
            return false;
        }
        for (const auto & auto_op : auto_ops) {
            if (!(context->get_cparams().*auto_op.enabled)) {
                disabled.push_back(auto_op.op);
            }
        }
        for (const auto & node : context->get_gf_res_reserve()->get_fused_nodes()) {
            ++counts[node.op][node.il];
        }
        llama_free(context);
        return true;
    };
    const auto layer_range = [](int32_t layer_start, int32_t layer_end) {
        std::set<int32_t> layers;
        for (int32_t il = layer_start; il < layer_end; ++il) {
            layers.insert(il);
        }
        return layers;
    };
    const auto counted_layers = [](const std::map<int32_t, size_t> & counts) {
        std::set<int32_t> layers;
        for (const auto & entry : counts) {
            layers.insert(entry.first);
        }
        return layers;
    };
    const auto counted_nodes = [](const std::map<int32_t, size_t> & counts) {
        size_t nodes = 0;
        for (const auto & entry : counts) {
            nodes += entry.second;
        }
        return nodes;
    };

    std::vector<llm_fused_op> whole_model_disabled;
    fused_node_counts whole_model_counts;
    if (!resolve_auto(skippy_graph_build_inputs(), whole_model_disabled, whole_model_counts) ||
            !whole_model_disabled.empty() ||
            counted_layers(whole_model_counts[LLM_FUSED_OP_FLASH_ATTN]) != layer_range(0, n_layer)) {
        llama_model_free(model);
        return 41;
    }

    const std::pair<int32_t, int32_t> stages[] = {{0, 1}, {1, n_layer}};
    for (const auto & [layer_start, layer_end] : stages) {
        skippy_graph_build_inputs inputs;
        std::vector<llm_fused_op> disabled;
        fused_node_counts counts;
        if (!make_stage_inputs(model, 1, layer_start, layer_end, inputs) ||
                !resolve_auto(inputs, disabled, counts)) {
            llama_model_free(model);
            return 42;
        }
        if (!disabled.empty()) {
            for (const llm_fused_op op : disabled) {
                std::fprintf(stderr, "stage %d..%d resolved auto %s to disabled\n",
                        layer_start, layer_end, fused_op_name(op));
            }
            llama_model_free(model);
            return 43;
        }
        // The whole-model counts restricted to this stage's layers are the exact
        // baseline: the same ops, the same layers, and the same number of nodes
        // per (op, layer).
        fused_node_counts expected;
        for (const auto & [op, op_counts] : whole_model_counts) {
            for (const auto & [il, node_count] : op_counts) {
                if (il >= layer_start && il < layer_end) {
                    expected[op][il] = node_count;
                }
            }
        }
        // A stage is a slice of the whole-model graph, so it cannot introduce an
        // op the whole model lacks; report one if that ever changes.
        for (const auto & [op, op_counts] : counts) {
            if (expected.count(op) == 0) {
                std::fprintf(stderr, "stage %d..%d kept unexpected %s nodes (%zu)\n",
                        layer_start, layer_end, fused_op_name(op), counted_nodes(op_counts));
                llama_model_free(model);
                return 45;
            }
        }
        const std::map<int32_t, size_t> no_nodes;
        for (const auto & [op, expected_counts] : expected) {
            const auto stage_it = counts.find(op);
            const std::map<int32_t, size_t> & stage_counts =
                    stage_it == counts.end() ? no_nodes : stage_it->second;
            if (stage_counts == expected_counts) {
                continue;
            }
            const std::set<int32_t> stage_layers = counted_layers(stage_counts);
            const std::set<int32_t> expected_layers = counted_layers(expected_counts);
            if (stage_layers != expected_layers) {
                std::fprintf(stderr, "stage %d..%d kept %s nodes for %zu layers, expected %zu\n",
                        layer_start, layer_end, fused_op_name(op), stage_layers.size(), expected_layers.size());
            } else {
                std::fprintf(stderr, "stage %d..%d kept %zu %s nodes, expected %zu\n",
                        layer_start, layer_end, counted_nodes(stage_counts), fused_op_name(op),
                        counted_nodes(expected_counts));
            }
            llama_model_free(model);
            return 44;
        }
    }

    llama_model_free(model);
    return 0;
}

static int check_decode_encode_fallback(const char * model_path) {
    llama_model * model = load_model(model_path);
    if (model == nullptr) {
        return 17;
    }
    skippy_graph_build_inputs per_call;
    if (!make_stage_inputs(model, 1, 0, llama_model_n_layer(model), per_call)) {
        llama_model_free(model);
        return 18;
    }
    llama_context * context = llama_init_from_model_with_graph_inputs(
            model, make_context_params(1), per_call);
    if (context == nullptr || llama_get_memory(context) != nullptr) {
        llama_free(context);
        llama_model_free(model);
        return 18;
    }

    llama_token token = 1;
    llama_batch batch = llama_batch_get_one(&token, 1);
    const int rc = context->decode(batch, per_call);
    if (rc != 0 ||
            !context->get_gf_res_reserve()->get_build_inputs().topology_equal(per_call) ||
            !context->get_gf_res_prev_active()->get_build_inputs().topology_equal(per_call)) {
        llama_free(context);
        llama_model_free(model);
        return 19;
    }

    llama_free(context);
    llama_model_free(model);
    return 0;
}

// No model files, backend, allocation or graph execution required.
static int check_decode_input_width() {
    llama_hparams hp = {};
    hp.n_embd = 256;
    hp.n_deepstack_layers = 1;
    hp.n_embd_out_impl = 1024;
    hp.n_embd_inp_enc_impl = 1536;
    struct width_case {
        const char * name;
        llm_arch arch;
        llama_context_type context;
        bool embeddings;
        int64_t expected;
    };
    const width_case cases[] = {
        {"ordinary projector", LLM_ARCH_QWEN3VL, LLAMA_CONTEXT_TYPE_DEFAULT, true, 512},
        {"token batch", LLM_ARCH_QWEN3VL, LLAMA_CONTEXT_TYPE_DEFAULT, false, 512},
        {"MTP hidden", LLM_ARCH_DEEPSEEK4, LLAMA_CONTEXT_TYPE_MTP, true, 1024},
        {"MTP token batch", LLM_ARCH_DEEPSEEK4, LLAMA_CONTEXT_TYPE_MTP, false, 512},
        {"DFlash encoder", LLM_ARCH_DFLASH, LLAMA_CONTEXT_TYPE_DEFAULT, true, 1536},
        {"DFlash token batch", LLM_ARCH_DFLASH, LLAMA_CONTEXT_TYPE_DEFAULT, false, 512},
    };
    for (const auto & test : cases) {
        const auto actual = llama_decode_input_width(hp, test.arch, test.context, test.embeddings);
        if (actual != test.expected) {
            fprintf(stderr, "decode width contract failed: %s\n", test.name);
            return 20;
        }
    }
    // Equal-width models retain the ordinary fallback in all relevant modes.
    hp.n_deepstack_layers = 0;
    hp.n_embd_out_impl = hp.n_embd_inp_enc_impl = 0;
    for (const auto & test : cases) {
        if (llama_decode_input_width(hp, test.arch, test.context, test.embeddings) != 256) {
            return 21;
        }
    }
    return 0;
}

static int check_embedding_execution_profiles() {
    skippy_graph_build_inputs inputs;
    inputs.selection.enabled      = true;
    inputs.selection.layer_end    = 2;
    inputs.resident_tensor_names  = { "block", "image_projection", "token_embedding" };
    const nlohmann::json contract = {
        { "version",            1                               },
        { "layer_start",        0                               },
        { "layer_end",          2                               },
        { "decoder",            { "block", "token_embedding" }  },
        { "decoder_embeddings", { "block", "image_projection" } },
        { "mtp",                nlohmann::json::array()         }
    };
    inputs.execution_contract = contract.dump();
    skippy_validate_execution_dependencies(inputs, false, 0, { "block", "token_embedding" });
    skippy_validate_execution_dependencies(inputs, false, 0, { "block", "image_projection" }, true);
    const auto rejects = [&](const skippy_graph_build_inputs & build, const std::vector<std::string> & parameters,
                             bool embeddings) {
        try {
            skippy_validate_execution_dependencies(build, false, 0, parameters, embeddings);
        } catch (const std::exception &) {
            return true;
        }
        return false;
    };
    // Neither input may silently use the other profile, omit an admitted weight,
    // or introduce a weight outside its own exact dependency set.
    if (!rejects(inputs, { "block", "image_projection" }, false) ||
        !rejects(inputs, { "block", "token_embedding" }, true) || !rejects(inputs, { "block" }, true) ||
        !rejects(inputs, { "block", "image_projection", "token_embedding" }, true)) {
        return 110;
    }
    auto invalid                  = inputs;
    invalid.resident_tensor_names = { "block", "token_embedding" };
    if (!rejects(invalid, { "block", "image_projection" }, true)) {
        return 111;
    }
    auto legacy = contract;
    legacy.erase("decoder_embeddings");
    invalid.execution_contract = legacy.dump();
    if (!rejects(invalid, { "block", "token_embedding" }, true)) {
        return 112;
    }
    // Tied input/output embeddings remain required when the image graph uses
    // the same tensor for its output projection.
    auto tied                  = contract;
    tied["decoder_embeddings"] = tied["decoder"];
    invalid.execution_contract = tied.dump();
    skippy_validate_execution_dependencies(invalid, false, 0, invalid.resident_tensor_names, true);
    if (!rejects(invalid, { "block" }, true)) {
        return 113;
    }
    return 0;
}

static int check_execution_profile_residency() {
    skippy_graph_build_inputs inputs;
    inputs.selection.enabled = true;
    inputs.selection.layer_start = 0;
    inputs.selection.layer_end = 2;
    inputs.resident_tensor_names = {"decoder", "mtp.0", "mtp.1", "shared"};
    const nlohmann::json contract = {{"version", 1}, {"layer_start", 0}, {"layer_end", 2},
        {"decoder", {"decoder", "shared"}},
        {"mtp", {{{"offset", 0}, {"parameters", {"mtp.0", "shared"}}},
                 {{"offset", 1}, {"parameters", {"mtp.1", "shared"}}}}}};
    inputs.execution_contract = contract.dump();
    skippy_validate_execution_dependencies(inputs, false, 0, {"decoder", "shared"});
    skippy_validate_execution_dependencies(inputs, true, 0, {"mtp.0", "shared"});
    skippy_validate_execution_dependencies(inputs, true, 1, {"mtp.1", "shared"});
    const auto rejects = [](const skippy_graph_build_inputs & build, bool auxiliary,
            int32_t offset, const std::vector<std::string> & parameters) {
        try { skippy_validate_execution_dependencies(build, auxiliary, offset, parameters); }
        catch (const std::exception &) { return true; }
        return false;
    };
    if (!rejects(inputs, false, 0, {"mtp.0", "shared"}) ||
            !rejects(inputs, true, 0, {"mtp.1", "shared"}) ||
            !rejects(inputs, true, 2, {"mtp.1", "shared"}) ||
            !rejects(inputs, false, 0, {"decoder"})) return 30;
    auto invalid = inputs;
    invalid.resident_tensor_names.pop_back();
    if (!rejects(invalid, false, 0, {"decoder", "shared"})) return 31;
    invalid = inputs;
    invalid.resident_tensor_names.push_back("unadmitted");
    if (!rejects(invalid, false, 0, {"decoder", "shared"})) return 32;
    invalid = inputs;
    invalid.selection.layer_start = 1;
    if (!rejects(invalid, false, 0, {"decoder", "shared"})) return 33;
    invalid = inputs;
    invalid.execution_contract.clear();
    if (!rejects(invalid, false, 0, {"decoder", "shared"})) return 34;
    if (inputs.topology_equal(invalid)) return 35;
    invalid.resident_tensor_names = {"decoder", "shared"};
    if (!rejects(invalid, false, 0, invalid.resident_tensor_names)) return 37;
    auto single_profile = contract;
    single_profile["mtp"] = nlohmann::json::array();
    invalid.execution_contract = single_profile.dump();
    skippy_validate_execution_dependencies(invalid, false, 0, invalid.resident_tensor_names);
    invalid = inputs;
    invalid.execution_contract = "{}";
    if (!rejects(invalid, false, 0, {"decoder", "shared"})) return 36;
    return 0;
}

int main(int argc, char ** argv) {
    if (const int rc = check_embedding_scale_predicate()) return rc;
    if (const int rc = check_prepared_export_token_axis()) return rc;
    if (const int rc = check_execution_profile_residency()) return rc;
    if (const int rc = check_embedding_execution_profiles()) {
        return rc;
    }
    if (argc == 2 && std::strcmp(argv[1], "--input-width") == 0) {
        return check_decode_input_width();
    }
    if (argc == 3 && std::strcmp(argv[1], "--stage-fused-ops") == 0) {
        llama_backend_init();
        const int rc = check_stage_fused_ops_auto(argv[2]);
        llama_backend_free();
        return rc;
    }
    if (argc != 3) {
        return 64;
    }
    if (const int rc = check_sequential_and_nested_lifetimes()) {
        return rc;
    }
    if (const int rc = check_topology_reuse_guards()) {
        return rc;
    }
    if (const int rc = check_concurrent_builds()) {
        return rc;
    }
    if (const int rc = check_mtp_request_input_registration()) {
        return rc;
    }
    if (const int rc = check_activation_port_binding()) {
        return rc;
    }
    if (const int rc = check_activation_port_alias_is_not_duplicated()) {
        return rc;
    }
    llama_backend_init();
    int rc = 0;
    if (rc == 0) rc = check_sched_reserve_and_memory_update(argv[1]);
    if (rc == 0) rc = check_stage_fused_ops_auto(argv[1]);
    if (rc == 0) {
        rc = check_decode_encode_fallback(argv[2]);
    }
    llama_backend_free();
    return rc;
}
