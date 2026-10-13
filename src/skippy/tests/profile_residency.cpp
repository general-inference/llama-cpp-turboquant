#include "skippy.h"
#include "skippy/runtime_types.h"
#include "skippy/execution_internal.h"
#include "skippy/graph_build_inputs.h"
#include "skippy/speculative_internal.h"
#include "llama-context.h"
#include "live_graph_reference.h"
#include "llama-model.h"
#include "llama-memory.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::string read_string(skippy_stage_plan * plan, skippy_stage_plan_string_ref_v1 ref) {
    const char * value = nullptr;
    size_t size = 0;
    skippy_error * error = nullptr;
    if (skippy_stage_plan_string_v1(plan, ref, &value, &size, &error) != SKIPPY_STATUS_OK) {
        skippy_error_free(error);
        return {};
    }
    return std::string(value, size);
}

struct mtp_receipt {
    live_graph_snapshot target, draft;
    std::vector<llama_token> tokens;
};

static int run(const char * path, bool integrated, mtp_receipt * receipt = nullptr,
        bool force_unfused_gdn = false, bool ordinary_reference = false,
        bool cross_sequence_geometry = false) {
    fresh_graph_scope fresh(ordinary_reference);
    ggml_context * tensor_context = nullptr;
    const gguf_init_params inspect = {true, &tensor_context};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_from_file(path, inspect), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors_owner(tensor_context, ggml_free);
    if (!metadata || !tensor_context) return 1;
    std::vector<skippy_stage_planner_tensor_v1> tensors;
    for (int64_t index = 0; index < gguf_get_n_tensors(metadata.get()); ++index) {
        const char * name = gguf_get_tensor_name(metadata.get(), index);
        const auto * tensor = ggml_get_tensor(tensor_context, name);
        skippy_stage_planner_tensor_v1 descriptor{};
        descriptor.abi_version = 1;
        descriptor.struct_size = sizeof(descriptor);
        descriptor.tensor_id = name;
        descriptor.native_name = name;
        descriptor.ggml_type = tensor->type;
        descriptor.dimension_count = ggml_n_dims(tensor);
        std::copy(tensor->ne, tensor->ne + descriptor.dimension_count, descriptor.dimensions);
        descriptor.data_offset = gguf_get_data_offset(metadata.get()) + gguf_get_tensor_offset(metadata.get(), index);
        descriptor.stored_length = ggml_nbytes(tensor);
        tensors.push_back(descriptor);
    }
    std::sort(tensors.begin(), tensors.end(), [](const auto & left, const auto & right) {
        return std::strcmp(left.tensor_id, right.tensor_id) < 0;
    });
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.use_extra_bufts = false;
    mp.load_mtp = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path, mp), llama_model_free);
    if (!model || (integrated && llama_model_n_layer_nextn(model.get()) == 0)) return 2;
    const int32_t layer_count = llama_model_n_layer(model.get());
    model.reset();
    const uint32_t prefill_capacity = force_unfused_gdn ? 34 : 8;
    const skippy_stage_planner_profile_v1 profiles[] = {
        {1, sizeof(skippy_stage_planner_profile_v1), "decode", 1, 1, 1, 0},
        {1, sizeof(skippy_stage_planner_profile_v1), "prefill", prefill_capacity, 1, prefill_capacity, 0},
    };
    const std::string package_id = "sha256:" + std::string(64, 'a');
    const skippy_stage_planner_config_v1 pc = {1, sizeof(skippy_stage_planner_config_v1),
        package_id.c_str(), &path, 1, tensors.data(), tensors.size(), profiles, 2, "test", "cpu"};
    skippy_error * error = nullptr;
    skippy_stage_planner * planner_raw = nullptr;
    if (skippy_stage_planner_create_v1(&pc, &planner_raw, &error) != SKIPPY_STATUS_OK) {
        fprintf(stderr, "planner: %s\n", error ? error->message : "unknown"); return 3;
    }
    std::unique_ptr<skippy_stage_planner, decltype(&skippy_stage_planner_free)> planner(planner_raw, skippy_stage_planner_free);
    skippy_stage_plan * plan_raw = nullptr;
    if (skippy_stage_planner_realize_v1(planner.get(), 0, layer_count, &plan_raw, &error) != SKIPPY_STATUS_OK) {
        fprintf(stderr, "plan: %s\n", error ? error->message : "unknown"); return 4;
    }
    std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> plan(plan_raw, skippy_stage_plan_free);
    skippy_stage_plan_desc_v1 descriptor{};
    if (skippy_stage_plan_describe_v1(plan.get(), &descriptor, &error) != SKIPPY_STATUS_OK) return 5;
    std::string contract = read_string(plan.get(), descriptor.execution_contract);
    std::vector<std::string> residents;
    for (size_t index = 0; index < descriptor.resident_tensor_count; ++index) {
        skippy_stage_plan_value_desc_v1 resident{};
        if (skippy_stage_plan_resident_tensor_at_v1(plan.get(), index, &resident, &error) != SKIPPY_STATUS_OK) return 6;
        residents.push_back(read_string(plan.get(), resident.identity));
    }
    std::vector<const char *> names;
    for (const auto & name : residents) names.push_back(name.c_str());
    // The runtime must own the contract after planner destruction.
    plan.reset(); planner.reset();
    skippy_runtime_config config = skippy_runtime_config_default();
    config.layer_start = 0;
    config.layer_end = layer_count;
    config.ctx_size = 32;
    config.n_batch = prefill_capacity;
    config.n_ubatch = prefill_capacity;
    config.n_gpu_layers = 0;
    config.n_threads = config.n_threads_batch = 1;
    config.flash_attn_type = 0;
    config.op_offload = SKIPPY_TRISTATE_FALSE;
    config.disable_repack = true;
    config.mtp_source = integrated ? SKIPPY_MTP_SOURCE_INTEGRATED : SKIPPY_MTP_SOURCE_DISABLED;
    config.resident_tensor_names = names.data();
    config.resident_tensor_name_count = names.size();
    skippy_model * runtime = nullptr;
    for (const char * missing : {static_cast<const char *>(nullptr), ""}) {
        config.execution_contract = missing;
        if (skippy_model_open(path, &config, &runtime, &error) != SKIPPY_STATUS_INVALID_ARGUMENT ||
                runtime != nullptr || error == nullptr ||
                std::strstr(error->message, "execution contract") == nullptr) return 14;
        skippy_error_free(error);
        error = nullptr;
    }
    config.execution_contract = contract.c_str();
    if (skippy_model_open(path, &config, &runtime, &error) != SKIPPY_STATUS_OK) {
        fprintf(stderr, "runtime: %s\n", error ? error->message : "unknown"); return 7;
    }
    contract.clear();
    if (runtime->ctx == nullptr || (integrated && runtime->mtp_ctx == nullptr) ||
            (!integrated && runtime->mtp_ctx != nullptr)) return 8;
    for (const auto & name : residents) {
        const auto found = std::find_if(runtime->model->tensors_by_name.begin(), runtime->model->tensors_by_name.end(),
                [&](const auto & entry) { return entry.first == name; });
        if (found == runtime->model->tensors_by_name.end() || found->second == nullptr ||
                (found->second->data == nullptr && found->second->buffer == nullptr)) return 13;
    }
    if (force_unfused_gdn) runtime->ctx->set_fused_gdn_for_test(false, false);
    skippy_session * session = nullptr;
    if (skippy_session_create(runtime, &session, &error) != SKIPPY_STATUS_OK) return 9;
    const size_t mtp_depth = integrated ? llama_model_n_layer_nextn(llama_get_model(runtime->mtp_ctx)) : 0;
    if (integrated) {
        for (const int32_t invalid : {-1, static_cast<int32_t>(mtp_depth)}) {
            try {
                runtime->mtp_ctx->set_nextn_layer_offset(invalid);
                return 27;
            } catch (const std::runtime_error & exception) {
                if (std::strstr(exception.what(), "outside the declared head count") == nullptr) return 27;
            }
            if (runtime->mtp_ctx->get_cparams().nextn_layer_offset != 0) return 27;
        }
    }
    if (receipt) {
        llama_memory_clear(llama_get_memory(runtime->ctx), true);
        if (integrated) llama_memory_clear(llama_get_memory(runtime->mtp_ctx), true);
    }
    std::vector<llama_token> execution_tokens = force_unfused_gdn
            ? std::vector<llama_token>{3, 7, 5, 9, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26}
            : std::vector<llama_token>{3, 7};
    llama_batch batch = llama_batch_init(execution_tokens.size(), 0, 1);
    batch.n_tokens = execution_tokens.size();
    for (size_t row = 0; row < execution_tokens.size(); ++row) {
        batch.token[row] = execution_tokens[row];
        batch.pos[row] = row;
        batch.n_seq_id[row] = 1;
        batch.seq_id[row][0] = session->seq_id;
        batch.logits[row] = 1;
    }
    const auto rc = skippy_decode_batch(session, batch, execution_tokens.size(), -1, &error, nullptr);
    llama_batch_free(batch);
    if (rc != 0) return 10;
    if (!integrated) {
        if (force_unfused_gdn) {
            llama_batch step = llama_batch_init(1, 0, 1);
            step.n_tokens = 1;
            step.token[0] = 11;
            step.pos[0] = execution_tokens.size();
            step.n_seq_id[0] = 1;
            step.seq_id[0][0] = session->seq_id;
            step.logits[0] = 1;
            const auto step_rc = skippy_decode_batch(session, step, 1, -1, &error, nullptr);
            llama_batch_free(step);
            if (step_rc != 0) return 23;
        }
        if (receipt && (force_unfused_gdn || cross_sequence_geometry) &&
                !live_graph_reference::snapshot(
                    runtime->ctx, session->seq_id, receipt->target)) return 25;
        skippy_session_free(session, &error);
        skippy_model_free(runtime, &error);
        skippy_error_free(error);
        return 0;
    }
    if (skippy_mtp_sync_target_inputs(session, execution_tokens.data(), nullptr,
                execution_tokens.size(), 0, &error) != SKIPPY_STATUS_OK) {
        fprintf(stderr, "MTP sync: %s\n", error ? error->message : "unknown"); return 11;
    }
    session->n_past = execution_tokens.size();
    const size_t draft_count = std::max<size_t>(2, std::min<size_t>(3, mtp_depth));
    skippy_native_mtp_draft draft{};
    if (skippy_mtp_propose_next(session, 31, draft_count, &draft, &error) != SKIPPY_STATUS_OK ||
            draft.token_count != static_cast<int32_t>(draft_count)) {
        fprintf(stderr, "MTP draft: %s\n", error ? error->message : "unknown"); return 12;
    }
    if (receipt) {
        if (!live_graph_reference::snapshot(runtime->ctx, session->seq_id, receipt->target) ||
                !live_graph_reference::snapshot(runtime->mtp_ctx, session->seq_id, receipt->draft)) return 18;
        receipt->tokens.assign(draft.token_ids, draft.token_ids + draft.token_count);
    }
    skippy_session_free(session, &error);
    skippy_model_free(runtime, &error);
    skippy_error_free(error);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) return 64;
    llama_backend_init();
    const bool force_unfused_gdn = argc == 3 && std::string(argv[2]) == "unfused-gdn";
    const bool cross_sequence_geometry = argc == 3 && std::string(argv[2]) == "cross-sequence-geometry";
    if (cross_sequence_geometry) {
        mtp_receipt actual;
        int result = run(argv[1], false, &actual, false, false, true);
        if (result == 0) {
            mtp_receipt reference;
            result = run(argv[1], false, &reference, false, true, true);
            if (result == 0 && !live_graph_reference::compare_snapshots(actual.target, reference.target)) result = 26;
        }
        llama_backend_free();
        return result;
    }
    if (force_unfused_gdn) {
        mtp_receipt actual;
        int result = run(argv[1], false, &actual, true);
        if (result == 0) {
            mtp_receipt reference;
            result = run(argv[1], false, &reference, true, true);
            if (result == 0 && !live_graph_reference::compare_snapshots(actual.target, reference.target)) result = 26;
        }
        llama_backend_free();
        return result;
    }
    int result = run(argv[1], false);
    mtp_receipt actual;
    if (result == 0) result = run(argv[1], true, &actual);
    if (result == 0) {
        mtp_receipt reference;
        result = run(argv[1], true, &reference, false, true);
        if (result == 0 && (actual.tokens != reference.tokens ||
                !live_graph_reference::compare_snapshots(actual.target, reference.target) ||
                !live_graph_reference::compare_snapshots(actual.draft, reference.draft))) result = 19;
    }
    llama_backend_free();
    return result;
}
