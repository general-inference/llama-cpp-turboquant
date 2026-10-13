#include "skippy.h"
#include "skippy/errors.h"
#include "skippy/environment.h"
#include "skippy/glm_dsa.h"
#include "skippy/graph_build_inputs.h"
#include "skippy-verify-checkpoint.h"
#include "skippy/devices.h"

#include <mutex>
#include "gguf.h"
#include "llama-arch.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "model_load_config_internal.h"
#include "skippy/graph_identity.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdio>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

#include "skippy/execution_internal.h"
#include "skippy/activation_internal.h"
#include "skippy/sampling_internal.h"
#include "skippy/speculative_internal.h"
#include "skippy/model_loading_internal.h"
#include "skippy/model_load_config_internal.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"

static uint64_t skippy_runtime_event_timestamp_mono_ns() {
    const int64_t now_us = ggml_time_us();
    return now_us > 0 ? static_cast<uint64_t>(now_us) * UINT64_C(1000) : 0;
}

static uint32_t skippy_runtime_event_failure_code_from_status(enum skippy_status status) {
    switch (status) {
        case SKIPPY_STATUS_OK:
            return SKIPPY_RUNTIME_EVENT_FAILURE_NONE;
        case SKIPPY_STATUS_INVALID_ARGUMENT:
            return SKIPPY_RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT;
        case SKIPPY_STATUS_IO_ERROR:
            return SKIPPY_RUNTIME_EVENT_FAILURE_IO_ERROR;
        case SKIPPY_STATUS_MODEL_ERROR:
            return SKIPPY_RUNTIME_EVENT_FAILURE_MODEL_ERROR;
        case SKIPPY_STATUS_RUNTIME_ERROR:
        case SKIPPY_STATUS_UNSUPPORTED:
            return SKIPPY_RUNTIME_EVENT_FAILURE_RUNTIME_ERROR;
        case SKIPPY_STATUS_ERROR:
        case SKIPPY_STATUS_BUFFER_TOO_SMALL:
            return SKIPPY_RUNTIME_EVENT_FAILURE_INTERNAL_ERROR;
    }

    return SKIPPY_RUNTIME_EVENT_FAILURE_INTERNAL_ERROR;
}

struct skippy_runtime_event_delivery {
    skippy_runtime_event_v1 event = {};
    skippy_runtime_event_callback callback = nullptr;
    void * user_data = nullptr;

    void dispatch() const {
        callback(&event, user_data);
    }
};

struct skippy_runtime_event_scope {
    mutable std::mutex state_mutex;
    skippy_runtime_event_reporter_v1 reporter = {};
    uint64_t sequence = 0;
    uint64_t stage_id = 0;
    uint64_t model_id = 0;
    uint64_t last_progress_step = 0;
    bool active = false;
    bool backend_emitted = false;

    skippy_runtime_event_scope(
            const skippy_runtime_config * config,
            const skippy_runtime_event_reporter_v1 * reporter_in) {
        if (config != nullptr && config->stage_index >= 0) {
            stage_id = static_cast<uint64_t>(config->stage_index);
        }

        if (reporter_in == nullptr || reporter_in->callback == nullptr) {
            return;
        }

        reporter = *reporter_in;
        active = true;
    }

    ~skippy_runtime_event_scope() {
        clear();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(state_mutex);
        active = false;
        reporter = {};
        sequence = 0;
        stage_id = 0;
        model_id = 0;
        last_progress_step = 0;
        backend_emitted = false;
    }

    void set_model_id(uint64_t value) {
        std::lock_guard<std::mutex> lock(state_mutex);
        model_id = value;
    }

    bool is_enabled() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        return is_enabled_locked();
    }

    void emit(
            uint32_t category,
            uint32_t kind,
            uint32_t emitter,
            uint64_t progress_current,
            uint64_t progress_total,
            uint32_t progress_unit,
            uint32_t failure_code,
            enum skippy_status status,
            const char * detail_ptr,
            uint64_t detail_len) {
        skippy_runtime_event_delivery delivery;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!is_enabled_locked()) {
                return;
            }
            build_event_locked(
                    delivery,
                    category,
                    kind,
                    emitter,
                    progress_current,
                    progress_total,
                    progress_unit,
                    failure_code,
                    status,
                    detail_ptr,
                    detail_len);
        }
        delivery.dispatch();
    }

    void emit_detail(
            uint32_t category,
            uint32_t kind,
            uint32_t emitter,
            enum skippy_status status,
            const char * detail_ptr) {
        emit(
                category,
                kind,
                emitter,
                0,
                0,
                SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                status,
                detail_ptr,
                detail_ptr != nullptr ? static_cast<uint64_t>(std::strlen(detail_ptr)) : 0);
    }

    void emit_started(const char * detail_ptr) {
        emit_detail(
                SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED,
                SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                SKIPPY_STATUS_OK,
                detail_ptr);
    }

    void emit_progress(float progress) {
        progress = std::max(0.0f, std::min(progress, 1.0f));
        const uint64_t scaled = static_cast<uint64_t>(std::llround(progress * 1000.0f));
        skippy_runtime_event_delivery delivery;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!is_enabled_locked() || scaled <= last_progress_step) {
                return;
            }
            last_progress_step = scaled;
            build_event_locked(
                    delivery,
                    SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS,
                    SKIPPY_RUNTIME_EVENT_EMITTER_WORKER_THREAD,
                    scaled,
                    1000,
                    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_STEPS,
                    SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                    SKIPPY_STATUS_OK,
                    nullptr,
                    0);
        }
        delivery.dispatch();
    }

    void emit_backend_selected(const char * detail_ptr) {
        skippy_runtime_event_delivery delivery;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!is_enabled_locked() || backend_emitted || detail_ptr == nullptr || detail_ptr[0] == '\0') {
                return;
            }
            backend_emitted = true;
            build_event_locked(
                    delivery,
                    SKIPPY_RUNTIME_EVENT_CATEGORY_BACKEND,
                    SKIPPY_RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED,
                    SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                    0,
                    0,
                    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                    SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                    SKIPPY_STATUS_OK,
                    detail_ptr,
                    static_cast<uint64_t>(std::strlen(detail_ptr)));
        }
        delivery.dispatch();
    }

    void emit_finished() {
        emit(
                SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED,
                SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                0,
                0,
                SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                SKIPPY_STATUS_OK,
                nullptr,
                0);
    }

    void emit_model_load_v2(
            uint32_t kind,
            uint64_t numeric_summary_0,
            uint64_t numeric_summary_1,
            const char * detail_ptr,
            uint64_t detail_len) {
        skippy_runtime_event_delivery delivery;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (!is_enabled_locked()) {
                return;
            }
            build_event_locked(
                    delivery,
                    SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                    kind,
                    SKIPPY_RUNTIME_EVENT_EMITTER_WORKER_THREAD,
                    0,
                    0,
                    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                    SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
                    SKIPPY_STATUS_OK,
                    detail_ptr,
                    detail_len);
            delivery.event.numeric_summary_0 = numeric_summary_0;
            delivery.event.numeric_summary_1 = numeric_summary_1;
        }
        delivery.dispatch();
    }

    void emit_failure(enum skippy_status status, const char * detail_ptr) {
        emit(
                SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED,
                SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD,
                0,
                0,
                SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE,
                skippy_runtime_event_failure_code_from_status(status),
                status,
                detail_ptr,
                detail_ptr != nullptr ? static_cast<uint64_t>(std::strlen(detail_ptr)) : 0);
        const auto diagnostic_kind =
                status == SKIPPY_STATUS_RUNTIME_ERROR || status == SKIPPY_STATUS_ERROR
                ? SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE
                : status == SKIPPY_STATUS_MODEL_ERROR || status == SKIPPY_STATUS_IO_ERROR
                ? SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE
                : SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED;
        const char * diagnostic_detail =
                diagnostic_kind == SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE
                ? "model load runtime failure"
                : diagnostic_kind == SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE
                ? "model load failed"
                : "model load request rejected";
        skippy_emit_diagnostic_event(
                diagnostic_kind,
                static_cast<skippy_runtime_event_failure_code>(
                        skippy_runtime_event_failure_code_from_status(status)),
                diagnostic_detail,
                static_cast<uint64_t>(std::strlen(diagnostic_detail)));
    }

private:

    bool is_enabled_locked() const {
        return active && reporter.callback != nullptr;
    }

    void build_event_locked(
            skippy_runtime_event_delivery & delivery,
            uint32_t category,
            uint32_t kind,
            uint32_t emitter,
            uint64_t progress_current,
            uint64_t progress_total,
            uint32_t progress_unit,
            uint32_t failure_code,
            enum skippy_status status,
            const char * detail_ptr,
            uint64_t detail_len) {
        delivery.event = {
            SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION,
            static_cast<uint32_t>(sizeof(skippy_runtime_event_v1)),
            category,
            kind,
            emitter,
            0,
            ++sequence,
            skippy_runtime_event_timestamp_mono_ns(),
            model_id,
            stage_id,
            0,
            progress_current,
            progress_total,
            progress_unit,
            failure_code,
            static_cast<int32_t>(status),
            0,
            detail_ptr,
            detail_len,
            0,
            0,
            0,
            0,
        };
        delivery.callback = reporter.callback;
        delivery.user_data = reporter.user_data;
    }
};

static void skippy_emit_model_load_observation(
        skippy_runtime_event_scope * event_scope,
        uint32_t kind,
        uint64_t model_id,
        uint64_t numeric_summary_0,
        uint64_t numeric_summary_1,
        const char * detail_ptr,
        uint64_t detail_len) {
    if (event_scope != nullptr && event_scope->is_enabled()) {
        event_scope->set_model_id(model_id);
        event_scope->emit_model_load_v2(
                kind, numeric_summary_0, numeric_summary_1, detail_ptr, detail_len);
        return;
    }
    skippy_emit_model_load_event_v2(
            kind, model_id, numeric_summary_0, numeric_summary_1, detail_ptr, detail_len);
}

static enum skippy_status skippy_validate_runtime_event_reporter(
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_error ** out_error) {
    if (reporter == nullptr) {
        return SKIPPY_STATUS_OK;
    }
    if (reporter->struct_size < sizeof(struct skippy_runtime_event_reporter_v1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "runtime event reporter struct is too small");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (reporter->callback == nullptr) {
        return SKIPPY_STATUS_OK;
    }
    if (reporter->abi_version != SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "unsupported runtime event reporter ABI version");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    return SKIPPY_STATUS_OK;
}

static bool skippy_model_open_progress_callback(float progress, void * user_data) {
    skippy_runtime_event_scope * scope = static_cast<skippy_runtime_event_scope *>(user_data);
    if (scope != nullptr) {
        scope->emit_progress(progress);
    }
    return true;
}

static bool skippy_is_full_model_config(const struct skippy_runtime_config * config) {
    if (config == nullptr) {
        return true;
    }

    return !skippy_runtime_has_stage_plan(config) && config->layer_start == 0;
}

static const char * skippy_validate_stage_plan_config(const struct skippy_runtime_config * config) {
    if (!skippy_runtime_has_stage_plan(config)) {
        return nullptr;
    }
    if (config->resident_tensor_name_count == 0 || config->resident_tensor_names == nullptr) {
        return "filtered stage loading requires explicit resident tensor names";
    }
    if (config->execution_contract == nullptr || config->execution_contract[0] == '\0') {
        return "filtered stage loading requires an admitted execution contract";
    }
    for (size_t i = 0; i < config->resident_tensor_name_count; ++i) {
        const char * name = config->resident_tensor_names[i];
        if (name == nullptr || name[0] == '\0') {
            return "resident tensor names must be non-empty";
        }
        for (size_t j = 0; j < i; ++j) {
            if (std::strcmp(name, config->resident_tensor_names[j]) == 0) {
                return "resident tensor names must be unique";
            }
        }
    }
    const auto validate_frontier = [](
            const char * const * identities,
            const char * const * bindings,
            size_t count) -> const char * {
        if (count == 0 || identities == nullptr || bindings == nullptr) {
            return "filtered stage loading requires exact planner frontier identities and bindings";
        }
        for (size_t i = 0; i < count; ++i) {
            if (identities[i] == nullptr || identities[i][0] == '\0' ||
                    bindings[i] == nullptr || bindings[i][0] == '\0') {
                return "planner frontier identities and bindings must be non-empty";
            }
            for (size_t j = 0; j < i; ++j) {
                if (std::strcmp(identities[i], identities[j]) == 0) return "planner frontier identities must be unique";
                if (std::strcmp(bindings[i], bindings[j]) == 0) return "planner frontier bindings must be unique";
            }
        }
        return nullptr;
    };
    if (!skippy_runtime_is_source_stage(config)) {
        if (const char * error = validate_frontier(
                    config->activation_import_identities, config->activation_import_bindings,
                    config->activation_import_identity_count)) return error;
    }
    if (!skippy_runtime_is_terminal_stage(config)) {
        if (const char * error = validate_frontier(
                    config->activation_export_identities, config->activation_export_bindings,
                    config->activation_export_identity_count)) return error;
    }
    return nullptr;
}

static llama_model_loader_stage_selection skippy_loader_selection(
        const skippy_runtime_config * config) {
    llama_model_loader_stage_selection selection;
    if (!skippy_runtime_has_stage_plan(config)) {
        return selection;
    }
    selection.layer_start = config->layer_start;
    selection.layer_end = config->layer_end;
    for (size_t i = 0; i < config->resident_tensor_name_count; ++i) {
        selection.resident_tensor_names.emplace(config->resident_tensor_names[i]);
    }
    return selection;
}

static const llama_model_loader_stage_selection * skippy_loader_selection_ptr(
        const skippy_runtime_config * config,
        const llama_model_loader_stage_selection & selection) {
    return skippy_runtime_has_stage_plan(config) ? &selection : nullptr;
}

static void skippy_emit_observable_backend_device(
        skippy_runtime_event_scope * event_scope,
        const struct skippy_runtime_config * config,
        const llama_model * model) {
    if (event_scope == nullptr || !event_scope->is_enabled()) {
        return;
    }
    if (config != nullptr && config->selected_backend_device != nullptr && config->selected_backend_device[0] != '\0') {
        event_scope->emit_backend_selected(config->selected_backend_device);
        return;
    }
    if (model != nullptr && !model->devices.empty()) {
        event_scope->emit_backend_selected(ggml_backend_dev_name(model->devices.front().dev));
    }
}

static bool skippy_should_create_native_mtp_sidecar(
        const llama_model * model,
        const skippy_runtime_config * config) {
    if (model == nullptr ||
        config == nullptr ||
        !skippy_runtime_is_terminal_stage(config) ||
        config->mtp_source != SKIPPY_MTP_SOURCE_INTEGRATED ||
        model->hparams.n_layer_nextn == 0) {
        return false;
    }

    // A declared auxiliary head must satisfy ordinary builder admission.
    // Missing dependencies are an error, not a family-specific silent opt-out.
    return true;
}

static void skippy_enable_integrated_mtp_tensors(
        const skippy_runtime_config * config,
        llama_model_params & params) {
    // Exact stage residency decides which optional weights are loaded, even
    // when drafting is disabled. Unfiltered loads retain the source policy.
    params.load_mtp = config != nullptr &&
                      skippy_runtime_is_terminal_stage(config) &&
                      (skippy_runtime_has_stage_plan(config) || config->mtp_source == SKIPPY_MTP_SOURCE_INTEGRATED);
}

// Applies a tri-state config override on top of a derived default. Auto
// (SKIPPY_TRISTATE_AUTO) keeps the derived value; false/true replace it.
static bool skippy_resolve_tristate(int32_t tristate, bool derived_default) {
    if (tristate == SKIPPY_TRISTATE_FALSE) {
        return false;
    }
    if (tristate == SKIPPY_TRISTATE_TRUE) {
        return true;
    }
    return derived_default;
}

static bool skippy_context_capacity(
        const skippy_runtime_config * config,
                         uint32_t   lane_count,
                         uint32_t & context_size_per_lane,
                         uint32_t & context_size_total) {
    context_size_per_lane = config != nullptr && config->ctx_size > 0
        ? static_cast<uint32_t>(config->ctx_size)
        : 512;
    lane_count = std::max<uint32_t>(lane_count, 1);
    if (context_size_per_lane > std::numeric_limits<uint32_t>::max() / lane_count) {
        return false;
    }
    context_size_total = context_size_per_lane * lane_count;
    return true;
}

extern "C" {

struct skippy_runtime_config skippy_runtime_config_default(void) {
    struct skippy_runtime_config config = {};
    config.cache_type_k = GGML_TYPE_F16;
    config.cache_type_v = GGML_TYPE_F16;
    config.kv_offload = SKIPPY_TRISTATE_AUTO;
    config.kv_unified = SKIPPY_TRISTATE_AUTO;
    config.swa_full = SKIPPY_TRISTATE_AUTO;
    config.op_offload = SKIPPY_TRISTATE_AUTO;
    config.split_mode = SKIPPY_TRISTATE_AUTO;
    return config;
}

enum skippy_status skippy_finish_model_open(
        llama_model * model,
        const struct skippy_runtime_config * config,
        skippy_runtime_event_scope * event_scope,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    if (skippy_runtime_has_stage_plan(config)) {
        const int32_t n_layer = skippy_stage_layer_count(model);
        if (config->layer_end > n_layer) {
            llama_model_free(model);
            const char * message = "layer_end exceeds model layer count";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (skippy_runtime_is_source_stage(config) != (config->layer_start == 0)) {
            llama_model_free(model);
            const char * message = "admitted activation imports disagree with the source-stage range";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (skippy_runtime_is_terminal_stage(config) != (config->layer_end == n_layer)) {
            llama_model_free(model);
            const char * message = "admitted activation exports disagree with the terminal-stage range";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }

    const bool encoder_decoder = llama_model_has_encoder(model) && llama_model_has_decoder(model);
    const auto capabilities = skippy_context_capabilities_of(model);
    const bool classifier_workload = capabilities.classifier;
    const bool embedding_workload = capabilities.pooled_embeddings;
    const uint32_t configured_lane_count = config != nullptr && config->lane_count > 0
        ? static_cast<uint32_t>(config->lane_count)
        : 1;
    // Encoder output is context-global in llama.cpp. Until that state can be
    // isolated per sequence, serialize encoder-decoder work on one lane.
    const uint32_t lane_count = encoder_decoder ? 1 : configured_lane_count;
    uint32_t context_size_per_lane = 0;
    uint32_t context_size_total = 0;
    if (!skippy_context_capacity(config, lane_count, context_size_per_lane, context_size_total)) {
        llama_model_free(model);
        const char * message = "ctx_size multiplied by lane_count exceeds the native context range";
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_model * stage_model = new skippy_model{};
    stage_model->model = model;
    if (config != nullptr) {
        stage_model->config = *config;
        stage_model->executable = true;
        const auto copy_frontier = [](const char * const * values, size_t count, std::vector<std::string> & storage,
                                      std::vector<const char *> & pointers) {
            storage.reserve(count);
            for (size_t i = 0; i < count; ++i) storage.emplace_back(values[i]);
            pointers.reserve(storage.size());
            for (const std::string & value : storage) pointers.push_back(value.c_str());
        };
        copy_frontier(config->activation_import_identities, config->activation_import_identity_count,
                stage_model->activation_import_identities, stage_model->activation_import_identity_ptrs);
        copy_frontier(config->activation_import_bindings, config->activation_import_identity_count,
                stage_model->activation_import_bindings, stage_model->activation_import_binding_ptrs);
        copy_frontier(config->activation_export_identities, config->activation_export_identity_count,
                stage_model->activation_export_identities, stage_model->activation_export_identity_ptrs);
        copy_frontier(config->activation_export_bindings, config->activation_export_identity_count,
                stage_model->activation_export_bindings, stage_model->activation_export_binding_ptrs);
        copy_frontier(config->resident_tensor_names, config->resident_tensor_name_count,
                stage_model->resident_tensor_names, stage_model->resident_tensor_name_ptrs);
        stage_model->config.activation_import_identities = stage_model->activation_import_identity_ptrs.data();
        stage_model->config.activation_import_bindings = stage_model->activation_import_binding_ptrs.data();
        stage_model->config.activation_export_identities = stage_model->activation_export_identity_ptrs.data();
        stage_model->config.activation_export_bindings = stage_model->activation_export_binding_ptrs.data();
        stage_model->config.resident_tensor_names = stage_model->resident_tensor_name_ptrs.data();
        stage_model->execution_contract = config->execution_contract == nullptr ? "" : config->execution_contract;
        stage_model->config.execution_contract = stage_model->execution_contract.c_str();
    }
    stage_model->lane_count = lane_count;

    llama_context_params params = llama_context_default_params();
    // ctx_size is a per-lane limit; reserve it for each admitted lane.
    params.n_ctx = context_size_total;
    params.n_batch = config != nullptr && config->n_batch > 0 ? static_cast<uint32_t>(config->n_batch) : context_size_per_lane;
    params.n_ubatch = config != nullptr && config->n_ubatch > 0 ? static_cast<uint32_t>(config->n_ubatch) : 0u;
    params.n_threads = config != nullptr && config->n_threads > 0 ? config->n_threads : skippy_default_thread_count();
    params.n_threads_batch = config != nullptr && config->n_threads_batch > 0 ? config->n_threads_batch : params.n_threads;
    params.n_seq_max = stage_model->lane_count;
    params.kv_unified = stage_model->lane_count > 1;
    if (config != nullptr) {
        params.kv_unified = skippy_resolve_tristate(config->kv_unified, params.kv_unified);
        params.offload_kqv = skippy_resolve_tristate(config->kv_offload, params.offload_kqv);
        params.swa_full = skippy_resolve_tristate(config->swa_full, params.swa_full);
        skippy_apply_context_hardware_config(config, params);
    }
    params.flash_attn_type = config != nullptr ? static_cast<llama_flash_attn_type>(config->flash_attn_type) : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (classifier_workload) {
        params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }
    const bool activation_export_stage =
            skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_terminal_stage(config);
    params.embeddings = embedding_workload || activation_export_stage;
    if (embedding_workload) {
        // Non-causal embedding graphs must fit their logical batch in one
        // physical batch, matching llama-embedding's runtime contract.
        params.n_ubatch = params.n_batch;
    }
    const bool glm_dsa_op_timing_enabled = skippy_glm_dsa_op_timing_enabled();
    const bool glm_dsa_tensor_trace_enabled = skippy_glm_dsa_tensor_trace_enabled();
    if (model->arch == LLM_ARCH_GLM_DSA && (glm_dsa_op_timing_enabled || glm_dsa_tensor_trace_enabled)) {
        stage_model->glm_dsa_timing.enabled = true;
        stage_model->glm_dsa_timing.print_timing = glm_dsa_op_timing_enabled;
        stage_model->glm_dsa_timing.trace_tensors = glm_dsa_tensor_trace_enabled;
        stage_model->glm_dsa_timing.trace_stats = skippy_glm_dsa_tensor_trace_stats_enabled();
        stage_model->glm_dsa_timing.trace_value_limit =
                skippy_env_u32("SKIPPY_GLM_DSA_TENSOR_TRACE_VALUES", 8, 0, 4096);
        stage_model->glm_dsa_timing.trace_node_limit =
                skippy_env_u32("SKIPPY_GLM_DSA_TENSOR_TRACE_NODES", 32, 1, 1024);
        stage_model->glm_dsa_timing.trace_stats_max_bytes =
                skippy_env_u32("SKIPPY_GLM_DSA_TENSOR_TRACE_STATS_MAX_BYTES", 32 * 1024 * 1024, 0, 1024 * 1024 * 1024);
        if (const char * trace_filter = std::getenv("SKIPPY_GLM_DSA_TENSOR_TRACE_FILTER")) {
            stage_model->glm_dsa_timing.trace_filter = trace_filter;
        }
        stage_model->glm_dsa_timing.stage_index = config != nullptr ? config->stage_index : -1;
        params.cb_eval = skippy_glm_dsa_op_timing_cb;
        params.cb_eval_user_data = stage_model;
    }
    if (llm_arch_is_recurrent(model->arch) || llm_arch_is_hybrid(model->arch)) {
        params.n_seq_max = std::max<uint32_t>(params.n_seq_max, std::max<uint32_t>(2, stage_model->lane_count * 2));
        params.n_rs_seq = std::max<uint32_t>(params.n_rs_seq, 2);
        params.kv_unified = true;
    }
    // Multi-lane activation exports rely on the shared KV pool's batch order.
    if (activation_export_stage && stage_model->lane_count > 1 && !params.kv_unified) {
        const char * message = "an exporting stage with multiple lanes requires a unified KV cache";
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        }
        llama_model_free(model);
        delete stage_model;
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const skippy_graph_build_inputs graph_build_inputs = skippy_graph_build_inputs_from_config(config);
    stage_model->ctx = llama_init_from_model_with_graph_inputs(model, params, graph_build_inputs);
    if (stage_model->ctx == nullptr) {
        const char * message = "failed to create llama context";
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_RUNTIME_ERROR, message);
        }
        llama_model_free(model);
        delete stage_model;
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, message);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    const auto fail_boundary_load = [&](const char * message) {
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_RUNTIME_ERROR, message);
        }
        llama_free(stage_model->mtp_ctx);
        stage_model->mtp_ctx = nullptr;
        llama_free(stage_model->ctx);
        stage_model->ctx = nullptr;
        llama_model_free(model);
        delete stage_model;
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, message);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    };

    const auto parse_identity = [](const char * value, uint8_t output[SKIPPY_ACTIVATION_IDENTITY_BYTES]) {
        if (value == nullptr) return false;
        const size_t length = std::strlen(value);
        if (length < SKIPPY_ACTIVATION_IDENTITY_BYTES * 2) return false;
        const char * hex = value + length - SKIPPY_ACTIVATION_IDENTITY_BYTES * 2;
        const auto nibble = [](char character) -> int {
            if (character >= '0' && character <= '9') return character - '0';
            if (character >= 'a' && character <= 'f') return character - 'a' + 10;
            return -1;
        };
        for (size_t index = 0; index < SKIPPY_ACTIVATION_IDENTITY_BYTES; ++index) {
            const int high = nibble(hex[index * 2]);
            const int low = nibble(hex[index * 2 + 1]);
            if (high < 0 || low < 0) return false;
            output[index] = static_cast<uint8_t>((high << 4) | low);
        }
        return true;
    };
    const auto build_boundary = [&](bool input, skippy_activation_boundary_desc & boundary) {
        std::vector<skippy_activation_port_shape> shapes;
        const bool observed = input ? stage_model->ctx->get_input_activation_frontier(shapes)
                                    : stage_model->ctx->get_activation_frontier(shapes);
        const char * const * identities = input ? stage_model->config.activation_import_identities
                                                 : stage_model->config.activation_export_identities;
        const char * const * bindings = input ? stage_model->config.activation_import_bindings
                                               : stage_model->config.activation_export_bindings;
        const size_t identity_count = input ? stage_model->config.activation_import_identity_count
                                            : stage_model->config.activation_export_identity_count;
        if (!observed || shapes.empty() || shapes.size() != identity_count ||
                shapes.size() > SKIPPY_ACTIVATION_MAX_PARTS || bindings == nullptr) return false;
        boundary = {};
        boundary.version = SKIPPY_ACTIVATION_BOUNDARY_DESC_VERSION;
        boundary.part_count = static_cast<uint32_t>(shapes.size());
        std::set<size_t> matched_bindings;
        for (size_t index = 0; index < shapes.size(); ++index) {
            size_t binding_index = identity_count;
            for (size_t candidate = 0; candidate < identity_count; ++candidate) {
                if (bindings[candidate] != nullptr && shapes[index].binding == bindings[candidate]) {
                    binding_index = candidate;
                    break;
                }
            }
            if (binding_index == identity_count || !matched_bindings.insert(binding_index).second) return false;
            auto & part = boundary.parts[index];
            std::string stable_identity;
            if (!skippy_build_activation_port_identity(bindings[binding_index], stable_identity) ||
                    stable_identity != identities[binding_index] ||
                    !parse_identity(identities[binding_index], part.identity)) return false;
            part.ggml_type = static_cast<uint32_t>(shapes[index].type);
            part.rank = shapes[index].rank;
            part.token_axis = shapes[index].token_axis;
            part.flags = shapes[index].optional ? SKIPPY_ACTIVATION_PART_OPTIONAL : 0;
            uint64_t stride = ggml_type_size(shapes[index].type);
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                part.dimensions[dimension] = shapes[index].dimensions[dimension];
                part.byte_strides[dimension] = stride;
                if (shapes[index].dimensions[dimension] > 0) stride *= static_cast<uint64_t>(shapes[index].dimensions[dimension]);
            }
        }
        return skippy_build_activation_frontier_identity(
                boundary.parts, boundary.part_count, boundary.frontier_identity);
    };

    if (skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_terminal_stage(config)) {
        if (!build_boundary(false, stage_model->output_activation_boundary)) {
            return fail_boundary_load("stage graph output frontier does not match its admitted planner identities");
        }
        stage_model->has_output_activation_boundary = true;
    }

    if (skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_source_stage(config)) {
        if (!build_boundary(true, stage_model->input_activation_boundary)) {
            return fail_boundary_load("stage graph input frontier does not match its admitted planner identities");
        }
        stage_model->has_input_activation_boundary = true;
    }

    if (skippy_should_create_native_mtp_sidecar(model, config)) {
        llama_context_params mtp_params = params;
        mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mtp_params.embeddings = false;
        stage_model->mtp_ctx = llama_init_from_model_with_graph_inputs(model, mtp_params, graph_build_inputs);
        if (stage_model->mtp_ctx != nullptr) {
            llama_set_embeddings_nextn(stage_model->ctx, true, false);
            llama_set_embeddings_nextn(stage_model->mtp_ctx, true, false);
        } else {
            fprintf(stderr, "skippy: native MTP sidecar unavailable for this final stage; continuing without drafts\n");
        }
    } else if (config != nullptr &&
               config->mtp_source == SKIPPY_MTP_SOURCE_INTEGRATED &&
               skippy_runtime_is_terminal_stage(config) &&
               model->hparams.n_layer_nextn > 0) {
        fprintf(stderr, "skippy: integrated MTP requested, but this split stage is missing required MTP graph tensors; continuing without drafts\n");
    }


    stage_model->lane_in_use.assign(stage_model->lane_count, false);
    stage_model->lane_resident_prefix_tokens.resize(stage_model->lane_count);

    const uint64_t native_model_id = reinterpret_cast<uint64_t>(stage_model);
    if (event_scope != nullptr) {
        event_scope->set_model_id(native_model_id);
        skippy_emit_observable_backend_device(event_scope, config, model);
    }

    // These are the concrete boundaries that are known after context
    // creation. They are sent to the operation reporter when one is active;
    // otherwise the process reporter receives the same observation. Do not
    // synthesize tensor/offload events here: llama.cpp does not expose a
    // stable count for those transitions at this boundary.
    skippy_emit_model_load_observation(
            event_scope,
            SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED,
            native_model_id,
            static_cast<uint64_t>(skippy_stage_layer_count(model)),
            0,
            "execution context ready",
            sizeof("execution context ready") - 1);
    skippy_emit_model_load_observation(
            event_scope,
            SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY,
            native_model_id,
            static_cast<uint64_t>(llama_vocab_n_tokens(&model->vocab)),
            0,
            "tokenizer ready",
            sizeof("tokenizer ready") - 1);
    if (stage_model->mtp_ctx != nullptr) {
        skippy_emit_model_load_observation(
                event_scope,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY,
                native_model_id,
                1,
                0,
                "mtp component ready",
                sizeof("mtp component ready") - 1);
    }
    if (event_scope != nullptr) {
        event_scope->emit_finished();
    }

    *out_model = stage_model;
    return skippy_success(out_error);
}

enum skippy_status skippy_model_open_impl(
        const char * path,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    enum skippy_status reporter_status = skippy_validate_runtime_event_reporter(reporter, out_error);
    if (reporter_status != SKIPPY_STATUS_OK) {
        return reporter_status;
    }

    skippy_runtime_event_scope event_scope(config, reporter);
    event_scope.emit_started(path);

    if (path == nullptr || out_model == nullptr) {
        const char * message = "path and out_model are required";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_model = nullptr;
    if (const char * message = skippy_validate_stage_plan_config(config)) {
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (skippy_runtime_has_stage_plan(config) && (config->layer_start < 0 || config->layer_start >= config->layer_end)) {
        const char * message = "layer_start must be non-negative and less than layer_end";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (!skippy_is_full_model_config(config) && !skippy_runtime_has_stage_plan(config)) {
        const char * message = "split loading requires an exact graph-admitted resident tensor closure";
        event_scope.emit_failure(SKIPPY_STATUS_UNSUPPORTED, message);
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, message);
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    llama_model_params params = llama_model_default_params();
    skippy_enable_integrated_mtp_tensors(config, params);
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
        skippy_apply_model_load_memory_config(config, params);
    }
    if (event_scope.is_enabled()) {
        params.progress_callback = skippy_model_open_progress_callback;
        params.progress_callback_user_data = &event_scope;
    }

    llama_backend_init();
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    std::vector<ggml_backend_dev_t> selected_devices;
    enum skippy_status device_status = skippy_apply_selected_backend_device(config, params, selected_devices, out_error);
    if (device_status != SKIPPY_STATUS_OK) {
        if (out_error != nullptr && *out_error != nullptr) {
            event_scope.emit_failure(device_status, (*out_error)->message);
        } else {
            event_scope.emit_failure(device_status, "failed to apply selected backend device");
        }
        return device_status;
    }
    const auto selection = skippy_loader_selection(config);
    llama_model * model = llama_model_load_from_file_with_selection(
            path, params, skippy_loader_selection_ptr(config, selection));
    if (model == nullptr) {
        const char * message = "failed to load llama model";
        event_scope.emit_failure(SKIPPY_STATUS_MODEL_ERROR, message);
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, message);
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    return skippy_finish_model_open(model, config, &event_scope, out_model, out_error);
}

enum skippy_status skippy_model_open_from_parts_impl(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    enum skippy_status reporter_status = skippy_validate_runtime_event_reporter(reporter, out_error);
    if (reporter_status != SKIPPY_STATUS_OK) {
        return reporter_status;
    }

    skippy_runtime_event_scope event_scope(config, reporter);
    const char * primary_path = paths != nullptr && path_count > 0 ? paths[0] : nullptr;
    event_scope.emit_started(primary_path);

    if (paths == nullptr || path_count == 0 || out_model == nullptr) {
        const char * message = "paths, path_count, and out_model are required";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < path_count; ++i) {
        if (paths[i] == nullptr) {
            const char * message = "package part path is null";
            event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }

    *out_model = nullptr;
    if (const char * message = skippy_validate_stage_plan_config(config)) {
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (skippy_runtime_has_stage_plan(config) && (config->layer_start < 0 || config->layer_start >= config->layer_end)) {
        const char * message = "layer_start must be non-negative and less than layer_end";
        event_scope.emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    if (!skippy_is_full_model_config(config) && !skippy_runtime_has_stage_plan(config)) {
        const char * message = "split loading requires an exact graph-admitted resident tensor closure";
        event_scope.emit_failure(SKIPPY_STATUS_UNSUPPORTED, message);
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, message);
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    llama_model_params params = llama_model_default_params();
    skippy_enable_integrated_mtp_tensors(config, params);
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
        skippy_apply_model_load_memory_config(config, params);
    }
    if (event_scope.is_enabled()) {
        params.progress_callback = skippy_model_open_progress_callback;
        params.progress_callback_user_data = &event_scope;
    }

    llama_backend_init();
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    std::vector<ggml_backend_dev_t> selected_devices;
    enum skippy_status device_status = skippy_apply_selected_backend_device(config, params, selected_devices, out_error);
    if (device_status != SKIPPY_STATUS_OK) {
        if (out_error != nullptr && *out_error != nullptr) {
            event_scope.emit_failure(device_status, (*out_error)->message);
        } else {
            event_scope.emit_failure(device_status, "failed to apply selected backend device");
        }
        return device_status;
    }
    const auto selection = skippy_loader_selection(config);
    llama_model * model = llama_model_load_from_parts_with_selection(
            paths, path_count, params, skippy_loader_selection_ptr(config, selection));
    if (model == nullptr) {
        const char * message = "failed to load llama model from GGUF parts";
        event_scope.emit_failure(SKIPPY_STATUS_MODEL_ERROR, message);
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, message);
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    return skippy_finish_model_open(model, config, &event_scope, out_model, out_error);
}

enum skippy_status skippy_model_attach_mtp_draft_model(
        struct skippy_model * target_model,
        const char * path,
        const struct skippy_runtime_config * config,
        struct skippy_error ** out_error) {
    if (target_model == nullptr || target_model->ctx == nullptr || path == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "target_model, target context, and path are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (config == nullptr || config->mtp_source != SKIPPY_MTP_SOURCE_EXTERNAL) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "MTP draft attachment requires an external MTP source");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (target_model->mtp_ctx != nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "target model already has an MTP draft context");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (const char * message = skippy_validate_stage_plan_config(config)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    uint32_t context_size_per_lane = 0;
    uint32_t context_size_total = 0;
    if (!skippy_context_capacity(
            config,
            target_model->lane_count,
            context_size_per_lane,
            context_size_total)) {
        skippy_set_error(
                out_error,
                SKIPPY_STATUS_INVALID_ARGUMENT,
                "ctx_size multiplied by lane_count exceeds the native context range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_model_params model_params = llama_model_default_params();
    // The guard above already established an external MTP source, so this model
    // backs an LLAMA_CONTEXT_TYPE_MTP draft context and must retain MTP tensors.
    model_params.load_mtp = true;
    if (config != nullptr) {
        model_params.n_gpu_layers = config->n_gpu_layers;
        skippy_apply_model_load_memory_config(config, model_params);
    }

    llama_backend_init();
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    std::vector<ggml_backend_dev_t> selected_devices;
    enum skippy_status device_status = skippy_apply_selected_backend_device(config, model_params, selected_devices, out_error);
    if (device_status != SKIPPY_STATUS_OK) {
        return device_status;
    }

    const auto selection = skippy_loader_selection(config);
    llama_model * draft_model = llama_model_load_from_file_with_selection(
            path, model_params, skippy_loader_selection_ptr(config, selection));
    if (draft_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to load MTP draft model");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.ctx_other = target_model->ctx;
    ctx_params.n_ctx = context_size_total;
    ctx_params.n_batch = config->n_batch > 0 ? static_cast<uint32_t>(config->n_batch) : context_size_per_lane;
    ctx_params.n_ubatch = config != nullptr && config->n_ubatch > 0 ? static_cast<uint32_t>(config->n_ubatch) : 0u;
    ctx_params.n_threads = config != nullptr && config->n_threads > 0 ? config->n_threads : skippy_default_thread_count();
    ctx_params.n_threads_batch = config != nullptr && config->n_threads_batch > 0 ? config->n_threads_batch : ctx_params.n_threads;
    ctx_params.n_seq_max = target_model->lane_count;
    ctx_params.kv_unified = target_model->lane_count > 1;
    if (config != nullptr) {
        ctx_params.kv_unified = skippy_resolve_tristate(config->kv_unified, ctx_params.kv_unified);
        ctx_params.offload_kqv = skippy_resolve_tristate(config->kv_offload, ctx_params.offload_kqv);
        ctx_params.swa_full = skippy_resolve_tristate(config->swa_full, ctx_params.swa_full);
        skippy_apply_context_hardware_config(config, ctx_params);
    }
    if (llm_arch_is_recurrent(draft_model->arch) || llm_arch_is_hybrid(draft_model->arch)) {
        ctx_params.kv_unified = true;
    }
    ctx_params.flash_attn_type = config != nullptr ? static_cast<llama_flash_attn_type>(config->flash_attn_type) : LLAMA_FLASH_ATTN_TYPE_AUTO;

    target_model->mtp_ctx = llama_init_from_model(draft_model, ctx_params);
    if (target_model->mtp_ctx == nullptr) {
        llama_model_free(draft_model);
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to create MTP draft context");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    llama_set_embeddings_nextn(target_model->ctx, true, false);
    target_model->mtp_model = draft_model;
    llama_set_embeddings_nextn(target_model->mtp_ctx, true, false);
    return skippy_success(out_error);
}

} // extern "C"
