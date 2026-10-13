#include "skippy.h"
#include "skippy/errors.h"
#include "skippy/environment.h"
#include "skippy/glm_dsa.h"
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
#include "skippy/sampling_internal.h"
#include "skippy/speculative_internal.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"


#include "skippy/model_loading_internal.h"

extern "C" {
enum skippy_status skippy_model_open(
        const char * path,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_impl(path, config, nullptr, out_model, out_error);
}

enum skippy_status skippy_model_open_with_events(
        const char * path,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_impl(path, config, reporter, out_model, out_error);
}

enum skippy_status skippy_model_open_from_parts(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_from_parts_impl(paths, path_count, config, nullptr, out_model, out_error);
}

enum skippy_status skippy_model_open_from_parts_with_events(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    return skippy_model_open_from_parts_impl(paths, path_count, config, reporter, out_model, out_error);
}

enum skippy_status skippy_model_free(
        struct skippy_model * model,
        struct skippy_error ** out_error) {
    if (model != nullptr) {
        const uint64_t native_model_id = reinterpret_cast<uint64_t>(model);
        skippy_emit_unload_event(
                SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_STARTED,
                native_model_id,
                0);
        if (model->mtp_ctx != nullptr) {
            llama_free(model->mtp_ctx);
        }
        if (model->mtp_model != nullptr) {
            llama_model_free(model->mtp_model);
        }
        if (model->ctx != nullptr) {
            llama_free(model->ctx);
        }
        if (model->chat_templates != nullptr && model->chat_templates_free != nullptr) {
            model->chat_templates_free(model->chat_templates);
        }
        llama_model_free(model->model);
        skippy_emit_unload_event(
                SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_COMPLETED,
                native_model_id,
                0);
        delete model;
    }
    return skippy_success(out_error);
}

const struct llama_model * skippy_model_llama_model(
        const struct skippy_model * model) {
    return model != nullptr ? model->model : nullptr;
}

bool skippy_model_output_activation_boundary(
        const struct skippy_model * model,
        struct skippy_activation_boundary_desc * out_desc) {
    if (model == nullptr || out_desc == nullptr || !model->has_output_activation_boundary) {
        return false;
    }
    *out_desc = model->output_activation_boundary;
    return true;
}

bool skippy_model_input_activation_boundary(
        const struct skippy_model * model,
        struct skippy_activation_boundary_desc * out_desc) {
    if (model == nullptr || out_desc == nullptr || !model->has_input_activation_boundary) {
        return false;
    }
    *out_desc = model->input_activation_boundary;
    return true;
}


} // extern "C"
