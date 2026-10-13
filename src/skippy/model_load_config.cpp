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
#include "skippy/model_loading_internal.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"

#include "skippy/model_load_config_internal.h"

#include <cstring>

int32_t skippy_default_thread_count() {
#ifdef __linux__
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings(
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break;
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t physical_cores = 0;
    size_t len = sizeof(physical_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &physical_cores, &len, NULL, 0) == 0 && physical_cores > 0) {
        return physical_cores;
    }
    if (sysctlbyname("hw.physicalcpu", &physical_cores, &len, NULL, 0) == 0 && physical_cores > 0) {
        return physical_cores;
    }
#endif
    const unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? static_cast<int32_t>(n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

enum skippy_status skippy_apply_selected_backend_device(
        const struct skippy_runtime_config * config,
        llama_model_params & params,
        std::vector<ggml_backend_dev_t> & selected_devices,
        struct skippy_error ** out_error) {
    if (config == nullptr || config->selected_backend_device == nullptr || config->selected_backend_device[0] == '\0') {
        return SKIPPY_STATUS_OK;
    }

    const std::string device_name(config->selected_backend_device);
    if (device_name == "CPU" || device_name == "CPU0") {
        params.devices = nullptr;
        params.main_gpu = -1;
        params.n_gpu_layers = 0;
        skippy_emit_device_event(
                SKIPPY_RUNTIME_EVENT_KIND_DEVICE_READY,
                0,
                "CPU",
                sizeof("CPU") - 1);
        return SKIPPY_STATUS_OK;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device_name.c_str());
    if (dev == nullptr) {
        skippy_emit_device_event(
                SKIPPY_RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE,
                0,
                "selected backend device unavailable",
                sizeof("selected backend device unavailable") - 1);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, ("unknown selected backend device: " + device_name).c_str());
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        skippy_emit_device_event(
                SKIPPY_RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE,
                0,
                "selected backend device is CPU",
                sizeof("selected backend device is CPU") - 1);
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, ("selected backend device is CPU; use CPU or CPU0: " + device_name).c_str());
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    selected_devices.clear();
    selected_devices.push_back(dev);
    selected_devices.push_back(nullptr);
    params.devices = selected_devices.data();
    params.main_gpu = 0;
    const char * selected_device_name = ggml_backend_dev_name(dev);
    if (selected_device_name == nullptr) {
        selected_device_name = "selected backend device";
    }
    skippy_emit_device_event(
            SKIPPY_RUNTIME_EVENT_KIND_DEVICE_READY,
            static_cast<uint64_t>(ggml_backend_dev_type(dev)),
            selected_device_name,
            static_cast<uint64_t>(std::strlen(selected_device_name)));
    return SKIPPY_STATUS_OK;
}

void skippy_apply_model_load_memory_config(
        const struct skippy_runtime_config * config,
        llama_model_params & params) {
    if (config == nullptr) {
        return;
    }
    if (config->use_direct_io) {
        params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    } else if (config->use_mlock) {
        params.load_mode = LLAMA_LOAD_MODE_MLOCK;
    } else if (config->has_mmap_override) {
        params.load_mode = config->use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    }
    params.check_tensors = config->check_tensors;
    params.no_host = config->no_host_buffer;
    if (config->has_main_gpu_override) {
        params.main_gpu = config->main_gpu;
    }
    if (config->split_mode != SKIPPY_TRISTATE_AUTO) {
        params.split_mode = static_cast<llama_split_mode>(config->split_mode);
    }
    if (config->disable_repack || skippy_runtime_has_stage_plan(config)) {
        params.use_extra_bufts = false;
    }
}

void skippy_apply_context_hardware_config(
        const struct skippy_runtime_config * config,
        llama_context_params & params) {
    if (config == nullptr) {
        return;
    }
    if (config->cache_type_k >= 0) {
        params.type_k = static_cast<ggml_type>(config->cache_type_k);
    }
    if (config->cache_type_v >= 0) {
        params.type_v = static_cast<ggml_type>(config->cache_type_v);
    }
    if (config->op_offload != SKIPPY_TRISTATE_AUTO) {
        params.op_offload = config->op_offload == SKIPPY_TRISTATE_TRUE;
    }
}

skippy_context_capabilities skippy_context_capabilities_of(const llama_model * model) {
    skippy_context_capabilities capabilities;
    if (model == nullptr) {
        return capabilities;
    }
    const bool encoder_decoder = llama_model_has_encoder(model) && llama_model_has_decoder(model);
    // A classifier workload serves the pooled classifier projection. A model
    // that also carries a generation head keeps its logits, because upstream
    // drops the LM head when a context runs in pooled embeddings mode.
    capabilities.classifier = (model->cls != nullptr || model->cls_out != nullptr) &&
                              model->output == nullptr;
    capabilities.pooled_embeddings = !encoder_decoder &&
            (capabilities.classifier ||
             (llama_model_has_encoder(model) ||
              (model->hparams.pooling_type != LLAMA_POOLING_TYPE_NONE &&
               model->hparams.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED)));
    return capabilities;
}
