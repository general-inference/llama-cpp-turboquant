#pragma once

#include "skippy/common.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdint>
#include <vector>

struct skippy_error;
struct skippy_runtime_config;

int32_t skippy_default_thread_count();
enum skippy_status skippy_apply_selected_backend_device(
        const skippy_runtime_config * config,
        llama_model_params & params,
        std::vector<ggml_backend_dev_t> & selected_devices,
        skippy_error ** out_error);
void skippy_apply_model_load_memory_config(
        const skippy_runtime_config * config,
        llama_model_params & params);
void skippy_apply_context_hardware_config(
        const skippy_runtime_config * config,
        llama_context_params & params);

// Model-derived context capabilities. The runtime derives its embedding and
// classifier mode from the loaded model, and stage planning must observe the
// same graph shape, so both read them through this one helper.
struct skippy_context_capabilities {
    bool classifier        = false;
    bool pooled_embeddings = false;
};
struct llama_model;
skippy_context_capabilities skippy_context_capabilities_of(const llama_model * model);
