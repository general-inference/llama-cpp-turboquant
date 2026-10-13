#include "skippy/model_load_config_internal.h"
#include "skippy/runtime.h"

#include <cstdio>

static bool require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

int main() {
    skippy_runtime_config config = skippy_runtime_config_default();
    bool passed = true;
    passed &= require(config.cache_type_k == GGML_TYPE_F16,
            "default K cache type must be F16");
    passed &= require(config.cache_type_v == GGML_TYPE_F16,
            "default V cache type must be F16");
    passed &= require(config.kv_offload == SKIPPY_TRISTATE_AUTO,
            "default kv_offload must preserve the derived default");
    passed &= require(config.kv_unified == SKIPPY_TRISTATE_AUTO,
            "default kv_unified must preserve the derived default");
    passed &= require(config.swa_full == SKIPPY_TRISTATE_AUTO,
            "default swa_full must preserve the derived default");
    passed &= require(config.op_offload == SKIPPY_TRISTATE_AUTO,
            "default op_offload must preserve the derived default");
    passed &= require(config.split_mode == SKIPPY_TRISTATE_AUTO,
            "default split_mode must preserve the derived default");

    config.use_direct_io = true;
    config.use_mlock = true;
    config.has_mmap_override = true;
    config.use_mmap = true;
    config.no_host_buffer = true;
    config.check_tensors = true;
    config.has_main_gpu_override = true;
    config.main_gpu = 2;
    config.split_mode = LLAMA_SPLIT_MODE_ROW;
    config.disable_repack = true;

    llama_model_params model_params = llama_model_default_params();
    skippy_apply_model_load_memory_config(&config, model_params);

    passed &= require(model_params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO,
            "direct I/O must take precedence over mmap and mlock");
    passed &= require(model_params.main_gpu == 2, "main_gpu override was not applied");
    passed &= require(model_params.split_mode == LLAMA_SPLIT_MODE_ROW, "split mode was not applied");
    passed &= require(model_params.no_host, "host-buffer policy was not applied");
    passed &= require(model_params.check_tensors, "tensor checking was not applied");
    passed &= require(!model_params.use_extra_bufts, "disabled repack must disable extra buffer types");

    llama_context_params context_params = llama_context_default_params();
    context_params.op_offload = true;
    config.op_offload = SKIPPY_TRISTATE_AUTO;
    skippy_apply_context_hardware_config(&config, context_params);
    passed &= require(context_params.op_offload, "auto op_offload must preserve the derived default");

    config.cache_type_k = GGML_TYPE_F32;
    config.cache_type_v = GGML_TYPE_F32;
    skippy_apply_context_hardware_config(&config, context_params);
    passed &= require(context_params.type_k == GGML_TYPE_F32,
            "explicit F32 K cache type must not fall back to F16");
    passed &= require(context_params.type_v == GGML_TYPE_F32,
            "explicit F32 V cache type must not fall back to F16");

    config.op_offload = SKIPPY_TRISTATE_FALSE;
    skippy_apply_context_hardware_config(&config, context_params);
    passed &= require(!context_params.op_offload, "false op_offload override was not applied");

    config.op_offload = SKIPPY_TRISTATE_TRUE;
    skippy_apply_context_hardware_config(&config, context_params);
    passed &= require(context_params.op_offload, "true op_offload override was not applied");

    return passed ? 0 : 1;
}
