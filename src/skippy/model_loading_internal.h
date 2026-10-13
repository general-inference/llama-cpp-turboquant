#pragma once

#include "skippy/common.h"

#include <cstddef>

struct skippy_error;
struct skippy_model;
struct skippy_runtime_config;
struct skippy_runtime_event_reporter_v1;
struct skippy_runtime_event_scope;
struct llama_model;

extern "C" {

enum skippy_status skippy_model_open_impl(
        const char * path,
        const skippy_runtime_config * config,
        const skippy_runtime_event_reporter_v1 * reporter,
        skippy_model ** out_model,
        skippy_error ** out_error);

enum skippy_status skippy_model_open_from_parts_impl(
        const char * const * paths,
        size_t path_count,
        const skippy_runtime_config * config,
        const skippy_runtime_event_reporter_v1 * reporter,
        skippy_model ** out_model,
        skippy_error ** out_error);

enum skippy_status skippy_finish_model_open(
        llama_model * model,
        const skippy_runtime_config * config,
        skippy_runtime_event_scope * event_scope,
        skippy_model ** out_model,
        skippy_error ** out_error);

} // extern "C"
