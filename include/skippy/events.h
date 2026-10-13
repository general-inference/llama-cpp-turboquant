#ifndef SKIPPY_EVENTS_H
#define SKIPPY_EVENTS_H

/**
 * @file skippy/events.h
 * @brief Versioned callbacks for model-open and runtime lifecycle progress.
 *
 * Event reporters are optional and operation-scoped. A reporter must remain
 * valid until the corresponding model-open call returns.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION 1

typedef uint32_t skippy_runtime_event_category;
enum {
    SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN = 1,
    SKIPPY_RUNTIME_EVENT_CATEGORY_BACKEND    = 2,
    SKIPPY_RUNTIME_EVENT_CATEGORY_SESSION    = 3,
    SKIPPY_RUNTIME_EVENT_CATEGORY_KV         = 4,
    SKIPPY_RUNTIME_EVENT_CATEGORY_WARNING    = 5,
    SKIPPY_RUNTIME_EVENT_CATEGORY_DEVICE     = 6,
    SKIPPY_RUNTIME_EVENT_CATEGORY_DIAGNOSTIC = 7,
    SKIPPY_RUNTIME_EVENT_CATEGORY_UNLOAD     = 8,
};

typedef uint32_t skippy_runtime_event_kind;
enum {
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED       = 1,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_PROGRESS      = 2,
    SKIPPY_RUNTIME_EVENT_KIND_BACKEND_DEVICE_SELECTED  = 3,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FINISHED      = 4,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_FAILED_HANDLED = 5,

    // SKIPPY_FEATURE_MODEL_LOAD_EVENTS_V2 (coarse load-phase facts, category
    // MODEL_OPEN). Kept in the shared kind space so a MODEL_OPEN-category
    // event's `kind` is decodable without a second enum lookup per category.
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED       = 100,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_MEMORY_ALLOCATED    = 101,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_TENSORS_OFFLOADED   = 102,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY     = 103,
    SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY = 104,

    // SKIPPY_FEATURE_KV_EVENTS (category KV).
    SKIPPY_RUNTIME_EVENT_KIND_KV_INITIALIZED                  = 200,
    SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED             = 201,
    SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED             = 202,
    SKIPPY_RUNTIME_EVENT_KIND_KV_CONTEXT_APPROACHING_CAPACITY = 203,
    SKIPPY_RUNTIME_EVENT_KIND_KV_CONTEXT_CAPACITY_EXHAUSTED   = 204,

    // SKIPPY_FEATURE_DEVICE_EVENTS (category DEVICE).
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_BACKEND_INITIALIZED = 300,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_READY               = 301,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_DEGRADED            = 302,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_UNAVAILABLE         = 303,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_RECOVERED           = 304,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_LOST                = 305,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_RESOURCE_ALLOCATED  = 306,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_OUT_OF_MEMORY       = 307,
    SKIPPY_RUNTIME_EVENT_KIND_DEVICE_FALLBACK_ACTIVATED  = 308,

    // SKIPPY_FEATURE_DIAGNOSTIC_EVENTS (category DIAGNOSTIC).
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED      = 400,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_CLEARED     = 401,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_RECOVERABLE_FAILURE = 402,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_FATAL_FAILURE       = 403,
    SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_INVARIANT_VIOLATION = 404,

    // SKIPPY_FEATURE_UNLOAD_EVENTS (category UNLOAD).
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_STARTED          = 500,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_COMPLETED        = 501,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_FAILED           = 502,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_FORCED           = 503,
    SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_SESSION_DRAINING = 504,
};

typedef uint32_t skippy_runtime_event_emitter_kind;
enum {
    SKIPPY_RUNTIME_EVENT_EMITTER_UNKNOWN       = 0,
    SKIPPY_RUNTIME_EVENT_EMITTER_OPEN_THREAD   = 1,
    SKIPPY_RUNTIME_EVENT_EMITTER_WORKER_THREAD = 2,
};

typedef uint32_t skippy_runtime_event_progress_unit;
enum {
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_NONE    = 0,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_BYTES   = 1,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_ITEMS   = 2,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_TENSORS = 3,
    SKIPPY_RUNTIME_EVENT_PROGRESS_UNIT_STEPS   = 4,
};

typedef uint32_t skippy_runtime_event_failure_code;
enum {
    SKIPPY_RUNTIME_EVENT_FAILURE_NONE             = 0,
    SKIPPY_RUNTIME_EVENT_FAILURE_INVALID_ARGUMENT = 1,
    SKIPPY_RUNTIME_EVENT_FAILURE_IO_ERROR         = 2,
    SKIPPY_RUNTIME_EVENT_FAILURE_MODEL_ERROR      = 3,
    SKIPPY_RUNTIME_EVENT_FAILURE_RUNTIME_ERROR    = 4,
    SKIPPY_RUNTIME_EVENT_FAILURE_BACKEND_ERROR    = 5,
    SKIPPY_RUNTIME_EVENT_FAILURE_CANCELLED        = 6,
    SKIPPY_RUNTIME_EVENT_FAILURE_INTERNAL_ERROR   = 7,
};

/** @brief Versioned event payload delivered to a runtime event callback. */
struct skippy_runtime_event_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    skippy_runtime_event_category category;
    skippy_runtime_event_kind kind;
    skippy_runtime_event_emitter_kind emitter;
    uint32_t reserved0;
    uint64_t sequence;
    uint64_t timestamp_mono_ns;
    uint64_t model_id;
    uint64_t stage_id;
    uint64_t session_id;
    uint64_t progress_current;
    uint64_t progress_total;
    skippy_runtime_event_progress_unit progress_unit;
    skippy_runtime_event_failure_code failure_code;
    int32_t status;
    uint32_t reserved1;
    const char * detail_ptr;
    uint64_t detail_len;

    // Append-only extension fields. Advertised by `struct_size` under the
    // unchanged SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION; a caller only reads a
    // field when `struct_size` covers its offset.
    uint64_t numeric_summary_0;
    uint64_t numeric_summary_1;
    uint64_t numeric_summary_2;
    uint64_t numeric_summary_3;
};

/** @brief Callback type used by a runtime event reporter. */
typedef void (*skippy_runtime_event_callback)(
        const struct skippy_runtime_event_v1 * event,
        void * user_data);

/** @brief Callback registration passed to model-open-with-events entrypoints. */
struct skippy_runtime_event_reporter_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    skippy_runtime_event_callback callback;
    void * user_data;
};

/**
 * @brief Installs one process-global runtime-scoped event reporter.
 *
 * Requires SKIPPY_FEATURE_RUNTIME_EVENT_REPORTER. Replaces any previously
 * installed global reporter. The per-call model-open reporter passed to a
 * `*_with_events` entrypoint still takes precedence for
 * SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN events during that call; the
 * global reporter receives events published through it by any capability
 * module (backend/device/KV/diagnostic/unload/model-load-v2, added by
 * sibling patches) via the shared internal dispatch path.
 *
 * Returns SKIPPY_STATUS_INVALID_ARGUMENT if `reporter` is null or its
 * `struct_size` is smaller than `sizeof(struct
 * skippy_runtime_event_reporter_v1)`, or if called from a reporter callback.
 * When replacing an existing reporter, waits until every in-flight callback
 * using the previous reporter has returned before installing the replacement.
 */
LLAMA_API enum skippy_status skippy_set_runtime_event_reporter(
        const struct skippy_runtime_event_reporter_v1 * reporter);

/**
 * @brief Removes the process-global runtime-scoped event reporter.
 *
 * Returns only after every in-flight callback invocation of the previously
 * installed reporter has returned, so no callback occurs after this call
 * returns. Idempotent: clearing when no reporter is installed is a no-op.
 * When called from a reporter callback, disables future callback admission
 * and returns immediately; the calling callback remains active until it
 * returns normally.
 */
LLAMA_API void skippy_clear_runtime_event_reporter(void);

/**
 * @brief Emits SKIPPY_FEATURE_MODEL_LOAD_EVENTS_V2 category MODEL_OPEN
 * events through the global reporter. No-op if no reporter is installed.
 */
LLAMA_API void skippy_emit_model_load_event_v2(
        skippy_runtime_event_kind kind,
        uint64_t model_id,
        uint64_t numeric_summary_0,
        uint64_t numeric_summary_1,
        const char * detail_ptr,
        uint64_t detail_len);

/** @brief Emits SKIPPY_FEATURE_KV_EVENTS category KV events. */
LLAMA_API void skippy_emit_kv_event(
        skippy_runtime_event_kind kind,
        uint64_t session_id,
        uint64_t numeric_summary_0,
        uint64_t numeric_summary_1);

/** @brief Emits SKIPPY_FEATURE_DEVICE_EVENTS category DEVICE events. */
LLAMA_API void skippy_emit_device_event(
        skippy_runtime_event_kind kind,
        uint64_t numeric_summary_0,
        const char * detail_ptr,
        uint64_t detail_len);

/** @brief Emits SKIPPY_FEATURE_DIAGNOSTIC_EVENTS category DIAGNOSTIC events. */
LLAMA_API void skippy_emit_diagnostic_event(
        skippy_runtime_event_kind kind,
        skippy_runtime_event_failure_code failure_code,
        const char * detail_ptr,
        uint64_t detail_len);

/** @brief Emits SKIPPY_FEATURE_UNLOAD_EVENTS category UNLOAD events. */
LLAMA_API void skippy_emit_unload_event(
        skippy_runtime_event_kind kind,
        uint64_t model_id,
        uint64_t numeric_summary_0);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_EVENTS_H
