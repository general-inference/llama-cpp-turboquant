#ifndef SKIPPY_COMMON_H
#define SKIPPY_COMMON_H

/**
 * @file skippy/common.h
 * @brief ABI versioning, feature discovery, status codes, and errors.
 *
 * Every other public Skippy header depends on this capability boundary. The
 * ABI is intentionally C-compatible so Rust and other native callers do not
 * depend on private llama.cpp C++ layouts.
 */

#include "../llama.h"
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"

#if defined(LLAMA_SHARED) && defined(_WIN32) && !defined(__MINGW32__)
#    ifdef SKIPPY_COMMON_BUILD
#        define SKIPPY_COMMON_API __declspec(dllexport)
#    else
#        define SKIPPY_COMMON_API __declspec(dllimport)
#    endif
#elif defined(LLAMA_SHARED)
#    define SKIPPY_COMMON_API __attribute__((visibility("default")))
#else
#    define SKIPPY_COMMON_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Experimental ABI for layer-range staged inference.
//
// This header is intentionally a capability boundary. It exposes opaque handles
// and plain C structs so Rust orchestration code can use llama.cpp model
// execution, tokenization, GGUF metadata, and GGUF writing without depending on
// private C++ layouts.
//
// Stability note:
// - ABI version 0 is experimental.
// - Callers must feature-probe before use.
// - Names, structs, and ownership rules may change until cross-family staged
//   runtime validation has completed.

#define SKIPPY_ABI_VERSION_MAJOR 0
#define SKIPPY_ABI_VERSION_MINOR 1
#define SKIPPY_ABI_VERSION_PATCH 73

#if defined(_MSC_VER)
#define SKIPPY_DEPRECATED(message) __declspec(deprecated(message))
#elif defined(__GNUC__) || defined(__clang__)
#define SKIPPY_DEPRECATED(message) __attribute__((deprecated(message)))
#else
#define SKIPPY_DEPRECATED(message)
#endif

enum skippy_feature {
    SKIPPY_FEATURE_RUNTIME_SLICE          = 1 << 0,
    SKIPPY_FEATURE_LAYER_PACKAGE          = 1 << 1,
    SKIPPY_FEATURE_ARTIFACT_SLICE         = 1 << 2,
    SKIPPY_FEATURE_MODEL_INTROSPECTION    = 1 << 3,
    SKIPPY_FEATURE_STATE_IMPORT_EXPORT    = 1 << 5,
    SKIPPY_FEATURE_TOKENIZE_DETOKENIZE    = 1 << 6,
    SKIPPY_FEATURE_ACTIVATION_FRAME       = 1 << 7,
    SKIPPY_FEATURE_NATIVE_KV_PAGE         = 1 << 8,
    SKIPPY_FEATURE_SESSION_RESET          = 1 << 9,
    SKIPPY_FEATURE_BATCH_VERIFY           = 1 << 10,
    SKIPPY_FEATURE_CHAT_TEMPLATE          = 1 << 11,
    SKIPPY_FEATURE_SAMPLING_CONFIG        = 1 << 12,
    SKIPPY_FEATURE_BATCH_VERIFY_FRAME     = 1 << 13,
    SKIPPY_FEATURE_RECURRENT_STATE        = 1 << 14,
    SKIPPY_FEATURE_LOGIT_BIAS             = 1 << 15,
    SKIPPY_FEATURE_SESSION_TRIM           = 1 << 16,
    SKIPPY_FEATURE_STAGE_PLAN             = 1 << 17,
    SKIPPY_FEATURE_PACKAGE_PART_LOAD      = 1 << 18,
    SKIPPY_FEATURE_GENERATION_SIGNALS     = 1 << 19,
    SKIPPY_FEATURE_EXTERNAL_MEDIA_PREFILL = 1 << 20,
    SKIPPY_FEATURE_CHAT_TEMPLATE_TOOLS    = 1 << 21,
    SKIPPY_FEATURE_CHAT_SAMPLING_GRAMMAR  = 1 << 22,
    SKIPPY_FEATURE_BACKEND_DEVICES        = 1 << 23,
    SKIPPY_FEATURE_RUNTIME_EVENTS         = 1 << 24,
    SKIPPY_FEATURE_NATIVE_MTP_N1          = 1 << 25,
    SKIPPY_FEATURE_NGRAM_CACHE_DRAFT      = 1 << 26,
    SKIPPY_FEATURE_MTP_MULTIMODAL         = 1 << 27,
    SKIPPY_FEATURE_ITERATION_BATCH        = 1 << 28,
    SKIPPY_FEATURE_ACTIVATION_BOUNDARY     = 1 << 29,
    SKIPPY_FEATURE_MODEL_SOURCE            = 1 << 30,
};

// Feature bit 31 is declared as a `uint64_t` constant, not an `enum
// skippy_feature` member: later bits in this same numbering space (32-36,
// added by sibling capability patches) exceed the range a C11 `enum`'s
// underlying type is portably guaranteed to hold (ISO C11 6.7.2.2p4 only
// requires the type to represent `int`-range values). Bit 31 is kept
// consistent with those siblings rather than split across two declaration
// styles. `skippy_abi_features()` already returns `uint64_t`, so callers
// combine these with the enum bits identically either way.
#define SKIPPY_FEATURE_RUNTIME_EVENT_REPORTER ((uint64_t)1 << 31)
#define SKIPPY_FEATURE_MODEL_LOAD_EVENTS_V2   ((uint64_t)1 << 32)
#define SKIPPY_FEATURE_KV_EVENTS              ((uint64_t)1 << 33)
#define SKIPPY_FEATURE_DEVICE_EVENTS          ((uint64_t)1 << 34)
#define SKIPPY_FEATURE_DIAGNOSTIC_EVENTS      ((uint64_t)1 << 35)
#define SKIPPY_FEATURE_UNLOAD_EVENTS          ((uint64_t)1 << 36)
/** @brief Full-model embedding, reranking, and encoder-decoder execution. */
#define SKIPPY_FEATURE_NON_CHAT_WORKLOADS     ((uint64_t)1 << 37)
#define SKIPPY_FEATURE_SYSTEM_ONE             ((uint64_t)1 << 38)

#define SKIPPY_FEATURE_CACHEGEN_KV_PAGE       (UINT64_C(1) << 39)
/** @brief Full-model Laya decision reads (`skippy/laya.h`). */
#define SKIPPY_FEATURE_LAYA_DECISIONS         (UINT64_C(1) << 40)
/** @brief Runtime Q8_0 K / Turbo4 V cache (ggml type 43); requires supported graph geometry and CPU or Metal placement. */
#define SKIPPY_FEATURE_TURBO4_KV              (UINT64_C(1) << 41)
/** @brief Runtime F16 K / Turbo4 V cache; requires supported graph geometry and CPU or Metal placement. */
#define SKIPPY_FEATURE_TURBO4_F16_KV          (UINT64_C(1) << 42)

enum skippy_status {
    SKIPPY_STATUS_OK                      = 0,
    SKIPPY_STATUS_ERROR                   = 1,
    SKIPPY_STATUS_INVALID_ARGUMENT        = 2,
    SKIPPY_STATUS_UNSUPPORTED             = 3,
    SKIPPY_STATUS_BUFFER_TOO_SMALL        = 4,
    SKIPPY_STATUS_IO_ERROR                = 5,
    SKIPPY_STATUS_MODEL_ERROR             = 6,
    SKIPPY_STATUS_RUNTIME_ERROR           = 7,
};

struct skippy_error {
    enum skippy_status status;
    const char * message;
};

struct skippy_abi_version {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

/** @brief Returns the native Skippy ABI version exported by this runtime. */
LLAMA_API struct skippy_abi_version skippy_abi_version(void);

/** @brief Returns the bitmask of optional capabilities compiled into the runtime. */
LLAMA_API uint64_t skippy_abi_features(void);

/** @brief Releases an error returned through an `out_error` parameter. */
LLAMA_API void skippy_error_free(struct skippy_error * error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_COMMON_H
