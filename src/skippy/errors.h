#pragma once

#include "skippy/common.h"

#include <cstdlib>
#include <cstring>
#include <exception>

static inline skippy_error * skippy_make_error(enum skippy_status status, const char * message) {
    skippy_error * error = new skippy_error{};
    error->status = status;

    if (message == nullptr) {
        error->message = nullptr;
        return error;
    }

    const size_t len = std::strlen(message);
    char * owned = static_cast<char *>(std::malloc(len + 1));
    if (owned == nullptr) {
        error->status = SKIPPY_STATUS_ERROR;
        error->message = nullptr;
        return error;
    }

    std::memcpy(owned, message, len + 1);
    error->message = owned;
    return error;
}

static inline void skippy_set_error(
        skippy_error ** out_error,
        enum skippy_status status,
        const char * message) {
    if (out_error != nullptr) {
        *out_error = skippy_make_error(status, message);
    }
}

static inline enum skippy_status skippy_success(skippy_error ** out_error) {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    return SKIPPY_STATUS_OK;
}
// A C++ exception must never unwind out of this ABI into Rust: Rust cannot
// catch a foreign exception, so its personality routine prints
// "fatal runtime error: Rust cannot catch foreign exceptions, aborting" and
// terminates the process. Every entry point that can execute graph
// code therefore contains an escaping exception here and reports it as an
// ordinary skippy failure. Pass nullptr for `out_error` when the caller
// reports the failure itself.
template <typename Body>
static inline bool skippy_contain_native_failure(skippy_error ** out_error, Body && body) {
    try {
        body();
        return true;
    } catch (const std::exception & exception) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, exception.what());
        return false;
    } catch (...) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "native graph execution failure without a diagnostic");
        return false;
    }
}

