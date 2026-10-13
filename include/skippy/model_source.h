#ifndef SKIPPY_MODEL_SOURCE_H
#define SKIPPY_MODEL_SOURCE_H

/**
 * @file skippy/model_source.h
 * @brief Callback-backed model construction and load-time quantization.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct skippy_model;
struct skippy_runtime_config;

#define SKIPPY_MODEL_TENSOR_SOURCE_V1_ABI_VERSION 1

/** @brief Supplies one canonical model tensor as F32 values during model construction.
 *
 * The callback is invoked synchronously and serially while
 * `skippy_model_open_from_source` is active. `tensor_name` and `destination`
 * are borrowed only for the callback duration; implementations must write
 * exactly `element_count` values and must not retain either pointer. On
 * failure, `out_message` may receive a borrowed NUL-terminated message that
 * remains valid until `skippy_model_open_from_source` returns.
 */
typedef enum skippy_status (*skippy_model_read_tensor_f32_callback)(
        const char * tensor_name,
        float * destination,
        size_t element_count,
        const char ** out_message,
        void * user_data);

/** @brief One tensor's importance weights for low-bit quantization.
 *
 * `tensor_name` identifies the canonical llama.cpp tensor. `values` contains
 * exactly the weights expected by llama.cpp for that tensor. Both buffers are
 * borrowed for the synchronous model-open call and are never retained.
 */
struct skippy_model_imatrix_entry_v1 {
    const char * tensor_name;
    const float * values;
    size_t value_count;
};

/** @brief Versioned callback source for direct checkpoint loading.
 *
 * The source and `user_data` are borrowed for the synchronous duration of
 * `skippy_model_open_from_source` and are never retained by the model.
 */
struct skippy_model_tensor_source_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    skippy_model_read_tensor_f32_callback read_tensor_f32;
    void * user_data;
    const struct skippy_model_imatrix_entry_v1 * imatrix;
    size_t imatrix_count;
};

/** @brief Opens a model from metadata-only GGUF and callback-supplied tensor values.
 *
 * `metadata_gguf`, `source`, and `source->user_data` must remain valid for
 * this synchronous call and are not retained; tensor callbacks execute
 * synchronously before the function returns. `quantization_ftype` is a
 * llama_ftype value, or -1 to preserve the tensor types declared by
 * `metadata_gguf`. On success, `out_model` owns the resulting model and must
 * be released with `skippy_model_free`.
 */
LLAMA_API enum skippy_status skippy_model_open_from_source(
        const void * metadata_gguf,
        size_t metadata_gguf_size,
        const struct skippy_model_tensor_source_v1 * source,
        int32_t quantization_ftype,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_MODEL_SOURCE_H
