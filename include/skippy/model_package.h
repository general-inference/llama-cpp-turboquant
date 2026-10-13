#ifndef SKIPPY_MODEL_PACKAGE_H
#define SKIPPY_MODEL_PACKAGE_H

/**
 * @file skippy/model_package.h
 * @brief Inspects GGUF tensors and writes metadata and multi-part packages.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

enum skippy_tensor_role {
    SKIPPY_TENSOR_ROLE_UNKNOWN            = 0,
    SKIPPY_TENSOR_ROLE_METADATA           = 1,
    SKIPPY_TENSOR_ROLE_TOKENIZER          = 2,
    SKIPPY_TENSOR_ROLE_EMBEDDING          = 3,
    SKIPPY_TENSOR_ROLE_LAYER              = 4,
    SKIPPY_TENSOR_ROLE_FINAL_NORM         = 5,
    SKIPPY_TENSOR_ROLE_OUTPUT             = 6,
};

struct skippy_model_info;

/** @brief Metadata for one tensor discovered in a GGUF model. */
struct skippy_tensor_info {
    const char * name;
    int32_t layer_index;
    enum skippy_tensor_role role;
    uint32_t ggml_type;
    uint64_t byte_size;
    uint64_t element_count;
};

/** @brief Opens a model for tensor-level inspection without creating a runtime. */
LLAMA_API enum skippy_status skippy_model_info_open(
        const char * path,
        struct skippy_model_info ** out_info,
        struct skippy_error ** out_error);

/** @brief Releases model-introspection state. */
LLAMA_API enum skippy_status skippy_model_info_free(
        struct skippy_model_info * info,
        struct skippy_error ** out_error);

/** @brief Returns the number of tensors visible to model introspection. */
LLAMA_API enum skippy_status skippy_model_info_tensor_count(
        struct skippy_model_info * info,
        size_t * out_count,
        struct skippy_error ** out_error);

/** @brief Copies one tensor's metadata into `out_tensor`. */
LLAMA_API enum skippy_status skippy_model_info_tensor_at(
        struct skippy_model_info * info,
        size_t index,
        struct skippy_tensor_info * out_tensor,
        struct skippy_error ** out_error);

/** @brief Writes complete GGUF model metadata and tensor descriptors without payload bytes. */
LLAMA_API enum skippy_status skippy_write_gguf_metadata_from_parts(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error);

/** @brief Composes ordered GGUF parts into one output GGUF file. */
LLAMA_API enum skippy_status skippy_write_gguf_from_parts(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error);

/**
 * @brief Materialize a GGUF while consuming scratch input parts.
 *
 * Each input is unlinked after its last selected tensor is copied. If the
 * operation fails, earlier inputs may already be deleted and the output may be
 * incomplete; later inputs remain available. Use only with disposable staging
 * parts, never published package files that must survive.
 */
LLAMA_API enum skippy_status skippy_write_gguf_from_parts_consuming(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_MODEL_PACKAGE_H
