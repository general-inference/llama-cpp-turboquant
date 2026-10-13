#ifndef SKIPPY_STATE_H
#define SKIPPY_STATE_H

/**
 * @file skippy/state.h
 * @brief Moves KV, recurrent, checkpoint, and resident-prefix state.
 */

#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Complete cache representations supported by the loaded memory adapters. */
enum skippy_memory_cache_capability {
    SKIPPY_MEMORY_CACHE_RESIDENT = 1 << 0,
    SKIPPY_MEMORY_CACHE_KV_RECURRENT = 1 << 1,
    SKIPPY_MEMORY_CACHE_FULL_STATE = 1 << 2,
};

/** @brief Returns complete cache representations admitted by the built graph and loaded memory.
 * Graph state effects restrict the loaded memory adapters' complete export capabilities.
 * Full-state serialization is admitted explicitly too. Zero means no complete
 * cache representation is available; callers cannot supply capability labels.
 */
LLAMA_API uint32_t skippy_model_memory_cache_capabilities(const struct skippy_model * model);

enum skippy_kv_page_flag {
    SKIPPY_KV_PAGE_FLAG_V_TRANSPOSED      = 1 << 0,
    SKIPPY_KV_PAGE_FLAG_HAS_K_IDX         = 1 << 1,
};

enum skippy_kv_page_codec {
    SKIPPY_KV_PAGE_CODEC_SINGLE_V1         = 1,
    SKIPPY_KV_PAGE_CODEC_ISWA_COMPOSITE_V1 = 2,
};

enum skippy_kv_page_component_role {
    SKIPPY_KV_PAGE_COMPONENT_BASE = 1,
    SKIPPY_KV_PAGE_COMPONENT_SWA  = 2,
};

#define SKIPPY_CACHEGEN_RECORD_V1_ABI_VERSION 1

enum skippy_cachegen_record_kind {
    SKIPPY_CACHEGEN_RECORD_F16            = 0,
    SKIPPY_CACHEGEN_RECORD_EXACT          = 1,
    SKIPPY_CACHEGEN_RECORD_F16_TRANSPOSED = 2,
    SKIPPY_CACHEGEN_RECORD_F32            = 3,
    SKIPPY_CACHEGEN_RECORD_F32_TRANSPOSED = 4,
    SKIPPY_CACHEGEN_RECORD_Q8_0           = 5,
    SKIPPY_CACHEGEN_RECORD_Q4_0           = 6,
};

/** @brief Describes one validated CacheGen record and its logical page destination. */
struct skippy_cachegen_record_v1 {
    uint32_t abi_version;
    uint32_t kind;
    uint32_t element_bytes;
    uint32_t reserved0;
    uint64_t output_offset;
    uint64_t decoded_bytes;
    uint64_t token_count;
    uint64_t token_start;
    uint64_t total_tokens;
    const void * payload;
    size_t payload_bytes;
};

/** @brief Describes one independently ranged cache in a composite KV page. */
struct skippy_kv_page_component_desc {
    uint32_t version;
    uint32_t role;
    uint64_t token_start;
    uint64_t token_count;
    uint32_t layer_count;
    uint32_t k_type;
    uint32_t v_type;
    uint32_t k_row_bytes;
    uint32_t v_row_bytes;
    uint32_t v_element_bytes;
    uint32_t k_idx_row_bytes;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint64_t flags;
};

/** @brief Describes the layout and ownership of an exported KV page payload. */
struct skippy_kv_page_desc {
    uint32_t version;
    int32_t layer_start;
    int32_t layer_end;
    uint64_t token_start;
    uint64_t token_count;
    uint32_t layer_count;
    uint32_t k_type;
    uint32_t v_type;
    uint32_t k_row_bytes;
    uint32_t v_row_bytes;
    uint32_t v_element_bytes;
    uint32_t k_idx_row_bytes;
    uint64_t payload_bytes;
    uint64_t flags;
    uint32_t codec;
    uint32_t component_count;
    struct skippy_kv_page_component_desc components[2];
};

/** @brief Exports the selected layer-range state for staged transfer. */
LLAMA_API enum skippy_status skippy_export_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

/** @brief Imports a layer-range state payload into a session. */
LLAMA_API enum skippy_status skippy_import_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

/** @brief Exports full KV state for the selected layer range. */
LLAMA_API enum skippy_status skippy_export_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

/** @brief Exports recurrent state for models that maintain recurrent memory. */
LLAMA_API enum skippy_status skippy_export_recurrent_state(
        struct skippy_session * session,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

/** @brief Imports full KV state for the selected layer range. */
LLAMA_API enum skippy_status skippy_import_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

/** @brief Imports recurrent state into a session. */
LLAMA_API enum skippy_status skippy_import_recurrent_state(
        struct skippy_session * session,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

/** @brief Trims the session's token history and associated state. */
LLAMA_API enum skippy_status skippy_trim_session(
        struct skippy_session * session,
        uint64_t token_count,
        struct skippy_error ** out_error);

/** @brief Retires a verified checkpoint after its tokens are accepted. */
LLAMA_API enum skippy_status skippy_retire_verify_checkpoint(
        struct skippy_session * session,
        uint64_t token_start,
        uint64_t token_count,
        struct skippy_error ** out_error);

/** @brief Exports a token/layer KV page with an explicit layout descriptor. */
LLAMA_API enum skippy_status skippy_export_kv_page(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        uint64_t token_start,
        uint64_t token_count,
        struct skippy_kv_page_desc * out_desc,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error);

/** @brief Imports a described KV page. Composite pages require a fresh session and are cleared on failure. */
LLAMA_API enum skippy_status skippy_import_kv_page(
        struct skippy_session * session,
        const struct skippy_kv_page_desc * desc,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error);

/** @brief Imports validated CacheGen records directly into resident KV storage. */
LLAMA_API enum skippy_status skippy_import_cachegen_kv_page_v1(
        struct skippy_session * session,
        const struct skippy_kv_page_desc * desc,
        const struct skippy_cachegen_record_v1 * records,
        size_t record_count,
        struct skippy_error ** out_error);

/** @brief Saves a session prefix into a resident cache sequence. */
LLAMA_API enum skippy_status skippy_session_save_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        uint64_t token_count,
        struct skippy_error ** out_error);

/** @brief Restores a session prefix from a resident cache sequence. */
LLAMA_API enum skippy_status skippy_session_restore_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error);

/** @brief Reports physically occupied cells in the session's unified KV cache. */
LLAMA_API enum skippy_status skippy_session_memory_used_cells(
        struct skippy_session * session,
        uint64_t * out_used_cells,
        struct skippy_error ** out_error);

/** @brief Drops a llama.cpp sequence from the session's cache. */
LLAMA_API enum skippy_status skippy_session_drop_sequence(
        struct skippy_session * session,
        int32_t seq_id,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_STATE_H
