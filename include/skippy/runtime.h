#ifndef SKIPPY_RUNTIME_H
#define SKIPPY_RUNTIME_H

/**
 * @file skippy/runtime.h
 * @brief Model loading, session lifecycle, and llama.cpp context access.
 */

#include "common.h"
#include "events.h"
#include "sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

enum skippy_load_mode {
    SKIPPY_LOAD_MODE_RUNTIME_SLICE        = 0,
    SKIPPY_LOAD_MODE_LAYER_PACKAGE        = 1,
    SKIPPY_LOAD_MODE_ARTIFACT_SLICE       = 2,
};

/** @brief Selects whether a target loads no MTP tensors, integrated MTP tensors, or an external MTP sidecar. */
enum skippy_mtp_source {
    SKIPPY_MTP_SOURCE_DISABLED   = 0,
    SKIPPY_MTP_SOURCE_INTEGRATED = 1,
    SKIPPY_MTP_SOURCE_EXTERNAL   = 2,
};

#define SKIPPY_GLM_DSA_POLICY_PROFILE_NONE     0
#define SKIPPY_GLM_DSA_POLICY_PROFILE_V1       1

#define SKIPPY_GLM_DSA_POLICY_DIRECT_SPARSE_ATTN                  (UINT32_C(1) << 0)
#define SKIPPY_GLM_DSA_POLICY_DIRECT_SPARSE_PREFILL               (UINT32_C(1) << 1)
#define SKIPPY_GLM_DSA_POLICY_DISABLE_COMPACT_FLASH_ATTN          (UINT32_C(1) << 2)
#define SKIPPY_GLM_DSA_POLICY_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL (UINT32_C(1) << 3)

struct skippy_model;
struct skippy_session;
struct skippy_activation_boundary_desc;

/** @brief Model, stage, memory, cache, device, and family-policy load options. */
struct skippy_runtime_config {
    int32_t stage_index;
    int32_t layer_start;
    int32_t layer_end;
    int32_t ctx_size;
    int32_t lane_count;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_threads;
    int32_t n_threads_batch;
    int32_t n_gpu_layers;
    bool has_mmap_override;
    bool use_mmap;
    bool use_mlock;
    // Cache types: 0 selects F32, a positive value selects that ggml type,
    // and a negative value keeps llama.cpp's derived default.
    int32_t cache_type_k;
    int32_t cache_type_v;
    int32_t flash_attn_type;

    enum skippy_load_mode load_mode;

    bool disable_repack;
    bool use_mmap_prefetch;
    bool use_mmap_buffer;
    // Exact tensor closure for a graph-admitted stage. An empty closure means
    // an unsplit full-model load; there is no range-based loader fallback.
    const char * const * resident_tensor_names;
    size_t resident_tensor_name_count;
    /** @brief Planner-produced execution dependency contract for shared residency. */
    const char * execution_contract;
    const char * const * activation_import_identities;
    size_t activation_import_identity_count;
    const char * const * activation_import_bindings;
    const char * const * activation_export_identities;
    size_t activation_export_identity_count;
    const char * const * activation_export_bindings;
    enum skippy_mtp_source mtp_source;

    // Optional ggml backend device name, for example "CUDA0", "MTL0",
    // "Vulkan1", or "CPU". When set, skippy loads the model only on that
    // backend device instead of relying on llama.cpp default device ordering.
    const char * selected_backend_device;

    // Optional GLM-DSA execution policy resolved from package generation
    // metadata. Zero values preserve llama.cpp built-in/env behavior.
    int32_t  glm_dsa_policy_profile;
    uint32_t glm_dsa_policy_flags;
    int32_t  glm_dsa_short_prefill_max_tokens;
    int32_t  glm_dsa_direct_sparse_decode_max_top_k;
    uint64_t glm_dsa_dense_sparse_mask_max_bytes;
    int32_t  glm_dsa_compact_flash_min_kv;

    // Tri-state KV/session controls resolved from `model_fit` config.
    /** @brief Whether K/V tensors are offloaded to the compute backend (mirrors upstream `offload_kqv`). */
    int32_t kv_offload;
    /** @brief Whether the KV cache is unified across sequences/lanes (mirrors upstream `kv_unified`). AUTO derives the setting from lane count; recurrent and hybrid models force it true. */
    int32_t kv_unified;
    /** @brief Whether sliding-window-attention layers use a full, unshifted cache window (mirrors upstream `swa_full`). */
    int32_t swa_full;

    // Model-load and context-creation controls resolved from `hardware`
    // config.
    /** @brief Whether the backend offloads host tensor operations to device (mirrors upstream `op_offload`). SKIPPY_TRISTATE_AUTO preserves llama.cpp's derived default (enabled); FALSE/TRUE force the value. */
    int32_t op_offload;
    /** @brief Whether model loading bypasses the host buffer, allowing extra buffers to be used (mirrors upstream `no_host`). */
    bool no_host_buffer;
    /** @brief Whether to validate model tensor data while loading (mirrors upstream `check_tensors`). */
    bool check_tensors;
    /** @brief Forces direct I/O for model loading (mirrors upstream `LLAMA_LOAD_MODE_DIRECT_IO`), taking precedence over `use_mmap`/`use_mlock` when set. */
    bool use_direct_io;
    /** @brief Whether `main_gpu` below should override llama.cpp's derived GPU index. */
    bool has_main_gpu_override;
    /** @brief Explicit GPU index used for the entire model when `split_mode` resolves to `none`. Ignored unless `has_main_gpu_override` is set, and overridden by `selected_backend_device` when that is also set. */
    int32_t main_gpu;
    /** @brief How to split the model across multiple GPUs (mirrors upstream `llama_split_mode`). SKIPPY_TRISTATE_AUTO (-1) preserves llama.cpp's derived default; 0=none, 1=layer, 2=row, 3=tensor. */
    int32_t split_mode;
};

/** @brief Returns a C-safe runtime configuration with all derived-default controls set to auto. */
LLAMA_API struct skippy_runtime_config skippy_runtime_config_default(void);

/** @brief Sentinel for `skippy_runtime_config` tri-state fields: keep the derived default. */
#define SKIPPY_TRISTATE_AUTO  (-1)
/** @brief Sentinel for `skippy_runtime_config` tri-state fields: force false. */
#define SKIPPY_TRISTATE_FALSE (0)
/** @brief Sentinel for `skippy_runtime_config` tri-state fields: force true. */
#define SKIPPY_TRISTATE_TRUE  (1)

/** @brief Opens a direct GGUF, runtime slice, layer package, or artifact slice. */
LLAMA_API enum skippy_status skippy_model_open(
        const char * path,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error);

/** @brief Opens a model and reports operation-scoped lifecycle events. */
LLAMA_API enum skippy_status skippy_model_open_with_events(
        const char * path,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error);

/** @brief Attaches a native MTP draft model to an opened target model. */
LLAMA_API enum skippy_status skippy_model_attach_mtp_draft_model(
        struct skippy_model * target_model,
        const char * path,
        const struct skippy_runtime_config * config,
        struct skippy_error ** out_error);

/** @brief Opens an ordered set of GGUF package parts as one model. */
LLAMA_API enum skippy_status skippy_model_open_from_parts(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error);

/** @brief Opens ordered GGUF parts and reports lifecycle events. */
LLAMA_API enum skippy_status skippy_model_open_from_parts_with_events(
        const char * const * paths,
        size_t path_count,
        const struct skippy_runtime_config * config,
        const struct skippy_runtime_event_reporter_v1 * reporter,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error);

/** @brief Releases an opened model handle. */
LLAMA_API enum skippy_status skippy_model_free(
        struct skippy_model * model,
        struct skippy_error ** out_error);

/** @brief Returns the underlying llama.cpp model for native read-only integrations. */
LLAMA_API const struct llama_model * skippy_model_llama_model(
        const struct skippy_model * model);

/** @brief Reads the configured stage graph's observed output boundary descriptor. */
LLAMA_API bool skippy_model_output_activation_boundary(
        const struct skippy_model * model,
        struct skippy_activation_boundary_desc * out_desc);

/** @brief Reads the configured stage graph's observed input boundary descriptor. */
LLAMA_API bool skippy_model_input_activation_boundary(
        const struct skippy_model * model,
        struct skippy_activation_boundary_desc * out_desc);

/** @brief Creates a fresh inference session for an opened model. */
LLAMA_API enum skippy_status skippy_session_create(
        struct skippy_model * model,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error);

/** @brief Creates a session restored from a resident prefix-cache sequence. */
LLAMA_API enum skippy_status skippy_session_create_from_resident_prefix(
        struct skippy_model * model,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error);

/** @brief Returns the underlying llama.cpp context for native integrations. */
LLAMA_API struct llama_context * skippy_session_llama_context(
        struct skippy_session * session);

/** @brief Returns the session's current token position. */
LLAMA_API int32_t skippy_session_position(
        const struct skippy_session * session);

/** @brief Returns the native batch width currently configured for the session. */
LLAMA_API int32_t skippy_session_batch_size(
        const struct skippy_session * session);

/** @brief Returns the llama.cpp sequence ID claimed by the session. */
LLAMA_API int32_t skippy_session_sequence_id(
        const struct skippy_session * session);

/** @brief Begins a caller-managed external decode section. */
LLAMA_API enum skippy_status skippy_session_begin_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error);

/** @brief Ends a caller-managed external decode section. */
LLAMA_API enum skippy_status skippy_session_end_external_decode(
        struct skippy_session * session,
        struct skippy_error ** out_error);

/** @brief Sets the session token position used by the next execution call. */
LLAMA_API enum skippy_status skippy_session_set_position(
        struct skippy_session * session,
        int32_t n_past,
        struct skippy_error ** out_error);

/** @brief Restores token history after importing cached model state.
 * The count must equal the imported session position and the IDs must describe
 * that prefix. Call before suffix prefill or sampling. Rebuilds sampler/grammar
 * history while retaining the current request's chat sampling configuration.
 */
LLAMA_API enum skippy_status skippy_session_restore_token_history(
        struct skippy_session * session,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error);

/** @brief Samples the current logits using the supplied sampling configuration. */
LLAMA_API enum skippy_status skippy_session_sample_current(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        llama_token * out_predicted_token,
        struct skippy_error ** out_error);

/** @brief Configures template-derived grammar and sampling metadata for a session. */
LLAMA_API enum skippy_status skippy_session_configure_chat_sampling(
        struct skippy_session * session,
        const struct skippy_sampling_config * sampling,
        const char * metadata_json,
        uint64_t prompt_token_count,
        struct skippy_error ** out_error);

/** @brief Clears session state so the session can be reused. */
LLAMA_API enum skippy_status skippy_session_reset(
        struct skippy_session * session,
        struct skippy_error ** out_error);

/** @brief Releases an inference session handle. */
LLAMA_API enum skippy_status skippy_session_free(
        struct skippy_session * session,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_RUNTIME_H
