#ifndef SKIPPY_LAYA_H
#define SKIPPY_LAYA_H

/**
 * @file skippy/laya.h
 * @brief Full-model Laya decision reads (encoder + typed decision head).
 *
 * Laya is non-autoregressive: one forward pass scores every option of every
 * question. It does not load through `skippy_model_open`; it owns its GGUF,
 * tokenizer, and execution context, which runs on the CPU backend or on a
 * device the caller names, with CPU fallback. Callers build the per-question token
 * sequences (see `skippy_laya_tokenize`) and read raw scorer and action-head
 * logits back; temperature and softmax stay on the caller side.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SKIPPY_LAYA_INFO_V1_ABI_VERSION 1
#define SKIPPY_LAYA_QTYPE_COUNT 3
#define SKIPPY_LAYA_MEMORY_V1_ABI_VERSION 1
/**
 * Longest question sequence a Laya GGUF may declare (`laya.max_len`). A model
 * declaring more is refused at open, because each pass builds dense attention
 * masks over up to `max_len` tokens.
 */
#define SKIPPY_LAYA_MAX_SEQUENCE_TOKENS 4096

struct skippy_laya_model;

/** @brief Question types, in the order the decision head was trained with. */
enum skippy_laya_qtype {
    SKIPPY_LAYA_QTYPE_CHOICE = 0,
    SKIPPY_LAYA_QTYPE_SCORE  = 1,
    SKIPPY_LAYA_QTYPE_NOUL   = 2,
};

/** @brief Sequence budgets, special tokens, and temperatures of an opened Laya model. */
struct skippy_laya_info_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t  max_len;          /**< maximum tokens per question sequence */
    int32_t  head_max_len;     /**< token budget for the head and options region */
    int32_t  max_markers;      /**< maximum options per question */
    int32_t  n_act;            /**< action-head classes */
    int32_t  cls_token_id;
    int32_t  sep_token_id;
    int32_t  mask_token_id;
    int32_t  n_embd;
    int32_t  n_layer;
    int64_t  parameter_count;
    /** Per-qtype temperature from the GGUF; 0 when absent. */
    float    temperature[SKIPPY_LAYA_QTYPE_COUNT];
};

/** @brief Measured memory of an opened Laya model. */
struct skippy_laya_memory_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t weights_bytes;     /**< the weight buffer */
    uint64_t compute_bytes;     /**< scheduler buffers after a full-length read */
    uint64_t host_scratch_bytes; /**< host-side attention masks one full pass builds */
    bool     on_accelerator;    /**< weights live on a non-CPU device */
};

/** @brief One question sequence inside a packed read. */
struct skippy_laya_sequence {
    size_t  token_offset;   /**< first token in the packed token array */
    size_t  token_count;
    int32_t qtype;          /**< `enum skippy_laya_qtype` */
    size_t  marker_offset;  /**< first marker in the packed marker array */
    size_t  marker_count;   /**< 1..max_markers option markers */
};

/**
 * @brief Opens a Laya GGUF with its own loader; not a `skippy_model`.
 *
 * `device` selects where the weights live: NULL or "" is the CPU backend,
 * "auto" is the first GPU device when one exists, and any other value names a
 * ggml backend device such as "CUDA0" or "MTL0". On a GPU the CPU backend
 * stays behind it for any unsupported op. `n_threads` sizes the CPU backend.
 * Fails
 * with SKIPPY_STATUS_MODEL_ERROR when the GGUF declares a `laya.max_len` above
 * SKIPPY_LAYA_MAX_SEQUENCE_TOKENS or a head budget larger than it.
 */
LLAMA_API enum skippy_status skippy_laya_model_open(
        const char * model_path,
        int32_t n_threads,
        const char * device,
        struct skippy_laya_model ** out_model,
        struct skippy_error ** out_error);

/** @brief Releases a model returned by `skippy_laya_model_open`. */
LLAMA_API void skippy_laya_model_free(struct skippy_laya_model * model);

/**
 * @brief Reports the model's sequence budgets, special tokens, and temperatures.
 *
 * `out_info` must be initialized with its `struct_size` and `abi_version`.
 */
LLAMA_API enum skippy_status skippy_laya_model_info_v1(
        const struct skippy_laya_model * model,
        struct skippy_laya_info_v1 * out_info,
        struct skippy_error ** out_error);

/**
 * @brief Measures the model's memory at its largest read.
 *
 * The first call runs one read of a full `max_len`-token sequence, so the
 * scheduler reserves the buffers the biggest pass needs, then reports them
 * with the weight buffer. `out_info` must be initialized with its
 * `struct_size` and `abi_version`.
 */
LLAMA_API enum skippy_status skippy_laya_model_memory_v1(
        struct skippy_laya_model * model,
        struct skippy_laya_memory_v1 * out_info,
        struct skippy_error ** out_error);

/**
 * @brief Tokenizes UTF-8 text with the model's own Metaspace + BPE tokenizer.
 *
 * No special tokens are added. Returns SKIPPY_STATUS_BUFFER_TOO_SMALL with
 * `out_token_count` set to the required length when `capacity` is short.
 */
LLAMA_API enum skippy_status skippy_laya_tokenize(
        const struct skippy_laya_model * model,
        const char * text,
        size_t text_len,
        int32_t * out_tokens,
        size_t capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);

/**
 * @brief Scores every question's options in one packed forward pass.
 *
 * `marker_positions` are sequence-local token indices. Outputs are row-major
 * per sequence: `out_logits` holds `sequence_count * max_markers` scorer
 * logits (unused markers are 0) and `out_act_logits` holds
 * `sequence_count * n_act` action-head logits. Calls are serialized per
 * model. Sequences are packed into passes of at most `max_len` tokens (see
 * `skippy_laya_model_info_v1`), which open caps at
 * SKIPPY_LAYA_MAX_SEQUENCE_TOKENS, so a pass's attention masks are bounded by
 * the model's own sequence budget.
 */
LLAMA_API enum skippy_status skippy_laya_read(
        struct skippy_laya_model * model,
        const int32_t * tokens,
        size_t token_count,
        const int32_t * marker_positions,
        size_t marker_count,
        const struct skippy_laya_sequence * sequences,
        size_t sequence_count,
        float * out_logits,
        size_t logits_capacity,
        float * out_act_logits,
        size_t act_logits_capacity,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_LAYA_H
