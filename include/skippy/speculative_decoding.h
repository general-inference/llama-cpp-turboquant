#ifndef SKIPPY_SPECULATIVE_DECODING_H
#define SKIPPY_SPECULATIVE_DECODING_H

/**
 * @file skippy/speculative_decoding.h
 * @brief Native MTP draft results and request-local n-gram drafting.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct skippy_ngram_cache;

#define SKIPPY_NATIVE_MTP_MAX_DRAFT_TOKENS 8

/** @brief A bounded native MTP proposal returned alongside target decoding. */
struct skippy_native_mtp_draft {
    uint32_t version;
    bool available;
    int32_t token_count;
    llama_token token_ids[SKIPPY_NATIVE_MTP_MAX_DRAFT_TOKENS];
    int64_t proposal_compute_us;
};

// Stateful, request-local adapter for llama.cpp's ngram-cache proposer.
// The cache is updated only with committed history. `continuation_prefix`
// supplies optional provisional tokens (such as native MTP output) used only
// while drafting; it never mutates the cache.
/** @brief Creates a request-local n-gram proposal cache. */
SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_create(
        uint16_t ngram_min,
        uint16_t ngram_max,
        struct skippy_ngram_cache ** out_cache,
        struct skippy_error ** out_error);

/** @brief Releases an n-gram proposal cache. */
SKIPPY_COMMON_API void skippy_ngram_cache_free(struct skippy_ngram_cache * cache);

/** @brief Replaces committed history in an n-gram proposal cache. */
SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_reset(
        struct skippy_ngram_cache * cache,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error);

/** @brief Appends newly committed tokens to an n-gram proposal cache. */
SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_append(
        struct skippy_ngram_cache * cache,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error);

/** @brief Drafts tokens from committed history plus an optional provisional prefix. */
SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_draft(
        struct skippy_ngram_cache * cache,
        const llama_token * continuation_prefix,
        size_t continuation_prefix_count,
        uint16_t max_draft_tokens,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_SPECULATIVE_DECODING_H
