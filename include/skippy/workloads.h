#ifndef SKIPPY_WORKLOADS_H
#define SKIPPY_WORKLOADS_H

/**
 * @file skippy/workloads.h
 * @brief Full-model embedding, reranking, and encoder-decoder execution.
 *
 * These operations deliberately reject filtered stage models. Their explicit
 * workload descriptor lets callers fail closed instead of inferring support
 * from a model name or architecture family.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct skippy_model;
struct skippy_session;

#define SKIPPY_WORKLOAD_INFO_V1_ABI_VERSION 1

enum skippy_workload_kind {
    SKIPPY_WORKLOAD_CAUSAL_GENERATION = 0,
    SKIPPY_WORKLOAD_EMBEDDING         = 1,
    SKIPPY_WORKLOAD_RERANK            = 2,
    SKIPPY_WORKLOAD_ENCODER_DECODER   = 3,
};

enum skippy_workload_pooling {
    SKIPPY_WORKLOAD_POOLING_UNSPECIFIED = -1,
    SKIPPY_WORKLOAD_POOLING_NONE        = 0,
    SKIPPY_WORKLOAD_POOLING_MEAN        = 1,
    SKIPPY_WORKLOAD_POOLING_CLS         = 2,
    SKIPPY_WORKLOAD_POOLING_LAST        = 3,
    SKIPPY_WORKLOAD_POOLING_RANK        = 4,
};

/** @brief Runtime-probed workload properties for an opened model. */
struct skippy_workload_info_v1 {
    uint32_t                     abi_version;
    uint32_t                     struct_size;
    enum skippy_workload_kind    kind;
    enum skippy_workload_pooling pooling;
    uint32_t                     output_dimensions;
    uint32_t                     classifier_outputs;
    bool                         has_encoder;
    bool                         has_decoder;
    bool                         full_model_only;
    uint8_t                      reserved0;
};

/**
 * @brief Describes the workload implemented by an opened model.
 *
 * The caller must initialize out_info->abi_version to SKIPPY_WORKLOAD_INFO_V1_ABI_VERSION
 * and out_info->struct_size to sizeof(struct skippy_workload_info_v1).
 * Incorrect values are rejected with SKIPPY_STATUS_INVALID_ARGUMENT.
 */
LLAMA_API enum skippy_status skippy_model_workload_info_v1(const struct skippy_model *      model,
                                                           struct skippy_workload_info_v1 * out_info,
                                                           struct skippy_error **           out_error);

/** @brief Computes one normalized embedding for a tokenized input. */
LLAMA_API enum skippy_status skippy_session_embed(struct skippy_session * session,
                                                  const llama_token *     token_ids,
                                                  size_t                  token_count,
                                                  float *                 output,
                                                  size_t                  output_capacity,
                                                  size_t *                out_dimensions,
                                                  struct skippy_error **  out_error);

/** @brief Computes one scalar relevance score for a query-document pair. */
LLAMA_API enum skippy_status skippy_session_rerank(struct skippy_session * session,
                                                   const char *            query,
                                                   const char *            document,
                                                   float *                 out_score,
                                                   size_t *                out_token_count,
                                                   struct skippy_error **  out_error);

/** @brief Encodes an encoder-decoder prompt and returns its first decoder token. */
LLAMA_API enum skippy_status skippy_session_encode_prompt(struct skippy_session * session,
                                                          const llama_token *     token_ids,
                                                          size_t                  token_count,
                                                          llama_token *           out_decoder_start_token,
                                                          struct skippy_error **  out_error);

#ifdef __cplusplus
}
#endif

#endif  // SKIPPY_WORKLOADS_H
