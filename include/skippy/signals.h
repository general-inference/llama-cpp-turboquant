#ifndef SKIPPY_SIGNALS_H
#define SKIPPY_SIGNALS_H

/**
 * @file skippy/signals.h
 * @brief Generation uncertainty and repetition signals for routing or policy.
 */

#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Statistics for the most recently sampled token. */
struct skippy_token_signal {
    float entropy;
    float top_logprob;
    float second_logprob;
    float margin;
    int32_t top_token;
    int32_t second_token;
};

/** @brief Aggregated uncertainty and repetition statistics over a token window. */
struct skippy_generation_signal_window {
    uint32_t token_count;
    float mean_entropy;
    float max_entropy;
    float mean_margin;
    float min_margin;
    uint32_t high_entropy_count;
    uint32_t repetition_count;
};

/** @brief Returns signal statistics for the most recently sampled token. */
LLAMA_API enum skippy_status skippy_session_last_token_signal(
        struct skippy_session * session,
        struct skippy_token_signal * out_signal,
        struct skippy_error ** out_error);

/** @brief Computes signal statistics over the requested recent token window. */
LLAMA_API enum skippy_status skippy_session_signal_window(
        struct skippy_session * session,
        uint32_t window_tokens,
        struct skippy_generation_signal_window * out_window,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_SIGNALS_H
