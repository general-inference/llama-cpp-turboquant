#ifndef SKIPPY_SAMPLING_H
#define SKIPPY_SAMPLING_H

/**
 * @file skippy/sampling.h
 * @brief Sampling parameters shared by single-token and batched decode calls.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SKIPPY_MAX_LOGIT_BIAS 256
#define SKIPPY_MAX_SAMPLERS 16
#define SKIPPY_MAX_DRY_SEQUENCE_BREAKERS 8
#define SKIPPY_MAX_DRY_SEQUENCE_BREAKER_BYTES 16
#define SKIPPY_SAMPLING_FLAG_ENABLED (1u << 0)
#define SKIPPY_SAMPLING_FLAG_IGNORE_EOS (1u << 1)

enum skippy_sampler_type {
    SKIPPY_SAMPLER_PENALTIES = 1,
    SKIPPY_SAMPLER_DRY = 2,
    SKIPPY_SAMPLER_TOP_N_SIGMA = 3,
    SKIPPY_SAMPLER_TOP_K = 4,
    SKIPPY_SAMPLER_TYPICAL_P = 5,
    SKIPPY_SAMPLER_TOP_P = 6,
    SKIPPY_SAMPLER_MIN_P = 7,
    SKIPPY_SAMPLER_XTC = 8,
    SKIPPY_SAMPLER_TEMPERATURE = 9,
};


/** @brief Versioned sampling configuration consumed by Skippy decode entrypoints. */
struct skippy_sampling_config {
    uint32_t version;
    uint32_t flags;
    uint32_t seed;
    int32_t top_k;
    int32_t penalty_last_n;
    float temperature;
    float top_p;
    float presence_penalty;
    float frequency_penalty;
    float repeat_penalty;
    uint32_t logit_bias_count;
    float min_p;
    float typical_p;
    float top_nsigma;
    float dynatemp_range;
    float dynatemp_exponent;
    float dry_multiplier;
    float dry_base;
    int32_t dry_allowed_length;
    int32_t dry_penalty_last_n;
    float xtc_probability;
    float xtc_threshold;
    int32_t mirostat_mode;
    float mirostat_entropy;
    float mirostat_learning_rate;
    uint32_t sampler_count;
    uint32_t samplers[SKIPPY_MAX_SAMPLERS];
    uint32_t ignore_eos;
    uint32_t dry_sequence_breaker_count;
    char dry_sequence_breakers[SKIPPY_MAX_DRY_SEQUENCE_BREAKERS][SKIPPY_MAX_DRY_SEQUENCE_BREAKER_BYTES];
    llama_logit_bias logit_bias[SKIPPY_MAX_LOGIT_BIAS];
    /** -1 disables enforcement; 0 closes reasoning immediately; positive values cap reasoning tokens. */
    int32_t reasoning_budget_tokens;
};

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_SAMPLING_H
