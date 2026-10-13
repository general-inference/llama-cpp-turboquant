#include "skippy/sampling_internal.h"
#include "skippy/errors.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

enum skippy_status skippy_store_sampled_token(
        llama_token token, llama_token * output, skippy_error ** out_error) {
    if (token < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "sampling produced no allowed token");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    *output = token;
    return skippy_success(out_error);
}

std::vector<llama_logit_bias> skippy_add_vocab_suppress_token_biases(
        std::vector<llama_logit_bias> biases,
        const llama_token * suppress_tokens,
        int32_t suppress_token_count) {
    if (suppress_tokens == nullptr || suppress_token_count <= 0) {
        return biases;
    }
    const llama_token * suppress_end = suppress_tokens + suppress_token_count;
    biases.erase(std::remove_if(biases.begin(), biases.end(), [=](const llama_logit_bias & bias) {
        return std::find(suppress_tokens, suppress_end, bias.token) != suppress_end;
    }), biases.end());
    biases.reserve(biases.size() + suppress_token_count);
    for (const llama_token * token = suppress_tokens; token != suppress_end; ++token) {
        biases.push_back({*token, -INFINITY});
    }
    return biases;
}

llama_token skippy_greedy_sample_allowed_logits(
        const float * logits,
        const llama_vocab * vocab,
        int32_t n_vocab,
        const llama_token * suppress_tokens,
        int32_t suppress_token_count,
        bool ignore_eog) {
    if (logits == nullptr || n_vocab <= 0) {
        return LLAMA_TOKEN_NULL;
    }
    std::vector<uint8_t> suppressed;
    if (suppress_tokens != nullptr && suppress_token_count > 0) {
        suppressed.resize(static_cast<size_t>(n_vocab), 0);
        for (int32_t i = 0; i < suppress_token_count; ++i) {
            const llama_token token = suppress_tokens[i];
            if (token >= 0 && token < n_vocab) {
                suppressed[static_cast<size_t>(token)] = 1;
            }
        }
    }

    llama_token best = LLAMA_TOKEN_NULL;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (llama_token token = 0; token < n_vocab; ++token) {
        if ((!suppressed.empty() && suppressed[static_cast<size_t>(token)] != 0) ||
                (ignore_eog && vocab != nullptr && llama_vocab_is_eog(vocab, token))) {
            continue;
        }
        if (best == LLAMA_TOKEN_NULL || logits[token] > best_logit) {
            best_logit = logits[token];
            best = token;
        }
    }
    return best;
}
