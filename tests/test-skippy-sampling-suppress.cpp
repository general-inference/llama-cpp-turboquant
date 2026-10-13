#include "../src/skippy/sampling_internal.h"

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool suppressed(const llama_logit_bias & bias) {
    return std::isinf(bias.bias) && std::signbit(bias.bias);
}

int main() {
    const llama_token suppress_tokens[] = {2, 3};
    const std::vector<llama_logit_bias> caller_biases = {
        {2, 100.0f},
        {4, -1.0f},
        {2, 50.0f},
    };
    auto merged = skippy_add_vocab_suppress_token_biases(
            caller_biases, suppress_tokens, 2);
    if (merged.size() != 3 || merged[0].token != 4 || merged[0].bias != -1.0f ||
            merged[1].token != 2 || !suppressed(merged[1]) ||
            merged[2].token != 3 || !suppressed(merged[2])) {
        std::fprintf(stderr, "vocab suppress tokens did not override caller logit biases\n");
        return 1;
    }

    llama_token_data candidates[] = {
        {0, 1.0f, 0.0f},
        {1, 2.0f, 0.0f},
        {2, 100.0f, 0.0f},
        {3, 50.0f, 0.0f},
        {4, 4.0f, 0.0f},
    };
    llama_token_data_array view = {candidates, 5, -1, false};
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(5, merged.size(), merged.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    llama_sampler_apply(chain, &view);
    llama_sampler_free(chain);
    if (!std::isinf(candidates[2].logit) || !std::signbit(candidates[2].logit) ||
            !std::isinf(candidates[3].logit) || !std::signbit(candidates[3].logit) ||
            candidates[4].logit != 3.0f || view.selected != 4) {
        std::fprintf(stderr, "vocab suppression did not reach the sampling chain\n");
        return 2;
    }

    const float logits[] = {1.0f, 2.0f, 100.0f, 50.0f, 4.0f};
    if (skippy_greedy_sample_allowed_logits(logits, nullptr, 5, suppress_tokens, 2, false) != 4 ||
            skippy_greedy_sample_allowed_logits(logits, nullptr, 5, nullptr, 0, false) != 2) {
        std::fprintf(stderr, "greedy fast path ignored vocab suppression\n");
        return 3;
    }
    const llama_token all_tokens[] = {0, 1, 2, 3, 4};
    const llama_token invalid_token[] = {99};
    if (skippy_greedy_sample_allowed_logits(logits, nullptr, 5, all_tokens, 5, false) != LLAMA_TOKEN_NULL ||
            skippy_greedy_sample_allowed_logits(logits, nullptr, 5, invalid_token, 1, false) != 2) {
        std::fprintf(stderr, "greedy fast path mishandled exhausted or invalid suppression\n");
        return 4;
    }

    llama_token output = 42;
    skippy_error * error = nullptr;
    const llama_token exhausted = skippy_greedy_sample_allowed_logits(
            logits, nullptr, 5, all_tokens, 5, false);
    if (skippy_store_sampled_token(exhausted, &output, &error) != SKIPPY_STATUS_RUNTIME_ERROR ||
            output != 42 || error == nullptr ||
            std::strstr(error->message, "no allowed token") == nullptr) {
        std::fprintf(stderr, "exhausted sampling leaked into the output buffer\n");
        skippy_error_free(error);
        return 6;
    }
    skippy_error_free(error);
    error = nullptr;
    if (skippy_store_sampled_token(0, &output, &error) != SKIPPY_STATUS_OK ||
            output != 0 || error != nullptr) {
        std::fprintf(stderr, "valid token zero was rejected\n");
        skippy_error_free(error);
        return 7;
    }

    merged = skippy_add_vocab_suppress_token_biases(caller_biases, nullptr, 0);
    if (merged.size() != caller_biases.size()) {
        std::fprintf(stderr, "models without suppress tokens lost caller logit biases\n");
        return 5;
    }
    return 0;
}
