#include "skippy/tokenization.h"
#include "skippy/errors.h"
#include "skippy/runtime_types.h"

#include "llama-model.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

using skippy_chat_templates_create_fn = void * (*)(const llama_model * model);
using skippy_chat_templates_free_fn = void (*)(void * templates);

extern "C" LLAMA_API void * skippy_model_chat_templates_get_or_init(
        struct skippy_model * model,
        skippy_chat_templates_create_fn create,
        skippy_chat_templates_free_fn free_templates);
extern "C" LLAMA_API void skippy_model_chat_templates_lock(struct skippy_model * model);
extern "C" LLAMA_API void skippy_model_chat_templates_unlock(struct skippy_model * model);

extern "C" LLAMA_API void * skippy_model_chat_templates_get_or_init(
        struct skippy_model * model,
        skippy_chat_templates_create_fn create,
        skippy_chat_templates_free_fn free_templates) {
    if (model == nullptr || model->model == nullptr || create == nullptr || free_templates == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(model->chat_templates_mutex);
    if (model->chat_templates == nullptr) {
        model->chat_templates = create(model->model);
        if (model->chat_templates == nullptr) {
            return nullptr;
        }
        model->chat_templates_free = free_templates;
    }
    return model->chat_templates;
}

extern "C" LLAMA_API void skippy_model_chat_templates_lock(struct skippy_model * model) {
    if (model != nullptr) {
        model->chat_templates_mutex.lock();
    }
}

extern "C" LLAMA_API void skippy_model_chat_templates_unlock(struct skippy_model * model) {
    if (model != nullptr) {
        model->chat_templates_mutex.unlock();
    }
}

extern "C" {

enum skippy_status skippy_tokenize(
        struct skippy_model * model,
        const char * text,
        bool add_special,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || text == nullptr || out_token_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model, text, and out_token_count are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (output_token_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token output capacity exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    const int32_t result = llama_tokenize(
            vocab,
            text,
            static_cast<int32_t>(std::strlen(text)),
            output_tokens,
            static_cast<int32_t>(output_token_capacity),
            add_special,
            true);
    if (result < 0) {
        *out_token_count = static_cast<size_t>(-result);
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    *out_token_count = static_cast<size_t>(result);
    return skippy_success(out_error);
}

enum skippy_status skippy_detokenize(
        struct skippy_model * model,
        const llama_token * tokens,
        size_t token_count,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || tokens == nullptr || out_text_bytes == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model, tokens, and out_text_bytes are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            output_text_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "detokenize inputs exceed int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    std::string text;
    for (size_t i = 0; i < token_count; ++i) {
        std::string piece;
        piece.resize(piece.capacity());
        int32_t piece_bytes = llama_token_to_piece(
                vocab,
                tokens[i],
                piece.data(),
                static_cast<int32_t>(piece.size()),
                0,
                true);
        if (piece_bytes < 0) {
            piece.resize(static_cast<size_t>(-piece_bytes));
            piece_bytes = llama_token_to_piece(
                    vocab,
                    tokens[i],
                    piece.data(),
                    static_cast<int32_t>(piece.size()),
                    0,
                    true);
        }
        if (piece_bytes < 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "token piece output buffer is too small");
            return SKIPPY_STATUS_BUFFER_TOO_SMALL;
        }
        piece.resize(static_cast<size_t>(piece_bytes));
        text += piece;
    }

    *out_text_bytes = text.size();
    if (output_text_capacity < text.size()) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "text output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    if (!text.empty() && output_text == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output_text is required when output capacity is sufficient");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (!text.empty()) {
        std::memcpy(output_text, text.data(), text.size());
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_token_is_eog(
        struct skippy_model * model,
        llama_token token,
        bool * out_is_eog,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || out_is_eog == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_is_eog are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model->model);
    *out_is_eog = llama_vocab_is_eog(vocab, token);
    return skippy_success(out_error);
}

const struct llama_model * skippy_model_native_model(
        const struct skippy_model * model) {
    if (model == nullptr) {
        return nullptr;
    }
    return model->model;
}


} // extern "C"
