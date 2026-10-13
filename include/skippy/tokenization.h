#ifndef SKIPPY_TOKENIZATION_H
#define SKIPPY_TOKENIZATION_H

/**
 * @file skippy/tokenization.h
 * @brief Token, detokenization, chat-template, and chat-response helpers.
 */

#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Converts UTF-8 text into model tokens. */
LLAMA_API enum skippy_status skippy_tokenize(
        struct skippy_model * model,
        const char * text,
        bool add_special,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error);

/** @brief Converts model tokens into UTF-8 text. */
LLAMA_API enum skippy_status skippy_detokenize(
        struct skippy_model * model,
        const llama_token * tokens,
        size_t token_count,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        struct skippy_error ** out_error);

/** @brief Reports whether a token is an end-of-generation token for the model. */
LLAMA_API enum skippy_status skippy_token_is_eog(
        struct skippy_model * model,
        llama_token token,
        bool * out_is_eog,
        struct skippy_error ** out_error);

/** @brief Exposes the underlying llama model for native integrations. */
LLAMA_API const struct llama_model * skippy_model_native_model(
        const struct skippy_model * model);

/** @brief Renders OpenAI-style JSON messages through the model chat template. */
SKIPPY_COMMON_API enum skippy_status skippy_apply_chat_template_json(
        struct skippy_model * model,
        const char * messages_json,
        const char * tools_json,
        const char * tool_choice_json,
        bool add_assistant,
        bool override_enable_thinking,
        bool enable_thinking,
        bool parallel_tool_calls,
        const char * reasoning_format,
        const char * chat_template_kwargs,
        const char * chat_template,
        bool use_jinja,
        const char * grammar,
        const char * json_schema,
        bool skip_chat_parsing,
        char * output_text,
        size_t output_text_capacity,
        size_t * out_text_bytes,
        char * output_metadata_json,
        size_t output_metadata_json_capacity,
        size_t * out_metadata_json_bytes,
        struct skippy_error ** out_error);

/** @brief Parses generated chat text and template metadata into an OpenAI message. */
SKIPPY_COMMON_API enum skippy_status skippy_parse_chat_response_json(
        const char * generated_text,
        const char * metadata_json,
        bool is_partial,
        char * output_message_json,
        size_t output_message_json_capacity,
        size_t * out_message_json_bytes,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_TOKENIZATION_H
