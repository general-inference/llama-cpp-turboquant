#include "skippy/workloads.h"

#include "llama.h"
#include "llama-model.h"
#include "model_load_config_internal.h"
#include "skippy/errors.h"
#include "skippy/rerank_template.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"
#include "skippy/workloads_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

enum skippy_workload_kind workload_kind(const skippy_model * model) {
    const bool                    has_encoder = llama_model_has_encoder(model->model);
    const bool                    has_decoder = llama_model_has_decoder(model->model);
    const enum llama_pooling_type pooling     = llama_pooling_type(model->ctx);

    if (skippy_context_capabilities_of(model->model).classifier || pooling == LLAMA_POOLING_TYPE_RANK) {
        return SKIPPY_WORKLOAD_RERANK;
    }
    if (has_encoder && has_decoder) {
        return SKIPPY_WORKLOAD_ENCODER_DECODER;
    }
    if (has_encoder || (pooling != LLAMA_POOLING_TYPE_NONE && pooling != LLAMA_POOLING_TYPE_UNSPECIFIED)) {
        return SKIPPY_WORKLOAD_EMBEDDING;
    }
    return SKIPPY_WORKLOAD_CAUSAL_GENERATION;
}

bool is_full_model(const skippy_model * model) {
    // Mirrors skippy_is_full_model_config(): a stage-plan config (or an
    // explicit layer range) means filtered stage execution, which non-chat
    // workloads do not support.
    return model != nullptr && !skippy_runtime_has_stage_plan(&model->config) &&
           model->config.layer_start == 0;
}

enum skippy_status require_workload(skippy_session *          session,
                                    enum skippy_workload_kind required,
                                    skippy_error **           out_error) {
    if (session == nullptr || session->stage_model == nullptr || session->stage_model->model == nullptr ||
        session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "an active session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (!is_full_model(session->stage_model)) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "non-chat workloads currently require a full model");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    if (workload_kind(session->stage_model) != required) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "model does not implement the requested workload");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    return SKIPPY_STATUS_OK;
}

enum skippy_status validate_tokens(skippy_session *    session,
                                   const llama_token * token_ids,
                                   size_t              token_count,
                                   skippy_error **     out_error) {
    if (token_ids == nullptr || token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "at least one token is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        token_count > static_cast<size_t>(llama_n_batch(session->ctx))) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token count exceeds the model batch capacity");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    return SKIPPY_STATUS_OK;
}

}  // namespace

void skippy_prepare_non_chat_workload(skippy_session * session) {
    skippy_synchronize_context_if_pending(session->stage_model);
    skippy_clear_chat_sampling(session);
    if (llama_memory_t memory = llama_get_memory(session->ctx)) {
        llama_memory_clear(memory, true);
    }
    session->n_past = 0;
    session->token_history.clear();
    session->signal_history.clear();
}

namespace {

llama_batch make_embedding_batch(skippy_session * session, const llama_token * token_ids, size_t token_count) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(token_count), 0, 1);
    batch.n_tokens    = static_cast<int32_t>(token_count);
    for (int32_t index = 0; index < batch.n_tokens; ++index) {
        batch.token[index]     = token_ids[index];
        batch.pos[index]       = index;
        batch.n_seq_id[index]  = 1;
        batch.seq_id[index][0] = session->seq_id;
        batch.logits[index]    = 1;
    }
    return batch;
}

enum skippy_status decode_embedding(skippy_session *    session,
                                    const llama_token * token_ids,
                                    size_t              token_count,
                                    const float **      out_embedding,
                                    skippy_error **     out_error) {
    skippy_prepare_non_chat_workload(session);
    llama_batch   batch         = make_embedding_batch(session, token_ids, token_count);
    int32_t decode_status = 0;
    const bool decoded = skippy_contain_native_failure(
            out_error, [&] { decode_status = llama_decode(session->ctx, batch); });
    llama_batch_free(batch);
    if (!decoded) {
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (decode_status != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed for non-chat workload");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    skippy_mark_context_work_pending(session->stage_model);

    const enum llama_pooling_type pooling   = llama_pooling_type(session->ctx);
    const float *                 embedding = pooling == LLAMA_POOLING_TYPE_NONE ?
                                                  llama_get_embeddings_ith(session->ctx, static_cast<int32_t>(token_count - 1)) :
                                                  llama_get_embeddings_seq(session->ctx, session->seq_id);
    skippy_mark_context_synchronized(session->stage_model);
    if (embedding == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "model produced no workload output");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    *out_embedding  = embedding;
    session->n_past = static_cast<int32_t>(token_count);
    session->token_history.assign(token_ids, token_ids + token_count);
    return SKIPPY_STATUS_OK;
}

void normalize_embedding(const float * source, float * output, size_t dimensions) {
    double squared_norm = 0.0;
    for (size_t index = 0; index < dimensions; ++index) {
        squared_norm += static_cast<double>(source[index]) * source[index];
    }
    const float scale = squared_norm > 0.0 ? static_cast<float>(1.0 / std::sqrt(squared_norm)) : 1.0f;
    for (size_t index = 0; index < dimensions; ++index) {
        output[index] = source[index] * scale;
    }
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    char    stack[128];
    int32_t length = llama_token_to_piece(vocab, token, stack, sizeof(stack), 0, true);
    if (length >= 0) {
        return std::string(stack, static_cast<size_t>(length));
    }
    std::string dynamic(static_cast<size_t>(-length), '\0');
    length = llama_token_to_piece(vocab, token, dynamic.data(), static_cast<int32_t>(dynamic.size()), 0, true);
    return length > 0 ? dynamic.substr(0, static_cast<size_t>(length)) : std::string();
}

bool tokenize(const llama_vocab * vocab, const std::string & text, std::vector<llama_token> & tokens) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    const int32_t text_length = static_cast<int32_t>(text.size());
    int32_t       count       = llama_tokenize(vocab, text.data(), text_length, nullptr, 0, true, true);
    if (count >= 0) {
        tokens.resize(static_cast<size_t>(count));
    } else {
        tokens.resize(static_cast<size_t>(-count));
    }
    count =
        llama_tokenize(vocab, text.data(), text_length, tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (count < 0) {
        return false;
    }
    tokens.resize(static_cast<size_t>(count));
    return !tokens.empty();
}

std::string rerank_prompt(const llama_model * model, const char * query, const char * document) {
    if (const char * prompt_template = llama_model_chat_template(model, "rerank")) {
        return skippy_render_rerank_template(prompt_template, query, document);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string         prompt(query);
    if (llama_vocab_get_add_eos(vocab)) {
        prompt += token_piece(vocab, llama_vocab_eos(vocab));
    }
    if (llama_vocab_get_add_sep(vocab)) {
        prompt += token_piece(vocab, llama_vocab_sep(vocab));
    }
    prompt += document;
    return prompt;
}

}  // namespace

extern "C" {

enum skippy_status skippy_model_workload_info_v1(const skippy_model *      model,
                                                 skippy_workload_info_v1 * out_info,
                                                 skippy_error **           out_error) {
    if (model == nullptr || model->model == nullptr || model->ctx == nullptr || out_info == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_info are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (out_info->abi_version != SKIPPY_WORKLOAD_INFO_V1_ABI_VERSION ||
        out_info->struct_size != sizeof(skippy_workload_info_v1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "incompatible workload descriptor ABI");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const skippy_workload_kind kind              = workload_kind(model);
    const uint32_t             output_dimensions = kind == SKIPPY_WORKLOAD_EMBEDDING ?
                                                       static_cast<uint32_t>(std::max(llama_model_n_embd_out(model->model), 0)) :
                                                       0;
    const uint32_t classifier_outputs = kind == SKIPPY_WORKLOAD_RERANK ? llama_model_n_cls_out(model->model) : 0;
    *out_info                         = {
        SKIPPY_WORKLOAD_INFO_V1_ABI_VERSION,
        static_cast<uint32_t>(sizeof(skippy_workload_info_v1)),
        kind,
        static_cast<skippy_workload_pooling>(llama_pooling_type(model->ctx)),
        output_dimensions,
        classifier_outputs,
        llama_model_has_encoder(model->model),
        llama_model_has_decoder(model->model),
        true,
        0,
    };
    return skippy_success(out_error);
}

enum skippy_status skippy_session_embed(skippy_session *    session,
                                        const llama_token * token_ids,
                                        size_t              token_count,
                                        float *             output,
                                        size_t              output_capacity,
                                        size_t *            out_dimensions,
                                        skippy_error **     out_error) {
    if (out_dimensions != nullptr) {
        *out_dimensions = 0;
    }
    enum skippy_status status = require_workload(session, SKIPPY_WORKLOAD_EMBEDDING, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (output == nullptr || out_dimensions == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "output and out_dimensions are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    status = validate_tokens(session, token_ids, token_count, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    const int32_t native_dimensions = llama_model_n_embd_out(session->stage_model->model);
    if (native_dimensions <= 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "model has no embedding output dimensions");
        return SKIPPY_STATUS_MODEL_ERROR;
    }
    const size_t dimensions = static_cast<size_t>(native_dimensions);
    *out_dimensions         = dimensions;
    if (output_capacity < dimensions) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "embedding output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    const float * embedding = nullptr;
    status                  = decode_embedding(session, token_ids, token_count, &embedding, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    normalize_embedding(embedding, output, dimensions);
    return skippy_success(out_error);
}

enum skippy_status skippy_session_rerank(skippy_session * session,
                                         const char *     query,
                                         const char *     document,
                                         float *          out_score,
                                         size_t *         out_token_count,
                                         skippy_error **  out_error) {
    if (out_token_count != nullptr) {
        *out_token_count = 0;
    }
    enum skippy_status status = require_workload(session, SKIPPY_WORKLOAD_RERANK, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (query == nullptr || document == nullptr || out_score == nullptr || out_token_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                         "query, document, and output pointers are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (llama_model_n_cls_out(session->stage_model->model) != 1) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "rerank API requires a single classifier output");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    std::vector<llama_token> tokens;
    const std::string        prompt = rerank_prompt(session->stage_model->model, query, document);
    if (!tokenize(llama_model_get_vocab(session->stage_model->model), prompt, tokens)) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to tokenize rerank input");
        return SKIPPY_STATUS_MODEL_ERROR;
    }
    status = validate_tokens(session, tokens.data(), tokens.size(), out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    const float * ranks = nullptr;
    status              = decode_embedding(session, tokens.data(), tokens.size(), &ranks, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    *out_score       = ranks[0];
    *out_token_count = tokens.size();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_encode_prompt(skippy_session *    session,
                                                const llama_token * token_ids,
                                                size_t              token_count,
                                                llama_token *       out_decoder_start_token,
                                                skippy_error **     out_error) {
    enum skippy_status status = require_workload(session, SKIPPY_WORKLOAD_ENCODER_DECODER, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (out_decoder_start_token == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_decoder_start_token is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    status = validate_tokens(session, token_ids, token_count, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (token_count > static_cast<size_t>(llama_n_ubatch(session->ctx))) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                         "encoder input exceeds the physical batch capacity");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_prepare_non_chat_workload(session);
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(token_ids), static_cast<int32_t>(token_count));
    if (llama_encode(session->ctx, batch) != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_encode failed for encoder-decoder workload");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    skippy_mark_context_work_pending(session->stage_model);

    llama_token start = llama_model_decoder_start_token(session->stage_model->model);
    if (start == LLAMA_TOKEN_NULL) {
        start = llama_vocab_bos(llama_model_get_vocab(session->stage_model->model));
    }
    if (start == LLAMA_TOKEN_NULL) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "model has no decoder start or BOS token");
        return SKIPPY_STATUS_MODEL_ERROR;
    }
    *out_decoder_start_token = start;
    return skippy_success(out_error);
}

}  // extern "C"
