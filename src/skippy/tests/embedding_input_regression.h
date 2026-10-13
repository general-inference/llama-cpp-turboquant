#pragma once

#include "llama.h"
#include "skippy/runtime.h"
#include "skippy/runtime_types.h"

#include <cmath>
#include <cstdio>
#include <exception>

// The synthetic GGUF's first stage must accept the same raw-embedding batch
// shape emitted by mtmd, then switch back to token input in the same session.
// No real model, projector, network, or GPU is required.
inline bool check_synthetic_embedding_input(skippy_model * model) {
    skippy_error *   error   = nullptr;
    skippy_session * session = nullptr;
    if (skippy_session_create(model, &session, &error) != SKIPPY_STATUS_OK) {
        skippy_error_free(error);
        return false;
    }
    if (skippy_session_begin_external_decode(session, &error) != SKIPPY_STATUS_OK) {
        skippy_session_free(session, &error);
        skippy_error_free(error);
        return false;
    }
    llama_batch prefix      = llama_batch_init(1, 0, 1);
    prefix.n_tokens         = 1;
    prefix.token[0]         = 1;
    prefix.pos[0]           = 0;
    prefix.n_seq_id[0]      = 1;
    prefix.seq_id[0][0]     = session->seq_id;
    prefix.logits[0]        = 1;
    const int prefix_status = llama_decode(session->ctx, prefix);
    llama_batch_free(prefix);
    if (prefix_status != 0) {
        skippy_session_end_external_decode(session, &error);
        skippy_session_free(session, &error);
        skippy_error_free(error);
        return false;
    }
    const int32_t width = llama_model_n_embd_inp(model->model);
    // Like mtmd, an M-RoPE model takes one position per section for each row.
    const llama_rope_type rope = llama_model_rope_type(model->model);
    const int32_t n_pos = rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
    llama_batch   image = llama_batch_init(2 * n_pos, width, 1);
    image.n_tokens      = 2;
    for (int32_t row = 0; row < 2; ++row) {
        for (int32_t section = 0; section < n_pos; ++section) {
            image.pos[section * 2 + row] = row + 1;
        }
        image.n_seq_id[row]  = 1;
        image.seq_id[row][0] = session->seq_id;
        image.logits[row]    = 1;
        for (int32_t col = 0; col < width; ++col) {
            image.embd[row * width + col] = 0.01f * float((row + col) % 7 - 3);
        }
    }
    bool passed = false;
    try {
        passed = llama_decode(session->ctx, image) == 0;
        for (int32_t row = 0; passed && row < 2; ++row) {
            const float * output = llama_get_embeddings_ith(session->ctx, row);
            passed               = output != nullptr;
            for (int32_t col = 0; passed && col < llama_model_n_embd(model->model); ++col) {
                passed = std::isfinite(output[col]);
            }
        }
        llama_batch text  = llama_batch_init(1, 0, 1);
        text.n_tokens     = 1;
        text.token[0]     = 1;
        text.pos[0]       = 3;
        text.n_seq_id[0]  = 1;
        text.seq_id[0][0] = session->seq_id;
        text.logits[0]    = 1;
        if (passed) {
            passed = llama_decode(session->ctx, text) == 0;
        }
        llama_batch_free(text);
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "synthetic embedding-input stage regression: %s\n", exception.what());
        passed = false;
    }
    llama_batch_free(image);
    skippy_session_end_external_decode(session, &error);
    skippy_session_free(session, &error);
    skippy_error_free(error);
    if (passed) {
        std::fprintf(stderr, "synthetic embedding-input stage and token transition passed\n");
    }
    return passed;
}
