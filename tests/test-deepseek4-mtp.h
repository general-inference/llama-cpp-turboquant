#pragma once

#include "skippy.h"
#include "../src/llama-model.h"
#include "../src/llama-context.h"
#include "../src/skippy/runtime_types.h"
#include "../src/skippy/speculative_internal.h"

#include <cmath>

static void test_deepseek4_mtp_rows(const std::string & path, int n_rows) {
    auto config = skippy_runtime_config_default();
    config.mtp_source = SKIPPY_MTP_SOURCE_INTEGRATED;
    config.n_gpu_layers = 0;
    config.selected_backend_device = "CPU";
    config.ctx_size = 256;
    config.n_batch = 16;
    config.n_ubatch = 4;
    config.n_threads = config.n_threads_batch = 2;
    skippy_model * model = nullptr;
    if (skippy_model_open(path.c_str(), &config, &model, nullptr) != SKIPPY_STATUS_OK || !model) {
        throw std::runtime_error("production model open failed");
    }
    std::unique_ptr<skippy_model, void(*)(skippy_model *)> model_guard(model, [](skippy_model * m) {
        skippy_model_free(m, nullptr);
    });
    if (model->model->hparams.n_embd != 512 || model->model->hparams.n_embd_out() != 2048 ||
            model->model->hparams.n_layer_nextn != 1 || !model->mtp_ctx) {
        throw std::runtime_error("required 512/2048 integrated MTP fixture unavailable");
    }
    skippy_session * session = nullptr;
    if (skippy_session_create(model, &session, nullptr) != SKIPPY_STATUS_OK || !session) {
        throw std::runtime_error("session creation failed");
    }
    std::unique_ptr<skippy_session, void(*)(skippy_session *)> session_guard(session, [](skippy_session * s) {
        skippy_session_free(s, nullptr);
    });
    if (!session->mtp_pending_h.empty() || session->mtp_has_pending_h) {
        throw std::runtime_error("new MTP session must not contain a pending hidden row");
    }
    llama_set_embeddings_nextn(session->ctx, true, false);
    for (int block = 0; block < 2; ++block) {
        llama_token tokens[] = { 3 + block*11, 7 + block*13, 19 + block*17, 41 + block*5, 59 + block*5 };
        llama_batch batch = llama_batch_init(n_rows, 0, 1);
        batch.n_tokens = n_rows;
        for (int row = 0; row < n_rows; ++row) {
            batch.token[row] = tokens[row];
            batch.pos[row] = block*n_rows + row;
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = session->seq_id;
            batch.logits[row] = 1;
        }
        const int rc = llama_decode(session->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error("production target decode failed");
        }
        fprintf(stderr, "MTP sync block=%d rows=%d n_ubatch=4 decoder=512 hidden=2048\n", block, n_rows);
        fflush(stderr);
        if (skippy_mtp_sync_target_inputs(session, tokens, nullptr, n_rows, block*n_rows, nullptr) != SKIPPY_STATUS_OK) {
            throw std::runtime_error("production MTP sync failed");
        }
        session->n_past = (block + 1)*n_rows;
    }
    fprintf(stderr, "MTP production draft begin\n");
    fflush(stderr);
    skippy_native_mtp_draft draft = {};
    if (skippy_mtp_propose_next(session, 31, 2, &draft, nullptr) != SKIPPY_STATUS_OK || draft.token_count != 2) {
        throw std::runtime_error("production MTP draft did not execute two steps");
    }
    const float * hidden = llama_get_embeddings_nextn_ith(session->ctx, n_rows - 1);
    if (!hidden || session->mtp_pending_h.size() != 2048) {
        throw std::runtime_error("pending hidden row is truncated");
    }
    for (size_t col = 0; col < 2048; ++col) {
        if (!std::isfinite(hidden[col]) || session->mtp_pending_h[col] != hidden[col]) {
            throw std::runtime_error("pending hidden row differs from full target output");
        }
    }
    fprintf(stderr, "MTP production sync/draft PASS\n");
}

static int test_deepseek4_mtp(const std::string & path) {
    for (int n_rows : {1, 2, 5}) {
        test_deepseek4_mtp_rows(path, n_rows);
    }
    return 0;
}
