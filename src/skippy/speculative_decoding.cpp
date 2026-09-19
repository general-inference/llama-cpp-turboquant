#include "skippy/speculative_decoding.h"
#include "skippy/environment.h"
#include "skippy/errors.h"
#include "skippy/graph_build_inputs.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"
#include "skippy/sampling_internal.h"
#include "skippy/speculative_internal.h"

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

thread_local skippy_session * g_skippy_external_decode_session = nullptr;

// This semantic identity belongs to the Inkling MTP producer/consumer. The
// activation transport treats it as an ordinary optional typed part.
static constexpr uint8_t SKIPPY_MTP_INPUT_EMBEDDING_IDENTITY[SKIPPY_ACTIVATION_IDENTITY_BYTES] = {
    0x0d, 0x5b, 0x22, 0xa5, 0xde, 0x94, 0xde, 0x79,
    0x94, 0xcb, 0x12, 0x03, 0x9a, 0x51, 0xd3, 0x41,
    0x4d, 0x8a, 0x65, 0x4f, 0x67, 0xbc, 0x21, 0xdb,
    0x43, 0x58, 0x56, 0xf7, 0x8c, 0x52, 0x9c, 0x89,
};

const float * skippy_mtp_input_embeddings_from_activation_frame(
        const skippy_activation_desc * desc,
        const void * payload,
        size_t token_count) {
    if (desc == nullptr || payload == nullptr) return nullptr;
    for (uint32_t index = 0; index < desc->part_count; ++index) {
        const auto & part = desc->parts[index];
        if (std::memcmp(part.identity, SKIPPY_MTP_INPUT_EMBEDDING_IDENTITY, sizeof(part.identity)) != 0) continue;
        if (part.ggml_type != GGML_TYPE_F32 || part.rank != 2 || part.token_axis != 1 ||
                part.dimensions[1] != static_cast<int64_t>(token_count) ||
                part.byte_strides[0] != sizeof(float)) return nullptr;
        return reinterpret_cast<const float *>(static_cast<const uint8_t *>(payload) + part.payload_offset);
    }
    return nullptr;
}

bool skippy_mtp_available(const skippy_session * session) {
    return session != nullptr &&
           session->stage_model != nullptr &&
           session->stage_model->mtp_ctx != nullptr &&
           skippy_runtime_is_terminal_stage(&session->stage_model->config);
}

static constexpr int32_t skippy_mtp_depth_for_step(size_t draft_step, uint32_t depth_count) {
    return depth_count > 0 ? static_cast<int32_t>(draft_step % depth_count) : 0;
}

static_assert(skippy_mtp_depth_for_step(0, 8) == 0);
static_assert(skippy_mtp_depth_for_step(7, 8) == 7);
static_assert(skippy_mtp_depth_for_step(8, 8) == 0);
static_assert(skippy_mtp_depth_for_step(5, 1) == 0);
static_assert(skippy_mtp_depth_for_step(5, 0) == 0);

struct skippy_mtp_depth_scope {
    explicit skippy_mtp_depth_scope(llama_context * ctx)
        : ctx(ctx), depth_count(ctx != nullptr ? llama_get_model(ctx)->hparams.n_layer_nextn : 0) {
        if (ctx != nullptr) {
            llama_set_nextn_layer_offset(ctx, 0);
        }
    }

    ~skippy_mtp_depth_scope() {
        if (ctx != nullptr) {
            llama_set_nextn_layer_offset(ctx, 0);
        }
    }

    void select(size_t draft_step) const {
        if (ctx != nullptr && depth_count > 0) {
            llama_set_nextn_layer_offset(ctx, skippy_mtp_depth_for_step(draft_step, depth_count));
        }
    }

    bool chains_heads() const {
        return depth_count > 1;
    }

    bool chains_at_same_position() const {
        return chains_heads() && llama_get_model(ctx)->arch == LLM_ARCH_INKLING;
    }

    llama_context * ctx;
    uint32_t depth_count;
};

bool skippy_mtp_greedy_sampling_fastpath_enabled() {
    return skippy_env_enabled("SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH");
}

void skippy_mtp_clear_session_state(skippy_session * session) {
    if (session == nullptr) {
        return;
    }
    session->mtp_next_pos = session->n_past;
    session->mtp_prefix_valid = session->n_past == 0;
    session->mtp_has_pending_h = false;
    session->mtp_pending_h.clear();
    session->mtp_has_pending_draft = false;
    session->mtp_pending_draft_pos = 0;
    session->mtp_pending_draft_token = -1;
    session->mtp_pending_draft_end = 0;
    session->mtp_recent_h.clear();
    session->mtp_recent_h_start = 0;
    if (skippy_mtp_available(session)) {
        if (llama_memory_t memory = session->stage_model->mtp_ctx->get_memory()) {
            llama_memory_seq_rm(memory, session->seq_id, -1, -1);
        }
    }
}

static bool skippy_mtp_shares_target_memory(
        const llama_context * mtp_ctx,
        const llama_context * target_ctx);

static constexpr size_t SKIPPY_MTP_RECENT_HIDDEN_ROWS = 16;

static size_t skippy_mtp_hidden_width(const skippy_session * session) {
    const int32_t n_hidden = llama_model_n_embd_out(session->stage_model->model);
    return n_hidden > 0 ? static_cast<size_t>(n_hidden) : 0;
}

// Keeps the trunk hidden rows at the tail of a batch that was just decoded.
static void skippy_mtp_remember_recent_hidden(skippy_session * session, size_t token_count, llama_pos token_start) {
    const size_t width = skippy_mtp_hidden_width(session);
    const size_t kept = std::min(token_count, SKIPPY_MTP_RECENT_HIDDEN_ROWS);
    session->mtp_recent_h.clear();
    session->mtp_recent_h_start = token_start + static_cast<llama_pos>(token_count - kept);
    for (size_t row = token_count - kept; row < token_count && width > 0; ++row) {
        const float * h = llama_get_embeddings_nextn_ith(session->ctx, static_cast<int32_t>(row));
        if (h == nullptr) {
            session->mtp_recent_h.clear();
            return;
        }
        session->mtp_recent_h.insert(session->mtp_recent_h.end(), h, h + width);
    }
}

static const float * skippy_mtp_recent_hidden_at(const skippy_session * session, llama_pos pos) {
    const size_t width = skippy_mtp_hidden_width(session);
    if (width == 0 || pos < session->mtp_recent_h_start) {
        return nullptr;
    }
    const size_t row = static_cast<size_t>(pos - session->mtp_recent_h_start);
    if ((row + 1) * width > session->mtp_recent_h.size()) {
        return nullptr;
    }
    return session->mtp_recent_h.data() + row * width;
}

// Rewinds the sidecar to `token_count` committed tokens. A single-head sidecar
// whose rows below the trim point are intact only loses the rejected suffix:
// its seed becomes the trunk hidden state of the last kept token. Anything
// else falls back to discarding the sidecar's state for this sequence.
void skippy_mtp_trim_session_state(skippy_session * session, llama_pos token_count) {
    if (session == nullptr) {
        return;
    }
    const bool single_head = skippy_mtp_available(session) &&
            llama_get_model(session->stage_model->mtp_ctx)->hparams.n_layer_nextn == 1;
    const float * seed = token_count > 0 ? skippy_mtp_recent_hidden_at(session, token_count - 1) : nullptr;
    const bool keep_prefix = single_head && session->mtp_prefix_valid && seed != nullptr &&
            session->mtp_next_pos >= token_count &&
            !skippy_mtp_shares_target_memory(session->stage_model->mtp_ctx, session->ctx);
    if (!keep_prefix) {
        skippy_mtp_clear_session_state(session);
        return;
    }
    if (llama_memory_t memory = session->stage_model->mtp_ctx->get_memory()) {
        llama_memory_seq_rm(memory, session->seq_id, token_count, -1);
    }
    session->mtp_pending_h.assign(seed, seed + skippy_mtp_hidden_width(session));
    session->mtp_has_pending_h = true;
    session->mtp_next_pos = token_count;
    session->mtp_has_pending_draft = false;
    session->mtp_pending_draft_pos = 0;
    session->mtp_pending_draft_token = -1;
    session->mtp_pending_draft_end = 0;
    const size_t width = skippy_mtp_hidden_width(session);
    const size_t keep_rows = static_cast<size_t>(token_count - session->mtp_recent_h_start);
    session->mtp_recent_h.resize(keep_rows * width);
}

static bool skippy_mtp_shares_target_memory(
        const llama_context * mtp_ctx,
        const llama_context * target_ctx) {
    if (mtp_ctx == nullptr || target_ctx == nullptr || llama_get_ctx_other(const_cast<llama_context *>(mtp_ctx)) != target_ctx) {
        return false;
    }
    // Inkling uses ctx_other only to borrow shared target weights. Its eight
    // attention/short-conv depth caches are independent of the target cache.
    return llama_get_model(mtp_ctx)->arch != LLM_ARCH_INKLING;
}

enum skippy_status skippy_mtp_sync_target_inputs(
        skippy_session * session,
        const llama_token * token_ids,
        const float * input_embeddings,
        size_t token_count,
        llama_pos token_start,
        struct skippy_error ** out_error) {
    if (!skippy_mtp_available(session) ||
            (token_ids == nullptr && input_embeddings == nullptr) ||
            token_count == 0) {
        return skippy_success(out_error);
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "MTP token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_context * mtp_ctx = session->stage_model->mtp_ctx;
    skippy_mtp_depth_scope mtp_depth_scope(mtp_ctx);
    const bool mtp_shares_target_memory = skippy_mtp_shares_target_memory(mtp_ctx, session->ctx);
    const bool chain_heads = mtp_depth_scope.chains_heads();
    const bool chain_at_same_position = mtp_depth_scope.chains_at_same_position();
    // MTP consumes complete pre-norm output rows, which can be wider than
    // token embeddings (for example DeepSeek4 hyperconnections).
    const int32_t n_hidden = llama_model_n_embd_out(session->stage_model->model);
    if (n_hidden <= 0 || n_hidden != llama_model_n_embd_out(llama_get_model(mtp_ctx))) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "target and MTP hidden widths differ");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const size_t row_bytes = static_cast<size_t>(n_hidden)*sizeof(float);
    if (session->mtp_pending_h.size() != static_cast<size_t>(n_hidden)) {
        session->mtp_pending_h.assign(static_cast<size_t>(n_hidden), 0.0f);
        session->mtp_has_pending_h = false;
    }

    if (!session->mtp_prefix_valid && !session->mtp_has_pending_h) {
        const float * h_last = llama_get_embeddings_nextn_ith(session->ctx, static_cast<int32_t>(token_count - 1));
        if (h_last == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "target pre-norm hidden row was not available");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        std::memcpy(session->mtp_pending_h.data(), h_last, row_bytes);
        session->mtp_has_pending_h = true;
        session->mtp_next_pos = token_start + static_cast<int32_t>(token_count);
        skippy_mtp_remember_recent_hidden(session, token_count, token_start);
        session->mtp_has_pending_draft = false;
        session->mtp_pending_draft_pos = 0;
        session->mtp_pending_draft_token = -1;
        session->mtp_pending_draft_end = 0;
        return skippy_success(out_error);
    }

    const bool reprime_prefix = !session->mtp_prefix_valid;

    if (chain_heads && session->mtp_has_pending_draft) {
        if (llama_memory_t memory = mtp_ctx->get_memory()) {
            llama_memory_seq_rm(memory, session->seq_id, token_start, -1);
        }
        session->mtp_next_pos = std::min(session->mtp_next_pos, static_cast<int32_t>(token_start));
        session->mtp_has_pending_draft = false;
        session->mtp_pending_draft_pos = 0;
        session->mtp_pending_draft_token = -1;
        session->mtp_pending_draft_end = 0;
    } else if (session->mtp_has_pending_draft) {
        const llama_pos token_end = token_start + static_cast<llama_pos>(token_count);
        if (session->mtp_pending_draft_pos >= token_start &&
                session->mtp_pending_draft_pos < token_end) {
            const size_t draft_index = static_cast<size_t>(session->mtp_pending_draft_pos - token_start);
            if (token_ids != nullptr && token_ids[draft_index] == session->mtp_pending_draft_token) {
                // Only the first drafted token is committed. A multi-token
                // proposal against a sidecar cache of its own also wrote rows
                // for the deeper drafts; those are not verified and nothing
                // else removes them. Left resident they make the next sidecar
                // decode non-consecutive (seq_pos_max is past the position we
                // resume at) and llama_batch_allocr::init rejects the batch.
                if (!mtp_shares_target_memory &&
                        session->mtp_pending_draft_end > session->mtp_pending_draft_pos + 1) {
                    if (llama_memory_t memory = mtp_ctx->get_memory()) {
                        llama_memory_seq_rm(
                                memory,
                                session->seq_id,
                                session->mtp_pending_draft_pos + 1,
                                -1);
                    }
                }
                session->mtp_next_pos = std::max(session->mtp_next_pos, session->mtp_pending_draft_pos + 1);
            } else {
                if (llama_memory_t memory = mtp_ctx->get_memory()) {
                    llama_memory_seq_rm(memory, session->seq_id, session->mtp_pending_draft_pos, -1);
                }
                session->mtp_next_pos = std::min(session->mtp_next_pos, static_cast<int32_t>(session->mtp_pending_draft_pos));
            }
        }
        session->mtp_has_pending_draft = false;
        session->mtp_pending_draft_pos = 0;
        session->mtp_pending_draft_token = -1;
        session->mtp_pending_draft_end = 0;
    }

    size_t first_index = 0;
    if (!chain_heads && session->mtp_next_pos > token_start) {
        first_index = static_cast<size_t>(std::min<llama_pos>(
                static_cast<llama_pos>(token_count),
                session->mtp_next_pos - token_start));
    }

    const int32_t n_decode = static_cast<int32_t>(token_count - first_index);
    if (!mtp_shares_target_memory && n_decode > 0) {
        llama_batch batch = llama_batch_init(n_decode, n_hidden, 1);
        if (token_ids != nullptr) {
            batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)*static_cast<size_t>(n_decode)));
            if (batch.token == nullptr) {
                llama_batch_free(batch);
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to allocate MTP token batch");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        }
        batch.n_tokens = n_decode;

        for (int32_t out_i = 0; out_i < n_decode; ++out_i) {
            const size_t src_i = first_index + static_cast<size_t>(out_i);
            if (batch.token != nullptr) {
                batch.token[out_i] = token_ids[src_i];
            }
            batch.pos[out_i] = token_start + static_cast<llama_pos>(src_i);
            batch.n_seq_id[out_i] = 1;
            batch.seq_id[out_i][0] = session->seq_id;
            batch.logits[out_i] = chain_at_same_position && out_i == n_decode - 1 ? 1 : 0;

            float * dst = batch.embd + static_cast<size_t>(out_i)*n_hidden;
            if (src_i == 0) {
                std::memcpy(dst, session->mtp_pending_h.data(), row_bytes);
            } else {
                const float * h_prev = llama_get_embeddings_nextn_ith(session->ctx, static_cast<int32_t>(src_i - 1));
                if (h_prev == nullptr) {
                    std::free(batch.token);
                    batch.token = nullptr;
                    llama_batch_free(batch);
                    skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "target pre-norm hidden row was not available");
                    return SKIPPY_STATUS_RUNTIME_ERROR;
                }
                std::memcpy(dst, h_prev, row_bytes);
            }
        }

        int32_t rc = 0;
        const uint32_t sync_depth_count = chain_heads ? mtp_depth_scope.depth_count : 1;
        const int32_t n_input = llama_model_n_embd(session->stage_model->model);
        const float * mtp_embeddings = input_embeddings != nullptr ?
                input_embeddings + first_index*static_cast<size_t>(n_input) : nullptr;
        skippy_graph_build_inputs graph_build_inputs = skippy_graph_build_inputs_from_config(
                &session->stage_model->config);
        skippy_graph_build_inputs_set_mtp_embeddings(
                graph_build_inputs,
                mtp_embeddings,
                static_cast<uint32_t>(n_decode),
                static_cast<uint32_t>(n_input),
                token_start + static_cast<llama_pos>(first_index));
        for (uint32_t depth = 0; depth < sync_depth_count; ++depth) {
            if (chain_heads) {
                if (llama_memory_t memory = mtp_ctx->get_memory()) {
                    llama_memory_seq_rm(memory, session->seq_id, batch.pos[0], -1);
                }
                mtp_depth_scope.select(depth);
            }

            if (!skippy_contain_native_failure(
                        nullptr, [&] { rc = mtp_ctx->decode(batch, graph_build_inputs); })) {
                rc = -1;
            }
            if (rc != 0) {
                session->mtp_prefix_valid = false;
            }
            if (rc != 0 || !chain_heads || depth + 1 == sync_depth_count) {
                break;
            }

            if (!chain_at_same_position) {
                continue;
            }

            for (int32_t out_i = 0; out_i < n_decode; ++out_i) {
                const float * h_next = llama_get_embeddings_nextn_ith(mtp_ctx, out_i);
                if (h_next == nullptr) {
                    rc = -1;
                    break;
                }
                std::memcpy(batch.embd + static_cast<size_t>(out_i)*n_hidden, h_next, row_bytes);
            }
            if (rc != 0) {
                break;
            }

            if (batch.token == nullptr) {
                continue;
            }
            for (int32_t out_i = 0; out_i + 1 < n_decode; ++out_i) {
                batch.token[out_i] = batch.token[out_i + 1];
            }
            const skippy_status sample_status = skippy_store_sampled_token(
                    skippy_greedy_sample_context(llama_get_model(mtp_ctx), mtp_ctx, -1),
                    &batch.token[n_decode - 1], out_error);
            if (sample_status != SKIPPY_STATUS_OK) {
                skippy_mtp_clear_session_state(session);
                std::free(batch.token);
                batch.token = nullptr;
                llama_batch_free(batch);
                return sample_status;
            }
        }
        if (batch.token != nullptr) {
            std::free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
        if (rc != 0) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed for MTP sidecar sync");
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        session->mtp_next_pos = token_start + static_cast<int32_t>(token_count);
    }

    const float * h_last = llama_get_embeddings_nextn_ith(session->ctx, static_cast<int32_t>(token_count - 1));
    if (h_last == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "target pre-norm hidden row was not available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    std::memcpy(session->mtp_pending_h.data(), h_last, row_bytes);
    session->mtp_has_pending_h = true;
    skippy_mtp_remember_recent_hidden(session, token_count, token_start);
    if (reprime_prefix) {
        session->mtp_prefix_valid = true;
    }
    return skippy_success(out_error);
}

int skippy_external_decode_observe(llama_context * ctx, const llama_batch & batch) {
    skippy_session * session = g_skippy_external_decode_session;
    if (session == nullptr || ctx != session->ctx || batch.n_tokens <= 0 ||
            session->stage_model == nullptr || session->stage_model->model == nullptr) {
        return 0;
    }
    skippy_mark_context_work_pending(session->stage_model);
    if (session->stage_model->model->arch != LLM_ARCH_INKLING) {
        return 0;
    }
    session->pending_activation_exports.clear();
    if (batch.embd != nullptr) {
        const size_t value_count = static_cast<size_t>(batch.n_tokens)*
                static_cast<size_t>(llama_model_n_embd(session->stage_model->model));
        skippy_pending_activation_part part;
        std::memcpy(part.desc.identity, SKIPPY_MTP_INPUT_EMBEDDING_IDENTITY, sizeof(part.desc.identity));
        part.desc.ggml_type = GGML_TYPE_F32;
        part.desc.rank = 2;
        part.desc.token_axis = 1;
        part.desc.flags = SKIPPY_ACTIVATION_PART_OPTIONAL;
        part.desc.dimensions[0] = llama_model_n_embd(session->stage_model->model);
        part.desc.dimensions[1] = batch.n_tokens;
        part.desc.byte_strides[0] = sizeof(float);
        part.desc.byte_strides[1] = static_cast<uint64_t>(part.desc.dimensions[0])*sizeof(float);
        part.desc.payload_bytes = value_count*sizeof(float);
        part.payload.resize(part.desc.payload_bytes);
        std::memcpy(part.payload.data(), batch.embd, part.payload.size());
        session->pending_activation_exports.push_back(std::move(part));
    }

    const llama_pos token_start = batch.pos != nullptr ? batch.pos[0] : session->n_past;
    skippy_error * error = nullptr;
    const skippy_status status = skippy_mtp_sync_target_inputs(
            session,
            batch.token,
            batch.embd,
            static_cast<size_t>(batch.n_tokens),
            token_start,
            &error);
    if (status == SKIPPY_STATUS_OK) {
        return 0;
    }

    fprintf(stderr, "skippy: external Inkling MTP sync failed: %s\n",
            error != nullptr && error->message != nullptr ? error->message : "unknown error");
    skippy_error_free(error);
    // The target decode has already committed. Degrade only the MTP sidecar
    // until a later target decode can re-prime it.
    skippy_mtp_clear_session_state(session);
    return -1;
}

skippy_graph_build_inputs skippy_external_decode_build_inputs(llama_context * ctx) {
    skippy_session * session = g_skippy_external_decode_session;
    if (session == nullptr || ctx != session->ctx || session->stage_model == nullptr) {
        return {};
    }
    return skippy_graph_build_inputs_from_config(&session->stage_model->config);
}

enum skippy_status skippy_mtp_propose_next(
        skippy_session * session,
        llama_token predicted_token,
        size_t max_draft_tokens,
        skippy_native_mtp_draft * out_mtp_draft,
        struct skippy_error ** out_error) {
    if (out_mtp_draft != nullptr) {
        *out_mtp_draft = {
            1,
            false,
            0,
            {},
            0,
        };
    }
    if (!skippy_mtp_available(session) ||
            !session->mtp_prefix_valid ||
            predicted_token < 0 ||
            !session->mtp_has_pending_h) {
        return skippy_success(out_error);
    }

    llama_context * mtp_ctx = session->stage_model->mtp_ctx;
    skippy_mtp_depth_scope mtp_depth_scope(mtp_ctx);
    const bool mtp_shares_target_memory = skippy_mtp_shares_target_memory(mtp_ctx, session->ctx);
    // MTP consumes complete pre-norm output rows, which can be wider than
    // token embeddings (for example DeepSeek4 hyperconnections).
    const int32_t n_hidden = llama_model_n_embd_out(session->stage_model->model);
    if (n_hidden <= 0 || n_hidden != llama_model_n_embd_out(llama_get_model(mtp_ctx))) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "target and MTP hidden widths differ");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session->mtp_pending_h.size() != static_cast<size_t>(n_hidden)) {
        return skippy_success(out_error);
    }

    size_t draft_limit = std::min<size_t>(
            max_draft_tokens,
            SKIPPY_NATIVE_MTP_MAX_DRAFT_TOKENS);
    const bool chain_heads = mtp_depth_scope.chains_heads();
    const bool chain_at_same_position = mtp_depth_scope.chains_at_same_position();
    if (chain_heads) {
        draft_limit = std::min<size_t>(draft_limit, mtp_depth_scope.depth_count);
    }
    llama_token token = predicted_token;
    int64_t elapsed_us = 0;
    int32_t token_count = 0;
    // One past the highest position this proposal writes. The chained branch
    // clears its own rows before every decode, so only the single-row loop can
    // leave an overhang behind the accepted prefix.
    llama_pos proposal_end_pos = session->n_past + 1;
    if (chain_heads && !chain_at_same_position) {
        std::vector<llama_token> prefix_tokens = { predicted_token };
        prefix_tokens.reserve(draft_limit + 1);
        std::vector<float> prefix_hidden = session->mtp_pending_h;
        prefix_hidden.reserve((draft_limit + 1)*static_cast<size_t>(n_hidden));

        for (size_t depth = 0; depth < draft_limit; ++depth) {
            const int32_t n_rows = static_cast<int32_t>(prefix_tokens.size());
            llama_batch batch = llama_batch_init(n_rows, n_hidden, 1);
            batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)*static_cast<size_t>(n_rows)));
            if (batch.token == nullptr) {
                llama_batch_free(batch);
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to allocate chained MTP proposal batch");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            batch.n_tokens = n_rows;
            for (int32_t row = 0; row < n_rows; ++row) {
                batch.token[row] = prefix_tokens[static_cast<size_t>(row)];
                batch.pos[row] = session->n_past + row;
                batch.n_seq_id[row] = 1;
                batch.seq_id[row][0] = session->seq_id;
                batch.logits[row] = row == n_rows - 1 ? 1 : 0;
                std::memcpy(
                        batch.embd + static_cast<size_t>(row)*n_hidden,
                        prefix_hidden.data() + static_cast<size_t>(row)*n_hidden,
                        static_cast<size_t>(n_hidden)*sizeof(float));
            }

            if (llama_memory_t memory = mtp_ctx->get_memory()) {
                llama_memory_seq_rm(memory, session->seq_id, session->n_past, -1);
            }
            mtp_depth_scope.select(depth);
            const int64_t t_start_us = ggml_time_us();
            const skippy_graph_build_inputs graph_build_inputs = skippy_graph_build_inputs_from_config(
                    &session->stage_model->config);
            int32_t rc = 0;
            const bool decoded = skippy_contain_native_failure(
                    out_error, [&] { rc = mtp_ctx->decode(batch, graph_build_inputs); });
            elapsed_us += ggml_time_us() - t_start_us;
            if (!decoded) {
                std::free(batch.token);
                batch.token = nullptr;
                llama_batch_free(batch);
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            if (rc != 0) {
                std::free(batch.token);
                batch.token = nullptr;
                llama_batch_free(batch);
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed for chained MTP proposal");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }

            token = skippy_greedy_sample_context(llama_get_model(mtp_ctx), mtp_ctx, -1);
            const skippy_status sample_status = skippy_store_sampled_token(token, &token, out_error);
            if (sample_status != SKIPPY_STATUS_OK) {
                skippy_mtp_clear_session_state(session);
                return sample_status;
            }
            if (out_mtp_draft != nullptr) {
                out_mtp_draft->token_ids[token_count] = token;
            }
            ++token_count;
            const float * h_next = llama_get_embeddings_nextn_ith(mtp_ctx, n_rows - 1);
            if (h_next != nullptr && depth + 1 < draft_limit) {
                prefix_tokens.push_back(token);
                prefix_hidden.insert(prefix_hidden.end(), h_next, h_next + n_hidden);
            }
            std::free(batch.token);
            batch.token = nullptr;
            llama_batch_free(batch);
            if (h_next == nullptr) {
                break;
            }
        }
    } else {
        llama_pos pos = session->n_past;
        int32_t n_seq_id = 1;
        llama_seq_id seq_id = session->seq_id;
        llama_seq_id * seq_ids = &seq_id;
        int8_t logits = 1;
        std::vector<float> proposal_h = session->mtp_pending_h;
        // Drop any sidecar row at or past the position this proposal resumes
        // at. The verify path only prunes a pending draft when its position
        // falls inside the token window the target just committed; every other
        // way a draft is discarded clears the session bookkeeping and leaves
        // the unverified rows resident. Those rows put seq_pos_max at or past
        // the resume position, and llama_batch_allocr::init then rejects the
        // batch outright: harmless for plain RoPE, fatal for M-RoPE, which
        // requires the batch to start strictly past seq_pos_max. Anything from
        // session->n_past onwards is speculative by construction, so this is
        // safe whether or not the sidecar shares the target's memory. The
        // chained branch above already clears the same way before each decode.
        if (llama_memory_t memory = mtp_ctx->get_memory()) {
            llama_memory_seq_rm(memory, session->seq_id, session->n_past, -1);
        }
        for (size_t depth = 0; depth < draft_limit; ++depth) {
            mtp_depth_scope.select(depth);
            llama_batch batch = {
                /*n_tokens =*/ 1,
                /*token    =*/ &token,
                /*embd     =*/ proposal_h.data(),
                /*pos      =*/ &pos,
                /*n_seq_id =*/ &n_seq_id,
                /*seq_id   =*/ &seq_ids,
                /*logits   =*/ &logits,
            };

            const int64_t t_start_us = ggml_time_us();
            const skippy_graph_build_inputs graph_build_inputs = skippy_graph_build_inputs_from_config(
                    &session->stage_model->config);
            int32_t rc = 0;
            const bool decoded = skippy_contain_native_failure(
                    out_error, [&] { rc = mtp_ctx->decode(batch, graph_build_inputs); });
            elapsed_us += ggml_time_us() - t_start_us;
            if (!decoded) {
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
            if (rc != 0) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "llama_decode failed for MTP sidecar proposal");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }

            token = skippy_greedy_sample_context(llama_get_model(mtp_ctx), mtp_ctx, -1);
            const skippy_status sample_status = skippy_store_sampled_token(token, &token, out_error);
            if (sample_status != SKIPPY_STATUS_OK) {
                skippy_mtp_clear_session_state(session);
                std::free(batch.token);
                batch.token = nullptr;
                llama_batch_free(batch);
                return sample_status;
            }
            if (out_mtp_draft != nullptr) {
                out_mtp_draft->token_ids[token_count] = token;
            }
            ++token_count;
            if (!mtp_shares_target_memory && !chain_at_same_position) {
                ++pos;
            }
            const float * h_next = llama_get_embeddings_nextn_ith(mtp_ctx, 0);
            if (h_next == nullptr) {
                break;
            }
            std::memcpy(proposal_h.data(), h_next, static_cast<size_t>(n_hidden)*sizeof(float));
        }
        if (!mtp_shares_target_memory && !chain_at_same_position) {
            proposal_end_pos = pos;
        }
    }

    if (out_mtp_draft != nullptr && token_count > 0) {
        out_mtp_draft->version = 1;
        out_mtp_draft->available = true;
        out_mtp_draft->token_count = token_count;
        out_mtp_draft->proposal_compute_us = elapsed_us;
        session->mtp_has_pending_draft = true;
        session->mtp_pending_draft_pos = session->n_past;
        session->mtp_pending_draft_token = out_mtp_draft->token_ids[0];
        session->mtp_pending_draft_end = proposal_end_pos;
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_mtp_propose_after_row(
        skippy_session * session,
        llama_pos token_start,
        size_t anchor_row,
        llama_token predicted_token,
        size_t max_draft_tokens,
        skippy_native_mtp_draft * out_mtp_draft,
        skippy_error ** out_error) {
    if (out_mtp_draft != nullptr) {
        *out_mtp_draft = {1, false, 0, {}, 0};
    }
    const llama_pos anchor_end = token_start + static_cast<llama_pos>(anchor_row) + 1;
    if (!skippy_mtp_available(session) || !session->mtp_prefix_valid || anchor_end > session->n_past) {
        return skippy_success(out_error);
    }
    const llama_pos verified_end = session->n_past;
    if (anchor_end < verified_end) {
        // The rows past the anchor stay in the trunk cache until the caller
        // trims them; the sidecar starts over from the anchor now.
        skippy_mtp_trim_session_state(session, anchor_end);
        if (!session->mtp_has_pending_h) {
            return skippy_success(out_error);
        }
    }
    session->n_past = anchor_end;
    const enum skippy_status status =
            skippy_mtp_propose_next(session, predicted_token, max_draft_tokens, out_mtp_draft, out_error);
    session->n_past = verified_end;
    return status;
}
