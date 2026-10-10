#pragma once

#include "skippy/common.h"

#include "llama.h"

#include <cstddef>

struct skippy_native_mtp_draft;
struct skippy_session;

extern thread_local skippy_session * g_skippy_external_decode_session;
bool skippy_mtp_available(const skippy_session * session);

bool skippy_mtp_greedy_sampling_fastpath_enabled();
enum skippy_status skippy_mtp_sync_target_inputs(
        skippy_session * session,
        const llama_token * token_ids,
        const float * input_embeddings,
        size_t token_count,
        llama_pos token_start,
        skippy_error ** out_error);
const float * skippy_mtp_input_embeddings_from_activation_frame(
        const skippy_activation_desc * desc,
        const void * payload,
        size_t token_count);
enum skippy_status skippy_mtp_propose_next(
        skippy_session * session,
        llama_token predicted_token,
        size_t max_draft_tokens,
        skippy_native_mtp_draft * out_mtp_draft,
        skippy_error ** out_error);
// Proposes after row `anchor_row` of the batch just verified at `token_start`,
// continuing from that row's predicted (correction) token. Rows past the
// anchor were conditioned on a rejected proposal; their sidecar rows are
// dropped first.
enum skippy_status skippy_mtp_propose_after_row(
        skippy_session * session,
        llama_pos token_start,
        size_t anchor_row,
        llama_token predicted_token,
        size_t max_draft_tokens,
        skippy_native_mtp_draft * out_mtp_draft,
        skippy_error ** out_error);
