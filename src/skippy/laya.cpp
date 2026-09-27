#include "skippy/laya.h"

#include "skippy/errors.h"

#include "laya.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct skippy_laya_model {
    laya_model * model = nullptr;
    laya_context * context = nullptr;
    bool warmed_up = false;
    // laya_encode reuses one scheduler and its compute buffers.
    std::mutex lane;

    ~skippy_laya_model() {
        if (context != nullptr) {
            laya_free(context);
        }
        if (model != nullptr) {
            laya_model_free(model);
        }
    }
};

namespace {

enum skippy_status laya_fail(
        skippy_error ** out_error,
        enum skippy_status status,
        const std::string & message) {
    skippy_set_error(out_error, status, message.c_str());
    return status;
}

// Runs one exported entry point. Everything the body does, including building
// error messages and allocating buffers, happens inside the containment, so no
// C++ exception can unwind into the Rust caller.
template <typename Body>
enum skippy_status laya_entry(skippy_error ** out_error, Body && body) {
    enum skippy_status status = SKIPPY_STATUS_RUNTIME_ERROR;
    if (!skippy_contain_native_failure(out_error, [&] { status = body(); })) {
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    return status;
}

// Rejects a GGUF whose declared budgets would make a single pass unbounded:
// each pass builds dense n x n attention masks, so n must stay small.
enum skippy_status validate_budgets(const laya_hparams & hp, skippy_error ** out_error) {
    const int32_t max_len = hp.max_len > 0 ? hp.max_len : 1024;
    const int32_t head_max_len = hp.head_max_len > 0 ? hp.head_max_len : 256;
    if (max_len > SKIPPY_LAYA_MAX_SEQUENCE_TOKENS) {
        return laya_fail(out_error, SKIPPY_STATUS_MODEL_ERROR,
                "laya.max_len " + std::to_string(max_len) + " exceeds the supported " +
                std::to_string(SKIPPY_LAYA_MAX_SEQUENCE_TOKENS) + " tokens");
    }
    if (head_max_len > max_len) {
        return laya_fail(out_error, SKIPPY_STATUS_MODEL_ERROR,
                "laya.head_max_len " + std::to_string(head_max_len) + " exceeds laya.max_len " +
                std::to_string(max_len));
    }
    if (hp.n_act <= 0) {
        return laya_fail(out_error, SKIPPY_STATUS_MODEL_ERROR, "laya model declares no action classes");
    }
    return skippy_success(out_error);
}

} // namespace

extern "C" enum skippy_status skippy_laya_model_open(
        const char * model_path,
        int32_t n_threads,
        const char * device,
        struct skippy_laya_model ** out_model,
        struct skippy_error ** out_error) {
    if (model_path == nullptr || out_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model_path and out_model are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_model = nullptr;
    skippy_laya_model * handle = nullptr;
    const enum skippy_status status = laya_entry(out_error, [&] {
        handle = new skippy_laya_model();
        handle->model = laya_model_load_from_file(model_path, device == nullptr ? "" : device);
        const enum skippy_status budgets = validate_budgets(laya_model_hparams(handle->model), out_error);
        if (budgets != SKIPPY_STATUS_OK) {
            return budgets;
        }
        handle->context = laya_init(handle->model, std::max(1, n_threads));
        return skippy_success(out_error);
    });
    if (status != SKIPPY_STATUS_OK) {
        delete handle;
        return status;
    }
    *out_model = handle;
    return SKIPPY_STATUS_OK;
}

extern "C" void skippy_laya_model_free(struct skippy_laya_model * model) {
    delete model;
}

extern "C" enum skippy_status skippy_laya_model_info_v1(
        const struct skippy_laya_model * model,
        struct skippy_laya_info_v1 * out_info,
        struct skippy_error ** out_error) {
    return laya_entry(out_error, [&] {
        if (model == nullptr || model->model == nullptr || out_info == nullptr) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_info are required");
        }
        if (out_info->struct_size != sizeof(skippy_laya_info_v1) ||
                out_info->abi_version != SKIPPY_LAYA_INFO_V1_ABI_VERSION) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "laya info descriptor uses an incompatible ABI");
        }
        const laya_hparams & hp = laya_model_hparams(model->model);
        out_info->max_len = hp.max_len > 0 ? hp.max_len : 1024;
        out_info->head_max_len = hp.head_max_len > 0 ? hp.head_max_len : 256;
        out_info->max_markers = LAYA_MAX_MARKERS;
        out_info->n_act = hp.n_act;
        out_info->cls_token_id = laya_vocab_bos(model->model);
        out_info->sep_token_id = laya_vocab_sep(model->model);
        out_info->mask_token_id = laya_vocab_mask(model->model);
        out_info->n_embd = hp.n_embd;
        out_info->n_layer = hp.n_layer;
        out_info->parameter_count = laya_model_n_params(model->model);
        for (int i = 0; i < SKIPPY_LAYA_QTYPE_COUNT; ++i) {
            out_info->temperature[i] = i < (int) hp.temperature.size() ? hp.temperature[i] : 0.0f;
        }
        return skippy_success(out_error);
    });
}

extern "C" enum skippy_status skippy_laya_tokenize(
        const struct skippy_laya_model * model,
        const char * text,
        size_t text_len,
        int32_t * out_tokens,
        size_t capacity,
        size_t * out_token_count,
        struct skippy_error ** out_error) {
    return laya_entry(out_error, [&] {
        if (model == nullptr || model->model == nullptr || out_token_count == nullptr ||
                (text == nullptr && text_len != 0) || (out_tokens == nullptr && capacity != 0)) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "invalid laya tokenize arguments");
        }
        const std::vector<int32_t> tokens =
                laya_tokenize(model->model, std::string(text == nullptr ? "" : text, text_len));
        *out_token_count = tokens.size();
        if (tokens.size() > capacity) {
            return laya_fail(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "laya token buffer is too small");
        }
        if (!tokens.empty()) {
            std::memcpy(out_tokens, tokens.data(), tokens.size() * sizeof(int32_t));
        }
        return skippy_success(out_error);
    });
}

namespace {

enum skippy_status validate_read(
        const laya_hparams & hp,
        size_t token_count,
        const int32_t * marker_positions,
        size_t marker_count,
        const struct skippy_laya_sequence * sequences,
        size_t sequence_count,
        size_t logits_capacity,
        size_t act_logits_capacity,
        skippy_error ** out_error) {
    const size_t n_act = (size_t) std::max(0, hp.n_act);
    const size_t max_len = (size_t) (hp.max_len > 0 ? hp.max_len : 1024);
    if (logits_capacity / LAYA_MAX_MARKERS < sequence_count ||
            (n_act > 0 && act_logits_capacity / n_act < sequence_count)) {
        return laya_fail(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "laya output buffers are too small");
    }
    for (size_t s = 0; s < sequence_count; ++s) {
        const skippy_laya_sequence & seq = sequences[s];
        if (seq.token_count == 0 || seq.token_count > max_len ||
                seq.token_offset > token_count || seq.token_count > token_count - seq.token_offset) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "laya sequence " + std::to_string(s) + " has an invalid token range");
        }
        if (seq.qtype < 0 || seq.qtype >= SKIPPY_LAYA_QTYPE_COUNT) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "laya sequence " + std::to_string(s) + " has an unknown question type");
        }
        if (seq.marker_count == 0 || seq.marker_count > LAYA_MAX_MARKERS ||
                seq.marker_offset > marker_count || seq.marker_count > marker_count - seq.marker_offset) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "laya sequence " + std::to_string(s) + " has an invalid marker range");
        }
        for (size_t m = 0; m < seq.marker_count; ++m) {
            const int32_t position = marker_positions[seq.marker_offset + m];
            if (position < 0 || (size_t) position >= seq.token_count) {
                return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "laya sequence " + std::to_string(s) + " has a marker outside its tokens");
            }
        }
    }
    return SKIPPY_STATUS_OK;
}

// Scores sequences [first, last) in one packed forward pass.
enum skippy_status read_pass(
        skippy_laya_model * model,
        const int32_t * tokens,
        const int32_t * marker_positions,
        const struct skippy_laya_sequence * sequences,
        size_t first,
        size_t last,
        size_t pass_tokens,
        float * out_logits,
        float * out_act_logits,
        skippy_error ** out_error) {
    const size_t n_seqs = last - first;
    const size_t n_act = (size_t) std::max(0, laya_model_hparams(model->model).n_act);

    std::vector<int32_t> batch_tokens(pass_tokens);
    std::vector<int32_t> positions(pass_tokens);
    std::vector<int32_t> seq_id(pass_tokens);
    std::vector<int32_t> qtype(pass_tokens);
    std::vector<int32_t> marker_pos(LAYA_MAX_MARKERS * n_seqs, 0);
    std::vector<int32_t> marker_mask(LAYA_MAX_MARKERS * n_seqs, 0);
    std::vector<int32_t> seq_start(n_seqs);

    size_t t = 0;
    for (size_t s = 0; s < n_seqs; ++s) {
        const skippy_laya_sequence & seq = sequences[first + s];
        seq_start[s] = (int32_t) t;
        for (size_t i = 0; i < seq.token_count; ++i) {
            batch_tokens[t] = tokens[seq.token_offset + i];
            positions[t] = (int32_t) i;
            seq_id[t] = (int32_t) s;
            qtype[t] = seq.qtype;
            ++t;
        }
        for (size_t m = 0; m < seq.marker_count; ++m) {
            marker_pos[m + LAYA_MAX_MARKERS * s] = seq_start[s] + marker_positions[seq.marker_offset + m];
            marker_mask[m + LAYA_MAX_MARKERS * s] = 1;
        }
    }

    laya_batch batch;
    batch.n_tokens = (int32_t) pass_tokens;
    batch.n_seqs = (int32_t) n_seqs;
    batch.tokens = batch_tokens.data();
    batch.positions = positions.data();
    batch.seq_id = seq_id.data();
    batch.qtype = qtype.data();
    batch.marker_pos = marker_pos.data();
    batch.marker_mask = marker_mask.data();
    batch.seq_start = seq_start.data();

    laya_result result;
    if (laya_encode(model->context, batch, result) != 0 || result.n_markers_max != LAYA_MAX_MARKERS ||
            (size_t) result.n_act != n_act || result.logits.size() != n_seqs * LAYA_MAX_MARKERS ||
            result.act_logits.size() != n_seqs * n_act) {
        return laya_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "laya forward pass failed");
    }
    for (size_t s = 0; s < n_seqs; ++s) {
        const size_t markers = sequences[first + s].marker_count;
        float * logits = out_logits + (first + s) * LAYA_MAX_MARKERS;
        for (size_t m = 0; m < LAYA_MAX_MARKERS; ++m) {
            logits[m] = m < markers ? result.logits[s * LAYA_MAX_MARKERS + m] : 0.0f;
        }
    }
    std::memcpy(out_act_logits + first * n_act, result.act_logits.data(), n_seqs * n_act * sizeof(float));
    return SKIPPY_STATUS_OK;
}

} // namespace

extern "C" enum skippy_status skippy_laya_read(
        struct skippy_laya_model * model,
        const int32_t * tokens,
        size_t token_count,
        const int32_t * marker_positions,
        size_t marker_count,
        const struct skippy_laya_sequence * sequences,
        size_t sequence_count,
        float * out_logits,
        size_t logits_capacity,
        float * out_act_logits,
        size_t act_logits_capacity,
        struct skippy_error ** out_error) {
    return laya_entry(out_error, [&] {
        if (model == nullptr || model->model == nullptr || model->context == nullptr ||
                tokens == nullptr || marker_positions == nullptr || sequences == nullptr ||
                out_logits == nullptr || out_act_logits == nullptr || sequence_count == 0) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "invalid laya read arguments");
        }
        const enum skippy_status valid = validate_read(laya_model_hparams(model->model), token_count,
                marker_positions, marker_count, sequences, sequence_count, logits_capacity,
                act_logits_capacity, out_error);
        if (valid != SKIPPY_STATUS_OK) {
            return valid;
        }

        // Every sequence fits in max_len, which open capped at
        // SKIPPY_LAYA_MAX_SEQUENCE_TOKENS, so packing sequences up to max_len
        // tokens bounds every pass by the model's own sequence budget.
        const laya_hparams & hp = laya_model_hparams(model->model);
        const size_t pass_budget = (size_t) (hp.max_len > 0 ? hp.max_len : 1024);
        std::lock_guard<std::mutex> lane(model->lane);
        size_t first = 0;
        while (first < sequence_count) {
            size_t last = first;
            size_t pass_tokens = 0;
            while (last < sequence_count &&
                    pass_tokens + sequences[last].token_count <= pass_budget) {
                pass_tokens += sequences[last].token_count;
                ++last;
            }
            const enum skippy_status pass = read_pass(model, tokens, marker_positions, sequences,
                    first, last, pass_tokens, out_logits, out_act_logits, out_error);
            if (pass != SKIPPY_STATUS_OK) {
                return pass;
            }
            first = last;
        }
        return skippy_success(out_error);
    });
}

extern "C" enum skippy_status skippy_laya_model_memory_v1(
        struct skippy_laya_model * model,
        struct skippy_laya_memory_v1 * out_info,
        struct skippy_error ** out_error) {
    return laya_entry(out_error, [&] {
        if (model == nullptr || model->model == nullptr || model->context == nullptr || out_info == nullptr) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_info are required");
        }
        if (out_info->struct_size != sizeof(skippy_laya_memory_v1) ||
                out_info->abi_version != SKIPPY_LAYA_MEMORY_V1_ABI_VERSION) {
            return laya_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "laya memory descriptor uses an incompatible ABI");
        }
        const laya_hparams & hp = laya_model_hparams(model->model);
        const size_t max_len = (size_t) (hp.max_len > 0 ? hp.max_len : 1024);
        if (!model->warmed_up) {
            // One full-length sequence makes the scheduler reserve the
            // buffers of the largest pass a read can build.
            std::vector<int32_t> tokens(max_len, laya_vocab_sep(model->model));
            tokens[0] = laya_vocab_bos(model->model);
            const int32_t marker = 1;
            skippy_laya_sequence sequence = { 0, max_len, SKIPPY_LAYA_QTYPE_NOUL, 0, 1 };
            std::vector<float> logits(LAYA_MAX_MARKERS);
            std::vector<float> act((size_t) std::max(1, hp.n_act));
            std::lock_guard<std::mutex> lane(model->lane);
            const enum skippy_status status = read_pass(model, tokens.data(), &marker, &sequence, 0, 1,
                    max_len, logits.data(), act.data(), out_error);
            if (status != SKIPPY_STATUS_OK) {
                return status;
            }
            model->warmed_up = true;
        }
        std::lock_guard<std::mutex> lane(model->lane);
        out_info->weights_bytes = laya_model_weights_bytes(model->model);
        out_info->compute_bytes = laya_context_compute_bytes(model->context);
        // Two dense f32 [n, n] masks, built on the host for every pass.
        out_info->host_scratch_bytes = (uint64_t) 2 * max_len * max_len * sizeof(float);
        out_info->on_accelerator = laya_model_on_accelerator(model->model);
        return skippy_success(out_error);
    });
}
