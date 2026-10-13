#pragma once

#include "llama-context.h"
#include "llama-model.h"
#include "skippy/runtime_types.h"
#include "skippy/execution.h"
#include "skippy/activation_internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

// Compare ordinary context graph reuse with a separately loaded context that
// rebuilds at each shape. Serialized sequence state avoids a second, per-cache
// description of which storage belongs to a model's recurrent/attention state.
struct live_graph_snapshot {
    std::vector<float> logits;
    std::vector<uint8_t> state;
};

class fresh_graph_scope {
public:
    explicit fresh_graph_scope(bool enabled) : enabled(enabled) {
        if (!enabled) return;
        if (const char * value = std::getenv("LLAMA_GRAPH_REUSE_DISABLE")) previous = value;
        set("1");
    }
    ~fresh_graph_scope() {
        if (enabled) set(previous ? previous->c_str() : nullptr);
    }
private:
    bool enabled;
    std::optional<std::string> previous;
    static void set(const char * value) {
#ifdef _WIN32
        _putenv_s("LLAMA_GRAPH_REUSE_DISABLE", value ? value : "");
#else
        if (value) setenv("LLAMA_GRAPH_REUSE_DISABLE", value, 1);
        else unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
#endif
    }
};

class live_graph_reference {
public:
    ~live_graph_reference() {
        skippy_error * error = nullptr;
        if (session) skippy_session_free(session, &error);
        skippy_error_free(error);
        error = nullptr;
        if (model) skippy_model_free(model, &error);
        skippy_error_free(error);
    }

    bool open(const std::vector<const char *> & parts, const skippy_runtime_config & config) {
        fresh_graph_scope fresh(true);
        skippy_error * error = nullptr;
        const auto status = skippy_model_open_from_parts(parts.data(), parts.size(), &config, &model, &error);
        if (!finished(status, error) || !model) return false;
        error = nullptr;
        const auto session_status = skippy_session_create(model, &session, &error);
        return finished(session_status, error);
    }

    void clear_with(skippy_model * actual) {
        llama_memory_clear(llama_get_memory(actual->ctx), true);
        llama_memory_clear(llama_get_memory(model->ctx), true);
    }

    bool prefill(const llama_token * tokens, size_t count, const live_graph_reference * source = nullptr) {
        skippy_error * error = nullptr;
        const auto status = skippy_prefill_chunk_frame(session, tokens, count,
                source ? &source->desc : nullptr, source ? source->payload.data() : nullptr,
                &desc, payload.data(), payload.size(), &bytes, &error);
        return finished(status, error);
    }

    bool step(llama_token token, const skippy_sampling_config * sampling,
            llama_token expected, const live_graph_reference * source = nullptr) {
        skippy_error * error = nullptr;
        llama_token prediction = -1;
        const auto status = skippy_decode_step_frame_sampled(session, token, sampling,
                source ? &source->desc : nullptr, source ? source->payload.data() : nullptr,
                &desc, payload.data(), payload.size(), &bytes, &prediction, &error);
        return finished(status, error) && (expected < 0 || prediction == expected);
    }

    bool compare(skippy_model * actual, bool logits) const {
        live_graph_snapshot left, right;
        return snapshot(actual->ctx, session->seq_id, left, logits) &&
               snapshot(model->ctx, session->seq_id, right, logits) && compare_snapshots(left, right);
    }

    static bool snapshot(llama_context * context, llama_seq_id seq, live_graph_snapshot & output,
            bool logits = true) {
        llama_synchronize(context);
        if (logits) {
            const auto count = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(context)));
            const float * values = llama_get_logits_ith(context, -1);
            if (!values) return false;
            output.logits.assign(values, values + count);
        }
        const auto size = llama_state_seq_get_size(context, seq);
        if (size == 0) return false;
        output.state.resize(size);
        return llama_state_seq_get_data(context, output.state.data(), size, seq) == size;
    }

    static bool compare_snapshots(const live_graph_snapshot & left, const live_graph_snapshot & right) {
        if (left.logits.size() != right.logits.size() || left.state != right.state) {
            std::fprintf(stderr, "fresh/reused graph sequence-state mismatch\n");
            return false;
        }
        for (size_t index = 0; index < left.logits.size(); ++index) {
            const float a = left.logits[index], b = right.logits[index];
            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > 1e-5f * (1 + std::abs(b))) {
                std::fprintf(stderr, "fresh/reused graph logit mismatch at %zu\n", index);
                return false;
            }
        }
        return true;
    }

    static bool check_capacity_preflight(skippy_session * actual, const llama_token * tokens, size_t count) {
        live_graph_snapshot before, after;
        if (!snapshot(actual->ctx, actual->seq_id, before, false)) return false;
        const auto position = actual->n_past;
        uint8_t payload = 0;
        size_t required = 0;
        skippy_activation_desc desc{};
        skippy_error * error = nullptr;
        const auto status = skippy_prefill_chunk_frame(actual, tokens, count, nullptr, nullptr,
                &desc, &payload, 1, &required, &error);
        skippy_error_free(error);
        return status == SKIPPY_STATUS_BUFFER_TOO_SMALL && required > 1 && actual->n_past == position &&
               !actual->reset_required && snapshot(actual->ctx, actual->seq_id, after, false) &&
               before.state == after.state;
    }

    static bool check_activation_prefill_sampling(skippy_session * actual,
            const llama_token * tokens, size_t count) {
        skippy_error * error = nullptr;
        auto status = skippy_session_configure_chat_sampling(
                actual, nullptr, "{\"grammar\":\"\"}", 0, &error);
        if (!finished(status, error)) return false;
        if (actual->sampling_backend_enabled) {
            std::fprintf(stderr, "activation-export stage attached a single-output backend sampler\n");
            return false;
        }
        std::vector<uint8_t> payload(1024 * 1024);
        for (int pass = 0; pass < 2; ++pass) {
            skippy_activation_desc desc{};
            size_t bytes = 0;
            status = skippy_prefill_chunk_frame(actual, tokens, count, nullptr, nullptr,
                    &desc, payload.data(), payload.size(), &bytes, &error);
            if (!finished(status, error) || bytes == 0 || desc.part_count == 0) return false;
            error = nullptr;
            status = skippy_session_reset(actual, &error);
            if (!finished(status, error) || actual->sampling_backend_enabled) return false;
            error = nullptr;
        }
        return true;
    }

    static bool check_reservation_overflow(skippy_session * actual) {
        uint8_t payload = 0;
        skippy_activation_desc desc{};
        skippy_error * error = nullptr;
        const auto status = skippy_prepare_output_activation_frame(actual, 1, &payload, 1,
                nullptr, &desc, nullptr, &error, skippy_activation_frame_shape::live_within_reservation);
        const bool rejected = status == SKIPPY_STATUS_RUNTIME_ERROR && actual->reset_required &&
                error != nullptr && error->status == SKIPPY_STATUS_RUNTIME_ERROR;
        skippy_error_free(error);
        return rejected;
    }

private:
    skippy_model * model = nullptr;
    skippy_session * session = nullptr;
    skippy_activation_desc desc{};
    std::vector<uint8_t> payload = std::vector<uint8_t>(1024 * 1024);
    size_t bytes = 0;

    static bool finished(skippy_status status, skippy_error * error) {
        if (status != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "fresh graph execution failed: %s\n", error ? error->message : "unknown");
        }
        skippy_error_free(error);
        return status == SKIPPY_STATUS_OK;
    }
};
