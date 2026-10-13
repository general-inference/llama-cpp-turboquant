#include "skippy.h"
#include "../runtime_types.h"
#include "llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "CHECK failed: %s at %s:%d\n", #condition, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

namespace {

bool require_recurrent = false;

void ok(skippy_status status, skippy_error * error) {
    if (status != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "native failure: %s\n", error ? error->message : "no detail");
        std::exit(1);
    }
    skippy_error_free(error);
}

void prefill(skippy_session * session, const std::vector<llama_token> & tokens) {
    size_t bytes = 0;
    skippy_error * error = nullptr;
    const auto status = skippy_prefill_chunk(session, tokens.data(), tokens.size(),
            nullptr, 0, nullptr, 0, &bytes, &error);
    ok(status, error);
}

skippy_sampling_config sampling_config() {
    skippy_sampling_config config{};
    config.version = 1;
    config.flags = SKIPPY_SAMPLING_FLAG_ENABLED;
    config.repeat_penalty = 1.0f;
    config.reasoning_budget_tokens = -1;
    config.logit_bias_count = 1;
    config.logit_bias[0] = {42, 100000.0f};
    return config;
}

void configure(skippy_session * session, const skippy_sampling_config & sampling,
        bool lazy, size_t prompt_count) {
    const char * metadata = lazy
        ? R"({"grammar":"root ::= <[42]> <[43]> <[44]>","grammar_lazy":true,"grammar_triggers":[{"type":0,"token":42}]})"
        : R"({"grammar":"root ::= <[42]> <[43]> <[44]>"})";
    skippy_error * error = nullptr;
    const auto status = skippy_session_configure_chat_sampling(
            session, &sampling, metadata, prompt_count, &error);
    ok(status, error);
    CHECK(session->grammar_sampler != nullptr);
}

llama_token decode(skippy_session * session, const skippy_sampling_config & sampling, llama_token token) {
    llama_token predicted = -1;
    size_t bytes = 0;
    skippy_error * error = nullptr;
    const auto status = skippy_decode_step_sampled(session, token, &sampling,
            nullptr, 0, nullptr, 0, &bytes, &predicted, &error);
    ok(status, error);
    return predicted;
}

struct snapshot {
    bool full;
    int32_t layers;
    skippy_kv_page_desc page{};
    std::vector<uint8_t> payload;
    std::vector<uint8_t> recurrent;

    snapshot(skippy_session * source, bool full_state) : full(full_state),
            layers(source->stage_model->model->hparams.n_layer()) {
        skippy_error * error = nullptr;
        size_t bytes = 0;
        auto export_payload = [&](void * output, size_t capacity) {
            return full ? skippy_export_full_state(source, 0, layers, output, capacity, &bytes, &error)
                : skippy_export_kv_page(source, 0, layers, 0, source->n_past, &page,
                        output, capacity, &bytes, &error);
        };
        const auto status = export_payload(nullptr, 0);
        CHECK(status == SKIPPY_STATUS_BUFFER_TOO_SMALL || status == SKIPPY_STATUS_OK);
        skippy_error_free(error);
        error = nullptr;
        CHECK(bytes > 0);
        payload.resize(bytes);
        const auto exported = export_payload(payload.data(), payload.size());
        ok(exported, error);
        if (!full) {
            error = nullptr;
            bytes = 0;
            const auto recurrent_status = skippy_export_recurrent_state(source, nullptr, 0, &bytes, &error);
            CHECK(recurrent_status == SKIPPY_STATUS_BUFFER_TOO_SMALL || recurrent_status == SKIPPY_STATUS_OK);
            skippy_error_free(error);
            error = nullptr;
            recurrent.resize(bytes);
            // Hybrid cases must restore real recurrent tensors, not quietly
            // exercise only the dense KV path with an empty side payload.
            CHECK(!require_recurrent || !recurrent.empty());
            if (bytes > 0) {
                const auto result = skippy_export_recurrent_state(source, recurrent.data(), bytes, &bytes, &error);
                ok(result, error);
            }
        }
    }

    void restore(skippy_session * session, const std::vector<llama_token> & prefix) const {
        skippy_error * error = nullptr;
        auto status = full
            ? skippy_import_full_state(session, 0, layers, payload.data(), payload.size(), &error)
            : skippy_import_kv_page(session, &page, payload.data(), payload.size(), &error);
        ok(status, error);
        if (!recurrent.empty()) {
            error = nullptr;
            status = skippy_import_recurrent_state(session, recurrent.data(), recurrent.size(), &error);
            ok(status, error);
        }
        error = nullptr;
        status = skippy_session_set_position(session, prefix.size(), &error);
        ok(status, error);
        error = nullptr;
        status = skippy_session_restore_token_history(session, prefix.data(), prefix.size(), &error);
        ok(status, error);
        CHECK(session->token_history == prefix);
    }
};

skippy_session * create(skippy_model * model) {
    skippy_session * session = nullptr;
    skippy_error * error = nullptr;
    const auto status = skippy_session_create(model, &session, &error);
    ok(status, error);
    return session;
}

void release(skippy_session * session) {
    skippy_error * error = nullptr;
    const auto status = skippy_session_free(session, &error);
    ok(status, error);
}

void grammar_restore(skippy_model * model, bool full, bool lazy) {
    const auto sampling = sampling_config();
    const std::vector<llama_token> prefix(256, 1);
    const std::vector<llama_token> suffix(94, 2);
    auto * cold = create(model);
    configure(cold, sampling, lazy, 351);
    prefill(cold, prefix);
    snapshot cached(cold, full);
    prefill(cold, suffix);
    std::vector<llama_token> expected;
    llama_token previous = 2;
    for (int i = 0; i < 3; ++i) {
        previous = decode(cold, sampling, previous);
        expected.push_back(previous);
    }
    CHECK(expected == std::vector<llama_token>({42, 43, 44}));
    release(cold);
    auto * warm = create(model);
    for (int attempt = 0; attempt < 3; ++attempt) {
        skippy_error * error = nullptr;
        const auto status = skippy_session_reset(warm, &error);
        ok(status, error);
        // Configure before importing state, just like the serving restore path.
        configure(warm, sampling, lazy, 351);
        cached.restore(warm, prefix);
        prefill(warm, suffix);
        previous = 2;
        for (const auto token : expected) {
            previous = decode(warm, sampling, previous);
            CHECK(previous == token);
        }
        CHECK(warm->token_history.size() == 353);
    }
    release(warm);
}

void generated_prefix_restore(skippy_model * model) {
    const auto sampling = sampling_config();
    auto * source = create(model);
    const std::vector<llama_token> prefix{1, 1, 1, 1, 42};
    prefill(source, prefix);
    snapshot cached(source, true);
    release(source);
    auto * restored = create(model);
    configure(restored, sampling, true, 4);
    cached.restore(restored, prefix);
    // The restored generated token 42 arms the lazy grammar; 43 advances it.
    CHECK(decode(restored, sampling, 43) == 44);
    const auto before = restored->token_history;
    skippy_error * error = nullptr;
    CHECK(skippy_session_restore_token_history(restored, prefix.data(), 1, &error)
            == SKIPPY_STATUS_INVALID_ARGUMENT);
    skippy_error_free(error);
    CHECK(restored->token_history == before);
    error = nullptr;
    CHECK(skippy_session_restore_token_history(restored, nullptr, before.size(), &error)
            == SKIPPY_STATUS_INVALID_ARGUMENT);
    skippy_error_free(error);
    CHECK(restored->token_history == before);
    release(restored);
}

void penalty_restore(skippy_model * model, bool full) {
    auto sampling = sampling_config();
    sampling.penalty_last_n = 512;
    sampling.frequency_penalty = 1.0f;
    sampling.presence_penalty = 1.0f;
    sampling.logit_bias_count = 2;
    sampling.logit_bias[0] = {42, 100.0f};
    sampling.logit_bias[1] = {43, 50.0f};
    const std::vector<llama_token> prefix(256, 42);
    auto * cold = create(model);
    skippy_error * error = nullptr;
    auto status = skippy_session_configure_chat_sampling(cold, &sampling, "{}", 257, &error);
    ok(status, error);
    prefill(cold, prefix);
    snapshot cached(cold, full);
    CHECK(decode(cold, sampling, 1) == 43);
    release(cold);
    auto * restored = create(model);
    error = nullptr;
    status = skippy_session_configure_chat_sampling(restored, &sampling, "{}", 257, &error);
    ok(status, error);
    cached.restore(restored, prefix);
    CHECK(decode(restored, sampling, 1) == 43);
    release(restored);
}


int32_t stage_layers(skippy_session * session) {
    return session->stage_model->model->hparams.n_layer();
}

std::vector<uint8_t> export_page(skippy_session * session, size_t token_count) {
    skippy_error * error = nullptr;
    size_t bytes = 0;
    skippy_kv_page_desc page{};
    const auto layers = stage_layers(session);
    auto status = skippy_export_kv_page(session, 0, layers, 0, token_count, &page,
            nullptr, 0, &bytes, &error);
    CHECK(status == SKIPPY_STATUS_BUFFER_TOO_SMALL || status == SKIPPY_STATUS_OK);
    skippy_error_free(error);
    error = nullptr;
    std::vector<uint8_t> payload(bytes);
    status = skippy_export_kv_page(session, 0, layers, 0, token_count, &page,
            payload.data(), payload.size(), &bytes, &error);
    ok(status, error);
    payload.resize(bytes);
    return payload;
}

std::vector<uint8_t> export_recurrent(skippy_session * session) {
    skippy_error * error = nullptr;
    size_t bytes = 0;
    auto status = skippy_export_recurrent_state(session, nullptr, 0, &bytes, &error);
    CHECK(status == SKIPPY_STATUS_BUFFER_TOO_SMALL || status == SKIPPY_STATUS_OK);
    skippy_error_free(error);
    error = nullptr;
    std::vector<uint8_t> payload(bytes);
    status = skippy_export_recurrent_state(session, payload.data(), payload.size(), &bytes, &error);
    ok(status, error);
    payload.resize(bytes);
    return payload;
}

void restore_recurrent(skippy_session * session, const std::vector<uint8_t> & payload,
        const std::vector<llama_token> & prefix) {
    skippy_error * error = nullptr;
    auto status = skippy_import_recurrent_state(session, payload.data(), payload.size(), &error);
    ok(status, error);
    error = nullptr;
    status = skippy_session_set_position(session, prefix.size(), &error);
    ok(status, error);
    error = nullptr;
    status = skippy_session_restore_token_history(session, prefix.data(), prefix.size(), &error);
    ok(status, error);
}

skippy_sampling_config greedy_config() {
    skippy_sampling_config config{};
    config.version = 1;
    config.flags = SKIPPY_SAMPLING_FLAG_ENABLED;
    config.repeat_penalty = 1.0f;
    config.temperature = 0.0f;
    config.top_k = 1;
    config.reasoning_budget_tokens = -1;
    return config;
}

// Decode one token and return the logits it produced.
std::vector<float> decode_logits(skippy_session * session, llama_token token) {
    const auto sampling = greedy_config();
    llama_token predicted = -1;
    size_t bytes = 0;
    skippy_error * error = nullptr;
    const auto status = skippy_decode_step_sampled(session, token, &sampling,
            nullptr, 0, nullptr, 0, &bytes, &predicted, &error);
    ok(status, error);
    const float * logits = llama_get_logits_ith(session->ctx, -1);
    CHECK(logits != nullptr);
    const int32_t n_vocab = static_cast<int32_t>(session->stage_model->model->vocab.n_tokens());
    return std::vector<float>(logits, logits + n_vocab);
}

// A declared-exact stage checkpoint must be reconstructible from its export
// alone. LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY is an alias of
// LLAMA_STATE_SEQ_FLAGS_SWA_ONLY: it writes the recurrent and sliding-window
// half of a hybrid memory and nothing from a full-attention KV cache, so a
// session restored from that half carries no full-attention cells and decodes a
// different continuation from an uninterrupted prefill. Compare the attention
// page the restored session exposes with the source's page, and the next-token
// logits of a restored prefix with those of the same prefix prefilled from
// scratch.
void state_round_trip(skippy_model * model) {
    const std::vector<llama_token> prefix(128, 1);
    auto * source = create(model);
    prefill(source, prefix);
    const auto source_page = export_page(source, prefix.size());
    CHECK(!source_page.empty());
    const auto recurrent = export_recurrent(source);
    CHECK(!recurrent.empty());
    release(source);

    auto * restored = create(model);
    restore_recurrent(restored, recurrent, prefix);
    const auto restored_page = export_page(restored, prefix.size());
    CHECK(restored_page.size() == source_page.size());
    CHECK(std::memcmp(restored_page.data(), source_page.data(), source_page.size()) == 0);
    release(restored);

    const std::vector<llama_token> suffix(8, 2);
    auto * cold = create(model);
    prefill(cold, prefix);
    prefill(cold, suffix);
    const auto cold_logits = decode_logits(cold, suffix.back());
    release(cold);
    auto * warm = create(model);
    restore_recurrent(warm, recurrent, prefix);
    prefill(warm, suffix);
    const auto warm_logits = decode_logits(warm, suffix.back());
    release(warm);
    CHECK(cold_logits.size() == warm_logits.size());
    float max_delta = 0.0f;
    size_t differing = 0;
    for (size_t i = 0; i < cold_logits.size(); ++i) {
        const float delta = std::fabs(cold_logits[i] - warm_logits[i]);
        if (delta != 0.0f) {
            ++differing;
        }
        max_delta = std::max(max_delta, delta);
    }
    std::fprintf(stderr, "state round trip: max|dlogit| = %.9g, differing = %zu/%zu\n",
            max_delta, differing, cold_logits.size());
    CHECK(max_delta == 0.0f);
}

} // namespace

int main(int argc, char ** argv) {
    CHECK(argc == 4);
    const bool grammar = std::strcmp(argv[2], "grammar") == 0;
    const bool state = std::strcmp(argv[2], "state") == 0;
    CHECK(grammar || state || std::strcmp(argv[2], "penalty") == 0);
    CHECK(std::strcmp(argv[3], "dense") == 0 || std::strcmp(argv[3], "recurrent") == 0);
    require_recurrent = std::strcmp(argv[3], "recurrent") == 0;
    auto config = skippy_runtime_config_default();
    config.selected_backend_device = "CPU";
    config.ctx_size = 512;
    skippy_model * model = nullptr;
    skippy_error * error = nullptr;
    const auto status = skippy_model_open(argv[1], &config, &model, &error);
    ok(status, error);
    if (state) {
        // A dense stage's page already carries its whole checkpoint, so the
        // partial side payload is intentionally empty there.
        CHECK(require_recurrent);
        state_round_trip(model);
    } else {
        for (const bool full : {false, true}) {
            if (grammar) {
                for (const bool lazy : {false, true}) {
                    grammar_restore(model, full, lazy);
                }
            } else {
                penalty_restore(model, full);
            }
        }
        if (grammar) {
            generated_prefix_restore(model);
        }
    }
    error = nullptr;
    const auto freed = skippy_model_free(model, &error);
    ok(freed, error);
    return 0;
}
