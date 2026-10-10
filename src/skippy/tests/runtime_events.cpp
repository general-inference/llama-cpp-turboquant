#include "skippy.h"

#include "../events_internal.h"
#include "../runtime_support.h"
#include "../runtime_types.h"
#include "../workloads_internal.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond,        \
                          __FILE__, __LINE__);                               \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

void set_detach_sampler_on_reset_environment(const char * value) {
#if defined(_WIN32)
    _putenv_s("SKIPPY_DETACH_SAMPLER_ON_RESET", value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv("SKIPPY_DETACH_SAMPLER_ON_RESET", value, 1);
    } else {
        unsetenv("SKIPPY_DETACH_SAMPLER_ON_RESET");
    }
#endif
}

struct capture {
    std::atomic<int> count{0};
    std::atomic<uint32_t> last_category{0};
    std::atomic<uint32_t> last_kind{0};
    std::atomic<uint64_t> last_numeric_summary_0{0};
};

struct production_capture {
    std::atomic<int> model_load_events{0};
    std::atomic<int> diagnostic_events{0};
    std::atomic<int> kv_initialized_events{0};
    std::atomic<int> kv_cleared_events{0};
    std::atomic<int> unload_started_events{0};
    std::atomic<int> unload_completed_events{0};
    std::atomic<int> last_kind{0};
};

struct replacement_capture {
    std::atomic<bool> callback_entered{false};
    std::atomic<bool> release_callback{false};
    std::atomic<bool> replacement_ready{false};
    std::atomic<bool> allow_replacement{false};
    std::atomic<bool> replacement_admitted{false};
    std::atomic<bool> replacement_finished{false};
    std::atomic<bool> replacement_succeeded{false};
    std::atomic<int> old_callback_count{0};
    std::atomic<int> replacement_callback_count{0};
};

struct reentrant_capture {
    std::atomic<int> callback_count{0};
    std::atomic<int> set_status{SKIPPY_STATUS_OK};
};

void capture_callback(const struct skippy_runtime_event_v1 * event, void * user_data) {
    auto * c = static_cast<capture *>(user_data);
    c->last_category.store(event->category);
    c->last_kind.store(event->kind);
    c->last_numeric_summary_0.store(event->numeric_summary_0);
    c->count.fetch_add(1);
}

void production_callback(const struct skippy_runtime_event_v1 * event, void * user_data) {
    auto * c = static_cast<production_capture *>(user_data);
    c->last_kind.store(static_cast<int>(event->kind));
    if (event->category == SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN &&
        event->kind >= SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED) {
        c->model_load_events.fetch_add(1);
    }
    if (event->category == SKIPPY_RUNTIME_EVENT_CATEGORY_DIAGNOSTIC) {
        c->diagnostic_events.fetch_add(1);
    }
    if (event->kind == SKIPPY_RUNTIME_EVENT_KIND_KV_INITIALIZED) {
        c->kv_initialized_events.fetch_add(1);
    }
    if (event->kind == SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CLEARED) {
        c->kv_cleared_events.fetch_add(1);
    }
    if (event->category == SKIPPY_RUNTIME_EVENT_CATEGORY_UNLOAD &&
        event->kind == SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_STARTED) {
        c->unload_started_events.fetch_add(1);
    }
    if (event->category == SKIPPY_RUNTIME_EVENT_CATEGORY_UNLOAD &&
        event->kind == SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_COMPLETED) {
        c->unload_completed_events.fetch_add(1);
    }
}

void blocking_callback(const struct skippy_runtime_event_v1 *, void * user_data) {
    auto * c = static_cast<replacement_capture *>(user_data);
    c->old_callback_count.fetch_add(1);
    c->callback_entered.store(true, std::memory_order_release);
    while (!c->release_callback.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void replacement_callback(const struct skippy_runtime_event_v1 *, void * user_data) {
    auto * c = static_cast<replacement_capture *>(user_data);
    c->replacement_callback_count.fetch_add(1);
}

void reentrant_lifecycle_callback(const struct skippy_runtime_event_v1 *, void * user_data) {
    auto * c = static_cast<reentrant_capture *>(user_data);
    struct skippy_runtime_event_reporter_v1 replacement{};
    replacement.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    replacement.struct_size = sizeof(replacement);
    replacement.callback = reentrant_lifecycle_callback;
    replacement.user_data = user_data;
    c->set_status.store(skippy_set_runtime_event_reporter(&replacement));
    skippy_clear_runtime_event_reporter();
    c->callback_count.fetch_add(1);
}

void mark_lifecycle_admitted(void * user_data) {
    auto * c = static_cast<replacement_capture *>(user_data);
    c->replacement_admitted.store(true, std::memory_order_release);
}

void install(capture & c) {
    struct skippy_runtime_event_reporter_v1 reporter{};
    reporter.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    reporter.struct_size = sizeof(reporter);
    reporter.callback = capture_callback;
    reporter.user_data = &c;
    CHECK(skippy_set_runtime_event_reporter(&reporter) == SKIPPY_STATUS_OK);
}

void install(production_capture & c) {
    struct skippy_runtime_event_reporter_v1 reporter{};
    reporter.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    reporter.struct_size = sizeof(reporter);
    reporter.callback = production_callback;
    reporter.user_data = &c;
    CHECK(skippy_set_runtime_event_reporter(&reporter) == SKIPPY_STATUS_OK);
}

// Exercises the reporter mechanism (install/dispatch/clear/quiescence) via
// the shared internal dispatch() seam directly, using the pre-existing
// MODEL_OPEN category/kind. No sibling-patch family emit entrypoint is
// required for this patch's own tests.
void emit_probe(uint64_t numeric_summary_0 = 0) {
    skippy::events::dispatch(
        SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN,
        SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED,
        0, 0, 0, SKIPPY_RUNTIME_EVENT_FAILURE_NONE,
        numeric_summary_0, 0, 0, 0, nullptr, 0);
}

int test_register_and_dispatch() {
    capture c;
    install(c);
    emit_probe(7);
    CHECK(c.count.load() == 1);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN);
    CHECK(c.last_kind.load() == SKIPPY_RUNTIME_EVENT_KIND_MODEL_OPEN_STARTED);
    CHECK(c.last_numeric_summary_0.load() == 7);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_malformed_reporter_rejected() {
    struct skippy_runtime_event_reporter_v1 reporter{};
    reporter.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    reporter.struct_size = 1; // smaller than the real struct
    reporter.callback = capture_callback;
    CHECK(skippy_set_runtime_event_reporter(&reporter) == SKIPPY_STATUS_INVALID_ARGUMENT);
    CHECK(skippy_set_runtime_event_reporter(nullptr) == SKIPPY_STATUS_INVALID_ARGUMENT);
    return 0;
}

int test_clear_is_quiescent() {
    capture c;
    install(c);
    skippy_clear_runtime_event_reporter();
    // No callback should fire after clear returns.
    emit_probe();
    CHECK(c.count.load() == 0);
    return 0;
}

int test_callback_lifecycle_is_nonblocking() {
    reentrant_capture c;
    struct skippy_runtime_event_reporter_v1 reporter{};
    reporter.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    reporter.struct_size = sizeof(reporter);
    reporter.callback = reentrant_lifecycle_callback;
    reporter.user_data = &c;
    CHECK(skippy_set_runtime_event_reporter(&reporter) == SKIPPY_STATUS_OK);

    emit_probe();
    CHECK(c.set_status.load() == SKIPPY_STATUS_INVALID_ARGUMENT);
    CHECK(c.callback_count.load() == 1);
    emit_probe();
    CHECK(c.callback_count.load() == 1);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_struct_size_extension() {
    struct skippy_runtime_event_v1 seen{};
    struct skippy_runtime_event_reporter_v1 reporter{};
    reporter.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    reporter.struct_size = sizeof(reporter);
    reporter.user_data = &seen;
    reporter.callback = [](const struct skippy_runtime_event_v1 * event, void * user_data) {
        std::memcpy(user_data, event, sizeof(*event));
    };
    CHECK(skippy_set_runtime_event_reporter(&reporter) == SKIPPY_STATUS_OK);

    emit_probe();
    CHECK(seen.struct_size == sizeof(struct skippy_runtime_event_v1));
    CHECK(seen.abi_version == SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION);

    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_replacement_waits_for_previous_callback() {
    replacement_capture c;
    struct skippy_runtime_event_reporter_v1 previous{};
    previous.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    previous.struct_size = sizeof(previous);
    previous.callback = blocking_callback;
    previous.user_data = &c;
    CHECK(skippy_set_runtime_event_reporter(&previous) == SKIPPY_STATUS_OK);

    std::thread callback_thread([] { emit_probe(); });
    while (!c.callback_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    struct skippy_runtime_event_reporter_v1 replacement{};
    replacement.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    replacement.struct_size = sizeof(replacement);
    replacement.callback = replacement_callback;
    replacement.user_data = &c;
    skippy::events::set_lifecycle_admission_hook(mark_lifecycle_admitted, &c);
    std::thread replacement_thread([&] {
        c.replacement_ready.store(true, std::memory_order_release);
        while (!c.allow_replacement.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        c.replacement_succeeded.store(
            skippy_set_runtime_event_reporter(&replacement) == SKIPPY_STATUS_OK,
            std::memory_order_release);
        c.replacement_finished.store(true, std::memory_order_release);
    });
    while (!c.replacement_ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    c.allow_replacement.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (!c.replacement_admitted.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool setter_admitted = c.replacement_admitted.load(std::memory_order_acquire);
    const bool replacement_blocked =
        setter_admitted && !c.replacement_finished.load(std::memory_order_acquire);

    c.release_callback.store(true, std::memory_order_release);
    callback_thread.join();
    replacement_thread.join();
    skippy::events::set_lifecycle_admission_hook(nullptr, nullptr);
    CHECK(setter_admitted);
    CHECK(replacement_blocked);
    CHECK(c.replacement_succeeded.load(std::memory_order_acquire));
    emit_probe();
    CHECK(c.old_callback_count.load() == 1);
    CHECK(c.replacement_callback_count.load() == 1);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_concurrent_dispatch() {
    capture c;
    install(c);

    constexpr int n_threads = 8;
    static constexpr int per_thread = 200;
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([] {
            for (int i = 0; i < per_thread; ++i) {
                emit_probe();
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
    CHECK(c.count.load() == n_threads * per_thread);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_model_load_events_v2_dispatch() {
    capture c;
    install(c);
    skippy_emit_model_load_event_v2(
        SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED, 42, 7, 0, nullptr, 0);
    CHECK(c.count.load() == 1);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN);
    CHECK(c.last_kind.load() == SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED);
    CHECK(c.last_numeric_summary_0.load() == 7);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_native_model_load_production(const char * fixture_path) {
    const char * old_detach_setting = std::getenv("SKIPPY_DETACH_SAMPLER_ON_RESET");
    const bool had_detach_setting = old_detach_setting != nullptr;
    const std::string saved_detach_setting = had_detach_setting ? old_detach_setting : "";
    set_detach_sampler_on_reset_environment("1");
    CHECK(fixture_path != nullptr && fixture_path[0] != '\0');

    // A legacy entrypoint with only the process reporter receives concrete
    // model-load observations from the production completion boundary.
    production_capture global;
    install(global);
    skippy_runtime_config config = skippy_runtime_config_default();
    config.selected_backend_device = "CPU";
    skippy_model * model = nullptr;
    skippy_error * error = nullptr;
    CHECK(skippy_model_open(fixture_path, &config, &model, &error) == SKIPPY_STATUS_OK);
    CHECK(model != nullptr);
    CHECK(global.model_load_events.load() >= 2);
    skippy_session * session = nullptr;
    CHECK(skippy_session_create(model, &session, &error) == SKIPPY_STATUS_OK);
    CHECK(session != nullptr);
    CHECK(global.kv_initialized_events.load() > 0);
    const llama_token prompt[] = {0, 1};
    size_t output_bytes = 0;
    CHECK(skippy_prefill_chunk(
            session, prompt, 2, nullptr, 0, nullptr, 0,
            &output_bytes, &error) == SKIPPY_STATUS_OK);
    CHECK(session->n_past == 2);
    skippy_sampling_config sampling{};
    sampling.version = 1;
    sampling.flags = SKIPPY_SAMPLING_FLAG_ENABLED;
    sampling.repeat_penalty = 1.0f;
    CHECK(skippy_session_configure_chat_sampling(
            session, &sampling, "{\"grammar\":\"\"}", 0, &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_backend_enabled);
    CHECK(skippy_session_reset(session, &error) == SKIPPY_STATUS_OK);
    CHECK(!session->sampling_backend_enabled);
    CHECK(session->sampling_chain == nullptr);

    // SKIPPY_DETACH_SAMPLER_ON_RESET=1 restores detach-on-reset, exercised
    // above. Reuse is the default; exercise it below with the setting unset.
    set_detach_sampler_on_reset_environment(nullptr);
    CHECK(skippy_session_configure_chat_sampling(
            session, &sampling, "{\"grammar\":\"\"}", 0, &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_backend_enabled);
    llama_sampler * sampling_chain = session->sampling_chain;
    CHECK(session->sampling_rebuild_count == 2);
    llama_token predicted = -1;
    CHECK(skippy_decode_step_sampled(
            session, 2, &sampling, nullptr, 0, nullptr, 0,
            &output_bytes, &predicted, &error) == SKIPPY_STATUS_OK);
    CHECK(predicted >= 0);
    const uint64_t reserves_before_reset =
            session->ctx->get_sched_reserve_count();
    CHECK(skippy_session_reset(session, &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_backend_enabled);
    CHECK(session->sampling_chain == sampling_chain);
    CHECK(session->sampling_reuse_hit_count == 1);
    CHECK(session->n_past == 0);
    CHECK(skippy_prefill_chunk(
            session, prompt, 2, nullptr, 0, nullptr, 0,
            &output_bytes, &error) == SKIPPY_STATUS_OK);
    CHECK(session->n_past == 2);
    CHECK(session->ctx->get_sched_reserve_count() == reserves_before_reset);

    // Seed and top-p do not affect a zero-temperature chain. Reconfiguring
    // those fields, or disabling explicit sampling, must keep the same greedy
    // sampler attached and avoid another scheduler reservation.
    sampling.seed = 17;
    sampling.top_p = 0.25f;
    CHECK(skippy_session_configure_chat_sampling(
            session, &sampling, "{\"grammar\":\"\"}", 2, &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_backend_enabled);
    CHECK(session->sampling_chain == sampling_chain);
    CHECK(session->ctx->get_sched_reserve_count() == reserves_before_reset);
    CHECK(skippy_session_reset(session, &error) == SKIPPY_STATUS_OK);
    CHECK(skippy_prefill_chunk(
            session, prompt, 2, nullptr, 0, nullptr, 0,
            &output_bytes, &error) == SKIPPY_STATUS_OK);
    CHECK(session->ctx->get_sched_reserve_count() == reserves_before_reset);
    CHECK(skippy_session_configure_chat_sampling(
            session, nullptr, "{\"grammar\":\"\"}", 2, &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_backend_enabled);
    CHECK(session->sampling_chain == sampling_chain);
    CHECK(session->sampling_config_valid);
    CHECK(session->ctx->get_sched_reserve_count() == reserves_before_reset);

    // Token verification requests every output row and samples those logits on
    // the CPU. It must detach and retire a sampler retained across reset before
    // decoding the multi-row verification batch.
    llama_token verified[2] = {-1, -1};
    size_t verified_count = 0;
    CHECK(skippy_verify_tokens(
            session, prompt, 2, verified, 2,
            &verified_count, &error) == SKIPPY_STATUS_OK);
    CHECK(verified_count == 2);
    CHECK(verified[0] >= 0);
    CHECK(verified[1] >= 0);
    CHECK(!session->sampling_backend_enabled);
    CHECK(session->sampling_backend_retired);

    // A session that verified a window keeps sampling on the CPU: attaching a
    // rebuilt chain would only be undone by the next window, re-reserving the
    // scheduler both times. A reset lets the next request offload again.
    const uint64_t reserves_after_verify = session->ctx->get_sched_reserve_count();
    CHECK(skippy_session_configure_chat_sampling(
            session, &sampling, "{\"grammar\":\"\"}", 2,
            &error) == SKIPPY_STATUS_OK);
    CHECK(!session->sampling_backend_enabled);
    CHECK(session->ctx->get_sched_reserve_count() == reserves_after_verify);
    CHECK(skippy_session_reset(session, &error) == SKIPPY_STATUS_OK);
    CHECK(skippy_session_configure_chat_sampling(
            session, &sampling, "{\"grammar\":\"\"}", 0,
            &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_backend_enabled);

    // Embedding and rerank batches request every output row. Their shared
    // preparation boundary must detach a sampler retained by the chat path,
    // after which a multi-output decode is valid.
    skippy_prepare_non_chat_workload(session);
    CHECK(!session->sampling_backend_enabled);
    CHECK(session->sampling_chain == nullptr);
    llama_batch multi_output = llama_batch_init(2, 0, 1);
    multi_output.n_tokens = 2;
    for (int32_t i = 0; i < multi_output.n_tokens; ++i) {
        multi_output.token[i] = prompt[i];
        multi_output.pos[i] = i;
        multi_output.n_seq_id[i] = 1;
        multi_output.seq_id[i][0] = session->seq_id;
        multi_output.logits[i] = 1;
    }
    CHECK(llama_decode(session->ctx, multi_output) == 0);
    llama_batch_free(multi_output);
    skippy_mark_context_work_pending(session->stage_model);

    // Sampled verification retires an initialized backend chain after
    // detaching it. Reset must discard that one-shot chain immediately rather
    // than reusing it for a CPU-sampled first token on the next request.
    CHECK(skippy_session_configure_chat_sampling(
            session, &sampling, "{\"grammar\":\"\"}", 0, &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_backend_enabled);
    llama_set_sampler(session->ctx, session->seq_id, nullptr);
    session->sampling_backend_enabled = false;
    session->sampling_backend_retired = true;
    CHECK(skippy_session_reset(session, &error) == SKIPPY_STATUS_OK);
    CHECK(session->sampling_chain == nullptr);
    CHECK(!session->sampling_backend_retired);
    CHECK(global.kv_cleared_events.load() > 0);
    CHECK(skippy_session_free(session, &error) == SKIPPY_STATUS_OK);
    CHECK(skippy_model_free(model, &error) == SKIPPY_STATUS_OK);
    CHECK(global.unload_started_events.load() == 1);
    CHECK(global.unload_completed_events.load() == 1);
    set_detach_sampler_on_reset_environment(
            had_detach_setting ? saved_detach_setting.c_str() : nullptr);
    if (error != nullptr) {
        skippy_error_free(error);
        error = nullptr;
    }
    skippy_clear_runtime_event_reporter();

    // The operation-scoped reporter owns model-load-v2 observations. The
    // process reporter must not receive a duplicate copy of that family.
    production_capture process;
    production_capture operation;
    install(process);
    skippy_runtime_event_reporter_v1 reporter{};
    reporter.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    reporter.struct_size = sizeof(reporter);
    reporter.callback = production_callback;
    reporter.user_data = &operation;
    model = nullptr;
    CHECK(skippy_model_open_with_events(
                  fixture_path, &config, &reporter, &model, &error) == SKIPPY_STATUS_OK);
    CHECK(model != nullptr);
    CHECK(operation.model_load_events.load() >= 2);
    CHECK(process.model_load_events.load() == 0);
    CHECK(skippy_model_free(model, &error) == SKIPPY_STATUS_OK);
    if (error != nullptr) {
        skippy_error_free(error);
        error = nullptr;
    }
    skippy_clear_runtime_event_reporter();

    // A deliberately failed production open emits a structured diagnostic;
    // this is a real failure path, not a callable-symbol coverage marker.
    production_capture failure;
    install(failure);
    model = nullptr;
    CHECK(skippy_model_open(
                  "/definitely/not/a/real/skippy-model.gguf",
                  &config,
                  &model,
                  &error) != SKIPPY_STATUS_OK);
    CHECK(model == nullptr);
    CHECK(failure.diagnostic_events.load() > 0);
    if (error != nullptr) {
        skippy_error_free(error);
    }
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_kv_events_dispatch() {
    capture c;
    install(c);
    skippy_emit_kv_event(SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED, 5, 9, 0);
    CHECK(c.count.load() == 1);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_KV);
    CHECK(c.last_kind.load() == SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED);
    CHECK(c.last_numeric_summary_0.load() == 9);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_device_events_dispatch() {
    capture c;
    install(c);
    skippy_emit_device_event(SKIPPY_RUNTIME_EVENT_KIND_DEVICE_READY, 1, "gpu0", 4);
    CHECK(c.count.load() == 1);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_DEVICE);
    CHECK(c.last_kind.load() == SKIPPY_RUNTIME_EVENT_KIND_DEVICE_READY);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_diagnostic_events_dispatch() {
    capture c;
    install(c);
    skippy_emit_diagnostic_event(
        SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED,
        SKIPPY_RUNTIME_EVENT_FAILURE_RUNTIME_ERROR, "oom risk", 8);
    CHECK(c.count.load() == 1);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_DIAGNOSTIC);
    CHECK(c.last_kind.load() == SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED);
    skippy_clear_runtime_event_reporter();
    return 0;
}

int test_unload_events_dispatch() {
    capture c;
    install(c);
    skippy_emit_unload_event(SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_COMPLETED, 42, 0);
    CHECK(c.count.load() == 1);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_UNLOAD);
    CHECK(c.last_kind.load() == SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_COMPLETED);
    skippy_clear_runtime_event_reporter();
    return 0;
}

// Cross-family check: now that all five families exist (this is the final
// patch in the series), prove each emit_* entrypoint dispatches its own
// distinct category rather than sharing one.
int test_families_independent() {
    capture c;
    install(c);

    skippy_emit_kv_event(SKIPPY_RUNTIME_EVENT_KIND_KV_PRESSURE_CROSSED, 5, 9, 0);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_KV);

    skippy_emit_device_event(SKIPPY_RUNTIME_EVENT_KIND_DEVICE_READY, 1, "gpu0", 4);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_DEVICE);

    skippy_emit_diagnostic_event(
        SKIPPY_RUNTIME_EVENT_KIND_DIAGNOSTIC_WARNING_RAISED,
        SKIPPY_RUNTIME_EVENT_FAILURE_RUNTIME_ERROR, "oom risk", 8);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_DIAGNOSTIC);

    skippy_emit_unload_event(SKIPPY_RUNTIME_EVENT_KIND_UNLOAD_COMPLETED, 42, 0);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_UNLOAD);

    skippy_emit_model_load_event_v2(
        SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED, 1, 0, 0, nullptr, 0);
    CHECK(c.last_category.load() == SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN);

    CHECK(c.count.load() == 5);
    skippy_clear_runtime_event_reporter();
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || (std::strcmp(argv[1], "native_model_load_production") == 0 && argc != 3) ||
        (std::strcmp(argv[1], "native_model_load_production") != 0 && argc != 2)) {
        std::fprintf(stderr, "usage: %s <test-name> [fixture-path]\n", argv[0]);
        return 2;
    }
    const std::string name = argv[1];
    if (name == "register_and_dispatch") return test_register_and_dispatch();
    if (name == "malformed_reporter_rejected") return test_malformed_reporter_rejected();
    if (name == "replacement_waits_for_previous_callback") return test_replacement_waits_for_previous_callback();
    if (name == "clear_is_quiescent") return test_clear_is_quiescent();
    if (name == "callback_lifecycle_is_nonblocking") return test_callback_lifecycle_is_nonblocking();
    if (name == "struct_size_extension") return test_struct_size_extension();
    if (name == "concurrent_dispatch") return test_concurrent_dispatch();
    if (name == "model_load_events_v2_dispatch") return test_model_load_events_v2_dispatch();
    if (name == "native_model_load_production") return test_native_model_load_production(argv[2]);
    if (name == "kv_events_dispatch") return test_kv_events_dispatch();
    if (name == "device_events_dispatch") return test_device_events_dispatch();
    if (name == "diagnostic_events_dispatch") return test_diagnostic_events_dispatch();
    if (name == "unload_events_dispatch") return test_unload_events_dispatch();
    if (name == "families_independent") return test_families_independent();
    std::fprintf(stderr, "unknown test: %s\n", name.c_str());
    return 2;
}
