#include "skippy/events.h"

#include "events_internal.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace {

struct global_reporter_state {
    // Lifecycle operations are serialized independently from dispatch. A
    // setter/clearer may wait for callbacks to drain, while a callback only
    // needs dispatch_mutex to snapshot its current target and retire its
    // in-flight count.
    std::mutex lifecycle_mutex;
    std::mutex dispatch_mutex;
    std::condition_variable drained;
    std::atomic<uint64_t> in_flight{0};
    skippy_runtime_event_callback callback = nullptr;
    void * user_data = nullptr;
    std::atomic<uint64_t> next_sequence{1};
};

global_reporter_state & state() {
    static global_reporter_state s;
    return s;
}

thread_local uint32_t callback_depth = 0;

struct callback_scope {
    callback_scope() { ++callback_depth; }
    ~callback_scope() { --callback_depth; }
};

#if defined(SKIPPY_RUNTIME_EVENTS_TEST_HOOKS)
using test_lifecycle_hook_fn = void (*)(void *);
std::atomic<test_lifecycle_hook_fn> test_lifecycle_hook{nullptr};
std::atomic<void *> test_lifecycle_hook_user_data{nullptr};
#endif

uint64_t now_mono_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void disable_dispatch(global_reporter_state & s) {
    // Disable admission before draining. Do not hold dispatch_mutex while
    // waiting: an in-flight callback must be able to finish and reacquire it
    // to retire its count.
    std::lock_guard<std::mutex> lock(s.dispatch_mutex);
    s.callback = nullptr;
    s.user_data = nullptr;
}

void quiesce(global_reporter_state & s) {
    disable_dispatch(s);
    std::unique_lock<std::mutex> lock(s.dispatch_mutex);
    s.drained.wait(lock, [&s] {
        return s.in_flight.load(std::memory_order_acquire) == 0;
    });
}

void callback_finished(global_reporter_state & s) {
    std::lock_guard<std::mutex> lock(s.dispatch_mutex);
    s.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    if (s.in_flight.load(std::memory_order_acquire) == 0) {
        s.drained.notify_all();
    }
}

#if defined(SKIPPY_RUNTIME_EVENTS_TEST_HOOKS)
void notify_lifecycle_admission_hook() {
    const auto hook = test_lifecycle_hook.load(std::memory_order_acquire);
    if (hook != nullptr) {
        hook(test_lifecycle_hook_user_data.load(std::memory_order_acquire));
    }
}
#endif
} // namespace

namespace skippy::events {

#if defined(SKIPPY_RUNTIME_EVENTS_TEST_HOOKS)
void set_lifecycle_admission_hook(lifecycle_admission_hook hook, void * user_data) {
    test_lifecycle_hook_user_data.store(user_data, std::memory_order_release);
    test_lifecycle_hook.store(hook, std::memory_order_release);
}
#endif

void dispatch(
        skippy_runtime_event_category category,
        skippy_runtime_event_kind kind,
        uint64_t model_id,
        uint64_t stage_id,
        uint64_t session_id,
        skippy_runtime_event_failure_code failure_code,
        uint64_t numeric_summary_0,
        uint64_t numeric_summary_1,
        uint64_t numeric_summary_2,
        uint64_t numeric_summary_3,
        const char * detail_ptr,
        uint64_t detail_len) {
    auto & s = state();

    // Snapshot the callback and admit the invocation atomically with respect
    // to lifecycle quiescence. The callback itself runs with both locks
    // released so it can complete while set/clear waits for its retirement.
    skippy_runtime_event_callback callback;
    void * user_data;
    {
        std::lock_guard<std::mutex> lock(s.dispatch_mutex);
        callback = s.callback;
        user_data = s.user_data;
        if (callback == nullptr) {
            return;
        }
        s.in_flight.fetch_add(1, std::memory_order_acq_rel);
    }

    skippy_runtime_event_v1 event{};
    event.abi_version = SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION;
    event.struct_size = static_cast<uint32_t>(sizeof(skippy_runtime_event_v1));
    event.category = category;
    event.kind = kind;
    event.emitter = SKIPPY_RUNTIME_EVENT_EMITTER_WORKER_THREAD;
    event.sequence = s.next_sequence.fetch_add(1, std::memory_order_relaxed);
    event.timestamp_mono_ns = now_mono_ns();
    event.model_id = model_id;
    event.stage_id = stage_id;
    event.session_id = session_id;
    event.failure_code = failure_code;
    event.numeric_summary_0 = numeric_summary_0;
    event.numeric_summary_1 = numeric_summary_1;
    event.numeric_summary_2 = numeric_summary_2;
    event.numeric_summary_3 = numeric_summary_3;
    event.detail_ptr = detail_ptr;
    event.detail_len = detail_len;

    {
        callback_scope scope;
        callback(&event, user_data);
    }

    callback_finished(s);
}

} // namespace skippy::events

enum skippy_status skippy_set_runtime_event_reporter(
        const struct skippy_runtime_event_reporter_v1 * reporter) {
    if (reporter == nullptr ||
        reporter->struct_size < sizeof(struct skippy_runtime_event_reporter_v1)) {
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (callback_depth != 0) {
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    auto & s = state();
    std::lock_guard<std::mutex> lifecycle_lock(s.lifecycle_mutex);
#if defined(SKIPPY_RUNTIME_EVENTS_TEST_HOOKS)
    notify_lifecycle_admission_hook();
#endif
    // A replacement has the same lifetime guarantee as clear: no callback
    // using the previous user_data can remain when this function returns.
    quiesce(s);
    {
        std::lock_guard<std::mutex> dispatch_lock(s.dispatch_mutex);
        s.callback = reporter->callback;
        s.user_data = reporter->user_data;
    }
    return SKIPPY_STATUS_OK;
}

void skippy_clear_runtime_event_reporter(void) {
    auto & s = state();
    if (callback_depth != 0) {
        disable_dispatch(s);
        return;
    }
    std::lock_guard<std::mutex> lifecycle_lock(s.lifecycle_mutex);
    quiesce(s);
}

void skippy_emit_model_load_event_v2(
        skippy_runtime_event_kind kind,
        uint64_t model_id,
        uint64_t numeric_summary_0,
        uint64_t numeric_summary_1,
        const char * detail_ptr,
        uint64_t detail_len) {
    skippy::events::dispatch(
        SKIPPY_RUNTIME_EVENT_CATEGORY_MODEL_OPEN, kind, model_id, 0, 0,
        SKIPPY_RUNTIME_EVENT_FAILURE_NONE, numeric_summary_0, numeric_summary_1, 0, 0,
        detail_ptr, detail_len);
}

void skippy_emit_kv_event(
        skippy_runtime_event_kind kind,
        uint64_t session_id,
        uint64_t numeric_summary_0,
        uint64_t numeric_summary_1) {
    skippy::events::dispatch(
        SKIPPY_RUNTIME_EVENT_CATEGORY_KV, kind, 0, 0, session_id,
        SKIPPY_RUNTIME_EVENT_FAILURE_NONE, numeric_summary_0, numeric_summary_1, 0, 0,
        nullptr, 0);
}

void skippy_emit_device_event(
        skippy_runtime_event_kind kind,
        uint64_t numeric_summary_0,
        const char * detail_ptr,
        uint64_t detail_len) {
    skippy::events::dispatch(
        SKIPPY_RUNTIME_EVENT_CATEGORY_DEVICE, kind, 0, 0, 0,
        SKIPPY_RUNTIME_EVENT_FAILURE_NONE, numeric_summary_0, 0, 0, 0,
        detail_ptr, detail_len);
}

void skippy_emit_diagnostic_event(
        skippy_runtime_event_kind kind,
        skippy_runtime_event_failure_code failure_code,
        const char * detail_ptr,
        uint64_t detail_len) {
    skippy::events::dispatch(
        SKIPPY_RUNTIME_EVENT_CATEGORY_DIAGNOSTIC, kind, 0, 0, 0,
        failure_code, 0, 0, 0, 0, detail_ptr, detail_len);
}

void skippy_emit_unload_event(
        skippy_runtime_event_kind kind,
        uint64_t model_id,
        uint64_t numeric_summary_0) {
    skippy::events::dispatch(
        SKIPPY_RUNTIME_EVENT_CATEGORY_UNLOAD, kind, model_id, 0, 0,
        SKIPPY_RUNTIME_EVENT_FAILURE_NONE, numeric_summary_0, 0, 0, 0, nullptr, 0);
}
