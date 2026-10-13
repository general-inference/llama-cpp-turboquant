#ifndef SKIPPY_EVENTS_INTERNAL_H
#define SKIPPY_EVENTS_INTERNAL_H

#include "skippy/events.h"

// Private capability-internal dispatch path shared by every
// `skippy_emit_*_event` family entrypoint (added by sibling patches). Not
// part of the installed public ABI: declared under src/skippy/, matching
// this codebase's private-header convention. Also used directly by the
// native runtime-event test binary (which shares this include path) so the
// reporter's own mechanism (install/clear/quiescence/struct-size extension)
// is testable without depending on any one family's emit entrypoint.
namespace skippy::events {

#if defined(SKIPPY_RUNTIME_EVENTS_TEST_HOOKS)
using lifecycle_admission_hook = void (*)(void *);
void set_lifecycle_admission_hook(lifecycle_admission_hook hook, void * user_data);
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
        uint64_t detail_len);

} // namespace skippy::events

#endif // SKIPPY_EVENTS_INTERNAL_H
