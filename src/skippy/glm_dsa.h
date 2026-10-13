#pragma once

#include "ggml.h"

#include <cstddef>

struct skippy_session;

bool skippy_glm_dsa_op_timing_enabled();
bool skippy_glm_dsa_tensor_trace_enabled();
bool skippy_glm_dsa_tensor_trace_stats_enabled();
void skippy_glm_dsa_op_timing_begin(skippy_session * session, size_t token_count);
bool skippy_glm_dsa_op_timing_cb(ggml_tensor * tensor, bool ask, void * user_data);
void skippy_glm_dsa_op_timing_end(skippy_session * session);
