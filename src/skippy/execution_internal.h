#pragma once

#include "skippy/common.h"

#include "llama.h"

#include <cstddef>
#include <cstdint>

struct skippy_session;
struct skippy_graph_build_inputs;

bool skippy_is_filtered(const skippy_session * session);
enum skippy_status skippy_decode_batch(
        skippy_session * session,
        llama_batch batch,
        size_t token_count,
        int32_t glm_dsa_phase_hint,
        skippy_error ** out_error,
        const skippy_graph_build_inputs * graph_build_inputs = nullptr);
