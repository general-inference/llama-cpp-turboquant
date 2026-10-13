#include "skippy.h"
#include "skippy/activation_internal.h"
#include "skippy/errors.h"
#include "skippy/environment.h"
#include "skippy/glm_dsa.h"
#include "skippy-verify-checkpoint.h"
#include "skippy/devices.h"

#include <mutex>
#include "gguf.h"
#include "llama-arch.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <cstdio>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

#include "skippy/execution_internal.h"
#include "skippy/sampling_internal.h"
#include "skippy/speculative_internal.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"

bool skippy_is_filtered(const skippy_session * session) {
    return session != nullptr &&
           session->stage_model != nullptr &&
           skippy_runtime_has_stage_plan(&session->stage_model->config);
}

bool skippy_emits_activation_frame(const skippy_session * session) {
    return skippy_is_filtered(session) &&
           !skippy_runtime_is_terminal_stage(&session->stage_model->config);
}

int64_t skippy_resolve_activation_part_dimension(
        const skippy_session * session,
        const skippy_activation_part_desc & part,
        uint32_t dimension,
        bool live_graph) {
    if (session == nullptr || session->ctx == nullptr || dimension >= GGML_MAX_DIMS) return -1;
    const auto * result = live_graph ? session->ctx->get_gf_res_prev_active() : session->ctx->get_gf_res_reserve();
    if (result == nullptr) return -1;
    const auto & ports = result->get_skippy_activation_exports();
    const auto & boundary = session->stage_model->output_activation_boundary;
    if (ports.size() != boundary.part_count) return -1;
    for (size_t index = 0; index < ports.size(); ++index) {
        if (std::memcmp(part.identity, boundary.parts[index].identity, SKIPPY_ACTIVATION_IDENTITY_BYTES) != 0) continue;
        const auto * tensor = ports[index].tensor;
        if (tensor == nullptr || static_cast<uint32_t>(tensor->type) != part.ggml_type ||
                ports[index].token_axis != part.token_axis) return -1;
        return tensor->ne[dimension];
    }
    return -1;
}

bool skippy_has_activation_payload(const skippy_activation_desc * desc, const void * payload) {
    return desc != nullptr && desc->payload_bytes > 0 && payload != nullptr;
}

const skippy_activation_part_desc * skippy_activation_part_at(const skippy_activation_desc * desc, size_t index) {
    if (desc == nullptr || index >= desc->part_count || index >= SKIPPY_ACTIVATION_MAX_PARTS) return nullptr;
    return &desc->parts[index];
}

const skippy_activation_part_desc * skippy_activation_find_part(
        const skippy_activation_desc * desc,
        const uint8_t identity[SKIPPY_ACTIVATION_IDENTITY_BYTES]) {
    if (desc == nullptr || identity == nullptr || desc->part_count > SKIPPY_ACTIVATION_MAX_PARTS) return nullptr;
    for (size_t index = 0; index < desc->part_count; ++index) {
        if (std::memcmp(desc->parts[index].identity, identity, SKIPPY_ACTIVATION_IDENTITY_BYTES) == 0) return &desc->parts[index];
    }
    return nullptr;
}

const void * skippy_activation_part_payload(
        const skippy_activation_desc * desc,
        const void * payload,
        const skippy_activation_part_desc * part) {
    if (desc == nullptr || payload == nullptr || part == nullptr || part->payload_offset > desc->payload_bytes ||
            part->payload_bytes > desc->payload_bytes - part->payload_offset) return nullptr;
    return static_cast<const uint8_t *>(payload) + part->payload_offset;
}

void * skippy_activation_part_payload(
        const skippy_activation_desc * desc,
        void * payload,
        const skippy_activation_part_desc * part) {
    return const_cast<void *>(skippy_activation_part_payload(desc, static_cast<const void *>(payload), part));
}

extern "C" {
enum skippy_status skippy_session_copy_output_activation_frame(
        struct skippy_session * session,
        size_t token_count,
        struct skippy_activation_desc * output_desc,
        void * output_payload,
        size_t output_payload_capacity,
        size_t * out_output_payload_bytes,
        struct skippy_error ** out_error) {
    if (token_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count must be greater than zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    enum skippy_status status = skippy_prepare_output_activation_frame(
            session,
            token_count,
            output_payload,
            output_payload_capacity,
            out_output_payload_bytes,
            output_desc,
            nullptr,
            out_error,
            skippy_activation_frame_shape::live);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    return skippy_copy_output_activation_frame(
            session, token_count, output_payload, nullptr, nullptr, out_output_payload_bytes, output_desc, out_error);
}

} // extern "C"
