#include "skippy/state.h"
#include "skippy/state_memory.h"
#include "skippy/errors.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"
#include "skippy-verify-checkpoint.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" {

// A hybrid or recurrent stage cannot be rebuilt from the partial
// (recurrent + sliding-window) sequence state: the full-attention half of the
// same stage is dropped, so a restored prefix decodes differently from an
// uninterrupted prefill. Pay the larger payload to keep these checkpoints exact.
static llama_state_seq_flags skippy_state_seq_flags(skippy_session * session) {
    if (session == nullptr || session->ctx == nullptr) {
        return LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    }
    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        return LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    }
    return skippy_memory_recurrent_adapter(memory) != nullptr
            ? LLAMA_STATE_SEQ_FLAGS_NONE
            : LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
}

static enum skippy_status skippy_validate_state_range(
        skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || session->stage_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    const skippy_runtime_config & config = session->stage_model->config;
    const int32_t expected_layer_start = skippy_runtime_has_stage_plan(&config) ? config.layer_start : 0;
    const int32_t expected_layer_end = skippy_runtime_has_stage_plan(&config) ?
            config.layer_end : skippy_stage_layer_count(session->stage_model->model);
    if (layer_start != expected_layer_start || layer_end != expected_layer_end) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "state range must match the session layer range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    return SKIPPY_STATUS_OK;
}

static void skippy_update_session_state_after_import(
        skippy_session * session,
        uint32_t imported_position) {
    session->n_past = std::max<int32_t>(
            session->n_past,
            static_cast<int32_t>(std::min<uint32_t>(
                    imported_position,
                    static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))));
    if (session->token_history.size() > static_cast<size_t>(session->n_past)) {
        session->token_history.resize(static_cast<size_t>(session->n_past));
    }
    if (session->signal_history.size() > static_cast<size_t>(session->n_past)) {
        session->signal_history.resize(static_cast<size_t>(session->n_past));
    }
    skippy_mtp_clear_session_state(session);
    session->verify_checkpoints.clear();
}

static enum skippy_status skippy_imported_session_position(
        skippy_session * session,
        bool partial_only,
        uint32_t * out_position,
        skippy_error ** out_error) {
    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable after state import");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    llama_memory_t restored_memory = memory;
    if (partial_only) {
        if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
            restored_memory = hybrid->get_mem_recr();
        } else if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
            restored_memory = hybrid_iswa->get_mem_recr();
        }
    }

    const llama_pos max_pos = llama_memory_seq_pos_max(restored_memory, session->seq_id);
    if (max_pos < 0) {
        *out_position = 0;
        return SKIPPY_STATUS_OK;
    }
    if (max_pos >= std::numeric_limits<int32_t>::max()) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "imported state position exceeds the supported range");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    *out_position = static_cast<uint32_t>(max_pos) + 1;
    return SKIPPY_STATUS_OK;
}

enum skippy_status skippy_export_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    if (out_bytes != nullptr) {
        *out_bytes = 0;
    }
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    session->ctx->synchronize();
    const llama_state_seq_flags state_flags = skippy_state_seq_flags(session);
    const size_t required = llama_state_seq_get_size_ext(
            session->ctx,
            session->seq_id,
            state_flags);
    if (out_bytes != nullptr) {
        *out_bytes = required;
    }
    if (required == 0) {
        return skippy_success(out_error);
    }
    if (output == nullptr || output_capacity < required) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "state output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t written = llama_state_seq_get_data_ext(
            session->ctx,
            static_cast<uint8_t *>(output),
            output_capacity,
            session->seq_id,
            state_flags);
    if (written != required) {
        if (out_bytes != nullptr) {
            *out_bytes = written;
        }
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to export sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_import_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (input == nullptr && input_bytes > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "state input is required when input_bytes is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    session->ctx->synchronize();
    const size_t read = llama_state_seq_set_data_ext(
            session->ctx,
            static_cast<const uint8_t *>(input),
            input_bytes,
            session->seq_id,
            skippy_state_seq_flags(session));
    if (read != input_bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to import sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->ctx->synchronize();
    uint32_t imported_position = 0;
    status = skippy_imported_session_position(session, true, &imported_position, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    skippy_update_session_state_after_import(session, imported_position);

    return skippy_success(out_error);
}

enum skippy_status skippy_export_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    if (out_bytes != nullptr) {
        *out_bytes = 0;
    }
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    session->ctx->synchronize();
    const size_t required = llama_state_seq_get_size(session->ctx, session->seq_id);
    if (out_bytes != nullptr) {
        *out_bytes = required;
    }
    if (required == 0) {
        return skippy_success(out_error);
    }
    if (output == nullptr || output_capacity < required) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "full state output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t written = llama_state_seq_get_data(
            session->ctx,
            static_cast<uint8_t *>(output),
            output_capacity,
            session->seq_id);
    if (written != required) {
        if (out_bytes != nullptr) {
            *out_bytes = written;
        }
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to export full sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_import_full_state(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    enum skippy_status status = skippy_validate_state_range(session, layer_start, layer_end, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (input == nullptr && input_bytes > 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "state input is required when input_bytes is non-zero");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    session->ctx->synchronize();
    const size_t read = llama_state_seq_set_data(
            session->ctx,
            static_cast<const uint8_t *>(input),
            input_bytes,
            session->seq_id);
    if (read != input_bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to import full sequence state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->ctx->synchronize();
    uint32_t imported_position = 0;
    status = skippy_imported_session_position(session, false, &imported_position, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    skippy_update_session_state_after_import(session, imported_position);

    return skippy_success(out_error);
}

static llama_memory_recurrent * skippy_get_recurrent_memory(
        skippy_session * session,
        skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return nullptr;
    }

    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return nullptr;
    }

    return skippy_memory_recurrent_adapter(memory);
}

enum skippy_status skippy_export_recurrent_state(
        struct skippy_session * session,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    if (out_bytes == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_bytes is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_bytes = 0;
    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent == nullptr) {
        return skippy_success(out_error);
    }

    session->ctx->synchronize();
    const size_t bytes = llama_state_seq_get_size_ext(
            session->ctx,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_NONE);
    *out_bytes = bytes;
    if (bytes == 0) {
        return skippy_success(out_error);
    }
    if (output == nullptr || output_capacity < bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "recurrent state output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t written = llama_state_seq_get_data_ext(
            session->ctx,
            static_cast<uint8_t *>(output),
            output_capacity,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_NONE);
    if (written != bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to export recurrent state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_import_recurrent_state(
        struct skippy_session * session,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    if (input_bytes == 0) {
        return skippy_success(out_error);
    }
    if (input == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "recurrent state input is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime has no recurrent memory");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    session->ctx->synchronize();
    const size_t read = llama_state_seq_set_data_ext(
            session->ctx,
            static_cast<const uint8_t *>(input),
            input_bytes,
            session->seq_id,
            LLAMA_STATE_SEQ_FLAGS_NONE);
    if (read != input_bytes) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to import recurrent state");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    session->ctx->synchronize();

    return skippy_success(out_error);
}

static llama_kv_cache * skippy_get_kv_cache(
        skippy_session * session,
        skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return nullptr;
    }

    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return nullptr;
    }

    return skippy_memory_kv_adapter(memory);
}

static llama_kv_cache_iswa * skippy_get_iswa_cache(skippy_session * session) {
    if (session == nullptr || session->ctx == nullptr) {
        return nullptr;
    }
    llama_memory_t memory = session->ctx->get_memory();
    return skippy_memory_iswa_adapter(memory);
}

static skippy_kv_page_component_desc skippy_component_from_page(
        const skippy_kv_page_desc & page,
        uint32_t role,
        uint64_t payload_offset) {
    skippy_kv_page_component_desc component = {};
    component.version = page.version;
    component.role = role;
    component.token_start = page.token_start;
    component.token_count = page.token_count;
    component.layer_count = page.layer_count;
    component.k_type = page.k_type;
    component.v_type = page.v_type;
    component.k_row_bytes = page.k_row_bytes;
    component.v_row_bytes = page.v_row_bytes;
    component.v_element_bytes = page.v_element_bytes;
    component.k_idx_row_bytes = page.k_idx_row_bytes;
    component.payload_offset = payload_offset;
    component.payload_bytes = page.payload_bytes;
    component.flags = page.flags;
    return component;
}

static skippy_kv_page_desc skippy_page_from_component(
        const skippy_kv_page_desc & composite,
        const skippy_kv_page_component_desc & component) {
    skippy_kv_page_desc page = {};
    page.version = component.version;
    page.layer_start = composite.layer_start;
    page.layer_end = composite.layer_end;
    page.token_start = component.token_start;
    page.token_count = component.token_count;
    page.layer_count = component.layer_count;
    page.k_type = component.k_type;
    page.v_type = component.v_type;
    page.k_row_bytes = component.k_row_bytes;
    page.v_row_bytes = component.v_row_bytes;
    page.v_element_bytes = component.v_element_bytes;
    page.k_idx_row_bytes = component.k_idx_row_bytes;
    page.payload_bytes = component.payload_bytes;
    page.flags = component.flags;
    page.codec = SKIPPY_KV_PAGE_CODEC_SINGLE_V1;
    return page;
}

static bool skippy_export_iswa_kv_page(
        skippy_session * session,
        llama_kv_cache_iswa * iswa,
        int32_t layer_start,
        int32_t layer_end,
        uint64_t token_start,
        uint64_t token_count,
        skippy_kv_page_desc * out_desc,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        std::string & error) {
    if (out_desc == nullptr || out_bytes == nullptr) {
        error = "out_desc and out_bytes are required";
        return false;
    }
    if (token_start + token_count < token_start) {
        error = "token range is too large";
        return false;
    }
    const uint64_t n_swa = iswa->get_swa()->get_n_swa();
    const uint64_t token_end = token_start + token_count;
    const uint64_t swa_start = n_swa == 0 || token_count <= n_swa ? token_start : token_end - n_swa;
    const uint64_t swa_count = token_end - swa_start;

    skippy_kv_page_desc base = {};
    skippy_kv_page_desc swa = {};
    size_t base_bytes = 0;
    size_t swa_bytes = 0;
    const bool base_sized = iswa->get_base()->stage_export_kv_page(session->seq_id, layer_start, layer_end, token_start, token_count, &base, nullptr, 0, &base_bytes, error);
    if (!base_sized && base_bytes == 0) {
        return false;
    }
    error.clear();
    const bool swa_sized = iswa->get_swa()->stage_export_kv_page(session->seq_id, layer_start, layer_end, swa_start, swa_count, &swa, nullptr, 0, &swa_bytes, error);
    if (!swa_sized && swa_bytes == 0) {
        return false;
    }
    if (base_bytes > std::numeric_limits<size_t>::max() - swa_bytes) {
        error = "composite KV page payload is too large";
        return false;
    }
    const size_t total = base_bytes + swa_bytes;
    *out_desc = {};
    out_desc->version = 2;
    out_desc->layer_start = layer_start;
    out_desc->layer_end = layer_end;
    out_desc->token_start = token_start;
    out_desc->token_count = token_count;
    out_desc->payload_bytes = total;
    out_desc->codec = SKIPPY_KV_PAGE_CODEC_ISWA_COMPOSITE_V1;
    out_desc->component_count = 2;
    out_desc->components[0] = skippy_component_from_page(base, SKIPPY_KV_PAGE_COMPONENT_BASE, 0);
    out_desc->components[1] = skippy_component_from_page(swa, SKIPPY_KV_PAGE_COMPONENT_SWA, base_bytes);
    *out_bytes = total;
    if (output == nullptr) {
        return true;
    }
    if (output_capacity < total) {
        error = "output buffer is too small";
        return false;
    }
    char * bytes = static_cast<char *>(output);
    return iswa->get_base()->stage_export_kv_page(session->seq_id, layer_start, layer_end, token_start, token_count, &base, bytes, base_bytes, &base_bytes, error) &&
           iswa->get_swa()->stage_export_kv_page(session->seq_id, layer_start, layer_end, swa_start, swa_count, &swa, bytes + base_bytes, swa_bytes, &swa_bytes, error);
}

enum skippy_status skippy_export_kv_page(
        struct skippy_session * session,
        int32_t layer_start,
        int32_t layer_end,
        uint64_t token_start,
        uint64_t token_count,
        struct skippy_kv_page_desc * out_desc,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    session->ctx->synchronize();
    std::string error;
    bool ok = false;
    if (auto * iswa = skippy_get_iswa_cache(session)) {
        ok = skippy_export_iswa_kv_page(session, iswa, layer_start, layer_end, token_start, token_count, out_desc, output, output_capacity, out_bytes, error);
    } else if (auto * kv = skippy_get_kv_cache(session, out_error)) {
        ok = kv->stage_export_kv_page(session->seq_id, layer_start, layer_end, token_start, token_count, out_desc, output, output_capacity, out_bytes, error);
        if (ok && out_desc != nullptr) {
            out_desc->codec = SKIPPY_KV_PAGE_CODEC_SINGLE_V1;
        }
    } else {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime memory type is not supported for native KV pages");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    if (!ok) {
        const bool too_small = out_bytes != nullptr && *out_bytes > output_capacity;
        const enum skippy_status status = too_small ? SKIPPY_STATUS_BUFFER_TOO_SMALL : SKIPPY_STATUS_RUNTIME_ERROR;
        skippy_set_error(out_error, status, error.c_str());
        return status;
    }
    if (output == nullptr || (out_bytes != nullptr && *out_bytes > output_capacity)) {
        skippy_set_error(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL, "native KV page output buffer is too small");
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    return skippy_success(out_error);
}

static bool skippy_validate_composite_desc(const skippy_kv_page_desc & desc, size_t input_bytes, std::string & error) {
    const uint64_t logical_end = desc.token_start + desc.token_count;
    const uint64_t swa_end = desc.components[1].token_start + desc.components[1].token_count;
    if (logical_end < desc.token_start || swa_end < desc.components[1].token_start ||
            desc.version != 2 || desc.component_count != 2 || desc.payload_bytes != input_bytes ||
            desc.components[0].version != 1 || desc.components[1].version != 1 ||
            desc.components[0].role != SKIPPY_KV_PAGE_COMPONENT_BASE || desc.components[1].role != SKIPPY_KV_PAGE_COMPONENT_SWA ||
            desc.components[0].payload_offset != 0 || desc.components[0].token_start != desc.token_start ||
            desc.components[0].token_count != desc.token_count || desc.components[1].token_start < desc.token_start ||
            swa_end != logical_end ||
            desc.components[1].payload_offset != desc.components[0].payload_bytes ||
            desc.components[1].payload_offset > desc.payload_bytes ||
            desc.components[1].payload_bytes != desc.payload_bytes - desc.components[1].payload_offset) {
        error = "invalid composite ISWA KV page descriptor";
        return false;
    }
    return true;
}

enum skippy_status skippy_import_kv_page(
        struct skippy_session * session,
        const struct skippy_kv_page_desc * desc,
        const void * input,
        size_t input_bytes,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || desc == nullptr || input == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session, descriptor, and payload are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    std::string error;
    bool ok = false;
    bool invalid_argument = false;
    if (desc->codec == SKIPPY_KV_PAGE_CODEC_ISWA_COMPOSITE_V1) {
        auto * iswa = skippy_get_iswa_cache(session);
        if (!skippy_validate_composite_desc(*desc, input_bytes, error)) {
            invalid_argument = true;
        } else if (iswa == nullptr) {
            error = "composite KV page requires ISWA memory";
        } else if (session->n_past != 0) {
            error = "composite ISWA KV page import requires a fresh session";
        } else {
            const auto base = skippy_page_from_component(*desc, desc->components[0]);
            const auto swa = skippy_page_from_component(*desc, desc->components[1]);
            const char * bytes = static_cast<const char *>(input);
            if (iswa->get_base()->stage_import_kv_page(session->seq_id, base, bytes, base.payload_bytes, error, true) &&
                    iswa->get_swa()->stage_import_kv_page(session->seq_id, swa, bytes + desc->components[1].payload_offset, swa.payload_bytes, error, true)) {
                ok = iswa->get_base()->stage_import_kv_page(session->seq_id, base, bytes, base.payload_bytes, error) &&
                     iswa->get_swa()->stage_import_kv_page(session->seq_id, swa, bytes + desc->components[1].payload_offset, swa.payload_bytes, error);
            }
        }
        if (!ok && iswa != nullptr) {
            iswa->get_base()->seq_rm(session->seq_id, -1, -1);
            iswa->get_swa()->seq_rm(session->seq_id, -1, -1);
        }
    } else if (desc->codec == 0 || desc->codec == SKIPPY_KV_PAGE_CODEC_SINGLE_V1) {
        if (auto * kv = skippy_get_kv_cache(session, out_error)) {
            ok = kv->stage_import_kv_page(session->seq_id, *desc, input, input_bytes, error);
        }
    } else {
        error = "unsupported native KV page codec";
        invalid_argument = true;
    }
    if (!ok) {
        const enum skippy_status status = invalid_argument ? SKIPPY_STATUS_INVALID_ARGUMENT : SKIPPY_STATUS_RUNTIME_ERROR;
        skippy_set_error(out_error, status, error.c_str());
        return status;
    }
    if (llm_graph_result * graph = session->ctx->get_gf_res_prev_active()) {
        graph->reset();
    }
    session->n_past = std::max<int32_t>(session->n_past, static_cast<int32_t>(std::min<uint64_t>(desc->token_start + desc->token_count, std::numeric_limits<int32_t>::max())));
    skippy_mtp_clear_session_state(session);
    session->verify_checkpoints.clear();
    session->ctx->synchronize();
    return skippy_success(out_error);
}

enum skippy_status skippy_import_cachegen_kv_page_v1(struct skippy_session *                  session,
                                                     const struct skippy_kv_page_desc *       desc,
                                                     const struct skippy_cachegen_record_v1 * records,
                                                     size_t                                   record_count,
                                                     struct skippy_error **                   out_error) {
    if (session == nullptr || session->ctx == nullptr || desc == nullptr || records == nullptr || record_count == 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                         "session, descriptor, and CacheGen records are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session->n_past != 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "CacheGen page import requires a fresh session");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    std::string error;
    bool        ok               = false;
    bool        unsupported      = false;
    bool        invalid_argument = false;
    if (desc->codec == SKIPPY_KV_PAGE_CODEC_ISWA_COMPOSITE_V1) {
        auto * iswa = skippy_get_iswa_cache(session);
        if (!skippy_validate_composite_desc(*desc, desc->payload_bytes, error)) {
            invalid_argument = true;
        } else if (iswa == nullptr) {
            unsupported = true;
            error       = "composite CacheGen page requires ISWA memory";
        } else {
            const auto     base         = skippy_page_from_component(*desc, desc->components[0]);
            const auto     swa          = skippy_page_from_component(*desc, desc->components[1]);
            const uint64_t split_offset = desc->components[1].payload_offset;
            size_t         split        = 0;
            while (split < record_count && records[split].output_offset < split_offset) {
                ++split;
            }
            if (split == 0 || split == record_count) {
                invalid_argument = true;
                error            = "CacheGen records do not cover both ISWA components";
            } else {
                bool       base_unsupported = false;
                bool       swa_unsupported  = false;
                bool       base_invalid     = false;
                bool       swa_invalid      = false;
                const bool base_valid       = iswa->get_base()->stage_import_cachegen_kv_page(
                    session->seq_id, base, records, split, 0, error, base_unsupported, base_invalid, true);
                const bool swa_valid =
                    base_valid && iswa->get_swa()->stage_import_cachegen_kv_page(
                                      session->seq_id, swa, records + split, record_count - split,
                                      static_cast<size_t>(split_offset), error, swa_unsupported, swa_invalid, true);
                unsupported      = base_unsupported || swa_unsupported;
                invalid_argument = base_invalid || swa_invalid;
                if (base_valid && swa_valid) {
                    bool ignored         = false;
                    bool ignored_invalid = false;
                    ok = iswa->get_base()->stage_import_cachegen_kv_page(session->seq_id, base, records, split, 0,
                                                                         error, ignored, ignored_invalid) &&
                         iswa->get_swa()->stage_import_cachegen_kv_page(
                             session->seq_id, swa, records + split, record_count - split,
                             static_cast<size_t>(split_offset), error, ignored, ignored_invalid);
                    invalid_argument = ignored_invalid;
                }
            }
        }
        if (!ok && iswa != nullptr) {
            iswa->get_base()->seq_rm(session->seq_id, -1, -1);
            iswa->get_swa()->seq_rm(session->seq_id, -1, -1);
        }
    } else if (desc->codec == 0 || desc->codec == SKIPPY_KV_PAGE_CODEC_SINGLE_V1) {
        if (auto * kv = skippy_get_kv_cache(session, out_error)) {
            ok = kv->stage_import_cachegen_kv_page(session->seq_id, *desc, records, record_count, 0, error, unsupported,
                                                   invalid_argument);
        } else {
            return SKIPPY_STATUS_UNSUPPORTED;
        }
    } else {
        invalid_argument = true;
        error            = "unsupported native KV page codec";
    }
    if (!ok) {
        const enum skippy_status status = invalid_argument ?
                                              SKIPPY_STATUS_INVALID_ARGUMENT :
                                              (unsupported ? SKIPPY_STATUS_UNSUPPORTED : SKIPPY_STATUS_RUNTIME_ERROR);
        skippy_set_error(out_error, status, error.c_str());
        return status;
    }
    if (llm_graph_result * graph = session->ctx->get_gf_res_prev_active()) {
        graph->reset();
    }
    session->n_past = std::max<int32_t>(session->n_past,
                                        static_cast<int32_t>(std::min<uint64_t>(desc->token_start + desc->token_count,
                                                                                std::numeric_limits<int32_t>::max())));
    skippy_mtp_clear_session_state(session);
    session->verify_checkpoints.clear();
    session->ctx->synchronize();
    return skippy_success(out_error);
}

enum skippy_status skippy_retire_verify_checkpoint(
        skippy_session * session,
        uint64_t token_start,
        uint64_t token_count,
        skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "active session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (!skippy_memory_needs_verify_checkpoint(session->ctx->get_memory())) {
        return skippy_success(out_error);
    }
    // No checkpoint is taken when the span rolls back in place, so there is
    // nothing to retire.
    if (skippy_verify_window_can_roll_back_in_place(session, static_cast<size_t>(token_count))) {
        return skippy_success(out_error);
    }
    if (!skippy_retire_verify_checkpoint_exact(
                session->verify_checkpoints, token_start, token_count)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "verify-window checkpoint was not found");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_trim_session(
        struct skippy_session * session,
        uint64_t token_count,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    enum skippy_status usable_status = skippy_require_usable_session(session, out_error);
    if (usable_status != SKIPPY_STATUS_OK) {
        return usable_status;
    }
    if (token_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds int32_t range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > static_cast<uint64_t>(session->n_past)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cannot trim session beyond current token count");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    session->ctx->synchronize();
    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    bool restored_verify_prefix = false;
    enum skippy_status status = skippy_restore_verify_prefix(
            session,
            token_count,
            &restored_verify_prefix,
            out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (!restored_verify_prefix) {
        const llama_pos p0 = static_cast<llama_pos>(token_count);
        if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
            if (!hybrid->seq_rm(session->seq_id, p0, -1)) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim hybrid memory suffix");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        } else if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
            if (!hybrid_iswa->seq_rm(session->seq_id, p0, -1)) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim hybrid ISWA memory suffix");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        } else if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
            if (!iswa->seq_rm(session->seq_id, p0, -1)) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim interleaved-SWA KV suffix");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        } else if (auto * dsa = dynamic_cast<llama_kv_cache_dsa *>(memory)) {
            if (!dsa->seq_rm(session->seq_id, p0, -1)) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim GLM-DSA KV suffix");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        } else if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
            if (!kv->seq_rm(session->seq_id, p0, -1)) {
                skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim native KV suffix");
                return SKIPPY_STATUS_RUNTIME_ERROR;
            }
        } else if (dynamic_cast<llama_memory_recurrent *>(memory) == nullptr) {
            skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "runtime memory type is not supported for trim");
            return SKIPPY_STATUS_UNSUPPORTED;
        }
    }
    session->n_past = static_cast<int32_t>(token_count);
    if (session->token_history.size() > token_count) {
        session->token_history.resize(static_cast<size_t>(token_count));
        skippy_clear_chat_sampling(session);
    }
    if (session->signal_history.size() > token_count) {
        session->signal_history.resize(static_cast<size_t>(token_count));
    }
    skippy_mtp_trim_session_state(session, static_cast<llama_pos>(token_count));
    session->verify_checkpoints.clear();
    session->ctx->synchronize();

    return skippy_success(out_error);
}

static bool skippy_is_reserved_sequence_id(
        const skippy_session * session,
        int32_t seq_id) {
    if (session == nullptr || session->stage_model == nullptr || seq_id < 0) {
        return true;
    }
    const int32_t reserved = static_cast<int32_t>(session->stage_model->lane_count) * 2;
    return seq_id < reserved;
}

static bool skippy_is_reserved_sequence_id_for_model(
        const skippy_model * model,
        int32_t seq_id) {
    if (model == nullptr || seq_id < 0) {
        return true;
    }
    const int32_t reserved = static_cast<int32_t>(model->lane_count) * 2;
    return seq_id < reserved;
}

static enum skippy_status skippy_get_resident_prefix_memory(
        skippy_session * session,
        llama_memory_t * out_memory,
        struct skippy_error ** out_error) {
    if (session == nullptr || session->ctx == nullptr || out_memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "session is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = session->ctx->get_memory();
    if (memory == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "runtime memory is unavailable");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_error * recurrent_error = nullptr;
    llama_memory_recurrent * recurrent = skippy_get_recurrent_memory(session, &recurrent_error);
    if (recurrent_error != nullptr) {
        if (out_error != nullptr) {
            *out_error = recurrent_error;
        } else {
            skippy_error_free(recurrent_error);
        }
        return out_error != nullptr && *out_error != nullptr ? (*out_error)->status : SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (recurrent != nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "resident prefix cache is not enabled for recurrent memory");
        return SKIPPY_STATUS_UNSUPPORTED;
    }

    *out_memory = memory;
    return SKIPPY_STATUS_OK;
}

enum skippy_status skippy_session_save_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        uint64_t token_count,
        struct skippy_error ** out_error) {
    if (token_count > static_cast<uint64_t>(std::numeric_limits<llama_pos>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds llama position range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (session == nullptr || token_count > static_cast<uint64_t>(session->n_past)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cannot save prefix beyond current session position");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_reserved_sequence_id(session, cache_seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cache sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(session, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    skippy_synchronize_context_if_pending(session->stage_model);
    if (!llama_memory_seq_rm(memory, cache_seq_id, -1, -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to clear resident prefix sequence");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (token_count > 0) {
        llama_memory_seq_cp(memory, session->seq_id, cache_seq_id, 0, static_cast<llama_pos>(token_count));
    }
    if (!session->borrowed_sequence && session->stage_model != nullptr && session->seq_id >= 0) {
        const size_t lane = static_cast<size_t>(session->seq_id);
        if (lane < session->stage_model->lane_resident_prefix_tokens.size()
                && session->token_history.size() >= static_cast<size_t>(token_count)) {
            auto & lane_tokens = session->stage_model->lane_resident_prefix_tokens[lane];
            lane_tokens.assign(
                    session->token_history.begin(),
                    session->token_history.begin() + static_cast<ptrdiff_t>(token_count));
            session->preserve_prefix_on_free = true;
            session->preserve_prefix_tokens = static_cast<int32_t>(token_count);
        }
    }
    return skippy_success(out_error);
}

enum skippy_status skippy_session_restore_prefix(
        struct skippy_session * session,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** out_error) {
    if (token_count > static_cast<size_t>(std::numeric_limits<llama_pos>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds llama position range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > 0 && token_ids == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_ids are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_reserved_sequence_id(session, cache_seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cache sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(session, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    skippy_synchronize_context_if_pending(session->stage_model);
    if (token_count > 0) {
        const llama_pos max_pos = llama_memory_seq_pos_max(memory, cache_seq_id);
        if (max_pos < static_cast<llama_pos>(token_count - 1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "resident prefix sequence is incomplete");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }
    if (!llama_memory_seq_rm(memory, session->seq_id, -1, -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to clear execution lane");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    if (token_count > 0) {
        llama_memory_seq_cp(memory, cache_seq_id, session->seq_id, 0, static_cast<llama_pos>(token_count));
    }
    session->n_past = static_cast<int32_t>(token_count);
    if (token_count == 0) {
        session->token_history.clear();
    } else {
        session->token_history.assign(token_ids, token_ids + token_count);
    }
    session->signal_history.clear();
    skippy_mtp_clear_session_state(session);
    session->verify_checkpoints.clear();
    return skippy_success(out_error);
}

enum skippy_status skippy_session_create_from_resident_prefix(
        struct skippy_model * model,
        int32_t cache_seq_id,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_session ** out_session,
        struct skippy_error ** out_error) {
    if (model == nullptr || model->model == nullptr || model->ctx == nullptr || out_session == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "model and out_session are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_session = nullptr;
    if (token_count > static_cast<size_t>(std::numeric_limits<llama_pos>::max())
            || token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_count exceeds supported position range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count > 0 && token_ids == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "token_ids are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_is_reserved_sequence_id_for_model(model, cache_seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "cache sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int32_t seq_id = skippy_claim_execution_lane(model);
    if (seq_id < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "no skippy execution lane is available");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_session temp{};
    temp.stage_model = model;
    temp.ctx = model->ctx;
    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(&temp, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        skippy_release_execution_lane(model, seq_id);
        return status;
    }

    skippy_synchronize_context_if_pending(model);
    if (token_count > 0) {
        const llama_pos max_pos = llama_memory_seq_pos_max(memory, cache_seq_id);
        if (max_pos < static_cast<llama_pos>(token_count - 1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "resident prefix sequence is incomplete");
            skippy_release_execution_lane(model, seq_id);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }
    bool lane_prefix_hit = false;
    const size_t lane = static_cast<size_t>(seq_id);
    if (lane < model->lane_resident_prefix_tokens.size()) {
        const auto & lane_tokens = model->lane_resident_prefix_tokens[lane];
        lane_prefix_hit = lane_tokens.size() == token_count
                && (token_count == 0 || std::equal(lane_tokens.begin(), lane_tokens.end(), token_ids));
    }
    if (!lane_prefix_hit) {
        if (!llama_memory_seq_rm(memory, seq_id, -1, -1)) {
            skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to clear borrowed execution lane");
            skippy_release_execution_lane(model, seq_id);
            return SKIPPY_STATUS_RUNTIME_ERROR;
        }
        if (token_count > 0) {
            llama_memory_seq_cp(memory, cache_seq_id, seq_id, 0, static_cast<llama_pos>(token_count));
        }
    }
    if (!llama_memory_seq_rm(memory, seq_id, static_cast<llama_pos>(token_count), -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to trim borrowed execution lane");
        skippy_release_execution_lane(model, seq_id);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    skippy_session * session = new skippy_session{};
    session->stage_model = model;
    session->ctx = model->ctx;
    session->seq_id = seq_id;
    session->n_past = static_cast<int32_t>(token_count);
    session->borrowed_sequence = false;
    session->borrowed_prefix_tokens = 0;
    if (token_count > 0) {
        session->token_history.assign(token_ids, token_ids + token_count);
    }
    skippy_mtp_clear_session_state(session);
    *out_session = session;
    return skippy_success(out_error);
}

enum skippy_status skippy_session_drop_sequence(
        struct skippy_session * session,
        int32_t seq_id,
        struct skippy_error ** out_error) {
    if (skippy_is_reserved_sequence_id(session, seq_id)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "sequence id overlaps execution lanes");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(session, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    skippy_synchronize_context_if_pending(session->stage_model);
    if (!llama_memory_seq_rm(memory, seq_id, -1, -1)) {
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "failed to drop resident prefix sequence");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
    return skippy_success(out_error);
}

static bool skippy_memory_used_cells(llama_memory_t memory, uint64_t * out_used_cells) {
    if (auto * cache = dynamic_cast<llama_kv_cache *>(memory)) {
        *out_used_cells = cache->get_used_cells();
        return true;
    }
    if (auto * dsa = dynamic_cast<llama_kv_cache_dsa *>(memory)) {
        *out_used_cells = std::max(dsa->get_mla()->get_used_cells(), dsa->get_lid()->get_used_cells());
        return true;
    }
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        *out_used_cells = std::max(iswa->get_base()->get_used_cells(), iswa->get_swa()->get_used_cells());
        return true;
    }
    if (auto * dsa_iswa = dynamic_cast<llama_kv_cache_dsa_iswa *>(memory)) {
        uint64_t dsa_used_cells = 0;
        if (!skippy_memory_used_cells(dsa_iswa->get_dsa(), &dsa_used_cells)) {
            return false;
        }
        *out_used_cells = std::max(dsa_used_cells, dsa_iswa->get_swa()->get_used_cells());
        return true;
    }
    if (auto * msa = dynamic_cast<llama_kv_cache_msa *>(memory)) {
        *out_used_cells = std::max(msa->get_base()->get_used_cells(), msa->get_idx()->get_used_cells());
        return true;
    }
    if (auto * dsv4 = dynamic_cast<llama_kv_cache_dsv4 *>(memory)) {
        uint64_t raw_used_cells = 0;
        if (!skippy_memory_used_cells(dsv4->get_raw(), &raw_used_cells)) {
            return false;
        }
        *out_used_cells = std::max({
                raw_used_cells,
                dsv4->get_csa()->get_used_cells(),
                dsv4->get_hca()->get_used_cells(),
                dsv4->get_lid()->get_used_cells(),
        });
        return true;
    }
    return false;
}

enum skippy_status skippy_session_memory_used_cells(
        struct skippy_session * session,
        uint64_t * out_used_cells,
        struct skippy_error ** out_error) {
    if (out_used_cells == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_used_cells is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_used_cells = 0;

    llama_memory_t memory = nullptr;
    enum skippy_status status = skippy_get_resident_prefix_memory(session, &memory, out_error);
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }
    if (!skippy_memory_used_cells(memory, out_used_cells)) {
        skippy_set_error(out_error, SKIPPY_STATUS_UNSUPPORTED, "physical occupancy is unavailable for this memory type");
        return SKIPPY_STATUS_UNSUPPORTED;
    }
    return skippy_success(out_error);
}

} // extern "C"
