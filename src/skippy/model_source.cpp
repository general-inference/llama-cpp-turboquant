#include "skippy/model_source.h"

#include "skippy/errors.h"
#include "skippy/model_load_config_internal.h"
#include "skippy/model_loading_internal.h"
#include "skippy/runtime.h"
#include "skippy/runtime_support.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using skippy_imatrix_map = std::unordered_map<std::string, const skippy_model_imatrix_entry_v1 *>;

static bool skippy_build_imatrix(
        const skippy_model_tensor_source_v1 * source,
        skippy_imatrix_map & entries,
        std::vector<llama_model_imatrix_data> & native_entries,
        std::string & error) {
    if (source->imatrix_count > 0 && source->imatrix == nullptr) {
        error = "importance matrix count requires a non-null entry array";
        return false;
    }
    native_entries.reserve(source->imatrix_count + 1);
    for (size_t i = 0; i < source->imatrix_count; ++i) {
        const skippy_model_imatrix_entry_v1 & entry = source->imatrix[i];
        if (entry.tensor_name == nullptr || entry.values == nullptr || entry.value_count == 0) {
            error = "importance matrix entries require a name and non-empty values";
            return false;
        }
        for (size_t value_index = 0; value_index < entry.value_count; ++value_index) {
            if (!std::isfinite(entry.values[value_index])) {
                error = "importance matrix contains a non-finite value";
                return false;
            }
        }
        if (!entries.emplace(entry.tensor_name, &entry).second) {
            error = std::string("duplicate importance matrix entry for ") + entry.tensor_name;
            return false;
        }
        native_entries.push_back({ entry.tensor_name, entry.values, entry.value_count });
    }
    native_entries.push_back({ nullptr, nullptr, 0 });
    return true;
}

static uint32_t skippy_gguf_u32_or(
        const gguf_context * metadata,
        const std::string & key,
        uint32_t fallback) {
    const int64_t key_id = gguf_find_key(metadata, key.c_str());
    if (key_id < 0) {
        return fallback;
    }
    switch (gguf_get_kv_type(metadata, key_id)) {
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(metadata, key_id);
        case GGUF_TYPE_INT32: {
            const int32_t value = gguf_get_val_i32(metadata, key_id);
            return value > 0 ? static_cast<uint32_t>(value) : fallback;
        }
        case GGUF_TYPE_ARRAY:
            break;
        default:
            return fallback;
    }

    uint32_t result = fallback;
    const size_t count = gguf_get_arr_n(metadata, key_id);
    switch (gguf_get_arr_type(metadata, key_id)) {
        case GGUF_TYPE_UINT32: {
            const auto * values = static_cast<const uint32_t *>(gguf_get_arr_data(metadata, key_id));
            for (size_t i = 0; i < count; ++i) {
                result = std::max(result, values[i]);
            }
            break;
        }
        case GGUF_TYPE_INT32: {
            const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(metadata, key_id));
            for (size_t i = 0; i < count; ++i) {
                if (values[i] > 0) {
                    result = std::max(result, static_cast<uint32_t>(values[i]));
                }
            }
            break;
        }
        default:
            break;
    }
    return result;
}

static bool skippy_imatrix_size(
        const ggml_tensor * tensor,
        size_t & result,
        std::string & error) {
    if (tensor->ne[0] <= 0 || tensor->ne[2] <= 0 ||
        static_cast<uint64_t>(tensor->ne[0]) > std::numeric_limits<size_t>::max() /
                static_cast<uint64_t>(tensor->ne[2])) {
        error = std::string("invalid importance matrix shape for tensor ") + ggml_get_name(tensor);
        return false;
    }
    result = static_cast<size_t>(tensor->ne[0]) * static_cast<size_t>(tensor->ne[2]);
    return true;
}

static bool skippy_apply_source_quantization(
        gguf_context * metadata,
        ggml_context * tensor_context,
        int32_t quantization_ftype,
        const skippy_imatrix_map & imatrix,
        const llama_model_imatrix_data * native_imatrix,
        std::string & error) {
    if (quantization_ftype < 0) {
        return true;
    }

    const int64_t architecture_id = gguf_find_key(metadata, "general.architecture");
    if (architecture_id < 0) {
        error = "metadata GGUF is missing general.architecture";
        return false;
    }
    if (gguf_get_kv_type(metadata, architecture_id) != GGUF_TYPE_STRING) {
        error = "metadata GGUF general.architecture must be a string";
        return false;
    }
    const char * architecture = gguf_get_val_str(metadata, architecture_id);
    const std::string prefix = architecture;
    const uint32_t n_embd = skippy_gguf_u32_or(metadata, prefix + ".embedding_length", 0);
    const uint32_t n_head = skippy_gguf_u32_or(metadata, prefix + ".attention.head_count", 0);
    llama_quant_model_desc desc = {
        /*.architecture =*/ architecture,
        /*.n_embd =*/ n_embd,
        /*.n_ff =*/ skippy_gguf_u32_or(metadata, prefix + ".feed_forward_length", 0),
        /*.n_layer =*/ skippy_gguf_u32_or(metadata, prefix + ".block_count", 0),
        /*.n_head =*/ n_head,
        /*.n_head_kv =*/ skippy_gguf_u32_or(metadata, prefix + ".attention.head_count_kv", n_head),
        /*.n_expert =*/ skippy_gguf_u32_or(metadata, prefix + ".expert_count", 0),
        /*.n_embd_head_k =*/ skippy_gguf_u32_or(
                metadata,
                prefix + ".attention.key_length",
                n_head > 0 ? n_embd / n_head : 0),
        /*.n_embd_head_v =*/ skippy_gguf_u32_or(
                metadata,
                prefix + ".attention.value_length",
                n_head > 0 ? n_embd / n_head : 0),
    };
    if (desc.n_layer == 0 || desc.n_layer > LLAMA_MAX_LAYERS) {
        error = "metadata GGUF has an invalid block count for quantization";
        return false;
    }
    llama_model * quant_model = llama_quant_model_from_metadata(&desc);
    if (quant_model == nullptr) {
        error = "failed to construct quantization policy from checkpoint metadata";
        return false;
    }
    llama_model_quantize_params quant_params = llama_model_quantize_default_params();
    quant_params.ftype = static_cast<llama_ftype>(quantization_ftype);
    quant_params.imatrix = native_imatrix;
    quantize_state_impl * quant_state = llama_quant_init(quant_model, &quant_params);
    if (quant_state == nullptr) {
        llama_model_free(quant_model);
        error = "failed to initialize tensor quantization policy";
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * tensor = ggml_get_first_tensor(tensor_context);
         tensor != nullptr;
         tensor = ggml_get_next_tensor(tensor_context, tensor)) {
        if (llama_quant_tensor_allows_quantization(quant_state, tensor)) {
            tensors.push_back(tensor);
        }
    }
    std::vector<ggml_type> result_types(tensors.size());
    if (!tensors.empty()) {
        llama_quant_compute_types(
                quant_state,
                static_cast<llama_ftype>(quantization_ftype),
                tensors.data(),
                result_types.data(),
                tensors.size());
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (llama_quant_tensor_requires_imatrix(
                    tensors[i], result_types[i], static_cast<llama_ftype>(quantization_ftype))) {
            const auto entry = imatrix.find(ggml_get_name(tensors[i]));
            size_t expected = 0;
            if (!skippy_imatrix_size(tensors[i], expected, error)) {
                llama_quant_free(quant_state);
                llama_model_free(quant_model);
                return false;
            }
            if (entry == imatrix.end()) {
                error = std::string("missing importance matrix for tensor ") + ggml_get_name(tensors[i]);
                llama_quant_free(quant_state);
                llama_model_free(quant_model);
                return false;
            }
            if (entry->second->value_count != expected) {
                error = std::string("importance matrix size mismatch for tensor ") + ggml_get_name(tensors[i]);
                llama_quant_free(quant_state);
                llama_model_free(quant_model);
                return false;
            }
        }
        gguf_set_tensor_type(metadata, ggml_get_name(tensors[i]), result_types[i]);
    }
    llama_quant_free(quant_state);
    llama_model_free(quant_model);
    return true;
}

struct skippy_model_source_state {
    const skippy_model_tensor_source_v1 * source = nullptr;
    const skippy_imatrix_map * imatrix = nullptr;
    std::string error;
};

static void skippy_set_model_tensor_from_source(ggml_tensor * tensor, void * user_data) {
    auto * state = static_cast<skippy_model_source_state *>(user_data);
    const int64_t element_count_i64 = ggml_nelements(tensor);
    if (element_count_i64 < 0) {
        throw std::runtime_error("source tensor has a negative element count");
    }
    const size_t element_count = static_cast<size_t>(element_count_i64);
    std::vector<float> source_values(element_count);
    const char * callback_message = nullptr;
    const skippy_status status = state->source->read_tensor_f32(
            ggml_get_name(tensor),
            source_values.data(),
            element_count,
            &callback_message,
            state->source->user_data);
    if (status != SKIPPY_STATUS_OK) {
        state->error = callback_message != nullptr
                ? callback_message
                : "model tensor source callback failed";
        throw std::runtime_error(state->error);
    }

    const int64_t elements_per_row = tensor->ne[0];
    if (elements_per_row <= 0 || element_count_i64 % elements_per_row != 0) {
        state->error = "source tensor has an invalid row shape";
        throw std::runtime_error(state->error);
    }
    std::vector<uint8_t> encoded(ggml_nbytes(tensor));
    const float * imatrix = nullptr;
    const auto imatrix_entry = state->imatrix->find(ggml_get_name(tensor));
    if (imatrix_entry != state->imatrix->end()) {
        size_t expected = 0;
        if (!skippy_imatrix_size(tensor, expected, state->error)) {
            throw std::runtime_error(state->error);
        }
        if (imatrix_entry->second->value_count != expected) {
            state->error = std::string("importance matrix size mismatch for tensor ") + ggml_get_name(tensor);
            throw std::runtime_error(state->error);
        }
        imatrix = imatrix_entry->second->values;
    }
    size_t written = 0;
    if (imatrix != nullptr && tensor->ne[2] > 1) {
        // llama.cpp records one importance vector per expert. Quantize 3-D
        // expert tensors one slice at a time so each expert sees its own
        // ne[0]-wide vector instead of reusing expert zero's weights.
        const int64_t rows_per_expert = tensor->ne[1];
        for (int64_t expert = 0; expert < tensor->ne[2]; ++expert) {
            written += ggml_quantize_chunk(
                    tensor->type,
                    source_values.data(),
                    encoded.data(),
                    expert * rows_per_expert * elements_per_row,
                    rows_per_expert,
                    elements_per_row,
                    imatrix + expert * elements_per_row);
        }
    } else {
        written = ggml_quantize_chunk(
                tensor->type,
                source_values.data(),
                encoded.data(),
                0,
                element_count_i64 / elements_per_row,
                elements_per_row,
                imatrix);
    }
    if (written != encoded.size()) {
        state->error = "source tensor encoding produced an unexpected byte count";
        throw std::runtime_error(state->error);
    }
    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
}

static llama_model_loader_stage_selection skippy_source_selection(
        const skippy_runtime_config * config) {
    llama_model_loader_stage_selection selection;
    if (!skippy_runtime_has_stage_plan(config)) {
        return selection;
    }
    selection.layer_start = config->layer_start;
    selection.layer_end = config->layer_end;
    if (config->resident_tensor_name_count == 0 || config->resident_tensor_names == nullptr) {
        throw std::invalid_argument("filtered stage loading requires explicit resident tensor names");
    }
    for (size_t i = 0; i < config->resident_tensor_name_count; ++i) {
        const char * name = config->resident_tensor_names[i];
        if (name == nullptr || name[0] == '\0') {
            throw std::invalid_argument("resident tensor names must be non-empty");
        }
        if (!selection.resident_tensor_names.emplace(name).second) {
            throw std::invalid_argument("resident tensor names must be unique");
        }
    }
    return selection;
}

static enum skippy_status skippy_model_open_from_source_impl(
        const void * metadata_gguf,
        size_t metadata_gguf_size,
        const struct skippy_model_tensor_source_v1 * source,
        int32_t quantization_ftype,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    if (metadata_gguf == nullptr || metadata_gguf_size == 0 || source == nullptr ||
        source->abi_version != SKIPPY_MODEL_TENSOR_SOURCE_V1_ABI_VERSION ||
        source->struct_size < sizeof(skippy_model_tensor_source_v1) ||
        source->read_tensor_f32 == nullptr || out_model == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "valid metadata, tensor source, and out_model are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_model = nullptr;
    if (skippy_runtime_has_stage_plan(config) &&
        (config->layer_start < 0 || config->layer_start >= config->layer_end)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "layer_start must be non-negative and less than layer_end");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (skippy_runtime_has_stage_plan(config) &&
            (config->execution_contract == nullptr || config->execution_contract[0] == '\0')) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "filtered stage loading requires an admitted execution contract");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    skippy_imatrix_map imatrix;
    std::vector<llama_model_imatrix_data> native_imatrix;
    std::string imatrix_error;
    if (!skippy_build_imatrix(source, imatrix, native_imatrix, imatrix_error)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, imatrix_error.c_str());
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    ggml_context * tensor_context = nullptr;
    gguf_init_params init_params = { true, &tensor_context };
    gguf_context * metadata = gguf_init_from_buffer(metadata_gguf, metadata_gguf_size, init_params);
    if (metadata == nullptr || tensor_context == nullptr) {
        if (tensor_context != nullptr) {
            ggml_free(tensor_context);
        }
        if (metadata != nullptr) {
            gguf_free(metadata);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to parse metadata-only GGUF");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    std::string quant_error;
    const llama_model_imatrix_data * native_imatrix_ptr =
            source->imatrix_count > 0 ? native_imatrix.data() : nullptr;
    if (!skippy_apply_source_quantization(
                metadata,
                tensor_context,
                quantization_ftype,
                imatrix,
                native_imatrix_ptr,
                quant_error)) {
        ggml_free(tensor_context);
        gguf_free(metadata);
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, quant_error.c_str());
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    llama_model_params params = llama_model_default_params();
    params.load_mtp = config != nullptr && skippy_runtime_is_terminal_stage(config) &&
                      (skippy_runtime_has_stage_plan(config) || config->mtp_source == SKIPPY_MTP_SOURCE_INTEGRATED);
    if (config != nullptr) {
        params.n_gpu_layers = config->n_gpu_layers;
        skippy_apply_model_load_memory_config(config, params);
    }
    llama_backend_init();
    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    std::vector<ggml_backend_dev_t> selected_devices;
    const skippy_status device_status = skippy_apply_selected_backend_device(
            config, params, selected_devices, out_error);
    if (device_status != SKIPPY_STATUS_OK) {
        ggml_free(tensor_context);
        gguf_free(metadata);
        return device_status;
    }

    skippy_model_source_state state = { source, &imatrix, {} };
    const auto selection = skippy_source_selection(config);
    llama_model * model = llama_model_init_from_user_with_selection(
            metadata,
            skippy_set_model_tensor_from_source,
            &state,
            params,
            skippy_runtime_has_stage_plan(config) ? &selection : nullptr);
    ggml_free(tensor_context);
    gguf_free(metadata);
    if (model == nullptr) {
        const char * message = state.error.empty()
                ? "failed to load model from tensor source"
                : state.error.c_str();
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, message);
        return SKIPPY_STATUS_MODEL_ERROR;
    }
    return skippy_finish_model_open(model, config, nullptr, out_model, out_error);
}

enum skippy_status skippy_model_open_from_source(
        const void * metadata_gguf,
        size_t metadata_gguf_size,
        const struct skippy_model_tensor_source_v1 * source,
        int32_t quantization_ftype,
        const struct skippy_runtime_config * config,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    try {
        return skippy_model_open_from_source_impl(
                metadata_gguf,
                metadata_gguf_size,
                source,
                quantization_ftype,
                config,
                out_model,
                out_error);
    } catch (const std::exception & exception) {
        if (out_model != nullptr) {
            *out_model = nullptr;
        }
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, exception.what());
        return SKIPPY_STATUS_RUNTIME_ERROR;
    } catch (...) {
        if (out_model != nullptr) {
            *out_model = nullptr;
        }
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, "unexpected failure while loading callback tensor source");
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }
}
