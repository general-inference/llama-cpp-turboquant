#include "skippy/model_package.h"
#include "skippy/errors.h"

#include "gguf.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct skippy_tensor_meta {
    std::string name;
    int32_t layer_index = -1;
    skippy_tensor_role role = SKIPPY_TENSOR_ROLE_UNKNOWN;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<int64_t> ne;
    uint64_t offset = 0;
    uint64_t size = 0;
};

static uint64_t skippy_tensor_element_count(const skippy_tensor_meta & tensor) {
    uint64_t count = 1;
    for (const int64_t dim : tensor.ne) {
        if (dim <= 0 || count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            return 0;
        }
        count *= static_cast<uint64_t>(dim);
    }
    return tensor.ne.empty() ? 0 : count;
}

struct skippy_source_tensor {
    const skippy_model_info * source = nullptr;
    const skippy_tensor_meta * tensor = nullptr;
    uint32_t part_index = 0;
};

struct skippy_model_info {
    gguf_context * ctx = nullptr;
    std::string path;
    std::vector<skippy_tensor_meta> tensors;
};

static bool skippy_gguf_read_u16(const gguf_context * ctx, const char * key, uint16_t & value) {
    const int index = gguf_find_key(ctx, key);
    if (index < 0 || gguf_get_kv_type(ctx, index) != GGUF_TYPE_UINT16) {
        return false;
    }
    value = gguf_get_val_u16(ctx, index);
    return true;
}


static int32_t skippy_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }

    static const std::regex blk_pattern("^blk\\.([0-9]+)\\.");
    static const std::regex cache_pattern("^cache_.*_l([0-9]+)$");

    std::cmatch match;
    if (std::regex_search(name, match, blk_pattern)) {
        return static_cast<int32_t>(std::stol(match[1].str()));
    }
    if (std::regex_search(name, match, cache_pattern)) {
        return static_cast<int32_t>(std::stol(match[1].str()));
    }

    return -1;
}

static enum skippy_tensor_role skippy_role_from_name(const char * name, int32_t layer_index) {
    if (name == nullptr) {
        return SKIPPY_TENSOR_ROLE_UNKNOWN;
    }

    const std::string tensor_name(name);
    if (layer_index >= 0) {
        return SKIPPY_TENSOR_ROLE_LAYER;
    }
    if (tensor_name == "token_embd.weight" || tensor_name == "per_layer_token_embd.weight") {
        return SKIPPY_TENSOR_ROLE_EMBEDDING;
    }
    if (tensor_name == "output.weight" || tensor_name == "output.bias") {
        return SKIPPY_TENSOR_ROLE_OUTPUT;
    }
    if (tensor_name == "output_norm.weight" || tensor_name == "output_norm.bias") {
        return SKIPPY_TENSOR_ROLE_FINAL_NORM;
    }

    return SKIPPY_TENSOR_ROLE_METADATA;
}

template <typename T>
static bool skippy_read(FILE * file, T & out) {
    return std::fread(&out, 1, sizeof(T), file) == sizeof(T);
}

static bool skippy_skip(FILE * file, uint64_t bytes) {
    return std::fseek(file, static_cast<long>(bytes), SEEK_CUR) == 0;
}

static bool skippy_read_string(FILE * file, std::string & out) {
    uint64_t len = 0;
    if (!skippy_read(file, len)) {
        return false;
    }
    out.resize(static_cast<size_t>(len));
    return len == 0 || std::fread(out.data(), 1, static_cast<size_t>(len), file) == len;
}

static bool skippy_gguf_type_size(gguf_type type, size_t & out_size) {
    switch (type) {
        case GGUF_TYPE_UINT8:   out_size = sizeof(uint8_t);  return true;
        case GGUF_TYPE_INT8:    out_size = sizeof(int8_t);   return true;
        case GGUF_TYPE_UINT16:  out_size = sizeof(uint16_t); return true;
        case GGUF_TYPE_INT16:   out_size = sizeof(int16_t);  return true;
        case GGUF_TYPE_UINT32:  out_size = sizeof(uint32_t); return true;
        case GGUF_TYPE_INT32:   out_size = sizeof(int32_t);  return true;
        case GGUF_TYPE_FLOAT32: out_size = sizeof(float);    return true;
        case GGUF_TYPE_BOOL:    out_size = sizeof(int8_t);   return true;
        case GGUF_TYPE_UINT64:  out_size = sizeof(uint64_t); return true;
        case GGUF_TYPE_INT64:   out_size = sizeof(int64_t);  return true;
        case GGUF_TYPE_FLOAT64: out_size = sizeof(double);   return true;
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
        case GGUF_TYPE_COUNT:
            return false;
    }

    return false;
}

static bool skippy_skip_gguf_value(FILE * file, gguf_type type);

static bool skippy_skip_gguf_array(FILE * file) {
    int32_t raw_type = 0;
    uint64_t count = 0;
    if (!skippy_read(file, raw_type) || !skippy_read(file, count)) {
        return false;
    }

    const gguf_type type = static_cast<gguf_type>(raw_type);
    if (type == GGUF_TYPE_STRING) {
        for (uint64_t i = 0; i < count; ++i) {
            std::string ignored;
            if (!skippy_read_string(file, ignored)) {
                return false;
            }
        }
        return true;
    }

    size_t element_size = 0;
    if (!skippy_gguf_type_size(type, element_size)) {
        return false;
    }
    if (count > UINT64_MAX / element_size) {
        return false;
    }
    return skippy_skip(file, count * element_size);
}

static bool skippy_skip_gguf_value(FILE * file, gguf_type type) {
    if (type == GGUF_TYPE_STRING) {
        std::string ignored;
        return skippy_read_string(file, ignored);
    }
    if (type == GGUF_TYPE_ARRAY) {
        return skippy_skip_gguf_array(file);
    }

    size_t value_size = 0;
    if (!skippy_gguf_type_size(type, value_size)) {
        return false;
    }
    return skippy_skip(file, value_size);
}

static bool skippy_parse_tensor_metadata(skippy_model_info * info) {
    FILE * file = ggml_fopen(info->path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }

    char magic[4] = {};
    uint32_t version = 0;
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    bool ok = std::fread(magic, 1, sizeof(magic), file) == sizeof(magic) &&
            std::memcmp(magic, GGUF_MAGIC, sizeof(magic)) == 0 &&
            skippy_read(file, version) &&
            version == GGUF_VERSION &&
            skippy_read(file, n_tensors) &&
            skippy_read(file, n_kv) &&
            n_tensors >= 0 &&
            n_kv >= 0;

    for (int64_t i = 0; ok && i < n_kv; ++i) {
        std::string key;
        int32_t raw_type = 0;
        ok = skippy_read_string(file, key) &&
                skippy_read(file, raw_type) &&
                skippy_skip_gguf_value(file, static_cast<gguf_type>(raw_type));
    }

    info->tensors.clear();
    info->tensors.reserve(static_cast<size_t>(std::max<int64_t>(n_tensors, 0)));
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        skippy_tensor_meta meta;
        uint32_t n_dims = 0;
        int32_t raw_type = 0;
        ok = skippy_read_string(file, meta.name) && skippy_read(file, n_dims);
        if (!ok || n_dims > GGML_MAX_DIMS) {
            ok = false;
            break;
        }

        meta.ne.resize(n_dims);
        for (uint32_t dim = 0; ok && dim < n_dims; ++dim) {
            ok = skippy_read(file, meta.ne[dim]);
        }
        ok = ok && skippy_read(file, raw_type) && skippy_read(file, meta.offset);
        if (!ok) {
            break;
        }

        meta.type = static_cast<ggml_type>(raw_type);
        meta.layer_index = skippy_layer_from_name(meta.name.c_str());
        meta.role = skippy_role_from_name(meta.name.c_str(), meta.layer_index);
        const int64_t tensor_id = gguf_find_tensor(info->ctx, meta.name.c_str());
        if (tensor_id < 0) {
            ok = false;
            break;
        }
        meta.size = static_cast<uint64_t>(gguf_get_tensor_size(info->ctx, tensor_id));
        info->tensors.push_back(std::move(meta));
    }

    std::fclose(file);
    return ok && info->tensors.size() == static_cast<size_t>(n_tensors);
}

static size_t skippy_align(size_t value, size_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

static void skippy_clear_split_metadata(gguf_context * ctx) {
    gguf_remove_key(ctx, "split.no");
    gguf_remove_key(ctx, "split.count");
    gguf_remove_key(ctx, "split.tensors.count");
}

static skippy_model_info * skippy_model_info_open_owned(const char * path) {
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx == nullptr) {
        return nullptr;
    }

    skippy_model_info * info = new skippy_model_info{};
    info->ctx = ctx;
    info->path = path;
    if (!skippy_parse_tensor_metadata(info)) {
        gguf_free(ctx);
        delete info;
        return nullptr;
    }
    return info;
}

static void skippy_model_info_delete(skippy_model_info * info) {
    if (info != nullptr) {
        gguf_free(info->ctx);
        delete info;
    }
}

static enum skippy_status skippy_add_tensor_to_context(
        ggml_context * ggml_ctx,
        gguf_context * out_ctx,
        const skippy_tensor_meta & meta,
        struct skippy_error ** out_error) {
    int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    for (size_t dim = 0; dim < meta.ne.size() && dim < GGML_MAX_DIMS; ++dim) {
        ne[dim] = meta.ne[dim];
    }
    ggml_tensor * tensor = ggml_new_tensor(
            ggml_ctx,
            meta.type,
            static_cast<int>(meta.ne.size()),
            ne);
    if (tensor == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate output tensor metadata");
        return SKIPPY_STATUS_ERROR;
    }
    ggml_set_name(tensor, meta.name.c_str());
    gguf_add_tensor(out_ctx, tensor);
    return SKIPPY_STATUS_OK;
}

static enum skippy_status skippy_copy_source_tensors(
        const std::vector<skippy_source_tensor> & selected,
        const char * output_path,
        gguf_context * out_ctx,
        bool consume_sources,
        struct skippy_error ** out_error) {
    if (!gguf_write_to_file(out_ctx, output_path, true)) {
        skippy_set_error(out_error, SKIPPY_STATUS_IO_ERROR, "failed to write output GGUF metadata");
        return SKIPPY_STATUS_IO_ERROR;
    }

    FILE * output = ggml_fopen(output_path, "ab");
    if (output == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_IO_ERROR, "failed to open output GGUF for tensor copy");
        return SKIPPY_STATUS_IO_ERROR;
    }

    const size_t output_alignment = gguf_get_alignment(out_ctx);
    std::vector<uint8_t> buffer(1024 * 1024);
    std::vector<uint8_t> zeroes(output_alignment, 0);
    bool ok = true;

    // When consuming, each source part is unlinked as soon as the last of its
    // tensors has been absorbed into the output. `selected` is grouped by
    // source part (the part loaders preserve input order), so a single
    // trailing scan after each tensor copy is enough: any source whose last
    // selected tensor was just copied can be released. Sources are opened
    // metadata-only (no_alloc), so nothing pins the file after the per-tensor
    // read handle is closed.
    std::set<const skippy_model_info *> consumed;
    const auto try_consume_source = [&consumed](const skippy_model_info * source) {
        if (consumed.insert(source).second) {
            std::remove(source->path.c_str());
        }
    };

    for (const skippy_source_tensor & item : selected) {
        FILE * input = ggml_fopen(item.source->path.c_str(), "rb");
        if (input == nullptr) {
            ok = false;
            break;
        }

        // std::fseek takes a long, which is 32-bit on Win32 MSVC, so GGUF files
        // over 4 GB wrap and tensor reads land at the wrong offset. Seek with a
        // 64-bit API where available, and validate the absolute offset first so
        // an out-of-range value fails loudly instead of wrapping.
        static_assert(std::numeric_limits<size_t>::digits <= std::numeric_limits<uint64_t>::digits,
                      "size_t wider than uint64_t is not supported here");
        const uint64_t source_data_offset =
            static_cast<uint64_t>(gguf_get_data_offset(item.source->ctx));
        const uint64_t tensor_offset = static_cast<uint64_t>(item.tensor->offset);
        bool seek_ok = false;
        if (tensor_offset <= (std::numeric_limits<uint64_t>::max)() - source_data_offset) {
            const uint64_t absolute_offset = source_data_offset + tensor_offset;
#if defined(_MSC_VER)
            seek_ok = absolute_offset <= static_cast<uint64_t>((std::numeric_limits<__int64>::max)()) &&
                      _fseeki64(input, static_cast<__int64>(absolute_offset), SEEK_SET) == 0;
#elif defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS == 64
            seek_ok = absolute_offset <= static_cast<uint64_t>((std::numeric_limits<off_t>::max)()) &&
                      fseeko(input, static_cast<off_t>(absolute_offset), SEEK_SET) == 0;
#else
            seek_ok = absolute_offset <= static_cast<uint64_t>((std::numeric_limits<long>::max)()) &&
                      std::fseek(input, static_cast<long>(absolute_offset), SEEK_SET) == 0;
#endif
        }
        if (!seek_ok) {
            std::fclose(input);
            ok = false;
            break;
        }

        uint64_t remaining = item.tensor->size;
        while (remaining > 0) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, buffer.size()));
            if (std::fread(buffer.data(), 1, chunk, input) != chunk ||
                    std::fwrite(buffer.data(), 1, chunk, output) != chunk) {
                ok = false;
                break;
            }
            remaining -= chunk;
        }

        if (std::fclose(input) != 0) {
            ok = false;
        }
        if (!ok) {
            break;
        }

        const size_t padded = skippy_align(static_cast<size_t>(item.tensor->size), output_alignment);
        const size_t padding = padded - static_cast<size_t>(item.tensor->size);
        if (padding > 0 && std::fwrite(zeroes.data(), 1, padding, output) != padding) {
            ok = false;
            break;
        }

        if (consume_sources) {
            const bool more_from_this_source = std::any_of(
                &item + 1, selected.data() + selected.size(),
                [&item](const skippy_source_tensor & candidate) {
                    return candidate.source == item.source;
                });
            if (!more_from_this_source) {
                try_consume_source(item.source);
            }
        }
    }

    if (std::fclose(output) != 0) {
        ok = false;
    }

    if (!ok) {
        skippy_set_error(out_error, SKIPPY_STATUS_IO_ERROR, "failed to copy selected GGUF tensor data");
        return SKIPPY_STATUS_IO_ERROR;
    }

    return SKIPPY_STATUS_OK;
}


enum skippy_status skippy_model_info_open(
        const char * path,
        struct skippy_model_info ** out_info,
        struct skippy_error ** out_error) {
    if (path == nullptr || out_info == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "path and out_info are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    *out_info = nullptr;

    skippy_model_info * info = skippy_model_info_open_owned(path);
    if (info == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to open GGUF model metadata");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    *out_info = info;
    return skippy_success(out_error);
}

enum skippy_status skippy_model_info_free(
        struct skippy_model_info * info,
        struct skippy_error ** out_error) {
    skippy_model_info_delete(info);
    return skippy_success(out_error);
}

enum skippy_status skippy_model_info_tensor_count(
        struct skippy_model_info * info,
        size_t * out_count,
        struct skippy_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || out_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "info and out_count are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int64_t n_tensors = gguf_get_n_tensors(info->ctx);
    if (n_tensors < 0) {
        skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "GGUF reported negative tensor count");
        return SKIPPY_STATUS_MODEL_ERROR;
    }

    *out_count = static_cast<size_t>(n_tensors);
    return skippy_success(out_error);
}

enum skippy_status skippy_model_info_tensor_at(
        struct skippy_model_info * info,
        size_t index,
        struct skippy_tensor_info * out_tensor,
        struct skippy_error ** out_error) {
    if (info == nullptr || info->ctx == nullptr || out_tensor == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "info and out_tensor are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int64_t n_tensors = gguf_get_n_tensors(info->ctx);
    if (index >= static_cast<size_t>(n_tensors)) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "tensor index is out of range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const int64_t tensor_id = static_cast<int64_t>(index);
    const char * name = gguf_get_tensor_name(info->ctx, tensor_id);
    const int32_t layer_index = skippy_layer_from_name(name);

    out_tensor->name = name;
    out_tensor->layer_index = layer_index;
    out_tensor->role = skippy_role_from_name(name, layer_index);
    out_tensor->ggml_type = static_cast<uint32_t>(gguf_get_tensor_type(info->ctx, tensor_id));
    out_tensor->byte_size = static_cast<uint64_t>(gguf_get_tensor_size(info->ctx, tensor_id));
    out_tensor->element_count = index < info->tensors.size() ?
            skippy_tensor_element_count(info->tensors[index]) : 0;
    return skippy_success(out_error);
}

enum skippy_status skippy_write_gguf_metadata_from_parts(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error) {
    if (input_paths == nullptr || input_count == 0 || output_path == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "input paths and output_path are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (input_count > std::numeric_limits<uint32_t>::max()) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "input part count exceeds uint32");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    std::vector<skippy_model_info *> sources;
    sources.reserve(input_count);
    bool split_metadata_expected = false;
    for (size_t i = 0; i < input_count; ++i) {
        if (input_paths[i] == nullptr) {
            for (skippy_model_info * source : sources) {
                skippy_model_info_delete(source);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "input path is null");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        skippy_model_info * source = skippy_model_info_open_owned(input_paths[i]);
        if (source == nullptr) {
            for (skippy_model_info * opened : sources) {
                skippy_model_info_delete(opened);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to open GGUF metadata source");
            return SKIPPY_STATUS_MODEL_ERROR;
        }
        sources.push_back(source);
        if (input_count > 1) {
            const bool has_split_no = gguf_find_key(source->ctx, "split.no") >= 0;
            const bool has_split_count = gguf_find_key(source->ctx, "split.count") >= 0;
            uint16_t split_no = 0;
            uint16_t split_count = 0;
            bool invalid_split_metadata = has_split_no != has_split_count;
            if (i == 0) {
                split_metadata_expected = has_split_no && has_split_count;
            } else if (has_split_no != split_metadata_expected) {
                invalid_split_metadata = true;
            }
            if (!invalid_split_metadata && split_metadata_expected &&
                    (!skippy_gguf_read_u16(source->ctx, "split.no", split_no) ||
                     !skippy_gguf_read_u16(source->ctx, "split.count", split_count) ||
                     split_no != i || split_count != input_count)) {
                invalid_split_metadata = true;
            }
            if (invalid_split_metadata) {
                for (skippy_model_info * opened : sources) {
                    skippy_model_info_delete(opened);
                }
                skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "input paths must consistently omit split metadata or use exact split.no order and split.count");
                return SKIPPY_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    std::vector<skippy_source_tensor> selected;
    std::set<std::string> names;
    for (size_t part_index = 0; part_index < sources.size(); ++part_index) {
        const skippy_model_info * source = sources[part_index];
        for (const skippy_tensor_meta & tensor : source->tensors) {
            if (names.insert(tensor.name).second) {
                selected.push_back({ source, &tensor, static_cast<uint32_t>(part_index) });
            }
        }
    }

    gguf_context * out_ctx = gguf_init_empty();
    if (out_ctx == nullptr) {
        for (skippy_model_info * source : sources) {
            skippy_model_info_delete(source);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate output GGUF context");
        return SKIPPY_STATUS_ERROR;
    }

    gguf_set_kv(out_ctx, sources.front()->ctx);
    skippy_clear_split_metadata(out_ctx);
    gguf_remove_key(out_ctx, "skippy.package.materialized");
    gguf_set_val_bool(out_ctx, "skippy.package.metadata_only", true);
    gguf_set_val_u32(out_ctx, "skippy.package.part_count", static_cast<uint32_t>(input_count));
    gguf_set_val_u32(out_ctx, "skippy.package.locator_schema", 1);

    std::vector<uint32_t> tensor_parts;
    std::vector<uint64_t> tensor_offsets;
    std::vector<uint64_t> tensor_sizes;
    std::vector<uint32_t> tensor_alignments;
    tensor_parts.reserve(selected.size());
    tensor_offsets.reserve(selected.size());
    tensor_sizes.reserve(selected.size());
    tensor_alignments.reserve(selected.size());
    for (const skippy_source_tensor & item : selected) {
        tensor_parts.push_back(item.part_index);
        tensor_offsets.push_back(static_cast<uint64_t>(gguf_get_data_offset(item.source->ctx)) + item.tensor->offset);
        tensor_sizes.push_back(item.tensor->size);
        tensor_alignments.push_back(static_cast<uint32_t>(gguf_get_alignment(item.source->ctx)));
    }
    gguf_set_arr_data(out_ctx, "skippy.package.tensor_part", GGUF_TYPE_UINT32,
            tensor_parts.data(), tensor_parts.size());
    gguf_set_arr_data(out_ctx, "skippy.package.tensor_offset", GGUF_TYPE_UINT64,
            tensor_offsets.data(), tensor_offsets.size());
    gguf_set_arr_data(out_ctx, "skippy.package.tensor_size", GGUF_TYPE_UINT64,
            tensor_sizes.data(), tensor_sizes.size());
    gguf_set_arr_data(out_ctx, "skippy.package.tensor_alignment", GGUF_TYPE_UINT32,
            tensor_alignments.data(), tensor_alignments.size());

    const size_t tensor_mem = ggml_tensor_overhead() * (selected.size() + 1) + 1024 * 1024;
    ggml_init_params ggml_params = {
        /*.mem_size   =*/ tensor_mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    if (ggml_ctx == nullptr) {
        gguf_free(out_ctx);
        for (skippy_model_info * source : sources) {
            skippy_model_info_delete(source);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate tensor metadata context");
        return SKIPPY_STATUS_ERROR;
    }

    for (const skippy_source_tensor & item : selected) {
        const enum skippy_status status = skippy_add_tensor_to_context(ggml_ctx, out_ctx, *item.tensor, out_error);
        if (status != SKIPPY_STATUS_OK) {
            ggml_free(ggml_ctx);
            gguf_free(out_ctx);
            for (skippy_model_info * source : sources) {
                skippy_model_info_delete(source);
            }
            return status;
        }
    }

    const bool written = gguf_write_to_file(out_ctx, output_path, true);
    ggml_free(ggml_ctx);
    gguf_free(out_ctx);
    for (skippy_model_info * source : sources) {
        skippy_model_info_delete(source);
    }
    if (!written) {
        skippy_set_error(out_error, SKIPPY_STATUS_IO_ERROR, "failed to write metadata-only GGUF");
        return SKIPPY_STATUS_IO_ERROR;
    }

    return skippy_success(out_error);
}

// Shared implementation for the from-parts writers. `consume_inputs` deletes
// each input part file as soon as its tensors have been absorbed into the
// output; callers that need the inputs to survive (bench materialization from
// published package files, metadata carriers) pass false.
static enum skippy_status skippy_write_gguf_parts_impl(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        bool consume_inputs,
        struct skippy_error ** out_error) {
    if (input_paths == nullptr || input_count == 0 || output_path == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "input paths and output_path are required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    std::vector<skippy_model_info *> sources;
    sources.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        if (input_paths[i] == nullptr) {
            for (skippy_model_info * source : sources) {
                skippy_model_info_delete(source);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "input path is null");
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        skippy_model_info * source = skippy_model_info_open_owned(input_paths[i]);
        if (source == nullptr) {
            for (skippy_model_info * opened : sources) {
                skippy_model_info_delete(opened);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_MODEL_ERROR, "failed to open GGUF package part");
            return SKIPPY_STATUS_MODEL_ERROR;
        }
        sources.push_back(source);
    }

    std::vector<skippy_source_tensor> selected;
    std::set<std::string> names;
    for (const skippy_model_info * source : sources) {
        for (const skippy_tensor_meta & tensor : source->tensors) {
            if (names.insert(tensor.name).second) {
                selected.push_back({ source, &tensor });
            }
        }
    }

    gguf_context * out_ctx = gguf_init_empty();
    if (out_ctx == nullptr) {
        for (skippy_model_info * source : sources) {
            skippy_model_info_delete(source);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate output GGUF context");
        return SKIPPY_STATUS_ERROR;
    }

    gguf_set_kv(out_ctx, sources.front()->ctx);
    skippy_clear_split_metadata(out_ctx);
    gguf_set_val_u32(out_ctx, GGUF_KEY_GENERAL_ALIGNMENT, static_cast<uint32_t>(gguf_get_alignment(out_ctx)));
    gguf_set_val_bool(out_ctx, "skippy.package.materialized", true);
    gguf_set_val_u32(out_ctx, "skippy.package.part_count", static_cast<uint32_t>(input_count));

    const size_t tensor_mem = ggml_tensor_overhead() * (selected.size() + 1) + 1024 * 1024;
    ggml_init_params ggml_params = {
        /*.mem_size   =*/ tensor_mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    if (ggml_ctx == nullptr) {
        gguf_free(out_ctx);
        for (skippy_model_info * source : sources) {
            skippy_model_info_delete(source);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_ERROR, "failed to allocate tensor metadata context");
        return SKIPPY_STATUS_ERROR;
    }

    for (const skippy_source_tensor & item : selected) {
        const enum skippy_status status = skippy_add_tensor_to_context(ggml_ctx, out_ctx, *item.tensor, out_error);
        if (status != SKIPPY_STATUS_OK) {
            ggml_free(ggml_ctx);
            gguf_free(out_ctx);
            for (skippy_model_info * source : sources) {
                skippy_model_info_delete(source);
            }
            return status;
        }
    }

    const enum skippy_status status = skippy_copy_source_tensors(selected, output_path, out_ctx, consume_inputs, out_error);
    ggml_free(ggml_ctx);
    gguf_free(out_ctx);
    for (skippy_model_info * source : sources) {
        skippy_model_info_delete(source);
    }
    if (status != SKIPPY_STATUS_OK) {
        return status;
    }

    return skippy_success(out_error);
}

enum skippy_status skippy_write_gguf_from_parts(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error) {
    return skippy_write_gguf_parts_impl(input_paths, input_count, output_path, false, out_error);
}

enum skippy_status skippy_write_gguf_from_parts_consuming(
        const char * const * input_paths,
        size_t input_count,
        const char * output_path,
        struct skippy_error ** out_error) {
    return skippy_write_gguf_parts_impl(input_paths, input_count, output_path, true, out_error);
}
