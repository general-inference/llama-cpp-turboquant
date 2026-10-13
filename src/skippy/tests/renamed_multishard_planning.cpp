#include "embedding_input_regression.h"
#include "live_graph_reference.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama.h"
#include "skippy/activation.h"
#include "skippy/execution.h"
#include "skippy/model_package.h"
#include "skippy/package_tensor_binding.h"
#include "skippy/runtime.h"
#include "skippy/runtime_types.h"
#include "skippy/stage_plan.h"
#include "skippy/stage_slice_plan.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char * split_no_key = "split.no";
constexpr const char * split_count_key = "split.count";
constexpr const char * split_tensor_count_key = "split.tensors.count";

using ggml_context_owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using gguf_context_owner = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

struct inspected_split {
    std::string path;
    uint16_t split_no = 0;
    uint16_t split_count = 0;
    int32_t tensor_count = 0;
    ggml_context_owner tensors { nullptr, ggml_free };
    gguf_context_owner metadata { nullptr, gguf_free };
};

struct package_inventory {
    std::vector<std::string> ordered_paths;
    std::vector<skippy_package_tensor_binding_input> tensors;
};

bool read_exact(std::ifstream & input, uint64_t offset, size_t length, std::vector<uint8_t> & bytes) {
    bytes.resize(length);
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(length));
    return input.good();
}

bool write_padding(std::ofstream & output, size_t length) {
    static const std::vector<uint8_t> zeros(GGUF_DEFAULT_ALIGNMENT, 0);
    output.write(reinterpret_cast<const char *>(zeros.data()), static_cast<std::streamsize>(length));
    return output.good();
}

bool write_two_renamed_splits(
        const std::string & source_path,
        const std::string & output_directory,
        std::vector<std::string> & paths,
        std::string & error) {
    std::filesystem::remove_all(output_directory);
    std::filesystem::create_directories(output_directory);
    ggml_context * source_tensors_raw = nullptr;
    gguf_init_params init = { true, &source_tensors_raw };
    gguf_context_owner source_metadata(
            gguf_init_from_file(source_path.c_str(), init),
            gguf_free);
    ggml_context_owner source_tensors(source_tensors_raw, ggml_free);
    if (!source_metadata || !source_tensors) {
        error = "cannot inspect source GGUF";
        return false;
    }
    const int64_t tensor_count = gguf_get_n_tensors(source_metadata.get());
    if (tensor_count < 2 || tensor_count > INT32_MAX) {
        error = "source GGUF cannot be split into two non-empty shards";
        return false;
    }
    const int64_t midpoint = tensor_count / 2;
    paths = {
        output_directory + "/renamed-metadata.gguf",
        output_directory + "/renamed-weights.gguf",
    };
    std::ifstream source(source_path, std::ios::binary);
    if (!source) {
        error = "cannot open source GGUF payload";
        return false;
    }
    for (uint16_t split_no = 0; split_no < 2; ++split_no) {
        gguf_context_owner output_metadata(gguf_init_empty(), gguf_free);
        if (!output_metadata) {
            error = "cannot allocate split metadata";
            return false;
        }
        if (split_no == 0) {
            gguf_set_kv(output_metadata.get(), source_metadata.get());
        }
        gguf_set_val_u16(output_metadata.get(), split_no_key, split_no);
        gguf_set_val_u16(output_metadata.get(), split_count_key, 2);
        gguf_set_val_i32(
                output_metadata.get(),
                split_tensor_count_key,
                static_cast<int32_t>(tensor_count));
        const int64_t begin = split_no == 0 ? 0 : midpoint;
        const int64_t end = split_no == 0 ? midpoint : tensor_count;
        for (int64_t index = begin; index < end; ++index) {
            const char * name = gguf_get_tensor_name(source_metadata.get(), index);
            ggml_tensor * tensor = ggml_get_tensor(source_tensors.get(), name);
            if (tensor == nullptr) {
                error = "source GGUF tensor metadata is incomplete";
                return false;
            }
            gguf_add_tensor(output_metadata.get(), tensor);
        }
        std::ofstream output(paths[split_no], std::ios::binary | std::ios::trunc);
        if (!output) {
            error = "cannot create renamed split";
            return false;
        }
        std::vector<uint8_t> metadata_bytes(gguf_get_meta_size(output_metadata.get()));
        gguf_get_meta_data(output_metadata.get(), metadata_bytes.data());
        output.write(
                reinterpret_cast<const char *>(metadata_bytes.data()),
                static_cast<std::streamsize>(metadata_bytes.size()));
        std::vector<uint8_t> payload;
        for (int64_t index = begin; index < end; ++index) {
            const char * name = gguf_get_tensor_name(source_metadata.get(), index);
            const ggml_tensor * tensor = ggml_get_tensor(source_tensors.get(), name);
            const size_t stored_length = ggml_nbytes(tensor);
            const uint64_t source_offset =
                    gguf_get_data_offset(source_metadata.get()) +
                    gguf_get_tensor_offset(source_metadata.get(), index);
            if (!read_exact(source, source_offset, stored_length, payload)) {
                error = "cannot read source tensor payload";
                return false;
            }
            output.write(
                    reinterpret_cast<const char *>(payload.data()),
                    static_cast<std::streamsize>(payload.size()));
            const size_t padding = GGML_PAD(stored_length, GGUF_DEFAULT_ALIGNMENT) - stored_length;
            if (!write_padding(output, padding)) {
                error = "cannot write split tensor padding";
                return false;
            }
        }
        if (!output.good()) {
            error = "cannot finish renamed split";
            return false;
        }
    }
    return true;
}

bool required_split_metadata(
        const gguf_context * metadata,
        uint16_t & split_no,
        uint16_t & split_count,
        int32_t & tensor_count,
        std::string & error) {
    const int no_key = gguf_find_key(metadata, split_no_key);
    const int count_key = gguf_find_key(metadata, split_count_key);
    const int tensors_key = gguf_find_key(metadata, split_tensor_count_key);
    if (no_key < 0 || count_key < 0 || tensors_key < 0 ||
            gguf_get_kv_type(metadata, no_key) != GGUF_TYPE_UINT16 ||
            gguf_get_kv_type(metadata, count_key) != GGUF_TYPE_UINT16 ||
            gguf_get_kv_type(metadata, tensors_key) != GGUF_TYPE_INT32) {
        error = "split metadata is missing or has the wrong type";
        return false;
    }
    split_no = gguf_get_val_u16(metadata, no_key);
    split_count = gguf_get_val_u16(metadata, count_key);
    tensor_count = gguf_get_val_i32(metadata, tensors_key);
    return split_count > 0 && tensor_count > 0;
}

bool inspect_package_splits(
        const std::vector<std::string> & unordered_paths,
        package_inventory & inventory,
        std::string & error) {
    inventory = {};
    if (unordered_paths.empty()) {
        error = "package split list is empty";
        return false;
    }
    std::vector<inspected_split> splits;
    splits.reserve(unordered_paths.size());
    for (const auto & path : unordered_paths) {
        inspected_split split;
        split.path = path;
        ggml_context * tensors_raw = nullptr;
        gguf_init_params init = { true, &tensors_raw };
        split.metadata.reset(gguf_init_from_file(path.c_str(), init));
        split.tensors.reset(tensors_raw);
        if (!split.metadata || !split.tensors ||
                !required_split_metadata(
                    split.metadata.get(),
                    split.split_no,
                    split.split_count,
                    split.tensor_count,
                    error)) {
            if (error.empty()) {
                error = "cannot inspect package split";
            }
            return false;
        }
        splits.push_back(std::move(split));
    }
    const uint16_t expected_count = splits.front().split_count;
    const int32_t expected_tensors = splits.front().tensor_count;
    if (expected_count != unordered_paths.size()) {
        error = "package split count differs from the supplied file count";
        return false;
    }
    std::sort(splits.begin(), splits.end(), [](const auto & left, const auto & right) {
        return left.split_no < right.split_no;
    });
    int64_t total_tensors = 0;
    std::set<std::string> native_names;
    for (size_t split_index = 0; split_index < splits.size(); ++split_index) {
        const auto & split = splits[split_index];
        if (split.split_no != split_index || split.split_count != expected_count ||
                split.tensor_count != expected_tensors) {
            error = "package split metadata is duplicated, unordered, or inconsistent";
            return false;
        }
        inventory.ordered_paths.push_back(split.path);
        const int64_t count = gguf_get_n_tensors(split.metadata.get());
        total_tensors += count;
        for (int64_t tensor_index = 0; tensor_index < count; ++tensor_index) {
            const char * name = gguf_get_tensor_name(split.metadata.get(), tensor_index);
            ggml_tensor * tensor = ggml_get_tensor(split.tensors.get(), name);
            if (name == nullptr || tensor == nullptr || !native_names.insert(name).second) {
                error = "package split tensor inventory is incomplete or duplicated";
                return false;
            }
            skippy_package_tensor_binding_input input;
            input.native_name = name;
            input.type = tensor->type;
            const int dimensions = ggml_n_dims(tensor);
            input.dimensions.assign(tensor->ne, tensor->ne + dimensions);
            input.split_no = split.split_no;
            input.data_offset =
                    gguf_get_data_offset(split.metadata.get()) +
                    gguf_get_tensor_offset(split.metadata.get(), tensor_index);
            input.stored_length = ggml_nbytes(tensor);
            inventory.tensors.push_back(std::move(input));
        }
    }
    if (total_tensors != expected_tensors) {
        error = "package split tensor total is incomplete";
        return false;
    }
    std::sort(inventory.tensors.begin(), inventory.tensors.end(), [](const auto & left, const auto & right) {
        return left.native_name < right.native_name;
    });
    for (size_t index = 0; index < inventory.tensors.size(); ++index) {
        inventory.tensors[index].tensor_id = "package.tensor." + std::to_string(index);
    }
    std::sort(inventory.tensors.begin(), inventory.tensors.end(), [](const auto & left, const auto & right) {
        return left.tensor_id < right.tensor_id;
    });
    return true;
}

bool write_metadata_carrier(
        const std::vector<std::string> & input_paths,
        const std::string & output_path,
        std::string & error) {
    std::vector<const char *> paths;
    for (const auto & path : input_paths) {
        paths.push_back(path.c_str());
    }
    skippy_error * native_error = nullptr;
    const skippy_status status = skippy_write_gguf_metadata_from_parts(
            paths.data(), paths.size(), output_path.c_str(), &native_error);
    if (status == SKIPPY_STATUS_OK) {
        return true;
    }
    error = native_error == nullptr ? "metadata carrier write failed" : native_error->message;
    skippy_error_free(native_error);
    return false;
}

bool write_selected_payload(
        const package_inventory & inventory,
        const std::vector<std::string> & resident_names,
        const std::string & output_path,
        std::string & error) {
    std::vector<gguf_context_owner> metadata;
    std::vector<ggml_context_owner> tensors;
    for (const auto & path : inventory.ordered_paths) {
        ggml_context * tensors_raw = nullptr;
        gguf_init_params init = { true, &tensors_raw };
        metadata.emplace_back(gguf_init_from_file(path.c_str(), init), gguf_free);
        tensors.emplace_back(tensors_raw, ggml_free);
        if (!metadata.back() || !tensors.back()) {
            error = "cannot inspect selected payload source";
            return false;
        }
    }

    std::set<std::string> requested(resident_names.begin(), resident_names.end());
    std::vector<const skippy_package_tensor_binding_input *> selected;
    gguf_context_owner output_metadata(gguf_init_empty(), gguf_free);
    for (const auto & source : inventory.tensors) {
        if (requested.erase(source.native_name) == 0) {
            continue;
        }
        if (source.split_no >= tensors.size()) {
            error = "selected payload source part is out of range";
            return false;
        }
        ggml_tensor * tensor = ggml_get_tensor(tensors[source.split_no].get(), source.native_name.c_str());
        if (tensor == nullptr) {
            error = "selected payload source tensor is absent";
            return false;
        }
        gguf_add_tensor(output_metadata.get(), tensor);
        selected.push_back(&source);
    }
    if (!requested.empty() || selected.empty()) {
        error = "selected payload resident closure is incomplete";
        return false;
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot create selected payload";
        return false;
    }
    std::vector<uint8_t> metadata_bytes(gguf_get_meta_size(output_metadata.get()));
    gguf_get_meta_data(output_metadata.get(), metadata_bytes.data());
    output.write(
            reinterpret_cast<const char *>(metadata_bytes.data()),
            static_cast<std::streamsize>(metadata_bytes.size()));
    std::vector<uint8_t> payload;
    for (const auto * source : selected) {
        std::ifstream input(inventory.ordered_paths[source->split_no], std::ios::binary);
        if (!input || !read_exact(input, source->data_offset, source->stored_length, payload)) {
            error = "cannot read selected payload tensor";
            return false;
        }
        output.write(
                reinterpret_cast<const char *>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
        const size_t padding = GGML_PAD(source->stored_length, GGUF_DEFAULT_ALIGNMENT) - source->stored_length;
        if (!write_padding(output, padding)) {
            error = "cannot write selected payload padding";
            return false;
        }
    }
    if (!output.good()) {
        error = "cannot finish selected payload";
        return false;
    }
    return true;
}

bool inspect_metadata_carrier(
        const std::string & path,
        const package_inventory & source,
        size_t expected_part_count,
        package_inventory & carrier,
        std::string & error) {
    ggml_context * tensors_raw = nullptr;
    gguf_init_params init = { true, &tensors_raw };
    gguf_context_owner metadata(gguf_init_from_file(path.c_str(), init), gguf_free);
    ggml_context_owner tensors(tensors_raw, ggml_free);
    if (!metadata || !tensors) {
        error = "cannot inspect metadata carrier";
        return false;
    }
    const int metadata_only_key = gguf_find_key(metadata.get(), "skippy.package.metadata_only");
    const int part_count_key = gguf_find_key(metadata.get(), "skippy.package.part_count");
    const int locator_schema_key = gguf_find_key(metadata.get(), "skippy.package.locator_schema");
    const int tensor_part_key = gguf_find_key(metadata.get(), "skippy.package.tensor_part");
    const int tensor_offset_key = gguf_find_key(metadata.get(), "skippy.package.tensor_offset");
    const int tensor_size_key = gguf_find_key(metadata.get(), "skippy.package.tensor_size");
    const int tensor_alignment_key = gguf_find_key(metadata.get(), "skippy.package.tensor_alignment");
    if (metadata_only_key < 0 || part_count_key < 0 || locator_schema_key < 0 ||
            tensor_part_key < 0 || tensor_offset_key < 0 || tensor_size_key < 0 || tensor_alignment_key < 0 ||
            gguf_get_kv_type(metadata.get(), metadata_only_key) != GGUF_TYPE_BOOL ||
            gguf_get_kv_type(metadata.get(), part_count_key) != GGUF_TYPE_UINT32 ||
            gguf_get_kv_type(metadata.get(), locator_schema_key) != GGUF_TYPE_UINT32 ||
            gguf_get_kv_type(metadata.get(), tensor_part_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(metadata.get(), tensor_part_key) != GGUF_TYPE_UINT32 ||
            gguf_get_kv_type(metadata.get(), tensor_offset_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(metadata.get(), tensor_offset_key) != GGUF_TYPE_UINT64 ||
            gguf_get_kv_type(metadata.get(), tensor_size_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(metadata.get(), tensor_size_key) != GGUF_TYPE_UINT64 ||
            gguf_get_kv_type(metadata.get(), tensor_alignment_key) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(metadata.get(), tensor_alignment_key) != GGUF_TYPE_UINT32 ||
            !gguf_get_val_bool(metadata.get(), metadata_only_key) ||
            gguf_get_val_u32(metadata.get(), part_count_key) != expected_part_count ||
            gguf_get_val_u32(metadata.get(), locator_schema_key) != 1 ||
            std::filesystem::file_size(path) != gguf_get_data_offset(metadata.get()) ||
            gguf_get_n_tensors(metadata.get()) != static_cast<int64_t>(source.tensors.size()) ||
            gguf_get_arr_n(metadata.get(), tensor_part_key) != source.tensors.size() ||
            gguf_get_arr_n(metadata.get(), tensor_offset_key) != source.tensors.size() ||
            gguf_get_arr_n(metadata.get(), tensor_size_key) != source.tensors.size() ||
            gguf_get_arr_n(metadata.get(), tensor_alignment_key) != source.tensors.size()) {
        error = "metadata carrier markers, size, or tensor count are invalid";
        return false;
    }
    const auto * tensor_parts = static_cast<const uint32_t *>(gguf_get_arr_data(metadata.get(), tensor_part_key));
    const auto * tensor_offsets = static_cast<const uint64_t *>(gguf_get_arr_data(metadata.get(), tensor_offset_key));
    const auto * tensor_sizes = static_cast<const uint64_t *>(gguf_get_arr_data(metadata.get(), tensor_size_key));
    const auto * tensor_alignments = static_cast<const uint32_t *>(gguf_get_arr_data(metadata.get(), tensor_alignment_key));
    std::map<std::string, const skippy_package_tensor_binding_input *> source_by_name;
    for (const auto & tensor : source.tensors) {
        source_by_name.emplace(tensor.native_name, &tensor);
    }
    carrier = {};
    carrier.ordered_paths.push_back(path);
    for (int64_t index = 0; index < gguf_get_n_tensors(metadata.get()); ++index) {
        const char * name = gguf_get_tensor_name(metadata.get(), index);
        ggml_tensor * tensor = name == nullptr ? nullptr : ggml_get_tensor(tensors.get(), name);
        const auto found = name == nullptr ? source_by_name.end() : source_by_name.find(name);
        if (tensor == nullptr || found == source_by_name.end()) {
            error = "metadata carrier tensor inventory differs from source";
            return false;
        }
        const auto & expected = *found->second;
        if (tensor_parts[index] != expected.split_no || tensor_offsets[index] != expected.data_offset ||
                tensor_sizes[index] != expected.stored_length || tensor_alignments[index] == 0 ||
                (tensor_alignments[index] & (tensor_alignments[index] - 1)) != 0) {
            error = "metadata carrier tensor locator differs from source";
            return false;
        }
        const int dimensions = ggml_n_dims(tensor);
        if (expected.dimensions.size() != static_cast<size_t>(dimensions) ||
                tensor->type != expected.type ||
                !std::equal(tensor->ne, tensor->ne + dimensions, expected.dimensions.begin(), expected.dimensions.end())) {
            error = "metadata carrier tensor descriptor differs from source";
            return false;
        }
        auto input = expected;
        input.split_no = 0;
        input.data_offset = gguf_get_data_offset(metadata.get()) + gguf_get_tensor_offset(metadata.get(), index);
        input.stored_length = ggml_nbytes(tensor);
        carrier.tensors.push_back(std::move(input));
    }
    std::sort(carrier.tensors.begin(), carrier.tensors.end(), [](const auto & left, const auto & right) {
        return left.tensor_id < right.tensor_id;
    });
    return true;
}

bool is_dummy_buffer(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->buffer != nullptr &&
           ggml_backend_buffer_get_size(tensor->buffer) == 0 &&
           ggml_backend_buffer_get_base(tensor->buffer) == nullptr;
}

bool check_model_buffers(const llama_model * model) {
    const auto & tensors = llama_internal_get_tensor_map(model);
    return !tensors.empty() && std::all_of(tensors.begin(), tensors.end(), [](const auto & item) {
        return is_dummy_buffer(item.second);
    });
}

bool check_recurrent_buffers(const llama_memory_recurrent * memory) {
    size_t count = 0;
    for (const auto * tensor : memory->r_l) {
        if (tensor != nullptr) {
            ++count;
            if (!is_dummy_buffer(tensor)) {
                return false;
            }
        }
    }
    for (const auto * tensor : memory->s_l) {
        if (tensor != nullptr) {
            ++count;
            if (!is_dummy_buffer(tensor)) {
                return false;
            }
        }
    }
    for (const auto * tensor : memory->p_l) {
        if (tensor != nullptr) {
            ++count;
            if (!is_dummy_buffer(tensor)) {
                return false;
            }
        }
    }
    return count > 0;
}

bool check_kv_buffers(const llama_kv_cache * memory, bool allow_empty = false) {
    if (memory == nullptr) {
        return false;
    }
    const auto layers = memory->get_layer_ids();
    return (allow_empty || !layers.empty()) && std::all_of(layers.begin(), layers.end(), [&](uint32_t layer) {
        const int32_t layer_id = static_cast<int32_t>(layer);
        ggml_tensor * value = memory->get_v_storage(layer_id);
        return is_dummy_buffer(memory->get_k_storage(layer_id)) &&
               (value == nullptr || is_dummy_buffer(value));
    });
}

bool check_k_only_buffers(const llama_kv_cache * memory) {
    if (memory == nullptr) {
        return false;
    }
    const auto layers = memory->get_layer_ids();
    return !layers.empty() && std::all_of(layers.begin(), layers.end(), [&](uint32_t layer) {
        return is_dummy_buffer(memory->get_k_storage(static_cast<int32_t>(layer)));
    });
}

bool check_memory_buffers(llama_context * context, bool expect_recurrent) {
    llama_memory_t memory = llama_get_memory(context);
    if (memory == nullptr) {
        return false;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        return check_kv_buffers(hybrid->get_mem_attn(), true) &&
               check_recurrent_buffers(hybrid->get_mem_recr());
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        auto * attention = hybrid->get_mem_attn();
        return check_kv_buffers(attention->get_base(), true) &&
               check_kv_buffers(attention->get_swa(), true) &&
               check_recurrent_buffers(hybrid->get_mem_recr());
    }
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory)) {
        return check_recurrent_buffers(recurrent);
    }
    if (auto * dsa_iswa = dynamic_cast<llama_kv_cache_dsa_iswa *>(memory)) {
        auto * dsa = dsa_iswa->get_dsa();
        return check_kv_buffers(dsa->get_mla()) &&
               check_kv_buffers(dsa->get_lid()) &&
               check_kv_buffers(dsa_iswa->get_swa());
    }
    if (auto * dsa = dynamic_cast<llama_kv_cache_dsa *>(memory)) {
        return check_kv_buffers(dsa->get_mla()) &&
               check_kv_buffers(dsa->get_lid());
    }
    if (auto * dsv4 = dynamic_cast<llama_kv_cache_dsv4 *>(memory)) {
        auto * raw = dsv4->get_raw();
        return raw != nullptr &&
               check_k_only_buffers(raw->get_swa()) &&
               check_k_only_buffers(dsv4->get_csa()) &&
               check_k_only_buffers(dsv4->get_hca()) &&
               check_k_only_buffers(dsv4->get_lid());
    }
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return check_kv_buffers(iswa->get_base()) &&
               check_kv_buffers(iswa->get_swa());
    }
    if (auto * msa = dynamic_cast<llama_kv_cache_msa *>(memory)) {
        return check_kv_buffers(msa->get_base()) &&
               check_kv_buffers(msa->get_idx());
    }
    return !expect_recurrent && check_kv_buffers(dynamic_cast<llama_kv_cache *>(memory));
}

bool check_compute_buffers(llama_context * context) {
    ggml_backend_sched_t scheduler = context->get_sched();
    for (int index = 0; index < ggml_backend_sched_get_n_backends(scheduler); ++index) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(scheduler, index);
        if (ggml_backend_sched_get_buffer_size(scheduler, backend) != 0) {
            return false;
        }
    }
    return true;
}

bool build_profile(
        llama_model * model,
        const std::vector<skippy_parameter_binding> & bindings,
        bool expect_recurrent,
        const char * profile_id,
        uint32_t n_tokens,
        uint32_t n_sequences,
        uint32_t n_outputs,
        std::vector<std::string> & dependencies,
        std::string & error,
        uint32_t n_rs_seq = 0) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 8;
    params.n_ubatch = 8;
    params.n_seq_max = n_sequences;
    params.n_rs_seq = n_rs_seq;
    params.flash_attn_type = expect_recurrent
            ? LLAMA_FLASH_ATTN_TYPE_DISABLED
            : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model, params),
            llama_free);
    if (!context) {
        error = "cannot create metadata-only context";
        return false;
    }
    llama_memory_context_ptr memory_context;
    if (llama_memory_t memory = llama_get_memory(context.get())) {
        memory_context = memory->init_full();
    }
    ggml_cgraph * graph = context->graph_reserve(
            n_tokens,
            n_sequences,
            n_outputs,
            memory_context.get(),
            true);
    if (graph == nullptr || ggml_graph_n_nodes(graph) == 0) {
        error = "metadata-only graph is empty";
        return false;
    }
    if (!check_model_buffers(model)) {
        error = "metadata-only graph allocated model payload storage";
        return false;
    }
    if (!check_memory_buffers(context.get(), expect_recurrent)) {
        llama_memory_t memory = llama_get_memory(context.get());
        error = "metadata-only graph allocated or omitted state storage type=" +
                (memory == nullptr ? std::string("<null>") : std::string(typeid(*memory).name()));
        return false;
    }
    if (!check_compute_buffers(context.get())) {
        error = "metadata-only graph allocated compute storage";
        return false;
    }
    const llm_graph_result * result = context->get_gf_res_reserve();
    const skippy_execution_profile_guard guard = {
        n_tokens,
        n_sequences,
        n_outputs,
        0,
        "renamed-multishard-v1",
        "cpu",
    };
    const std::vector<skippy_stage_profile_input> profiles = {{
        profile_id,
        graph,
        &result->get_block_boundaries(),
        &result->get_state_effects(),
        &bindings,
        &result->get_request_input_bindings(),
        llama_model_n_layer(model),
        guard,
        result->get_logits() != nullptr ? result->get_logits() : result->get_embd(),
    }};
    skippy_guarded_stage_slice_plan plan;
    if (!skippy_build_guarded_stage_slice_plan(
                profiles,
                0,
                llama_model_n_layer(model),
                plan,
                error) ||
            plan.profiles.size() != 1 ||
            plan.resident_parameter_identities.empty()) {
        if (error.empty()) {
            error = "metadata-only graph did not produce a guarded package-ID plan";
        }
        return false;
    }
    const std::set<std::string> package_ids = [&] {
        std::set<std::string> ids;
        for (const auto & binding : bindings) {
            ids.insert(binding.identity);
        }
        return ids;
    }();
    for (const auto & identity : plan.resident_parameter_identities) {
        if (package_ids.count(identity) == 0) {
            error = "guarded plan contains a non-package parameter identity";
            return false;
        }
    }
    if (expect_recurrent) {
        // Shared delta-net family contract: every recurrent layer must declare
        // ownership of BOTH its conv (cache_r_l*) and SSM (cache_s_l*) storage,
        // with at least one ordered write per kind per layer.
        const auto & effects = result->get_state_effects();
        const int32_t n_layer = llama_model_n_layer(model);
        for (int32_t il = 0; il < n_layer; ++il) {
            if (!model->hparams.is_recr(il)) {
                continue; // hybrid families: full-attention layers own no recurrent storage
            }
            bool conv_read = false, ssm_read = false;
            int64_t conv_writes = 0, ssm_writes = 0;
            for (const auto & effect : effects) {
                if (effect.layer != il) {
                    continue;
                }
                if (effect.kind == skippy_state_kind::recurrent_conv) {
                    conv_read |= effect.access == skippy_state_access::read;
                    conv_writes += effect.access == skippy_state_access::write;
                } else if (effect.kind == skippy_state_kind::recurrent_ssm) {
                    ssm_read |= effect.access == skippy_state_access::read;
                    ssm_writes += effect.access == skippy_state_access::write;
                }
            }
            if (!conv_read || conv_writes == 0 || !ssm_read || ssm_writes == 0) {
                error = "delta-net recurrent state ownership incomplete at layer " + std::to_string(il) +
                    " (conv read=" + (conv_read ? "1" : "0") + " writes=" + std::to_string(conv_writes) +
                    " ssm read=" + (ssm_read ? "1" : "0") + " writes=" + std::to_string(ssm_writes) + ")";
                return false;
            }
            // Every declared write must reference its real graph node: a CPY (or
            // zeroing SCALE for ordinal 0) whose destination is the registered
            // storage. A discarded copy node or a non-write tensor registered as
            // the effect node fails here instead of mis-ordering stage plans.
            for (const auto & effect : effects) {
                if (effect.layer != il || effect.access != skippy_state_access::write) {
                    continue;
                }
                const ggml_tensor * node = effect.effect_node;
                if (node == nullptr) {
                    error = "delta-net state write at layer " + std::to_string(il) + " has no effect node";
                    return false;
                }
                const ggml_op expect = effect.write_ordinal == 0 ? GGML_OP_SCALE : GGML_OP_CPY;
                if (node->op != expect) {
                    error = "delta-net state write at layer " + std::to_string(il) +
                        " registers a non-write node (op=" + std::to_string(static_cast<int>(node->op)) + ")";
                    return false;
                }
                if (skippy_state_canonical_storage(node->src[node->op == GGML_OP_CPY ? 1 : 0]) != effect.storage) {
                    error = "delta-net state write at layer " + std::to_string(il) +
                        " does not target its registered storage";
                    return false;
                }
                bool in_graph = false;
                for (int i = 0; i < ggml_graph_n_nodes(graph) && !in_graph; ++i) {
                    in_graph = ggml_graph_node(graph, i) == node;
                }
                if (!in_graph) {
                    error = "delta-net state write node at layer " + std::to_string(il) +
                        " is not present in the graph";
                    return false;
                }
            }
        }
    }
    dependencies = plan.resident_parameter_identities;
    return true;
}

bool rejects_invalid_bindings(
        const llama_model * model,
        const std::vector<skippy_package_tensor_binding_input> & valid) {
    std::vector<skippy_parameter_binding> bindings;
    std::string error;
    auto rejects = [&](std::vector<skippy_package_tensor_binding_input> changed) {
        error.clear();
        return !skippy_bind_package_tensors(model, changed, bindings, error) && !error.empty();
    };
    auto changed = valid;
    changed[1].tensor_id = changed[0].tensor_id;
    if (!rejects(changed)) { return false; }
    changed = valid;
    std::reverse(changed.begin(), changed.end());
    if (!rejects(changed)) { return false; }
    changed = valid;
    changed[1].native_name = changed[0].native_name;
    if (!rejects(changed)) { return false; }
    changed = valid;
    changed.pop_back();
    if (!rejects(changed)) { return false; }
    changed = valid;
    changed.push_back(valid.back());
    changed.back().tensor_id = "package.tensor.extra";
    changed.back().native_name = "native.tensor.extra";
    if (!rejects(changed)) { return false; }
    changed = valid;
    changed[0].type = changed[0].type == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    if (!rejects(changed)) { return false; }
    changed = valid;
    ++changed[0].dimensions[0];
    if (!rejects(changed)) { return false; }
    changed = valid;
    ++changed[0].stored_length;
    if (!rejects(changed)) { return false; }
    changed = valid;
    changed[0].split_no = static_cast<uint16_t>(1 - changed[0].split_no);
    if (!rejects(changed)) { return false; }
    changed = valid;
    ++changed[0].data_offset;
    return rejects(changed);
}

// Exercise physical source metadata separately from loader-created logical shapes.
// The fixture model is already loaded from GGUF, so source dimensions must stay
// immutable while a native tensor is reshaped as TENSOR_ALLOW_RESHAPE does.
bool accepts_source_shape_after_native_reshape(
        llama_model * model,
        const std::vector<skippy_package_tensor_binding_input> & inputs) {
    for (const auto & entry : llama_internal_get_tensor_map(model)) {
        ggml_tensor * tensor = entry.second;
        if (tensor->type != GGML_TYPE_F32 || ggml_nelements(tensor) < 4 ||
                ggml_nelements(tensor) % 2 != 0 || !ggml_is_contiguous(tensor)) {
            continue;
        }
        const auto original = *tensor;
        const int64_t elements = ggml_nelements(tensor);
        tensor->ne[0] = original.ne[0] == elements ? elements / 2 : elements;
        tensor->ne[1] = elements / tensor->ne[0];
        tensor->ne[2] = tensor->ne[3] = 1;
        tensor->nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            tensor->nb[i] = tensor->nb[i - 1] * tensor->ne[i - 1];
        }
        std::vector<skippy_parameter_binding> bindings;
        std::string error;
        bool ok = skippy_bind_package_tensors(model, inputs, bindings, error);
        auto forged = inputs;
        for (auto & input : forged) {
            if (input.native_name == entry.first) {
                input.dimensions.assign(tensor->ne, tensor->ne + GGML_MAX_DIMS);
            }
        }
        // Matching the new native shape is not evidence that the package's
        // physical layout matches GGUF, even when element/byte counts agree.
        ok = ok && !skippy_bind_package_tensors(model, forged, bindings, error);
        // Corrupt an active dimension; singleton strides are intentionally ignored.
        tensor->nb[0] += sizeof(float);
        ok = ok && !skippy_bind_package_tensors(model, inputs, bindings, error);
        *tensor = original;
        return ok;
    }
    return false;
}

std::string plan_string(const skippy_stage_plan * plan, skippy_stage_plan_string_ref_v1 reference) {
    const char * data = nullptr;
    size_t length = 0;
    skippy_error * error = nullptr;
    if (skippy_stage_plan_string_v1(plan, reference, &data, &length, &error) != SKIPPY_STATUS_OK) {
        skippy_error_free(error);
        return {};
    }
    return std::string(data, length);
}

struct runtime_stage_plan {
    std::string execution_contract;
    std::string source_execution_contract;
    std::string full_execution_contract;
    int32_t layer_start = 0;
    int32_t layer_end = 0;
    std::vector<std::string> full_resident_names;
    std::vector<std::string> resident_names;
    std::vector<std::string> import_identities;
    std::vector<std::string> import_bindings;
    std::vector<std::string> source_resident_names;
    std::vector<std::string> export_identities;
    std::vector<std::string> export_bindings;
};

bool check_public_stage_plan_abi(
        const package_inventory & inventory,
        const std::vector<std::string> & paths,
        int32_t layer_count,
        runtime_stage_plan * runtime_stage = nullptr) {
    std::vector<const char *> shard_paths;
    for (const auto & path : paths) {
        shard_paths.push_back(path.c_str());
    }
    std::vector<skippy_stage_planner_tensor_v1> tensors;
    for (const auto & source : inventory.tensors) {
        skippy_stage_planner_tensor_v1 tensor{};
        tensor.abi_version = 1;
        tensor.struct_size = sizeof(tensor);
        tensor.tensor_id = source.tensor_id.c_str();
        tensor.native_name = source.native_name.c_str();
        tensor.ggml_type = source.type;
        tensor.dimension_count = source.dimensions.size();
        std::copy(source.dimensions.begin(), source.dimensions.end(), tensor.dimensions);
        tensor.split_no = source.split_no;
        tensor.data_offset = source.data_offset;
        tensor.stored_length = source.stored_length;
        tensors.push_back(tensor);
    }
    const std::vector<skippy_stage_planner_profile_v1> profiles = {
        {1, sizeof(skippy_stage_planner_profile_v1), "batched", 8, 2, 8, 0},
        {1, sizeof(skippy_stage_planner_profile_v1), "decode", 1, 1, 1, 0},
        {1, sizeof(skippy_stage_planner_profile_v1), "prefill", 8, 1, 8, 0},
    };
    const std::string package_id = "sha256:" + std::string(64, 'a');
    const std::string graph_configuration_id = "skippy.test.graph.v1";
    const std::string backend_id = "skippy.test.backend";
    skippy_stage_planner_config_v1 config = {1, sizeof(skippy_stage_planner_config_v1),
        package_id.c_str(), shard_paths.data(), shard_paths.size(), tensors.data(), tensors.size(),
        profiles.data(), profiles.size(), graph_configuration_id.c_str(), backend_id.c_str()};
    skippy_error * error = nullptr;
    skippy_stage_planner * planner = nullptr;
    if (skippy_stage_planner_create_v1(&config, &planner, &error) != SKIPPY_STATUS_OK || planner == nullptr) {
        std::fprintf(stderr, "public stage planner creation failed: %s\n", error == nullptr ? "" : error->message);
        skippy_error_free(error);
        return false;
    }
    std::unique_ptr<skippy_stage_planner, decltype(&skippy_stage_planner_free)> planner_owner(
            planner, skippy_stage_planner_free);
    skippy_stage_plan * source = nullptr;
    skippy_stage_plan * final = nullptr;
    for (int32_t cut = 1; cut < layer_count; ++cut) {
        skippy_stage_plan * cut_source = nullptr;
        skippy_stage_plan * cut_final = nullptr;
        if (skippy_stage_planner_realize_v1(planner, 0, cut, &cut_source, &error) != SKIPPY_STATUS_OK ||
                skippy_stage_planner_realize_v1(planner, cut, layer_count, &cut_final, &error) != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "public stage realization failed at cut %d: %s\n",
                    cut, error == nullptr ? "" : error->message);
            skippy_error_free(error);
            skippy_stage_plan_free(cut_source);
            skippy_stage_plan_free(cut_final);
            return false;
        }
        const skippy_stage_plan * cut_chain[] = {cut_source, cut_final};
        if (skippy_stage_plan_validate_chain_v1(cut_chain, 2, &error) != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "public stage chain failed at cut %d: %s\n",
                    cut, error == nullptr ? "" : error->message);
            skippy_error_free(error);
            skippy_stage_plan_free(cut_source);
            skippy_stage_plan_free(cut_final);
            return false;
        }
        if (cut == 1) {
            source = cut_source;
            final = cut_final;
        } else {
            skippy_stage_plan_free(cut_source);
            skippy_stage_plan_free(cut_final);
        }
    }
    if (layer_count >= 3) {
        skippy_stage_plan * three_source = nullptr;
        skippy_stage_plan * three_middle = nullptr;
        skippy_stage_plan * three_final = nullptr;
        if (skippy_stage_planner_realize_v1(planner, 0, 1, &three_source, &error) != SKIPPY_STATUS_OK ||
                skippy_stage_planner_realize_v1(planner, 1, 2, &three_middle, &error) != SKIPPY_STATUS_OK ||
                skippy_stage_planner_realize_v1(planner, 2, layer_count, &three_final, &error) != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "public three-stage realization failed: %s\n",
                    error == nullptr ? "" : error->message);
            skippy_error_free(error);
            skippy_stage_plan_free(three_source);
            skippy_stage_plan_free(three_middle);
            skippy_stage_plan_free(three_final);
            return false;
        }
        const skippy_stage_plan * three_chain[] = {three_source, three_middle, three_final};
        const skippy_status three_status = skippy_stage_plan_validate_chain_v1(three_chain, 3, &error);
        skippy_stage_plan_free(three_source);
        skippy_stage_plan_free(three_middle);
        skippy_stage_plan_free(three_final);
        if (three_status != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "public three-stage chain failed: %s\n",
                    error == nullptr ? "" : error->message);
            skippy_error_free(error);
            return false;
        }
    }
    std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> source_owner(source, skippy_stage_plan_free);
    std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> final_owner(final, skippy_stage_plan_free);
    const skippy_stage_plan * chain[] = {source, final};
    if (skippy_stage_plan_validate_chain_v1(chain, 2, &error) != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "public stage chain failed: %s\n", error == nullptr ? "" : error->message);
        skippy_error_free(error);
        return false;
    }
    skippy_stage_plan_desc_v1 descriptor{};
    skippy_stage_plan_desc_v1 final_descriptor{};
    skippy_stage_plan_profile_desc_v1 profile{};
    skippy_stage_plan_value_desc_v1 resident{};
    if (skippy_stage_plan_describe_v1(source, &descriptor, &error) != SKIPPY_STATUS_OK ||
        skippy_stage_plan_describe_v1(final, &final_descriptor, &error) != SKIPPY_STATUS_OK ||
        descriptor.abi_version != 1 || descriptor.profile_count != profiles.size() * 2 ||
        descriptor.resident_tensor_count == 0 || plan_string(source, descriptor.package_id) != package_id ||
        plan_string(source, descriptor.plan_id).compare(0, 15, "skippy-plan:v1:") != 0 ||
        skippy_stage_plan_profile_at_v1(source, 0, &profile, &error) != SKIPPY_STATUS_OK ||
        plan_string(source, profile.profile_id) != "batched" ||
        plan_string(source, profile.graph_configuration_id) != graph_configuration_id ||
        plan_string(source, profile.backend_id) != backend_id ||
        skippy_stage_plan_resident_tensor_at_v1(source, 0, &resident, &error) != SKIPPY_STATUS_OK ||
        plan_string(source, resident.identity).compare(0, 15, "package.tensor.") != 0) {
        std::fprintf(stderr, "public stage descriptors are incomplete\n");
        skippy_error_free(error);
        return false;
    }
    skippy_stage_plan_profile_desc_v1 final_profile{};
    if (skippy_stage_plan_profile_at_v1(final, 0, &final_profile, &error) != SKIPPY_STATUS_OK ||
            profile.activation_export_count == 0 ||
            profile.activation_export_count != final_profile.activation_import_count) {
        std::fprintf(stderr, "public stage frontier descriptors are incomplete\n");
        skippy_error_free(error);
        return false;
    }
    std::vector<std::pair<std::string, std::string>> stable_frontier;
    for (size_t index = 0; index < profile.activation_export_count; ++index) {
        skippy_stage_plan_value_desc_v1 exported{};
        skippy_stage_plan_value_desc_v1 imported{};
        if (skippy_stage_plan_value_at_v1(source, 0, SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT,
                    index, &exported, &error) != SKIPPY_STATUS_OK ||
                skippy_stage_plan_value_at_v1(final, 0, SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_IMPORT,
                    index, &imported, &error) != SKIPPY_STATUS_OK ||
                plan_string(source, exported.identity) != plan_string(final, imported.identity) ||
                plan_string(source, exported.identity).compare(0, 15, "skippy-port:v1:") != 0 ||
                plan_string(source, exported.binding).empty() ||
                plan_string(source, exported.binding) != plan_string(final, imported.binding)) {
            std::fprintf(stderr, "public stage frontier bindings are incomplete\n");
            skippy_error_free(error);
            return false;
        }
        stable_frontier.emplace_back(
                plan_string(source, exported.identity), plan_string(source, exported.binding));
    }
    if (runtime_stage != nullptr) {
        *runtime_stage = {};
        runtime_stage->execution_contract = plan_string(final, final_descriptor.execution_contract);
        runtime_stage->source_execution_contract = plan_string(source, descriptor.execution_contract);
        skippy_stage_plan * full = nullptr;
        if (skippy_stage_planner_realize_v1(planner, 0, layer_count, &full, &error) != SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            return false;
        }
        std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> full_owner(full, skippy_stage_plan_free);
        skippy_stage_plan_desc_v1 full_descriptor{};
        if (skippy_stage_plan_describe_v1(full, &full_descriptor, &error) != SKIPPY_STATUS_OK) {
            skippy_error_free(error);
            return false;
        }
        runtime_stage->full_execution_contract = plan_string(full, full_descriptor.execution_contract);
        runtime_stage->layer_start = 1;
        runtime_stage->layer_end = layer_count;
        for (size_t index = 0; index < full_descriptor.resident_tensor_count; ++index) {
            skippy_stage_plan_value_desc_v1 resident_value{};
            if (skippy_stage_plan_resident_tensor_at_v1(full, index, &resident_value, &error) !=
                    SKIPPY_STATUS_OK) {
                skippy_error_free(error);
                return false;
            }
            const std::string tensor_id = plan_string(full, resident_value.identity);
            const auto found = std::find_if(
                    inventory.tensors.begin(), inventory.tensors.end(), [&](const auto & candidate) {
                        return candidate.tensor_id == tensor_id;
                    });
            if (found == inventory.tensors.end()) {
                std::fprintf(stderr, "public full-stage resident identity is absent from the package inventory\n");
                return false;
            }
            runtime_stage->full_resident_names.push_back(found->native_name);
        }
        for (size_t index = 0; index < descriptor.resident_tensor_count; ++index) {
            skippy_stage_plan_value_desc_v1 resident_value{};
            if (skippy_stage_plan_resident_tensor_at_v1(source, index, &resident_value, &error) !=
                    SKIPPY_STATUS_OK) {
                skippy_error_free(error);
                return false;
            }
            const std::string tensor_id = plan_string(source, resident_value.identity);
            const auto found = std::find_if(
                    inventory.tensors.begin(), inventory.tensors.end(), [&](const auto & candidate) {
                        return candidate.tensor_id == tensor_id;
                    });
            if (found == inventory.tensors.end()) {
                std::fprintf(stderr, "public source-stage resident identity is absent from the package inventory\n");
                return false;
            }
            runtime_stage->source_resident_names.push_back(found->native_name);
        }
        for (size_t index = 0; index < final_descriptor.resident_tensor_count; ++index) {
            skippy_stage_plan_value_desc_v1 resident_value{};
            if (skippy_stage_plan_resident_tensor_at_v1(final, index, &resident_value, &error) !=
                    SKIPPY_STATUS_OK) {
                skippy_error_free(error);
                return false;
            }
            const std::string tensor_id = plan_string(final, resident_value.identity);
            const auto found = std::find_if(
                    inventory.tensors.begin(), inventory.tensors.end(), [&](const auto & candidate) {
                        return candidate.tensor_id == tensor_id;
                    });
            if (found == inventory.tensors.end()) {
                std::fprintf(stderr, "public stage resident identity is absent from the package inventory\n");
                return false;
            }
            runtime_stage->resident_names.push_back(found->native_name);
        }
        for (size_t index = 0; index < final_profile.activation_import_count; ++index) {
            skippy_stage_plan_value_desc_v1 imported{};
            skippy_stage_plan_value_desc_v1 exported{};
            if (skippy_stage_plan_value_at_v1(final, 0, SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_IMPORT,
                        index, &imported, &error) != SKIPPY_STATUS_OK ||
                    skippy_stage_plan_value_at_v1(source, 0, SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT,
                        index, &exported, &error) != SKIPPY_STATUS_OK) {
                skippy_error_free(error);
                return false;
            }
            runtime_stage->import_identities.push_back(plan_string(final, imported.identity));
            runtime_stage->import_bindings.push_back(plan_string(final, imported.binding));
            runtime_stage->export_identities.push_back(plan_string(source, exported.identity));
            runtime_stage->export_bindings.push_back(plan_string(source, exported.binding));
        }
    }
    std::vector<std::string> expected_profiles;
    for (const auto & requested : profiles) {
        expected_profiles.emplace_back(requested.profile_id);
        expected_profiles.push_back("embeddings:" + std::string(requested.profile_id));
    }
    std::sort(expected_profiles.begin(), expected_profiles.end());
    for (size_t index = 0; index < descriptor.profile_count; ++index) {
        if (skippy_stage_plan_profile_at_v1(source, index, &profile, &error) != SKIPPY_STATUS_OK ||
            plan_string(source, profile.profile_id) != expected_profiles[index] ||
            plan_string(source, profile.graph_configuration_id) != graph_configuration_id ||
            plan_string(source, profile.backend_id) != backend_id) {
            skippy_error_free(error);
            return false;
        }
        if (profile.activation_export_count != stable_frontier.size()) {
            std::fprintf(stderr, "public stage frontier count varies across execution profiles\n");
            return false;
        }
        for (size_t frontier_index = 0; frontier_index < stable_frontier.size(); ++frontier_index) {
            skippy_stage_plan_value_desc_v1 exported{};
            if (skippy_stage_plan_value_at_v1(source, index,
                        SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT, frontier_index,
                        &exported, &error) != SKIPPY_STATUS_OK ||
                    plan_string(source, exported.identity) != stable_frontier[frontier_index].first ||
                    plan_string(source, exported.binding) != stable_frontier[frontier_index].second) {
                std::fprintf(stderr, "public stage frontier identity varies across execution profiles\n");
                skippy_error_free(error);
                return false;
            }
        }
    }
    skippy_stage_plan * repeated = nullptr;
    if (skippy_stage_planner_realize_v1(planner, 0, 1, &repeated, &error) != SKIPPY_STATUS_OK) {
        skippy_error_free(error);
        return false;
    }
    std::unique_ptr<skippy_stage_plan, decltype(&skippy_stage_plan_free)> repeated_owner(
            repeated, skippy_stage_plan_free);
    skippy_stage_plan_desc_v1 repeated_descriptor{};
    if (skippy_stage_plan_describe_v1(repeated, &repeated_descriptor, &error) != SKIPPY_STATUS_OK ||
            plan_string(source, descriptor.plan_id) != plan_string(repeated, repeated_descriptor.plan_id)) {
        skippy_error_free(error);
        return false;
    }
    const std::string original_plan_id = plan_string(source, descriptor.plan_id);
    const auto identity_plan_id = [&](const skippy_stage_planner_config_v1 & identity_config) {
        skippy_stage_planner * identity_planner = nullptr;
        skippy_stage_plan * identity_plan = nullptr;
        skippy_stage_plan_desc_v1 identity_descriptor{};
        skippy_error * identity_error = nullptr;
        if (skippy_stage_planner_create_v1(&identity_config, &identity_planner, &identity_error) != SKIPPY_STATUS_OK ||
                identity_planner == nullptr ||
                skippy_stage_planner_realize_v1(identity_planner, 0, 1, &identity_plan, &identity_error) !=
                        SKIPPY_STATUS_OK ||
                identity_plan == nullptr ||
                skippy_stage_plan_describe_v1(identity_plan, &identity_descriptor, &identity_error) !=
                        SKIPPY_STATUS_OK) {
            skippy_error_free(identity_error);
            skippy_stage_plan_free(identity_plan);
            skippy_stage_planner_free(identity_planner);
            return std::string{};
        }
        const std::string identity = plan_string(identity_plan, identity_descriptor.plan_id);
        skippy_stage_plan_free(identity_plan);
        skippy_stage_planner_free(identity_planner);
        return identity;
    };
    auto changed_identity = config;
    changed_identity.graph_configuration_id = "skippy.test.graph.v2";
    const std::string changed_graph_plan_id = identity_plan_id(changed_identity);
    changed_identity = config;
    changed_identity.backend_id = "skippy.test.other-backend";
    const std::string changed_backend_plan_id = identity_plan_id(changed_identity);
    if (changed_graph_plan_id.empty() || changed_backend_plan_id.empty() ||
            changed_graph_plan_id == original_plan_id || changed_backend_plan_id == original_plan_id ||
            changed_graph_plan_id == changed_backend_plan_id) {
        return false;
    }
    skippy_stage_plan * partial = reinterpret_cast<skippy_stage_plan *>(1);
    if (skippy_stage_planner_realize_v1(planner, -1, 1, &partial, &error) == SKIPPY_STATUS_OK || partial != nullptr) {
        skippy_error_free(error);
        return false;
    }
    skippy_error_free(error);
    error = nullptr;
    const skippy_stage_plan * invalid_chain[] = {source, source};
    if (skippy_stage_plan_validate_chain_v1(invalid_chain, 2, &error) == SKIPPY_STATUS_OK) {
        skippy_error_free(error);
        return false;
    }
    skippy_error_free(error);
    error = nullptr;
    auto malformed = config;
    malformed.abi_version = 0;
    skippy_stage_planner * partial_planner = reinterpret_cast<skippy_stage_planner *>(1);
    if (skippy_stage_planner_create_v1(&malformed, &partial_planner, &error) == SKIPPY_STATUS_OK ||
            partial_planner != nullptr) {
        skippy_error_free(error);
        return false;
    }
    skippy_error_free(error);
    error = nullptr;
    malformed = config;
    malformed.struct_size = offsetof(skippy_stage_planner_config_v1, graph_configuration_id);
    partial_planner = nullptr;
    if (skippy_stage_planner_create_v1(&malformed, &partial_planner, &error) == SKIPPY_STATUS_OK ||
            partial_planner != nullptr) {
        skippy_error_free(error);
        return false;
    }
    skippy_error_free(error);
    error = nullptr;
    malformed = config;
    malformed.graph_configuration_id = "";
    partial_planner = nullptr;
    if (skippy_stage_planner_create_v1(&malformed, &partial_planner, &error) == SKIPPY_STATUS_OK ||
            partial_planner != nullptr) {
        skippy_error_free(error);
        return false;
    }
    skippy_error_free(error);
    error = nullptr;
    malformed = config;
    malformed.backend_id = nullptr;
    partial_planner = nullptr;
    if (skippy_stage_planner_create_v1(&malformed, &partial_planner, &error) == SKIPPY_STATUS_OK ||
            partial_planner != nullptr) {
        skippy_error_free(error);
        return false;
    }
    skippy_error_free(error);
    error = nullptr;
    if (shard_paths.size() > 1) {
        auto reversed_paths = shard_paths;
        std::reverse(reversed_paths.begin(), reversed_paths.end());
        malformed = config;
        malformed.shard_paths = reversed_paths.data();
        partial_planner = nullptr;
        if (skippy_stage_planner_create_v1(&malformed, &partial_planner, &error) == SKIPPY_STATUS_OK ||
                partial_planner != nullptr) {
            skippy_error_free(error);
            return false;
        }
        skippy_error_free(error);
    }
    return true;
}

int run_proof(const char * source_path, const char * output_directory, bool expect_recurrent) {
    std::string error;
    std::vector<std::string> generated_paths;
    if (!write_two_renamed_splits(source_path, output_directory, generated_paths, error)) {
        std::fprintf(stderr, "renamed split generation failed: %s\n", error.c_str());
        return 1;
    }
    const std::vector<std::string> reversed = {generated_paths[1], generated_paths[0]};
    package_inventory inventory;
    if (!inspect_package_splits(reversed, inventory, error) ||
            inventory.ordered_paths != generated_paths) {
        std::fprintf(stderr, "split inspection/order failed: %s\n", error.c_str());
        return 2;
    }
    package_inventory rejected;
    if (inspect_package_splits({generated_paths[0]}, rejected, error) ||
            inspect_package_splits({generated_paths[0], generated_paths[0]}, rejected, error) ||
            inspect_package_splits(
                {generated_paths[0], generated_paths[1], generated_paths[1]}, rejected, error)) {
        std::fprintf(stderr, "invalid split inventory was accepted\n");
        return 3;
    }
    std::vector<const char *> load_paths;
    for (const auto & path : inventory.ordered_paths) {
        load_paths.push_back(path.c_str());
    }
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.use_extra_bufts = false;
    model_params.no_alloc = true;
    llama_file_read_accounting accounting;
    llama_file_read_accounting_begin(&accounting);
    llama_model * model_raw = llama_model_load_from_splits(
            load_paths.data(),
            load_paths.size(),
            model_params);
    llama_file_read_accounting_end(&accounting);
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(model_raw, llama_model_free);
    if (!model || accounting.calls != 0 || accounting.bytes != 0 || !check_model_buffers(model.get())) {
        std::fprintf(stderr, "metadata-only split load read or allocated weight payloads\n");
        return 4;
    }
    std::vector<skippy_parameter_binding> bindings;
    if (!skippy_bind_package_tensors(model.get(), inventory.tensors, bindings, error)) {
        std::fprintf(stderr, "exact package tensor binding failed: %s\n", error.c_str());
        return 5;
    }
    if (!rejects_invalid_bindings(model.get(), inventory.tensors)) {
        std::fprintf(stderr, "invalid package tensor binding was accepted\n");
        return 6;
    }
    if (!accepts_source_shape_after_native_reshape(model.get(), inventory.tensors)) {
        std::fprintf(stderr, "physical/logical package tensor shape regression failed\n");
        return 6;
    }
    std::vector<std::string> prefill;
    std::vector<std::string> batched;
    std::vector<std::string> decode;
    if (!build_profile(model.get(), bindings, expect_recurrent, "prefill", 8, 1, 8, prefill, error) ||
            !build_profile(model.get(), bindings, expect_recurrent, "batched", 8, 2, 8, batched, error) ||
            !build_profile(model.get(), bindings, expect_recurrent, "decode", 1, 1, 1, decode, error)) {
        std::fprintf(stderr, "renamed multishard graph proof failed: %s\n", error.c_str());
        return 7;
    }
    if (expect_recurrent) {
        // Rollback lane: n_rs_seq > 0 routes delta-net conv state writes through
        // the multi-snapshot loop; ownership must hold there too (qwen35 and the
        // other rollback-allowlisted delta-net archs do not clamp n_rs_seq).
        std::vector<std::string> rollback;
        if (!build_profile(model.get(), bindings, expect_recurrent, "rollback", 8, 1, 8, rollback, error, 2)) {
            std::fprintf(stderr, "renamed multishard rollback proof failed: %s\n", error.c_str());
            return 9;
        }
    }
    if (prefill != batched || prefill != decode) {
        std::fprintf(stderr, "package-ID dependency closure differs across graph profiles\n");
        return 8;
    }
    if (!check_public_stage_plan_abi(inventory, inventory.ordered_paths, llama_model_n_layer(model.get()))) {
        std::fprintf(stderr, "public stage-plan ABI proof failed\n");
        return 9;
    }
    const std::string carrier_path = std::string(output_directory) + "/metadata-only.gguf";
    std::vector<std::string> reversed_paths = inventory.ordered_paths;
    std::reverse(reversed_paths.begin(), reversed_paths.end());
    if (write_metadata_carrier(
            reversed_paths,
            std::string(output_directory) + "/metadata-only-reversed.gguf",
            error)) {
        std::fprintf(stderr, "metadata carrier accepted out-of-order GGUF splits\n");
        return 10;
    }
    error.clear();
    package_inventory carrier;
    runtime_stage_plan terminal_stage;
    if (!write_metadata_carrier(inventory.ordered_paths, carrier_path, error) ||
            !inspect_metadata_carrier(carrier_path, inventory, inventory.ordered_paths.size(), carrier, error) ||
            !check_public_stage_plan_abi(
                carrier,
                carrier.ordered_paths,
                llama_model_n_layer(model.get()),
                &terminal_stage)) {
        std::fprintf(stderr, "metadata carrier stage-plan ABI proof failed: %s\n", error.c_str());
        return 10;
    }

    std::vector<const char *> resident_name_ptrs;
    for (const auto & name : terminal_stage.full_resident_names) {
        resident_name_ptrs.push_back(name.c_str());
    }
    std::vector<std::string> carrier_parts = {carrier_path};
    carrier_parts.insert(carrier_parts.end(), inventory.ordered_paths.begin(), inventory.ordered_paths.end());
    std::vector<const char *> carrier_part_ptrs;
    for (const auto & path : carrier_parts) {
        carrier_part_ptrs.push_back(path.c_str());
    }
    skippy_runtime_config runtime_config = skippy_runtime_config_default();
    runtime_config.layer_start = 0;
    runtime_config.layer_end = llama_model_n_layer(model.get());
    runtime_config.ctx_size = 32;
    runtime_config.lane_count = 1;
    runtime_config.n_batch = 8;
    runtime_config.n_ubatch = 8;
    runtime_config.n_threads = 1;
    runtime_config.n_threads_batch = 1;
    runtime_config.n_gpu_layers = 0;
    runtime_config.flash_attn_type = expect_recurrent
            ? LLAMA_FLASH_ATTN_TYPE_DISABLED
            : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    runtime_config.load_mode = SKIPPY_LOAD_MODE_RUNTIME_SLICE;
    runtime_config.execution_contract = terminal_stage.full_execution_contract.c_str();
    runtime_config.resident_tensor_names = resident_name_ptrs.data();
    runtime_config.resident_tensor_name_count = resident_name_ptrs.size();
    skippy_model * carrier_model = nullptr;
    skippy_error * runtime_error = nullptr;
    const skippy_status runtime_status = skippy_model_open_from_parts(
            carrier_part_ptrs.data(), carrier_part_ptrs.size(), &runtime_config, &carrier_model, &runtime_error);
    if (runtime_status != SKIPPY_STATUS_OK || carrier_model == nullptr) {
        std::fprintf(
                stderr,
                "metadata carrier runtime load failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 11;
    }
    skippy_model_free(carrier_model, &runtime_error);
    if (runtime_error != nullptr) {
        std::fprintf(stderr, "metadata carrier runtime free failed: %s\n", runtime_error->message);
        skippy_error_free(runtime_error);
        return 12;
    }

    const std::string selected_payload_path = std::string(output_directory) + "/terminal-stage.gguf";
    if (!write_selected_payload(inventory, terminal_stage.resident_names, selected_payload_path, error)) {
        std::fprintf(stderr, "terminal stage payload write failed: %s\n", error.c_str());
        return 13;
    }
    std::vector<const char *> terminal_resident_ptrs;
    for (const auto & name : terminal_stage.resident_names) {
        terminal_resident_ptrs.push_back(name.c_str());
    }
    std::vector<const char *> terminal_import_identity_ptrs;
    for (const auto & identity : terminal_stage.import_identities) {
        terminal_import_identity_ptrs.push_back(identity.c_str());
    }
    std::vector<const char *> terminal_import_binding_ptrs;
    for (const auto & binding : terminal_stage.import_bindings) {
        terminal_import_binding_ptrs.push_back(binding.c_str());
    }
    const std::vector<const char *> terminal_part_ptrs = {
        carrier_path.c_str(),
        selected_payload_path.c_str(),
    };
    runtime_config.stage_index = 1;
    runtime_config.layer_start = terminal_stage.layer_start;
    runtime_config.layer_end = terminal_stage.layer_end;
    runtime_config.execution_contract = terminal_stage.execution_contract.c_str();
    runtime_config.resident_tensor_names = terminal_resident_ptrs.data();
    runtime_config.resident_tensor_name_count = terminal_resident_ptrs.size();
    runtime_config.activation_import_identities = terminal_import_identity_ptrs.data();
    runtime_config.activation_import_identity_count = terminal_import_identity_ptrs.size();
    runtime_config.activation_import_bindings = terminal_import_binding_ptrs.data();
    if (expect_recurrent) {
        runtime_config.n_batch = 512;
        runtime_config.n_ubatch = 128;
    }
    skippy_model * terminal_model = nullptr;
    runtime_error = nullptr;
    const skippy_status terminal_status = skippy_model_open_from_parts(
            terminal_part_ptrs.data(), terminal_part_ptrs.size(), &runtime_config, &terminal_model, &runtime_error);
    if (terminal_status != SKIPPY_STATUS_OK || terminal_model == nullptr) {
        std::fprintf(
                stderr,
                "terminal stage runtime load failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 14;
    }
    live_graph_reference terminal_reference;
    if (!terminal_reference.open(terminal_part_ptrs, runtime_config)) return 43;
    terminal_reference.clear_with(terminal_model);
    skippy_session * terminal_session = nullptr;
    if (skippy_session_create(terminal_model, &terminal_session, &runtime_error) != SKIPPY_STATUS_OK ||
            terminal_session == nullptr) {
        std::fprintf(stderr, "terminal stage session create failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 15;
    }

    const std::string source_payload_path = std::string(output_directory) + "/source-stage.gguf";
    if (!write_selected_payload(inventory, terminal_stage.source_resident_names, source_payload_path, error)) {
        std::fprintf(stderr, "source stage payload write failed: %s\n", error.c_str());
        return 16;
    }
    std::vector<const char *> source_resident_ptrs;
    for (const auto & name : terminal_stage.source_resident_names) {
        source_resident_ptrs.push_back(name.c_str());
    }
    std::vector<const char *> source_export_identity_ptrs;
    for (const auto & identity : terminal_stage.export_identities) {
        source_export_identity_ptrs.push_back(identity.c_str());
    }
    std::vector<const char *> source_export_binding_ptrs;
    for (const auto & binding : terminal_stage.export_bindings) {
        source_export_binding_ptrs.push_back(binding.c_str());
    }
    const std::vector<const char *> source_part_ptrs = {
        carrier_path.c_str(),
        source_payload_path.c_str(),
    };
    runtime_config.stage_index = 0;
    runtime_config.layer_start = 0;
    runtime_config.layer_end = 1;
    runtime_config.execution_contract = terminal_stage.source_execution_contract.c_str();
    runtime_config.resident_tensor_names = source_resident_ptrs.data();
    runtime_config.resident_tensor_name_count = source_resident_ptrs.size();
    runtime_config.activation_import_identities = nullptr;
    runtime_config.activation_import_identity_count = 0;
    runtime_config.activation_import_bindings = nullptr;
    runtime_config.activation_export_identities = source_export_identity_ptrs.data();
    runtime_config.activation_export_identity_count = source_export_identity_ptrs.size();
    runtime_config.activation_export_bindings = source_export_binding_ptrs.data();
    carrier_model = nullptr;
    runtime_error = nullptr;
    const skippy_status source_status = skippy_model_open_from_parts(
            source_part_ptrs.data(), source_part_ptrs.size(), &runtime_config, &carrier_model, &runtime_error);
    if (source_status != SKIPPY_STATUS_OK || carrier_model == nullptr) {
        std::fprintf(stderr, "source stage runtime load failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 17;
    }
    if (!check_synthetic_embedding_input(carrier_model)) {
        return 40;
    }
    live_graph_reference source_reference;
    if (!source_reference.open(source_part_ptrs, runtime_config)) return 44;
    source_reference.clear_with(carrier_model);
    skippy_session * session = nullptr;
    if (skippy_session_create(carrier_model, &session, &runtime_error) != SKIPPY_STATUS_OK || session == nullptr) {
        std::fprintf(stderr, "source stage session create failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 18;
    }
    const llama_token tokens[] = {1, 2};
    if (!live_graph_reference::check_capacity_preflight(session, tokens, 2)) return 47;
    std::vector<uint8_t> output_payload(1024 * 1024);
    skippy_activation_desc output_desc{};
    size_t output_bytes = 0;
    try {
        const skippy_status prefill_status = skippy_prefill_chunk_frame(
                session, tokens, 2, nullptr, nullptr, &output_desc, output_payload.data(),
                output_payload.size(), &output_bytes, &runtime_error);
        if (prefill_status != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "source stage prefill failed: %s\n",
                    runtime_error == nullptr ? "" : runtime_error->message);
            skippy_error_free(runtime_error);
            return 19;
        }
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "source stage prefill threw: %s\n", exception.what());
        return 20;
    }
    if (output_desc.part_count == 0 || output_bytes == 0 || output_desc.payload_bytes != output_bytes) {
        std::fprintf(stderr, "source stage prefill returned an empty activation frontier\n");
        return 21;
    }
    skippy_activation_desc terminal_output_desc{};
    std::vector<uint8_t> terminal_output_payload(1024 * 1024);
    size_t terminal_output_bytes = 0;
    if (skippy_prefill_chunk_frame(
                terminal_session, tokens, 2, &output_desc, output_payload.data(),
                &terminal_output_desc, terminal_output_payload.data(), terminal_output_payload.size(),
                &terminal_output_bytes, &runtime_error) != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "terminal stage prefill failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 22;
    }

    if (!source_reference.prefill(tokens, 2) ||
            !terminal_reference.prefill(tokens, 2, &source_reference) ||
            !source_reference.compare(carrier_model, false) ||
            !terminal_reference.compare(terminal_model, false)) return 45;

    skippy_sampling_config sampling{};
    sampling.version = 1;
    sampling.flags = SKIPPY_SAMPLING_FLAG_ENABLED;
    sampling.seed = 1;
    sampling.repeat_penalty = 1.0f;
    llama_token next_token = 3;
    for (int iteration = 0; iteration < 2; ++iteration) {
        skippy_activation_desc step_desc{};
        size_t step_bytes = 0;
        llama_token ignored_prediction = -1;
        if (skippy_decode_step_frame_sampled(
                    session, next_token, nullptr, nullptr, nullptr,
                    &step_desc, output_payload.data(), output_payload.size(), &step_bytes,
                    &ignored_prediction, &runtime_error) != SKIPPY_STATUS_OK) {
            std::fprintf(stderr, "source stage sampled decode setup failed: %s\n",
                    runtime_error == nullptr ? "" : runtime_error->message);
            skippy_error_free(runtime_error);
            return 23;
        }
        llama_token predicted_token = -1;
        terminal_output_bytes = 0;
        if (skippy_decode_step_frame_sampled(
                    terminal_session, next_token, &sampling, &step_desc, output_payload.data(),
                    &terminal_output_desc, terminal_output_payload.data(), terminal_output_payload.size(),
                    &terminal_output_bytes, &predicted_token, &runtime_error) != SKIPPY_STATUS_OK ||
                predicted_token < 0) {
            std::fprintf(stderr, "terminal stage sampled decode failed: %s\n",
                    runtime_error == nullptr ? "" : runtime_error->message);
            skippy_error_free(runtime_error);
            return 24;
        }
        if (!source_reference.step(next_token, nullptr, -1) ||
                !terminal_reference.step(next_token, &sampling, predicted_token, &source_reference) ||
                !source_reference.compare(carrier_model, false) ||
                !terminal_reference.compare(terminal_model, true)) return 46;
        next_token = predicted_token;
    }
    if (!live_graph_reference::check_activation_prefill_sampling(session, tokens, 2)) return 49;
    if (!live_graph_reference::check_reservation_overflow(session)) return 48;
    if (skippy_session_free(session, &runtime_error) != SKIPPY_STATUS_OK || runtime_error != nullptr) {
        std::fprintf(stderr, "source stage session free failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 25;
    }
    if (skippy_model_free(carrier_model, &runtime_error) != SKIPPY_STATUS_OK || runtime_error != nullptr) {
        std::fprintf(stderr, "source stage model free failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 26;
    }
    if (skippy_session_free(terminal_session, &runtime_error) != SKIPPY_STATUS_OK || runtime_error != nullptr) {
        std::fprintf(stderr, "terminal stage session free failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 27;
    }
    if (skippy_model_free(terminal_model, &runtime_error) != SKIPPY_STATUS_OK || runtime_error != nullptr) {
        std::fprintf(stderr, "terminal stage model free failed: %s\n",
                runtime_error == nullptr ? "" : runtime_error->message);
        skippy_error_free(runtime_error);
        return 28;
    }
    std::printf(
            "renamed multishard metadata planning: tensors=%zu dependencies=%zu payload_reads=%zu payload_bytes=%zu recurrent=%d\n",
            bindings.size(),
            prefill.size(),
            accounting.calls,
            accounting.bytes,
            expect_recurrent ? 1 : 0);
    return 0;
}

}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MODEL_PATH OUTPUT_DIRECTORY EXPECT_RECURRENT\n", argv[0]);
        return 64;
    }
    llama_backend_init();
    const int result = run_proof(argv[1], argv[2], std::string(argv[3]) == "1");
    llama_backend_free();
    return result;
}
