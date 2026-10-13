#include "skippy/environment.h"
#include "skippy/glm_dsa.h"
#include "skippy/runtime_types.h"

#include "ggml-backend.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

bool skippy_glm_dsa_op_timing_enabled() {
    return skippy_env_enabled("SKIPPY_GLM_DSA_OP_TIMING");
}

bool skippy_glm_dsa_tensor_trace_enabled() {
    return skippy_env_enabled("SKIPPY_GLM_DSA_TENSOR_TRACE");
}

bool skippy_glm_dsa_tensor_trace_stats_enabled() {
    return skippy_env_enabled("SKIPPY_GLM_DSA_TENSOR_TRACE_STATS");
}

static bool skippy_glm_dsa_direct_sparse_decision_log_enabled() {
    return skippy_env_enabled("SKIPPY_GLM_DSA_LOG_DIRECT_SPARSE_DECISIONS");
}

static bool skippy_glm_dsa_direct_sparse_attn_enabled() {
    if (skippy_env_disabled("SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_ATTN") ||
        skippy_env_enabled("SKIPPY_GLM_DSA_DISABLE_DIRECT_SPARSE_ATTN")) {
        return false;
    }

    return true;
}

static bool skippy_glm_dsa_direct_sparse_prefill_enabled() {
    if (skippy_env_disabled("SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_PREFILL") ||
        skippy_env_enabled("SKIPPY_GLM_DSA_DISABLE_DIRECT_SPARSE_PREFILL")) {
        return false;
    }

    return true;
}

static bool skippy_glm_dsa_unproven_large_direct_sparse_prefill_enabled() {
    return skippy_env_enabled("SKIPPY_GLM_DSA_ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL");
}

static uint32_t skippy_glm_dsa_direct_sparse_prefill_max_tokens() {
    return skippy_env_u32("SKIPPY_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS", 8, 1, 4096);
}

static uint64_t skippy_glm_dsa_dense_sparse_mask_max_bytes() {
    const char * value = std::getenv("SKIPPY_GLM_DSA_DENSE_SPARSE_MASK_MAX_BYTES");
    if (value == nullptr || value[0] == '\0') {
        return 512ULL * 1024ULL * 1024ULL;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
        return UINT64_MAX;
    }
    return parsed;
}

static constexpr int64_t SKIPPY_GLM_DSA_DIRECT_SPARSE_DECODE_DEFAULT_MAX_TOP_K = 256;

static int64_t skippy_glm_dsa_direct_sparse_decode_max_top_k() {
    const char * value = std::getenv("SKIPPY_GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K");
    if (value == nullptr || value[0] == '\0') {
        return SKIPPY_GLM_DSA_DIRECT_SPARSE_DECODE_DEFAULT_MAX_TOP_K;
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return SKIPPY_GLM_DSA_DIRECT_SPARSE_DECODE_DEFAULT_MAX_TOP_K;
    }

    return parsed <= 0 ? INT64_MAX : parsed;
}


static bool skippy_name_starts_with(const char * name, const char * prefix) {
    return name != nullptr && std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

static const char * skippy_glm_dsa_op_name(skippy_glm_dsa_op_kind kind) {
    switch (kind) {
        case SKIPPY_GLM_DSA_OP_INDEXER:
            return "indexer";
        case SKIPPY_GLM_DSA_OP_TOP_K:
            return "top_k";
        case SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL:
            return "sparse_mask_fill";
        case SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK:
            return "sparse_mask_topk";
        case SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD:
            return "sparse_mask_add";
        case SKIPPY_GLM_DSA_OP_DSA_SPARSE_ATTN:
            return "dsa_sparse_attn";
        case SKIPPY_GLM_DSA_OP_COMPACT_GET_ROWS:
            return "compact_get_rows";
        case SKIPPY_GLM_DSA_OP_MLA_ATTENTION:
            return "mla_attention";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE:
            return "routed_moe_route";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE_UP:
            return "routed_moe_gate_up";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE:
            return "routed_moe_gate";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_UP:
            return "routed_moe_up";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_ACT:
            return "routed_moe_act";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_DOWN:
            return "routed_moe_down";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_WEIGHTED:
            return "routed_moe_weighted";
        case SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE:
            return "routed_moe_aggregate";
        case SKIPPY_GLM_DSA_OP_SHARED_EXPERT:
            return "shared_expert";
        case SKIPPY_GLM_DSA_OP_COUNT:
        case SKIPPY_GLM_DSA_OP_UNKNOWN:
            break;
    }
    return "unknown";
}

static skippy_glm_dsa_op_kind skippy_glm_dsa_op_kind_for_tensor(const char * name) {
    if (skippy_name_starts_with(name, "top_k")) {
        return SKIPPY_GLM_DSA_OP_TOP_K;
    }
    if (skippy_name_starts_with(name, "indexer_")) {
        return SKIPPY_GLM_DSA_OP_INDEXER;
    }
    if (skippy_name_starts_with(name, "dsa_sparse_mask_fill")) {
        return SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL;
    }
    if (skippy_name_starts_with(name, "dsa_sparse_mask_topk")) {
        return SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK;
    }
    if (skippy_name_starts_with(name, "dsa_sparse_mask")) {
        return SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD;
    }
    if (skippy_name_starts_with(name, "dsa_sparse_attn")) {
        return SKIPPY_GLM_DSA_OP_DSA_SPARSE_ATTN;
    }
    if (skippy_name_starts_with(name, "dsa_compact_")) {
        return SKIPPY_GLM_DSA_OP_COMPACT_GET_ROWS;
    }
    if (skippy_name_starts_with(name, "kqv_out")) {
        return SKIPPY_GLM_DSA_OP_MLA_ATTENTION;
    }
    if (skippy_name_starts_with(name, "ffn_moe_gate_up")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE_UP;
    }
    if (skippy_name_starts_with(name, "ffn_moe_gate")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE;
    }
    if (skippy_name_starts_with(name, "ffn_moe_up")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_UP;
    }
    if (skippy_name_starts_with(name, "ffn_moe_silu") ||
            skippy_name_starts_with(name, "ffn_moe_swiglu") ||
            skippy_name_starts_with(name, "ffn_moe_geglu") ||
            skippy_name_starts_with(name, "ffn_moe_gelu") ||
            skippy_name_starts_with(name, "ffn_moe_reglu") ||
            skippy_name_starts_with(name, "ffn_moe_relu")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_ACT;
    }
    if (skippy_name_starts_with(name, "ffn_moe_down")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_DOWN;
    }
    if (skippy_name_starts_with(name, "ffn_moe_weighted")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_WEIGHTED;
    }
    if (skippy_name_starts_with(name, "ffn_moe_out")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE;
    }
    if (skippy_name_starts_with(name, "ffn_moe_")) {
        return SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE;
    }
    if (skippy_name_starts_with(name, "ffn_shexp") ||
            skippy_name_starts_with(name, "ffn_up") ||
            skippy_name_starts_with(name, "ffn_gate") ||
            skippy_name_starts_with(name, "ffn_silu") ||
            skippy_name_starts_with(name, "ffn_swiglu") ||
            skippy_name_starts_with(name, "ffn_down")) {
        return SKIPPY_GLM_DSA_OP_SHARED_EXPERT;
    }
    return SKIPPY_GLM_DSA_OP_UNKNOWN;
}

static void skippy_glm_dsa_group_name_for_tensor(
        const char * tensor_name,
        char * group_name,
        size_t group_name_size) {
    if (group_name == nullptr || group_name_size == 0) {
        return;
    }

    group_name[0] = '\0';
    if (tensor_name == nullptr || tensor_name[0] == '\0') {
        std::snprintf(group_name, group_name_size, "global");
        return;
    }

    const char * suffix = std::strrchr(tensor_name, '-');
    if (suffix == nullptr || suffix[1] < '0' || suffix[1] > '9') {
        std::snprintf(group_name, group_name_size, "global");
        return;
    }

    const char * layer = suffix + 1;
    size_t layer_len = 0;
    while (layer[layer_len] >= '0' && layer[layer_len] <= '9') {
        ++layer_len;
    }
    std::snprintf(group_name, group_name_size, "layer_%.*s", static_cast<int>(layer_len), layer);
}

static int skippy_glm_dsa_layer_from_group_name(const char * group_name) {
    const char * prefix = "layer_";
    if (!skippy_name_starts_with(group_name, prefix)) {
        return -1;
    }

    const char * layer = group_name + std::strlen(prefix);
    if (layer[0] == '\0') {
        return -1;
    }

    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(layer, &end, 10);
    if (errno != 0 || end == layer || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return -1;
    }

    return static_cast<int>(parsed);
}


static skippy_glm_dsa_op_group * skippy_glm_dsa_op_timing_group(
        skippy_glm_dsa_op_timing & timing,
        const char * group_name) {
    if (group_name == nullptr || group_name[0] == '\0') {
        return nullptr;
    }

    skippy_glm_dsa_op_group * unused = nullptr;
    for (size_t i = 0; i < SKIPPY_GLM_DSA_TIMING_GROUP_LIMIT; ++i) {
        skippy_glm_dsa_op_group & group = timing.groups[i];
        if (!group.used) {
            if (unused == nullptr) {
                unused = &group;
            }
            continue;
        }
        if (std::strncmp(group.name, group_name, sizeof(group.name)) == 0) {
            return &group;
        }
    }

    if (unused == nullptr) {
        return nullptr;
    }

    unused->used = true;
    std::snprintf(unused->name, sizeof(unused->name), "%s", group_name);
    return unused;
}

static void skippy_glm_dsa_record_group_timing(
        skippy_glm_dsa_op_timing & timing,
        skippy_glm_dsa_op_kind kind,
        const char * tensor_name,
        int64_t elapsed_us) {
    char group_name[SKIPPY_GLM_DSA_TIMING_GROUP_NAME_SIZE] = {};
    skippy_glm_dsa_group_name_for_tensor(tensor_name, group_name, sizeof(group_name));
    skippy_glm_dsa_op_group * group = skippy_glm_dsa_op_timing_group(timing, group_name);
    if (group == nullptr) {
        return;
    }

    group->stats[kind].nodes += 1;
    group->stats[kind].elapsed_us += elapsed_us;
}

static void skippy_glm_dsa_record_hot_tensor(
        skippy_glm_dsa_op_timing & timing,
        skippy_glm_dsa_op_kind kind,
        const ggml_tensor * tensor,
        int64_t elapsed_us) {
    if (tensor == nullptr || elapsed_us <= 0) {
        return;
    }

    size_t slot = SKIPPY_GLM_DSA_HOT_TENSOR_LIMIT;
    for (size_t i = 0; i < SKIPPY_GLM_DSA_HOT_TENSOR_LIMIT; ++i) {
        if (!timing.hot_tensors[i].used || elapsed_us > timing.hot_tensors[i].elapsed_us) {
            slot = i;
            break;
        }
    }
    if (slot == SKIPPY_GLM_DSA_HOT_TENSOR_LIMIT) {
        return;
    }

    for (size_t i = SKIPPY_GLM_DSA_HOT_TENSOR_LIMIT - 1; i > slot; --i) {
        timing.hot_tensors[i] = timing.hot_tensors[i - 1];
    }

    skippy_glm_dsa_hot_tensor & record = timing.hot_tensors[slot];
    record.used = true;
    record.kind = kind;
    record.op = tensor->op;
    record.elapsed_us = elapsed_us;
    for (int i = 0; i < 4; ++i) {
        record.ne[i] = tensor->ne[i];
    }
    std::snprintf(record.name, sizeof(record.name), "%s", tensor->name);
}

static bool skippy_glm_dsa_tensor_has_buffer(const ggml_tensor * tensor) {
    return tensor != nullptr &&
           tensor->data != nullptr &&
           (tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer) != nullptr;
}

static bool skippy_glm_dsa_trace_filter_matches(const std::string & filter, const char * name) {
    if (filter.empty()) {
        return true;
    }
    if (name == nullptr || name[0] == '\0') {
        return false;
    }

    size_t start = 0;
    while (start <= filter.size()) {
        const size_t end = filter.find(',', start);
        const size_t len = (end == std::string::npos ? filter.size() : end) - start;
        if (len > 0 && std::string(name).find(filter.substr(start, len)) != std::string::npos) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return false;
}

template<typename T>
static bool skippy_glm_dsa_get_tensor_prefix(
        const ggml_tensor * tensor,
        std::vector<T> & values,
        size_t value_count) {
    if (tensor->nb[0] != sizeof(T)) {
        return false;
    }

    values.resize(value_count);
    size_t copied = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3] && copied < value_count; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2] && copied < value_count; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1] && copied < value_count; ++i1) {
                const size_t row_values = std::min<size_t>(
                        static_cast<size_t>(tensor->ne[0]),
                        value_count - copied);
                const size_t offset =
                        static_cast<size_t>(i1) * tensor->nb[1] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(i3) * tensor->nb[3];
                ggml_backend_tensor_get(
                        tensor,
                        values.data() + copied,
                        offset,
                        row_values * sizeof(T));
                copied += row_values;
            }
        }
    }
    return copied == value_count;
}

static void skippy_glm_dsa_trace_tensor_values(std::string & line, const ggml_tensor * tensor, uint32_t limit) {
    const size_t value_count = static_cast<size_t>(
            std::min<int64_t>(static_cast<int64_t>(limit), std::max<int64_t>(0, ggml_nelements(tensor))));

    line += " values=[";
    if (value_count == 0) {
        line += "]";
        return;
    }

    if (!skippy_glm_dsa_tensor_has_buffer(tensor)) {
        line += "unallocated]";
        return;
    }

    switch (tensor->type) {
        case GGML_TYPE_F32: {
            std::vector<float> values;
            if (!skippy_glm_dsa_get_tensor_prefix(tensor, values, value_count)) {
                line += "unsupported-stride";
                break;
            }
            for (size_t i = 0; i < value_count; ++i) {
                line += format("%s%.9g", i == 0 ? "" : ",", static_cast<double>(values[i]));
            }
            break;
        }
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: {
            const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
            if (traits == nullptr || traits->to_float == nullptr) {
                line += format("unsupported:%s", ggml_type_name(tensor->type));
                break;
            }
            std::vector<uint16_t> raw;
            std::vector<float> values(value_count);
            if (!skippy_glm_dsa_get_tensor_prefix(tensor, raw, value_count)) {
                line += "unsupported-stride";
                break;
            }
            traits->to_float(raw.data(), values.data(), value_count);
            for (size_t i = 0; i < value_count; ++i) {
                line += format("%s%.9g", i == 0 ? "" : ",", static_cast<double>(values[i]));
            }
            break;
        }
        case GGML_TYPE_I32: {
            std::vector<int32_t> values;
            if (!skippy_glm_dsa_get_tensor_prefix(tensor, values, value_count)) {
                line += "unsupported-stride";
                break;
            }
            for (size_t i = 0; i < value_count; ++i) {
                line += format("%s%d", i == 0 ? "" : ",", values[i]);
            }
            break;
        }
        default:
            line += format("unsupported:%s", ggml_type_name(tensor->type));
            break;
    }

    line += "]";
}

static uint64_t skippy_glm_dsa_fnv1a64(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

static void skippy_glm_dsa_trace_numeric_stats_from_f32(std::string & line, const float * values, size_t count) {
    double sum = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(values[i]);
        sum += value;
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const double mean = count == 0 ? 0.0 : sum / static_cast<double>(count);
    line += format(
            " count=%zu sum=%.17g mean=%.17g max_abs=%.17g",
            count,
            sum,
            mean,
            max_abs);
}

static void skippy_glm_dsa_trace_tensor_stats(std::string & line, const ggml_tensor * tensor, uint32_t max_bytes) {
    if (!skippy_glm_dsa_tensor_has_buffer(tensor)) {
        line += " stats=unallocated";
        return;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    if (nbytes > static_cast<size_t>(max_bytes)) {
        line += format(" stats=skipped:nbytes>%u", max_bytes);
        return;
    }

    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(tensor, raw.data(), 0, nbytes);
    const uint64_t hash = skippy_glm_dsa_fnv1a64(raw.data(), raw.size());
    line += format(" stats=fnv64:%016llx", static_cast<unsigned long long>(hash));

    const size_t count = static_cast<size_t>(std::max<int64_t>(0, ggml_nelements(tensor)));
    switch (tensor->type) {
        case GGML_TYPE_F32:
            skippy_glm_dsa_trace_numeric_stats_from_f32(line, reinterpret_cast<const float *>(raw.data()), count);
            break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: {
            const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
            if (traits == nullptr || traits->to_float == nullptr) {
                break;
            }
            std::vector<float> values(count);
            traits->to_float(raw.data(), values.data(), count);
            skippy_glm_dsa_trace_numeric_stats_from_f32(line, values.data(), count);
            break;
        }
        case GGML_TYPE_I32: {
            double sum = 0.0;
            double max_abs = 0.0;
            const int32_t * values = reinterpret_cast<const int32_t *>(raw.data());
            for (size_t i = 0; i < count; ++i) {
                const double value = static_cast<double>(values[i]);
                sum += value;
                max_abs = std::max(max_abs, std::fabs(value));
            }
            const double mean = count == 0 ? 0.0 : sum / static_cast<double>(count);
            line += format(
                    " count=%zu sum=%.17g mean=%.17g max_abs=%.17g",
                    count,
                    sum,
                    mean,
                    max_abs);
            break;
        }
        default:
            break;
    }
}

static void skippy_glm_dsa_dump_trace_tensor(
        const skippy_glm_dsa_op_timing & timing,
        const ggml_tensor * tensor) {
    const char * dump_dir = std::getenv("SKIPPY_GLM_DSA_TENSOR_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0' || timing.token_count != 1 ||
            !skippy_glm_dsa_tensor_has_buffer(tensor)) {
        return;
    }

    std::string filename = tensor->name;
    for (char & c : filename) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            c = '_';
        }
    }
    filename += format(
            "_%lld_%lld_%lld_%lld_%s.bin",
            static_cast<long long>(tensor->ne[0]),
            static_cast<long long>(tensor->ne[1]),
            static_cast<long long>(tensor->ne[2]),
            static_cast<long long>(tensor->ne[3]),
            ggml_type_name(tensor->type));

    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(tensor, raw.data(), 0, nbytes);
    std::ofstream output(std::string(dump_dir) + "/" + filename, std::ios::binary | std::ios::trunc);
    if (output) {
        output.write(reinterpret_cast<const char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
    }
}

static void skippy_glm_dsa_trace_tensor(
        skippy_glm_dsa_op_timing & timing,
        skippy_glm_dsa_op_kind kind,
        const ggml_tensor * tensor) {
    if (!timing.trace_tensors ||
            timing.trace_nodes >= timing.trace_node_limit ||
            !skippy_glm_dsa_trace_filter_matches(timing.trace_filter, tensor->name)) {
        return;
    }

    timing.trace_nodes += 1;
    skippy_glm_dsa_dump_trace_tensor(timing, tensor);
    std::string line = format(
            "skippy: glm_dsa_tensor_trace stage=%d tokens=%zu op=%s node=%u name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] contiguous=%d nbytes=%zu",
            timing.stage_index,
            timing.token_count,
            skippy_glm_dsa_op_name(kind),
            timing.trace_nodes,
            tensor->name,
            ggml_type_name(tensor->type),
            static_cast<long long>(tensor->ne[0]),
            static_cast<long long>(tensor->ne[1]),
            static_cast<long long>(tensor->ne[2]),
            static_cast<long long>(tensor->ne[3]),
            static_cast<size_t>(tensor->nb[0]),
            static_cast<size_t>(tensor->nb[1]),
            static_cast<size_t>(tensor->nb[2]),
            static_cast<size_t>(tensor->nb[3]),
            ggml_is_contiguous(tensor) ? 1 : 0,
            ggml_nbytes(tensor));
    skippy_glm_dsa_trace_tensor_values(line, tensor, timing.trace_value_limit);
    if (timing.trace_stats) {
        skippy_glm_dsa_trace_tensor_stats(line, tensor, timing.trace_stats_max_bytes);
    }
    LLAMA_LOG_INFO("%s\n", line.c_str());
}

bool skippy_glm_dsa_op_timing_cb(ggml_tensor * tensor, bool ask, void * user_data) {
    skippy_model * model = static_cast<skippy_model *>(user_data);
    if (model == nullptr || !model->glm_dsa_timing.enabled || tensor == nullptr) {
        return false;
    }

    skippy_glm_dsa_op_timing & timing = model->glm_dsa_timing;
    const skippy_glm_dsa_op_kind kind = skippy_glm_dsa_op_kind_for_tensor(tensor->name);
    const bool trace_match = timing.trace_tensors &&
            skippy_glm_dsa_trace_filter_matches(timing.trace_filter, tensor->name);
    if (kind == SKIPPY_GLM_DSA_OP_UNKNOWN && !trace_match) {
        return false;
    }

    // Tensor tracing can be used as a narrow synchronization probe. Keep the
    // normal op-timing callback broad, but only split the graph at tensors the
    // trace filter actually requests when timing is disabled.
    if (timing.trace_tensors && !timing.print_timing && !trace_match) {
        return false;
    }

    if (ask) {
        timing.pending_kind = kind;
        timing.pending_start_us = ggml_time_us();
        return true;
    }

    const int64_t elapsed_us = std::max<int64_t>(0, ggml_time_us() - timing.pending_start_us);
    if (kind != SKIPPY_GLM_DSA_OP_UNKNOWN) {
        timing.stats[kind].nodes += 1;
        timing.stats[kind].elapsed_us += elapsed_us;
        skippy_glm_dsa_record_group_timing(timing, kind, tensor->name, elapsed_us);
        skippy_glm_dsa_record_hot_tensor(timing, kind, tensor, elapsed_us);
    }
    skippy_glm_dsa_trace_tensor(timing, kind, tensor);
    timing.pending_kind = SKIPPY_GLM_DSA_OP_UNKNOWN;
    timing.pending_start_us = 0;
    return true;
}

static skippy_glm_dsa_op_stat skippy_glm_dsa_sparse_mask_stat_from_stats(const skippy_glm_dsa_op_stat stats[]) {
    skippy_glm_dsa_op_stat stat = {};
    stat.nodes += stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL].nodes;
    stat.nodes += stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK].nodes;
    stat.nodes += stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD].nodes;
    stat.elapsed_us += stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL].elapsed_us;
    stat.elapsed_us += stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK].elapsed_us;
    stat.elapsed_us += stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD].elapsed_us;
    return stat;
}

static skippy_glm_dsa_op_stat skippy_glm_dsa_sparse_mask_stat(const skippy_glm_dsa_op_timing & timing) {
    return skippy_glm_dsa_sparse_mask_stat_from_stats(timing.stats);
}

static skippy_glm_dsa_op_stat skippy_glm_dsa_indexer_topk_stat_from_stats(const skippy_glm_dsa_op_stat stats[]) {
    skippy_glm_dsa_op_stat stat = {};
    stat.nodes += stats[SKIPPY_GLM_DSA_OP_INDEXER].nodes;
    stat.nodes += stats[SKIPPY_GLM_DSA_OP_TOP_K].nodes;
    stat.elapsed_us += stats[SKIPPY_GLM_DSA_OP_INDEXER].elapsed_us;
    stat.elapsed_us += stats[SKIPPY_GLM_DSA_OP_TOP_K].elapsed_us;
    return stat;
}

static skippy_glm_dsa_op_stat skippy_glm_dsa_indexer_topk_stat(const skippy_glm_dsa_op_timing & timing) {
    return skippy_glm_dsa_indexer_topk_stat_from_stats(timing.stats);
}

static skippy_glm_dsa_op_stat skippy_glm_dsa_routed_moe_stat_from_stats(const skippy_glm_dsa_op_stat stats[]) {
    skippy_glm_dsa_op_stat stat = {};
    for (int i = SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE; i <= SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE; ++i) {
        stat.nodes += stats[i].nodes;
        stat.elapsed_us += stats[i].elapsed_us;
    }
    return stat;
}

static skippy_glm_dsa_op_stat skippy_glm_dsa_routed_moe_stat(const skippy_glm_dsa_op_timing & timing) {
    return skippy_glm_dsa_routed_moe_stat_from_stats(timing.stats);
}

static void skippy_glm_dsa_append_op_stat(std::string & line, const char * name, const skippy_glm_dsa_op_stat & stat) {
    line += format(
            " %s_nodes=%llu %s_us=%lld",
            name,
            static_cast<unsigned long long>(stat.nodes),
            name,
            static_cast<long long>(stat.elapsed_us));
}

static void skippy_glm_dsa_append_op_kind(
        std::string & line,
        skippy_glm_dsa_op_kind kind,
        const skippy_glm_dsa_op_stat & stat) {
    skippy_glm_dsa_append_op_stat(line, skippy_glm_dsa_op_name(kind), stat);
}

static void skippy_glm_dsa_print_op_group(
        const skippy_glm_dsa_op_timing & timing,
        const skippy_glm_dsa_op_group & group) {
    uint64_t total_nodes = 0;
    int64_t total_us = 0;
    for (int i = 0; i < SKIPPY_GLM_DSA_OP_COUNT; ++i) {
        total_nodes += group.stats[i].nodes;
        total_us += group.stats[i].elapsed_us;
    }
    if (total_nodes == 0) {
        return;
    }

    std::string line = format(
            "skippy: glm_dsa_group_timing stage=%d tokens=%zu group=%s total_us=%lld",
            timing.stage_index,
            timing.token_count,
            group.name,
            static_cast<long long>(total_us));
    skippy_glm_dsa_append_op_stat(line, "indexer_topk", skippy_glm_dsa_indexer_topk_stat_from_stats(group.stats));
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_INDEXER, group.stats[SKIPPY_GLM_DSA_OP_INDEXER]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_TOP_K, group.stats[SKIPPY_GLM_DSA_OP_TOP_K]);
    skippy_glm_dsa_append_op_stat(line, "sparse_mask", skippy_glm_dsa_sparse_mask_stat_from_stats(group.stats));
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL, group.stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK, group.stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD, group.stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_DSA_SPARSE_ATTN, group.stats[SKIPPY_GLM_DSA_OP_DSA_SPARSE_ATTN]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_COMPACT_GET_ROWS, group.stats[SKIPPY_GLM_DSA_OP_COMPACT_GET_ROWS]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_MLA_ATTENTION, group.stats[SKIPPY_GLM_DSA_OP_MLA_ATTENTION]);
    skippy_glm_dsa_append_op_stat(line, "routed_moe", skippy_glm_dsa_routed_moe_stat_from_stats(group.stats));
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE_UP, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE_UP]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_UP, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_UP]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_ACT, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_ACT]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_DOWN, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_DOWN]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_WEIGHTED, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_WEIGHTED]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE, group.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SHARED_EXPERT, group.stats[SKIPPY_GLM_DSA_OP_SHARED_EXPERT]);
    LLAMA_LOG_INFO("%s\n", line.c_str());
}

static void skippy_glm_dsa_print_hot_tensors(const skippy_glm_dsa_op_timing & timing) {
    for (size_t i = 0; i < SKIPPY_GLM_DSA_HOT_TENSOR_LIMIT; ++i) {
        const skippy_glm_dsa_hot_tensor & record = timing.hot_tensors[i];
        if (!record.used) {
            return;
        }

        LLAMA_LOG_INFO(
                "skippy: glm_dsa_hot_tensor stage=%d tokens=%zu rank=%zu op=%s kind=%s elapsed_us=%lld name=%s ne0=%lld ne1=%lld ne2=%lld ne3=%lld\n",
                timing.stage_index,
                timing.token_count,
                i + 1,
                ggml_op_name(record.op),
                skippy_glm_dsa_op_name(record.kind),
                static_cast<long long>(record.elapsed_us),
                record.name[0] == '\0' ? "<unnamed>" : record.name,
                static_cast<long long>(record.ne[0]),
                static_cast<long long>(record.ne[1]),
                static_cast<long long>(record.ne[2]),
                static_cast<long long>(record.ne[3]));
    }
}

void skippy_glm_dsa_op_timing_begin(skippy_session * session, size_t token_count) {
    if (session == nullptr || session->stage_model == nullptr || !session->stage_model->glm_dsa_timing.enabled) {
        return;
    }

    skippy_glm_dsa_op_timing & timing = session->stage_model->glm_dsa_timing;
    const bool enabled = timing.enabled;
    const bool print_timing = timing.print_timing;
    const bool trace_tensors = timing.trace_tensors;
    const bool trace_stats = timing.trace_stats;
    const int32_t stage_index = timing.stage_index;
    const uint32_t trace_value_limit = timing.trace_value_limit;
    const uint32_t trace_node_limit = timing.trace_node_limit;
    const uint32_t trace_stats_max_bytes = timing.trace_stats_max_bytes;
    const std::string trace_filter = timing.trace_filter;
    timing = skippy_glm_dsa_op_timing{};
    timing.enabled = enabled;
    timing.print_timing = print_timing;
    timing.trace_tensors = trace_tensors;
    timing.trace_stats = trace_stats;
    timing.stage_index = stage_index;
    timing.token_count = token_count;
    timing.trace_value_limit = trace_value_limit;
    timing.trace_node_limit = trace_node_limit;
    timing.trace_stats_max_bytes = trace_stats_max_bytes;
    timing.trace_filter = trace_filter;
}

void skippy_glm_dsa_op_timing_end(skippy_session * session) {
    if (session == nullptr || session->stage_model == nullptr || !session->stage_model->glm_dsa_timing.enabled) {
        return;
    }

    const skippy_glm_dsa_op_timing & timing = session->stage_model->glm_dsa_timing;
    uint64_t total_nodes = 0;
    int64_t total_us = 0;
    for (int i = 0; i < SKIPPY_GLM_DSA_OP_COUNT; ++i) {
        total_nodes += timing.stats[i].nodes;
        total_us += timing.stats[i].elapsed_us;
    }
    if (total_nodes == 0) {
        return;
    }
    if (!timing.print_timing) {
        return;
    }

    std::string line = format(
            "skippy: glm_dsa_op_timing stage=%d tokens=%zu total_us=%lld",
            timing.stage_index,
            timing.token_count,
            static_cast<long long>(total_us));
    skippy_glm_dsa_append_op_stat(line, "indexer_topk", skippy_glm_dsa_indexer_topk_stat(timing));
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_INDEXER, timing.stats[SKIPPY_GLM_DSA_OP_INDEXER]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_TOP_K, timing.stats[SKIPPY_GLM_DSA_OP_TOP_K]);
    skippy_glm_dsa_append_op_stat(line, "sparse_mask", skippy_glm_dsa_sparse_mask_stat(timing));
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL, timing.stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_FILL]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK, timing.stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_TOPK]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD, timing.stats[SKIPPY_GLM_DSA_OP_SPARSE_MASK_ADD]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_DSA_SPARSE_ATTN, timing.stats[SKIPPY_GLM_DSA_OP_DSA_SPARSE_ATTN]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_COMPACT_GET_ROWS, timing.stats[SKIPPY_GLM_DSA_OP_COMPACT_GET_ROWS]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_MLA_ATTENTION, timing.stats[SKIPPY_GLM_DSA_OP_MLA_ATTENTION]);
    skippy_glm_dsa_append_op_stat(line, "routed_moe", skippy_glm_dsa_routed_moe_stat(timing));
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_ROUTE]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE_UP, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE_UP]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_GATE]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_UP, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_UP]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_ACT, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_ACT]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_DOWN, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_DOWN]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_WEIGHTED, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_WEIGHTED]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE, timing.stats[SKIPPY_GLM_DSA_OP_ROUTED_MOE_AGGREGATE]);
    skippy_glm_dsa_append_op_kind(line, SKIPPY_GLM_DSA_OP_SHARED_EXPERT, timing.stats[SKIPPY_GLM_DSA_OP_SHARED_EXPERT]);
    LLAMA_LOG_INFO("%s\n", line.c_str());

    for (size_t i = 0; i < SKIPPY_GLM_DSA_TIMING_GROUP_LIMIT; ++i) {
        if (timing.groups[i].used) {
            skippy_glm_dsa_print_op_group(timing, timing.groups[i]);
        }
    }
    skippy_glm_dsa_print_hot_tensors(timing);
}

