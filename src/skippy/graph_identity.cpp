#include "graph_identity.h"

#include "block_boundaries.h"
#include "skippy/activation.h"
#include "stage_slice_plan.h"
#include "state_effects.h"

#include "ggml.h"
#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>

namespace {

constexpr uint32_t identity_version = 1;
constexpr int32_t known_flags = GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT |
        GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_LOSS | GGML_TENSOR_FLAG_COMPUTE;

uint32_t rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
}

class sha256 {
public:
    sha256() : state{
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {
    }

    void update(const uint8_t * data, size_t size) {
        total_size += size;
        while (size > 0) {
            const size_t copy = std::min(size, block.size() - block_size);
            std::memcpy(block.data() + block_size, data, copy);
            block_size += copy;
            data += copy;
            size -= copy;
            if (block_size == block.size()) {
                transform(block.data());
                block_size = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t bits = static_cast<uint64_t>(total_size) * 8;
        block[block_size++] = 0x80;
        if (block_size > 56) {
            std::fill(block.begin() + block_size, block.end(), 0);
            transform(block.data());
            block_size = 0;
        }
        std::fill(block.begin() + block_size, block.begin() + 56, 0);
        for (int index = 0; index < 8; ++index) {
            block[63 - index] = static_cast<uint8_t>(bits >> (8 * index));
        }
        transform(block.data());
        std::array<uint8_t, 32> digest{};
        for (size_t index = 0; index < state.size(); ++index) {
            for (int byte = 0; byte < 4; ++byte) {
                digest[4 * index + byte] = static_cast<uint8_t>(state[index] >> (24 - 8 * byte));
            }
        }
        return digest;
    }

private:
    void transform(const uint8_t * data) {
        static constexpr uint32_t constants[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t words[64];
        for (int index = 0; index < 16; ++index) {
            words[index] = static_cast<uint32_t>(data[4 * index]) << 24 |
                    static_cast<uint32_t>(data[4 * index + 1]) << 16 |
                    static_cast<uint32_t>(data[4 * index + 2]) << 8 |
                    static_cast<uint32_t>(data[4 * index + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const uint32_t s0 = rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (int index = 0; index < 64; ++index) {
            const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t first = h + s1 + choice + constants[index] + words[index];
            const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t second = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::array<uint32_t, 8> state;
    std::array<uint8_t, 64> block{};
    size_t block_size = 0;
    size_t total_size = 0;
};

class encoding {
public:
    void raw(const uint8_t * value, size_t size) {
        bytes.insert(bytes.end(), value, value + size);
    }

    void u8(uint8_t value) {
        bytes.push_back(value);
    }

    void u32(uint32_t value) {
        for (int byte = 0; byte < 4; ++byte) {
            bytes.push_back(static_cast<uint8_t>(value >> (8 * byte)));
        }
    }

    void u64(uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            bytes.push_back(static_cast<uint8_t>(value >> (8 * byte)));
        }
    }

    void i32(int32_t value) {
        u32(static_cast<uint32_t>(value));
    }

    void i64(int64_t value) {
        u64(static_cast<uint64_t>(value));
    }

    bool text(const std::string & value) {
        if (value.size() > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        u32(static_cast<uint32_t>(value.size()));
        bytes.insert(bytes.end(), value.begin(), value.end());
        return true;
    }

    std::string digest(const char * domain) const {
        sha256 hash;
        hash.update(bytes.data(), bytes.size());
        const auto value = hash.finish();
        static constexpr char hex[] = "0123456789abcdef";
        std::string result = domain;
        result += ":v1:";
        result.reserve(result.size() + 64);
        for (uint8_t byte : value) {
            result.push_back(hex[byte >> 4]);
            result.push_back(hex[byte & 15]);
        }
        return result;
    }

    std::array<uint8_t, 32> raw_digest() const {
        sha256 hash;
        hash.update(bytes.data(), bytes.size());
        return hash.finish();
    }

private:
    std::vector<uint8_t> bytes;
};

enum class leaf_class : uint8_t {
    parameter = 1,
    request = 2,
    state = 3,
    immutable_constant = 4,
};

struct leaf_identity {
    leaf_class kind = leaf_class::parameter;
    std::string identity;
    skippy_state_kind state_kind = skippy_state_kind::kv_key;
    int32_t layer = -1;
    uint32_t state_ordinal = 0;
    uint32_t constant_ordinal = 0;
};

bool valid_tensor_metadata(const ggml_tensor * tensor, std::string & error) {
    if (tensor == nullptr || tensor->type < 0 || tensor->type >= GGML_TYPE_COUNT ||
            tensor->op < 0 || tensor->op >= GGML_OP_COUNT) {
        error = "semantic tensor has an unknown type or operation";
        return false;
    }
    if ((tensor->flags & ~known_flags) != 0) {
        error = "semantic tensor has unknown flags";
        return false;
    }
    if (tensor->op == GGML_OP_MAP_CUSTOM1 || tensor->op == GGML_OP_MAP_CUSTOM2 ||
            tensor->op == GGML_OP_MAP_CUSTOM3 || tensor->op == GGML_OP_CUSTOM) {
        error = "semantic tensor uses an opaque custom operation";
        return false;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (tensor->ne[dimension] < 0 || tensor->nb[dimension] > std::numeric_limits<uint64_t>::max()) {
            error = "semantic tensor has invalid dimensions or strides";
            return false;
        }
    }
    return true;
}

void encode_tensor_metadata(encoding & out, const ggml_tensor * tensor) {
    out.i32(static_cast<int32_t>(tensor->op));
    out.i32(static_cast<int32_t>(tensor->type));
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        out.i64(tensor->ne[dimension]);
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        out.u64(static_cast<uint64_t>(tensor->nb[dimension]));
    }
    for (size_t index = 0; index < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++index) {
        out.i32(tensor->op_params[index]);
    }
    out.i32(tensor->flags & known_flags);
    out.u64(static_cast<uint64_t>(tensor->view_offs));
}

bool tensor_metadata_identity(
        const ggml_tensor * tensor,
        std::string & identity,
        std::string & error) {
    if (!valid_tensor_metadata(tensor, error)) {
        return false;
    }
    encoding out;
    out.text("skippy.semantic.tensor-metadata");
    out.u32(identity_version);
    encode_tensor_metadata(out, tensor);
    identity = out.digest("skippy-tensor-metadata");
    return true;
}

bool encode_leaf(encoding & out, const leaf_identity & leaf) {
    out.u8(static_cast<uint8_t>(leaf.kind));
    if (leaf.kind == leaf_class::state) {
        out.i32(static_cast<int32_t>(leaf.state_kind));
        out.i32(leaf.layer);
        out.u32(leaf.state_ordinal);
        return true;
    }
    if (leaf.kind == leaf_class::immutable_constant) {
        out.u32(leaf.constant_ordinal);
        return true;
    }
    return out.text(leaf.identity);
}

class graph_encoder {
public:
    graph_encoder(
            const ggml_cgraph * graph,
            const std::vector<skippy_block_boundary> & boundaries,
            const std::vector<skippy_state_effect> & effects,
            const std::vector<skippy_parameter_binding> & parameters,
            const std::vector<skippy_request_input_binding> & request_inputs,
            skippy_graph_semantics & result,
            std::string & error) :
        graph(graph), boundaries(boundaries), effects(effects), parameters(parameters),
        request_inputs(request_inputs), result(result), error(error) {
    }

    bool build() {
        if (graph == nullptr || graph->n_nodes <= 0 || graph->n_leafs < 0) {
            error = "semantic graph is empty or invalid";
            return false;
        }
        if (!skippy_validate_block_boundaries(boundaries, 0, static_cast<int32_t>(boundaries.size() / 2), error) ||
                !skippy_validate_state_effects(effects, graph, error) || !index_nodes() ||
                !bind_parameters() || !bind_requests() || !bind_state()) {
            return false;
        }
        encoding graph_encoding;
        graph_encoding.text("skippy.semantic.graph");
        graph_encoding.u32(identity_version);
        graph_encoding.i32(static_cast<int32_t>(graph->order));
        graph_encoding.u32(static_cast<uint32_t>(graph->n_nodes));
        for (int32_t index = 0; index < graph->n_nodes; ++index) {
            if (!encode_node(index, graph_encoding)) {
                return false;
            }
        }
        if (!encode_boundaries(graph_encoding) || !encode_effects(graph_encoding)) {
            return false;
        }
        result.identity = graph_encoding.digest("skippy-graph");
        return true;
    }

private:
    bool index_nodes() {
        for (int32_t index = 0; index < graph->n_nodes; ++index) {
            const ggml_tensor * node = graph->nodes[index];
            if (node == nullptr || !node_indices.emplace(node, index).second) {
                error = "semantic graph has null or duplicate nodes";
                return false;
            }
        }
        for (int32_t index = 0; index < graph->n_leafs; ++index) {
            const ggml_tensor * leaf = skippy_state_canonical_storage(graph->leafs[index]);
            if (leaf == nullptr) {
                error = "semantic graph has a cyclic leaf view";
                return false;
            }
            graph_leaves.insert(leaf);
        }
        return true;
    }

    bool add_binding(const ggml_tensor * tensor, const leaf_identity & leaf, const char * incomplete) {
        if (tensor == nullptr || (leaf.kind != leaf_class::state && leaf.identity.empty())) {
            error = incomplete;
            return false;
        }
        const ggml_tensor * canonical = skippy_state_canonical_storage(tensor);
        if (canonical == nullptr) {
            error = "semantic binding has cyclic storage";
            return false;
        }
        if (!bound_tensors.insert(tensor).second) {
            error = "duplicate semantic tensor binding";
            return false;
        }
        const auto existing = leaves.find(canonical);
        if (existing != leaves.end()) {
            error = "conflicting semantic tensor bindings";
            return false;
        }
        if (leaf.kind != leaf_class::state) {
            const auto key = std::make_pair(static_cast<uint8_t>(leaf.kind), leaf.identity);
            const bool inserted = binding_identities.insert(key).second;
            // A tied package parameter may be materialized as distinct tensors
            // in separate backend contexts. They intentionally share one stable
            // package identity. Request identities remain one-to-one.
            if (!inserted && leaf.kind != leaf_class::parameter) {
                error = "duplicate semantic binding identity: " + leaf.identity;
                return false;
            }
        }
        leaves.emplace(canonical, leaf);
        return true;
    }

    bool bind_parameters() {
        for (const auto & parameter : parameters) {
            if (!add_binding(parameter.tensor, {leaf_class::parameter, parameter.identity}, "parameter binding is incomplete")) {
                return false;
            }
            const ggml_tensor * canonical = skippy_state_canonical_storage(parameter.tensor);
            if ((parameter.tensor->flags & GGML_TENSOR_FLAG_INPUT) != 0 ||
                    (canonical->flags & GGML_TENSOR_FLAG_INPUT) != 0) {
                error = "parameter binding conflicts with request-input flags";
                return false;
            }
        }
        return true;
    }

    bool bind_requests() {
        for (const auto & request : request_inputs) {
            if (!add_binding(request.tensor, {leaf_class::request, request.identity}, "request-input binding is incomplete")) {
                return false;
            }
            const ggml_tensor * canonical = skippy_state_canonical_storage(request.tensor);
            if ((request.tensor->flags & GGML_TENSOR_FLAG_INPUT) == 0 &&
                    (canonical->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                error = "request-input binding does not target an input tensor";
                return false;
            }
        }
        return true;
    }

    bool bind_state() {
        std::map<std::pair<int32_t, int32_t>, uint32_t> ordinals;
        std::set<const ggml_tensor *> bound_storages;
        for (const auto & effect : effects) {
            if (effect.access != skippy_state_access::read) {
                continue;
            }
            // Architectures that share one KV cache tensor across layers bind
            // the storage once; later readers resolve to the same leaf.
            if (!bound_storages.insert(effect.storage).second) {
                continue;
            }
            leaf_identity leaf;
            leaf.kind = leaf_class::state;
            leaf.state_kind = effect.kind;
            leaf.layer = effect.layer;
            leaf.state_ordinal = ordinals[{static_cast<int32_t>(effect.kind), effect.layer}]++;
            if (!add_binding(effect.storage, leaf, "state owner binding is incomplete")) {
                return false;
            }
        }
        return true;
    }

    bool tensor_value(const ggml_tensor * tensor, int32_t consumer, std::string & identity) {
        if (tensor == nullptr) {
            identity.clear();
            return true;
        }
        const auto encoded = result.value_identities.find(tensor);
        if (encoded != result.value_identities.end()) {
            identity = encoded->second;
            return true;
        }
        const auto node = node_indices.find(tensor);
        if (node != node_indices.end()) {
            error = node->second >= consumer ? "semantic graph contains a forward reference" : "semantic graph reference was not encoded";
            return false;
        }
        const ggml_tensor * canonical = skippy_state_canonical_storage(tensor);
        if (canonical == nullptr) {
            error = "semantic graph contains a cyclic view";
            return false;
        }
        const auto leaf = leaves.find(canonical);
        if (leaf == leaves.end() && graph_leaves.find(canonical) != graph_leaves.end() &&
                canonical->op == GGML_OP_NONE && canonical->buffer == nullptr &&
                (canonical->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
            leaf_identity constant;
            constant.kind = leaf_class::immutable_constant;
            constant.constant_ordinal = next_constant_ordinal++;
            leaves.emplace(canonical, constant);
        }
        const auto classified_leaf = leaves.find(canonical);
        if (classified_leaf == leaves.end()) {
            error = ((tensor->flags & GGML_TENSOR_FLAG_INPUT) != 0 ||
                    (canonical->flags & GGML_TENSOR_FLAG_INPUT) != 0
                    ? "semantic graph has an unregistered request input"
                    : "semantic graph has an unresolved structural leaf") +
                    std::string(": ") + ggml_get_name(canonical) +
                    " type=" + std::to_string(static_cast<int>(canonical->type)) +
                    " ne=" + std::to_string(canonical->ne[0]) + "x" +
                    std::to_string(canonical->ne[1]) + "x" +
                    std::to_string(canonical->ne[2]) + "x" +
                    std::to_string(canonical->ne[3]) +
                    " flags=" + std::to_string(canonical->flags) +
                    " buffered=" + (canonical->buffer != nullptr ? "1" : "0");
            return false;
        }
        if (tensor != canonical) {
            if (tensor->view_src == nullptr || !external_visiting.insert(tensor).second) {
                error = "semantic graph has an unresolved external alias";
                return false;
            }
            std::string view;
            std::array<std::string, GGML_MAX_SRC> sources;
            if (!tensor_value(tensor->view_src, consumer, view)) {
                return false;
            }
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                if (!tensor_value(tensor->src[source], consumer, sources[source])) {
                    return false;
                }
            }
            external_visiting.erase(tensor);
            encoding alias;
            alias.text("skippy.semantic.external-alias");
            alias.u32(identity_version);
            if (!valid_tensor_metadata(tensor, error)) {
                return false;
            }
            encode_tensor_metadata(alias, tensor);
            alias.text(view);
            for (const auto & source : sources) {
                alias.text(source);
            }
            identity = alias.digest("skippy-value");
            result.value_identities.emplace(tensor, identity);
            return true;
        }
        if (!valid_tensor_metadata(tensor, error)) {
            return false;
        }
        encoding value;
        value.text("skippy.semantic.leaf");
        value.u32(identity_version);
        encode_leaf(value, classified_leaf->second);
        encode_tensor_metadata(value, tensor);
        identity = value.digest("skippy-value");
        result.value_identities.emplace(tensor, identity);
        return true;
    }

    bool encode_node(int32_t index, encoding & graph_encoding) {
        const ggml_tensor * node = graph->nodes[index];
        if (!valid_tensor_metadata(node, error)) {
            return false;
        }
        std::string view;
        std::array<std::string, GGML_MAX_SRC> sources;
        if (!tensor_value(node->view_src, index, view)) {
            return false;
        }
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            // ggml_cast (and friends) point src[1] back at the node itself;
            // that self-reference carries no semantic dependency.
            if (node->src[source] == node) {
                continue;
            }
            if (!tensor_value(node->src[source], index, sources[source])) {
                return false;
            }
        }
        const ggml_tensor * canonical = skippy_state_canonical_storage(node);
        const auto leaf = canonical == nullptr ? leaves.end() : leaves.find(canonical);
        if (node->op == GGML_OP_NONE && leaf == leaves.end()) {
            error = "semantic graph has an unbound operation-free node";
            return false;
        }
        if ((node->flags & GGML_TENSOR_FLAG_INPUT) != 0 &&
                (leaf == leaves.end() || leaf->second.kind != leaf_class::request)) {
            error = "semantic graph has an unregistered request input node";
            return false;
        }
        encoding value;
        value.text("skippy.semantic.node");
        value.u32(identity_version);
        encode_tensor_metadata(value, node);
        value.u8(leaf == leaves.end() ? 0 : 1);
        if (leaf != leaves.end() && !encode_leaf(value, leaf->second)) {
            return false;
        }
        value.text(view);
        for (const auto & source : sources) {
            value.text(source);
        }
        const std::string identity = value.digest("skippy-value");
        result.value_identities.emplace(node, identity);
        graph_encoding.i32(index);
        encode_tensor_metadata(graph_encoding, node);
        graph_encoding.u8(leaf == leaves.end() ? 0 : 1);
        if (leaf != leaves.end()) {
            encode_leaf(graph_encoding, leaf->second);
        }
        graph_encoding.text(view);
        for (const auto & source : sources) {
            graph_encoding.text(source);
        }
        graph_encoding.text(identity);
        return true;
    }

    bool encode_boundaries(encoding & out) {
        out.u32(static_cast<uint32_t>(boundaries.size()));
        for (const auto & boundary : boundaries) {
            std::string identity;
            if (!tensor_value(boundary.tensor, graph->n_nodes, identity)) {
                return false;
            }
            out.i32(static_cast<int32_t>(boundary.kind));
            out.i32(boundary.layer);
            out.text(identity);
        }
        return true;
    }

    bool encode_effects(encoding & out) {
        std::vector<const skippy_state_effect *> ordered;
        ordered.reserve(effects.size());
        for (const auto & effect : effects) {
            ordered.push_back(&effect);
        }
        std::sort(ordered.begin(), ordered.end(), [&](const auto * left, const auto * right) {
            const int left_position = left->effect_node == nullptr ? -1 : node_indices.at(left->effect_node);
            const int right_position = right->effect_node == nullptr ? -1 : node_indices.at(right->effect_node);
            return std::tie(left->layer, left->kind, left->access, left->write_ordinal, left_position) <
                    std::tie(right->layer, right->kind, right->access, right->write_ordinal, right_position);
        });
        out.u32(static_cast<uint32_t>(ordered.size()));
        for (const auto * effect : ordered) {
            std::string storage;
            std::string node;
            if (!tensor_value(effect->storage, graph->n_nodes, storage) ||
                    !tensor_value(effect->effect_node, graph->n_nodes, node)) {
                return false;
            }
            out.i32(static_cast<int32_t>(effect->kind));
            out.i32(static_cast<int32_t>(effect->access));
            out.i32(effect->layer);
            out.i32(static_cast<int32_t>(effect->residency));
            out.i64(effect->write_ordinal);
            out.text(storage);
            out.text(node);
        }
        return true;
    }

    const ggml_cgraph * graph;
    const std::vector<skippy_block_boundary> & boundaries;
    const std::vector<skippy_state_effect> & effects;
    const std::vector<skippy_parameter_binding> & parameters;
    const std::vector<skippy_request_input_binding> & request_inputs;
    skippy_graph_semantics & result;
    std::string & error;
    std::unordered_map<const ggml_tensor *, int32_t> node_indices;
    std::unordered_set<const ggml_tensor *> graph_leaves;
    std::unordered_map<const ggml_tensor *, leaf_identity> leaves;
    uint32_t next_constant_ordinal = 0;
    std::unordered_set<const ggml_tensor *> bound_tensors;
    std::set<std::pair<uint8_t, std::string>> binding_identities;
    std::unordered_set<const ggml_tensor *> external_visiting;
};

bool build_source_snapshot_identity(
        const skippy_stage_slice_selection & selection,
        std::string & identity,
        std::string & error) {
    if (selection.source_value_identities.empty() ||
            selection.source_value_identities.size() != selection.source_value_metadata_identities.size() ||
            selection.source_value_identities.size() != selection.source_value_references.size()) {
        error = "stage slice source semantic snapshot is incomplete";
        return false;
    }
    using snapshot_record = std::tuple<std::string, std::string, std::vector<std::string>>;
    std::vector<snapshot_record> records;
    records.reserve(selection.source_value_identities.size());
    for (const auto & [tensor, value_identity] : selection.source_value_identities) {
        const auto metadata = selection.source_value_metadata_identities.find(tensor);
        const auto references = selection.source_value_references.find(tensor);
        std::string current_metadata;
        if (tensor == nullptr || metadata == selection.source_value_metadata_identities.end() ||
                references == selection.source_value_references.end() ||
                references->second.size() != 1 + GGML_MAX_SRC ||
                references->second[0] != tensor->view_src ||
                !std::equal(std::begin(tensor->src), std::end(tensor->src),
                        references->second.begin() + 1) ||
                !tensor_metadata_identity(tensor, current_metadata, error) ||
                current_metadata != metadata->second) {
            if (error.empty()) {
                error = "stage slice source semantic snapshot is stale";
            }
            return false;
        }
        std::vector<std::string> reference_identities;
        reference_identities.reserve(references->second.size());
        for (const ggml_tensor * reference : references->second) {
            if (reference == nullptr) {
                reference_identities.emplace_back();
                continue;
            }
            const auto found = selection.source_value_identities.find(reference);
            if (found == selection.source_value_identities.end()) {
                error = "stage slice source semantic snapshot omits a transitive reference";
                return false;
            }
            reference_identities.push_back(found->second);
        }
        records.emplace_back(value_identity, metadata->second, std::move(reference_identities));
    }
    std::sort(records.begin(), records.end());
    encoding out;
    out.text("skippy.semantic.source-snapshot");
    out.u32(identity_version);
    out.u32(static_cast<uint32_t>(records.size()));
    for (const auto & [value, metadata, references] : records) {
        out.text(value);
        out.text(metadata);
        out.u32(static_cast<uint32_t>(references.size()));
        for (const auto & reference : references) {
            out.text(reference);
        }
    }
    identity = out.digest("skippy-source-snapshot");
    return true;
}

bool slice_encoding(const skippy_stage_slice_selection & selection, encoding & out, std::string & error) {
    if (selection.graph_identity.empty() || selection.source_snapshot_identity.empty() ||
            selection.layer_start < 0 ||
            selection.layer_start >= selection.layer_end ||
            selection.node_indices.size() != selection.node_identities.size() ||
            selection.activation_imports.size() != selection.activation_import_identities.size() ||
            selection.activation_exports.size() != selection.activation_export_identities.size() ||
            selection.request_inputs.size() != selection.request_input_identities.size() ||
            selection.state_effects.size() != selection.state_effect_identities.size()) {
        error = "stage slice semantic identity inputs are incomplete";
        return false;
    }
    out.text("skippy.semantic.slice");
    out.u32(identity_version);
    out.text(selection.graph_identity);
    out.text(selection.source_snapshot_identity);
    out.i32(selection.layer_start);
    out.i32(selection.layer_end);
    out.u32(static_cast<uint32_t>(selection.node_indices.size()));
    for (size_t index = 0; index < selection.node_indices.size(); ++index) {
        out.i32(selection.node_indices[index]);
        out.text(selection.node_identities[index]);
    }
    auto identities = [&](const std::vector<std::string> & values) {
        out.u32(static_cast<uint32_t>(values.size()));
        for (const auto & value : values) {
            out.text(value);
        }
    };
    identities(selection.activation_import_identities);
    identities(selection.activation_export_identities);
    identities(selection.request_input_identities);
    identities(selection.parameter_identities);
    identities(selection.state_effect_identities);
    return true;
}

bool effect_identity(
        const skippy_graph_semantics & graph,
        const skippy_state_effect & effect,
        std::string & identity,
        std::string & error) {
    const auto storage = graph.value_identities.find(effect.storage);
    if (storage == graph.value_identities.end()) {
        error = "stage state effect has no semantic storage identity";
        return false;
    }
    std::string node;
    if (effect.effect_node != nullptr) {
        const auto found = graph.value_identities.find(effect.effect_node);
        if (found == graph.value_identities.end()) {
            error = "stage state effect has no semantic node identity";
            return false;
        }
        node = found->second;
    }
    encoding out;
    out.text("skippy.semantic.state-effect");
    out.u32(identity_version);
    out.i32(static_cast<int32_t>(effect.kind));
    out.i32(static_cast<int32_t>(effect.access));
    out.i32(effect.layer);
    out.i32(static_cast<int32_t>(effect.residency));
    out.i64(effect.write_ordinal);
    out.text(storage->second);
    out.text(node);
    identity = out.digest("skippy-state-effect");
    return true;
}

}

bool skippy_build_activation_port_identity(
        const std::string & binding,
        std::string & identity) {
    if (binding.empty()) {
        identity.clear();
        return false;
    }
    encoding encoded;
    encoded.text("skippy.activation.port");
    encoded.u32(identity_version);
    encoded.text(binding);
    identity = encoded.digest("skippy-port");
    return true;
}

bool skippy_build_activation_frontier_identity(
        const skippy_activation_part_desc * parts,
        size_t part_count,
        uint8_t output[32]) {
    if (parts == nullptr || part_count == 0 ||
            part_count > SKIPPY_ACTIVATION_MAX_PARTS || output == nullptr) {
        return false;
    }
    encoding encoded;
    encoded.text("skippy.activation.frontier");
    encoded.u32(SKIPPY_ACTIVATION_BOUNDARY_DESC_VERSION);
    encoded.u32(static_cast<uint32_t>(part_count));
    std::vector<size_t> order(part_count);
    for (size_t index = 0; index < part_count; ++index) order[index] = index;
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        return std::memcmp(parts[left].identity, parts[right].identity,
                SKIPPY_ACTIVATION_IDENTITY_BYTES) < 0;
    });
    for (size_t index : order) {
        const auto & part = parts[index];
        encoded.raw(part.identity, SKIPPY_ACTIVATION_IDENTITY_BYTES);
        encoded.u32(part.ggml_type);
        encoded.u32(part.rank);
        encoded.i32(part.token_axis);
        encoded.u32(part.flags);
        for (int64_t dimension : part.dimensions) encoded.i64(dimension);
        for (uint64_t stride : part.byte_strides) encoded.u64(stride);
    }
    const auto digest = encoded.raw_digest();
    std::memcpy(output, digest.data(), digest.size());
    return true;
}

bool skippy_build_graph_semantics(
        const ggml_cgraph * graph,
        const std::vector<skippy_block_boundary> & boundaries,
        const std::vector<skippy_state_effect> & state_effects,
        const std::vector<skippy_parameter_binding> & parameters,
        const std::vector<skippy_request_input_binding> & request_inputs,
        skippy_graph_semantics & result,
        std::string & error) {
    result = {};
    error.clear();
    return graph_encoder(graph, boundaries, state_effects, parameters, request_inputs, result, error).build();
}

bool skippy_finalize_stage_slice_identity(
        const skippy_graph_semantics & graph,
        skippy_stage_slice_selection & selection,
        std::string & error) {
    selection.graph_identity = graph.identity;
    selection.node_identities.clear();
    selection.activation_import_identities.clear();
    selection.activation_export_identities.clear();
    selection.request_input_identities.clear();
    selection.state_effect_identities.clear();
    selection.source_snapshot_identity.clear();
    selection.source_value_identities.clear();
    selection.source_value_metadata_identities.clear();
    selection.source_value_references.clear();
    std::function<bool(const ggml_tensor *)> remember;
    remember = [&](const ggml_tensor * tensor) {
        if (tensor == nullptr || selection.source_value_identities.count(tensor) != 0) {
            return true;
        }
        const auto found = graph.value_identities.find(tensor);
        std::string metadata;
        if (found == graph.value_identities.end() ||
                !tensor_metadata_identity(tensor, metadata, error)) {
            if (error.empty()) {
                error = "selected tensor has no semantic identity";
            }
            return false;
        }
        std::vector<const ggml_tensor *> references;
        references.reserve(1 + GGML_MAX_SRC);
        references.push_back(tensor->view_src);
        references.insert(references.end(), std::begin(tensor->src), std::end(tensor->src));
        selection.source_value_identities.emplace(tensor, found->second);
        selection.source_value_metadata_identities.emplace(tensor, std::move(metadata));
        selection.source_value_references.emplace(tensor, std::move(references));
        if (!remember(tensor->view_src)) {
            return false;
        }
        for (const ggml_tensor * source : tensor->src) {
            if (!remember(source)) {
                return false;
            }
        }
        return true;
    };
    auto append = [&](const ggml_tensor * tensor, std::vector<std::string> & identities) {
        if (!remember(tensor)) {
            return false;
        }
        identities.push_back(graph.value_identities.at(tensor));
        return true;
    };
    for (int32_t index : selection.node_indices) {
        if (selection.source_graph == nullptr || index < 0 || index >= selection.source_graph->n_nodes ||
                !append(selection.source_graph->nodes[index], selection.node_identities)) {
            return false;
        }
    }
    for (const ggml_tensor * tensor : selection.activation_imports) {
        if (!append(tensor, selection.activation_import_identities)) {
            return false;
        }
    }
    for (const ggml_tensor * tensor : selection.activation_exports) {
        if (!append(tensor, selection.activation_export_identities)) {
            return false;
        }
    }
    for (const ggml_tensor * tensor : selection.request_inputs) {
        if (!append(tensor, selection.request_input_identities)) {
            return false;
        }
    }
    for (const auto & effect : selection.state_effects) {
        if (!remember(effect.storage) ||
                (effect.effect_node != nullptr && !remember(effect.effect_node))) {
            return false;
        }
        std::string identity;
        if (!effect_identity(graph, effect, identity, error)) {
            return false;
        }
        selection.state_effect_identities.push_back(std::move(identity));
    }
    if (!build_source_snapshot_identity(selection, selection.source_snapshot_identity, error)) {
        return false;
    }
    encoding out;
    if (!slice_encoding(selection, out, error)) {
        return false;
    }
    selection.slice_identity = out.digest("skippy-slice");
    return true;
}

bool skippy_validate_stage_slice_identity(
        const skippy_stage_slice_selection & selection,
        std::string & error) {
    encoding out;
    if (!slice_encoding(selection, out, error)) {
        return false;
    }
    if (selection.slice_identity.empty() || selection.slice_identity != out.digest("skippy-slice")) {
        error = "stage slice semantic identity does not match its contents";
        return false;
    }
    if (selection.source_graph == nullptr || selection.source_value_identities.empty() ||
            selection.source_value_metadata_identities.empty() ||
            selection.source_value_references.empty()) {
        error = "stage slice has no source semantic value bindings";
        return false;
    }
    std::string source_snapshot_identity;
    if (!build_source_snapshot_identity(selection, source_snapshot_identity, error) ||
            source_snapshot_identity != selection.source_snapshot_identity) {
        if (error.empty()) {
            error = "stage slice source semantic snapshot identity does not match";
        }
        return false;
    }
    auto matches = [&](const ggml_tensor * tensor, const std::string & identity) {
        const auto found = selection.source_value_identities.find(tensor);
        const auto metadata = selection.source_value_metadata_identities.find(tensor);
        const auto references = selection.source_value_references.find(tensor);
        std::string current_metadata;
        if (found == selection.source_value_identities.end() || found->second != identity ||
                metadata == selection.source_value_metadata_identities.end() ||
                references == selection.source_value_references.end() ||
                references->second.size() != 1 + GGML_MAX_SRC ||
                references->second[0] != tensor->view_src ||
                !tensor_metadata_identity(tensor, current_metadata, error) ||
                metadata->second != current_metadata) {
            return false;
        }
        return std::equal(std::begin(tensor->src), std::end(tensor->src),
                references->second.begin() + 1);
    };
    for (size_t position = 0; position < selection.node_indices.size(); ++position) {
        const int32_t index = selection.node_indices[position];
        if (index < 0 || index >= selection.source_graph->n_nodes ||
                !matches(selection.source_graph->nodes[index], selection.node_identities[position])) {
            error = "stage slice node pointer does not match its source semantic identity";
            return false;
        }
    }
    auto frontier_matches = [&](const std::vector<const ggml_tensor *> & tensors,
                                const std::vector<std::string> & identities) {
        for (size_t position = 0; position < tensors.size(); ++position) {
            if (!matches(tensors[position], identities[position])) {
                return false;
            }
        }
        return true;
    };
    if (!frontier_matches(selection.activation_imports, selection.activation_import_identities) ||
            !frontier_matches(selection.activation_exports, selection.activation_export_identities) ||
            !frontier_matches(selection.request_inputs, selection.request_input_identities)) {
        error = "stage slice pointer does not match its source semantic identity";
        return false;
    }
    skippy_graph_semantics source_semantics;
    source_semantics.identity = selection.graph_identity;
    source_semantics.value_identities = selection.source_value_identities;
    for (size_t position = 0; position < selection.state_effects.size(); ++position) {
        std::string identity;
        if (!effect_identity(source_semantics, selection.state_effects[position], identity, error) ||
                identity != selection.state_effect_identities[position]) {
            if (error.empty()) {
                error = "stage slice state effect pointer does not match its source semantic identity";
            }
            return false;
        }
    }
    return true;
}

bool skippy_build_profile_identity(
        const std::string & profile_id,
        const std::string & graph_identity,
        const skippy_execution_profile_guard & guard,
        std::string & identity,
        std::string & error) {
    if (profile_id.empty() || graph_identity.empty() || guard.n_tokens == 0 ||
            guard.n_sequences == 0 || guard.n_outputs == 0 ||
            guard.n_outputs > guard.n_tokens || guard.graph_configuration_id.empty() ||
            guard.backend_id.empty()) {
        error = "profile semantic identity inputs are incomplete";
        return false;
    }
    encoding out;
    out.text("skippy.semantic.profile");
    out.u32(identity_version);
    out.text(profile_id);
    out.text(graph_identity);
    out.u32(guard.n_tokens);
    out.u32(guard.n_sequences);
    out.u32(guard.n_outputs);
    out.u32(guard.n_recurrent_rollback_sequences);
    out.text(guard.graph_configuration_id);
    out.text(guard.backend_id);
    identity = out.digest("skippy-profile");
    return true;
}

bool skippy_build_stage_plan_identity(
        const std::string & package_id,
        const skippy_guarded_stage_slice_plan & plan,
        std::string & identity,
        std::string & error) {
    identity.clear();
    error.clear();
    if (package_id.empty() || plan.layer_count <= 0 || plan.layer_start < 0 ||
            plan.layer_start >= plan.layer_end || plan.layer_end > plan.layer_count ||
            plan.profiles.empty() || plan.resident_parameter_identities.empty()) {
        error = "stage plan identity inputs are incomplete";
        return false;
    }
    encoding out;
    out.text("skippy.semantic.stage-plan");
    out.u32(identity_version);
    out.text(package_id);
    out.i32(plan.layer_count);
    out.i32(plan.layer_start);
    out.i32(plan.layer_end);
    out.u32(static_cast<uint32_t>(plan.profiles.size()));
    for (const auto & profile : plan.profiles) {
        if (!out.text(profile.profile_id) || !out.text(profile.profile_identity) ||
                !out.text(profile.slice.slice_identity)) {
            error = "stage plan profile identity is too large";
            return false;
        }
    }
    out.u32(static_cast<uint32_t>(plan.auxiliary_profiles.size()));
    for (const auto & profile : plan.auxiliary_profiles) {
        if (!out.text(profile.profile_id) || !out.text(profile.graph_identity)) {
            error = "stage plan auxiliary profile identity is too large";
            return false;
        }
    }
    out.u32(static_cast<uint32_t>(plan.resident_parameter_identities.size()));
    for (const auto & parameter : plan.resident_parameter_identities) {
        if (!out.text(parameter)) {
            error = "stage plan parameter identity is too large";
            return false;
        }
    }
    identity = out.digest("skippy-plan");
    return true;
}
