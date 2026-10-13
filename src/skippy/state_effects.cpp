#include "state_effects.h"

#include "block_boundaries.h"

#include "ggml.h"
#include "../../ggml/src/ggml-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

const ggml_tensor * skippy_state_canonical_storage(const ggml_tensor * tensor) {
    const ggml_tensor * slow = tensor;
    const ggml_tensor * fast = tensor;
    while (fast != nullptr && fast->view_src != nullptr) {
        slow = slow->view_src;
        fast = fast->view_src->view_src;
        if (slow == fast) {
            return nullptr;
        }
    }
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static std::string effect_error(const char * fmt, size_t index) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), fmt, index);
    return buf;
}

static bool state_kind_known(skippy_state_kind kind) {
    switch (kind) {
        case skippy_state_kind::kv_key:
        case skippy_state_kind::kv_value:
        case skippy_state_kind::recurrent_conv:
        case skippy_state_kind::recurrent_ssm:
        case skippy_state_kind::derived_persistent:
            return true;
    }
    return false;
}

static bool state_access_known(skippy_state_access access) {
    switch (access) {
        case skippy_state_access::read:
        case skippy_state_access::write:
            return true;
    }
    return false;
}

static bool state_residency_known(skippy_state_residency residency) {
    switch (residency) {
        case skippy_state_residency::layer_local:
        case skippy_state_residency::per_stage:
            return true;
    }
    return false;
}

static bool checked_add(size_t a, size_t b, size_t & result) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}

static bool checked_mul(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

/** Compute the strided byte extent without overflowing ggml_nbytes arithmetic. */
static bool tensor_span(const ggml_tensor * tensor, size_t & span) {
    if (tensor == nullptr || tensor->type < 0 || tensor->type >= GGML_TYPE_COUNT) {
        return false;
    }
    bool empty = false;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (tensor->ne[d] < 0 || static_cast<uint64_t>(tensor->ne[d]) > std::numeric_limits<size_t>::max()) {
            return false;
        }
        empty |= tensor->ne[d] == 0;
    }
    const size_t block = ggml_blck_size(tensor->type);
    const size_t element = ggml_type_size(tensor->type);
    if (block == 0 || element == 0 || tensor->ne[0] % block != 0) {
        return false;
    }
    span = 0;
    if (empty) {
        return true;
    }
    span = element;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const size_t count = static_cast<size_t>(tensor->ne[d]) / (d == 0 ? block : 1);
        size_t extent;
        if (!checked_mul(count - 1, tensor->nb[d], extent) || !checked_add(span, extent, span)) {
            return false;
        }
    }
    return true;
}

struct state_region {
    const ggml_tensor * base = nullptr;
    size_t offset = 0;
    size_t span = 0;
};

static bool tensor_region(const ggml_tensor * tensor, state_region & region) {
    region = {};
    region.base = skippy_state_canonical_storage(tensor);
    if (region.base == nullptr || !tensor_span(tensor, region.span)) {
        return false;
    }
    while (tensor->view_src != nullptr) {
        size_t child_span;
        size_t parent_span;
        if (!tensor_span(tensor, child_span) || !tensor_span(tensor->view_src, parent_span) ||
                tensor->view_offs > parent_span || child_span > parent_span - tensor->view_offs ||
                !checked_add(region.offset, tensor->view_offs, region.offset)) {
            return false;
        }
        tensor = tensor->view_src;
    }
    size_t base_span;
    return tensor->view_offs == 0 && tensor_span(tensor, base_span) &&
            region.offset <= base_span && region.span <= base_span - region.offset;
}

/** ggml_cast uses CPY with src[1] == result; it is not a persistent mutation. */
static const ggml_tensor * write_destination(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_CPY:
            return node->src[1] != node ? node->src[1] : nullptr;
        case GGML_OP_SET_ROWS:
            return node->src[2];
        case GGML_OP_SCALE:
            return node->view_src != nullptr ? node->src[0] : nullptr;
        default:
            return nullptr;
    }
}

static bool valid_write_destination(const ggml_tensor * node, const ggml_tensor * storage) {
    const ggml_tensor * destination = write_destination(node);
    state_region result;
    state_region target;
    if (destination == nullptr || node->view_src == nullptr ||
            !tensor_region(node, result) || !tensor_region(destination, target) ||
            result.base != storage || target.base != storage ||
            result.offset != target.offset || result.span != target.span || node->type != destination->type) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (node->ne[d] != destination->ne[d] || node->nb[d] != destination->nb[d]) {
            return false;
        }
    }
    return true;
}

bool skippy_validate_state_effects(
        const std::vector<skippy_state_effect> & effects,
        const ggml_cgraph * graph,
        std::string & error) {
    if (graph == nullptr) {
        error = "state effects require the exact graph";
        return false;
    }

    std::unordered_map<const ggml_tensor *, int> graph_positions;
    for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph)); ++i) {
        if (!graph_positions.emplace(ggml_graph_node(const_cast<ggml_cgraph *>(graph), i), i).second) {
            error = "state effects require distinct exact graph nodes";
            return false;
        }
    }

    using owner = std::tuple<int32_t, int32_t, const ggml_tensor *>;
    std::set<owner> seen_reads;
    std::unordered_set<const ggml_tensor *> seen_write_nodes;
    std::unordered_map<const ggml_tensor *, std::set<owner>> storage_owners;
    std::set<owner> reads;
    std::map<owner, std::map<int64_t, int>> writes;

    for (size_t i = 0; i < effects.size(); ++i) {
        const skippy_state_effect & effect = effects[i];
        if (!state_kind_known(effect.kind) || !state_access_known(effect.access) ||
                !state_residency_known(effect.residency) || effect.layer < 0) {
            error = effect_error("state effect %zu has an unknown kind/access or invalid layer", i);
            return false;
        }
        state_region storage;
        if (!tensor_region(effect.storage, storage) || storage.base != effect.storage) {
            error = effect_error("state effect %zu has invalid or noncanonical storage", i);
            return false;
        }
        const owner key = {static_cast<int32_t>(effect.kind), effect.layer, effect.storage};
        // Architectures may share one KV cache tensor across several attention
        // layers (for example gemma3n's reused per-layer caches), so the same
        // state kind may own a storage from several layers; a kind change on
        // the same storage remains a conflict.
        const auto existing = storage_owners.find(effect.storage);
        if (existing != storage_owners.end()) {
            for (const auto & prior : existing->second) {
                if (std::get<0>(prior) != std::get<0>(key)) {
                    error = effect_error("state effect %zu conflicts with another storage owner", i);
                    return false;
                }
            }
        }
        storage_owners[effect.storage].insert(key);
        if (effect.access == skippy_state_access::read) {
            if (effect.effect_node != nullptr || effect.write_ordinal != -1 ||
                    !seen_reads.insert(key).second) {
                error = effect_error("state effect %zu has an invalid or duplicate read owner", i);
                return false;
            }
            reads.insert(key);
        } else {
            const auto position = graph_positions.find(effect.effect_node);
            if (position == graph_positions.end() || !(effect.effect_node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                error = effect_error("state effect %zu write node is not executable in the exact graph", i);
                return false;
            }
            if (!seen_write_nodes.insert(effect.effect_node).second) {
                error = effect_error("state effect %zu duplicates an exact write node", i);
                return false;
            }
            if (!valid_write_destination(effect.effect_node, effect.storage)) {
                error = effect_error("state effect %zu has an unsupported write or invalid destination", i);
                return false;
            }
            if (effect.write_ordinal < 0 || !writes[key].emplace(effect.write_ordinal, position->second).second) {
                error = effect_error("state effect %zu has a missing or duplicate write ordinal", i);
                return false;
            }
        }
    }

    for (const auto & owner_writes : writes) {
        if (reads.find(owner_writes.first) == reads.end()) {
            error = "state write has no matching read owner";
            return false;
        }
        int64_t next = 0;
        int previous_position = -1;
        for (const auto & write : owner_writes.second) {
            if (write.first != next++ || write.second <= previous_position) {
                error = "state write ordinals are not contiguous in executable graph order";
                return false;
            }
            previous_position = write.second;
        }
    }

    for (const auto & node_position : graph_positions) {
        const ggml_tensor * node = node_position.first;
        if (node->op != GGML_OP_CPY && node->op != GGML_OP_SET_ROWS && node->op != GGML_OP_SCALE) {
            continue;
        }
        const ggml_tensor * destination = write_destination(node);
        const ggml_tensor * storage = skippy_state_canonical_storage(destination);
        const ggml_tensor * result_storage = skippy_state_canonical_storage(node);
        if ((storage_owners.find(storage) != storage_owners.end() ||
                storage_owners.find(result_storage) != storage_owners.end()) &&
                (!valid_write_destination(node, storage) || seen_write_nodes.find(node) == seen_write_nodes.end())) {
            error = effect_error("graph node %zu mutates owned state without a valid write record", node_position.second);
            return false;
        }
    }
    return true;
}

bool skippy_infer_state_effects(
        const std::vector<skippy_state_effect> & explicit_effects,
        const ggml_cgraph * graph,
        const std::vector<skippy_block_boundary> & boundaries,
        std::vector<skippy_state_effect> & effects,
        std::string & error) {
    effects = explicit_effects;
    error.clear();
    if (graph == nullptr || boundaries.empty() || boundaries.size() % 2 != 0) {
        error = "state inference requires an exact graph and paired block boundaries";
        return false;
    }

    std::unordered_map<const ggml_tensor *, int32_t> positions;
    for (int32_t index = 0; index < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph)); ++index) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(graph), index);
        if (node == nullptr) {
            error = "state inference graph contains a null node";
            return false;
        }
        positions.emplace(node, index);
    }

    // A raw-embedding entry is a graph leaf, before every executable node.
    bool initial_entry_is_leaf = false;
    for (int32_t index = 0; index < graph->n_leafs; ++index) {
        initial_entry_is_leaf = initial_entry_is_leaf || graph->leafs[index] == boundaries.front().tensor;
    }

    struct explicit_owner {
        skippy_state_kind kind = skippy_state_kind::derived_persistent;
        int32_t layer = -1;
        int64_t next_write_ordinal = 0;
        int32_t last_write_position = -1;
        bool initialized = false;
        bool consistent = true;
    };
    std::unordered_map<const ggml_tensor *, explicit_owner> explicit_owners;
    std::vector<const ggml_tensor *> explicit_storage_order;
    std::unordered_set<const ggml_tensor *> explicit_write_nodes;
    for (const auto & effect : explicit_effects) {
        auto & owner = explicit_owners[effect.storage];
        if (!owner.initialized) {
            owner.kind = effect.kind;
            owner.layer = effect.layer;
            owner.initialized = true;
            explicit_storage_order.push_back(effect.storage);
        } else if (owner.kind != effect.kind || owner.layer != effect.layer) {
            owner.consistent = false;
        }
        if (effect.access == skippy_state_access::write) {
            explicit_write_nodes.insert(effect.effect_node);
            owner.next_write_ordinal = std::max(owner.next_write_ordinal, effect.write_ordinal + 1);
            const auto position = positions.find(effect.effect_node);
            if (position != positions.end()) {
                owner.last_write_position = std::max(owner.last_write_position, position->second);
            }
        }
    }

    struct inferred_owner {
        int32_t layer = -1;
        std::vector<const ggml_tensor *> writes;
    };
    std::unordered_map<const ggml_tensor *, inferred_owner> inferred;
    std::vector<const ggml_tensor *> storage_order;
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> supplemental_writes;

    for (int32_t index = 0; index < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph)); ++index) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(graph), index);
        if (node->op != GGML_OP_CPY && node->op != GGML_OP_SET_ROWS && node->op != GGML_OP_SCALE) {
            continue;
        }
        const ggml_tensor * storage = skippy_state_canonical_storage(write_destination(node));
        if (storage == nullptr || positions.find(storage) != positions.end()) {
            continue;
        }
        const auto explicit_owner_it = explicit_owners.find(storage);
        if (explicit_owner_it != explicit_owners.end()) {
            if (explicit_write_nodes.count(node) != 0) {
                continue;
            }
            if (!valid_write_destination(node, storage)) {
                error = effect_error("graph node %zu has an unsupported registered-state mutation", index);
                return false;
            }
            if (explicit_owner_it->second.consistent &&
                    index <= explicit_owner_it->second.last_write_position) {
                error = effect_error("graph node %zu is an unregistered state write before a registered write", index);
                return false;
            }
            supplemental_writes[storage].push_back(node);
            continue;
        }
        if (storage->buffer == nullptr) {
            continue;
        }
        if (!valid_write_destination(node, storage)) {
            error = effect_error("graph node %zu has an unsupported external-storage mutation", index);
            return false;
        }

        int32_t owner_layer = -1;
        for (size_t layer = 0; layer < boundaries.size() / 2; ++layer) {
            const auto entry = positions.find(boundaries[2 * layer].tensor);
            const auto exit = positions.find(boundaries[2 * layer + 1].tensor);
            const bool initial_leaf = layer == 0 && initial_entry_is_leaf && entry == positions.end();
            if ((!initial_leaf && entry == positions.end()) || exit == positions.end()) {
                error = "state inference could not locate a block boundary in the exact graph";
                return false;
            }
            const int32_t entry_position = initial_leaf ? -1 : entry->second;
            if (index >= entry_position && index <= exit->second) {
                owner_layer = static_cast<int32_t>(layer);
                break;
            }
        }
        if (owner_layer < 0) {
            error = effect_error("graph node %zu mutates external storage outside a block", index);
            return false;
        }
        const auto inserted = inferred.emplace(storage, inferred_owner{owner_layer, {}});
        if (inserted.second) {
            storage_order.push_back(storage);
        } else if (inserted.first->second.layer != owner_layer) {
            error = "inferred persistent storage spans multiple blocks";
            return false;
        }
        inserted.first->second.writes.push_back(node);
    }

    for (const ggml_tensor * storage : explicit_storage_order) {
        const auto supplemental = supplemental_writes.find(storage);
        if (supplemental == supplemental_writes.end()) {
            continue;
        }
        auto & owner = explicit_owners.at(storage);
        if (!owner.consistent) {
            continue;
        }
        for (const ggml_tensor * write : supplemental->second) {
            effects.push_back({owner.kind, skippy_state_access::write, owner.layer,
                    storage, write, owner.next_write_ordinal++,
                    skippy_state_residency::layer_local});
        }
    }

    for (const ggml_tensor * storage : storage_order) {
        const auto & owner = inferred.at(storage);
        effects.push_back({skippy_state_kind::derived_persistent,
                skippy_state_access::read, owner.layer, storage, nullptr, -1,
                skippy_state_residency::layer_local});
        for (size_t ordinal = 0; ordinal < owner.writes.size(); ++ordinal) {
            effects.push_back({skippy_state_kind::derived_persistent,
                    skippy_state_access::write, owner.layer, storage,
                    owner.writes[ordinal], static_cast<int64_t>(ordinal),
                    skippy_state_residency::layer_local});
        }
    }
    return skippy_validate_state_effects(effects, graph, error);
}
