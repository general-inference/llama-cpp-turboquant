#pragma once

// Stage-independent persistent-state access/effect records for the
// executable graph.
//
// Generic graph helpers register every persistent-state read and ordered
// write they establish, keyed by semantic state kind and layer ordinal and
// owned by the exact graph result that built them. Records carry no stage
// range, endpoint ownership, family policy, tensor-name rule, or split
// decision. Storage identity is the canonical backing tensor: views are
// resolved through view_src before recording.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;
struct skippy_block_boundary;

enum class skippy_state_kind : int32_t {
    kv_key         = 0,
    kv_value       = 1,
    recurrent_conv = 2,
    recurrent_ssm  = 3,
    derived_persistent = 4,
};

enum class skippy_state_access : int32_t {
    read  = 0,
    write = 1,
};

enum class skippy_state_residency : int32_t {
    layer_local = 0,
    per_stage   = 1,
};

struct skippy_state_effect {
    skippy_state_kind kind = skippy_state_kind::kv_key;
    skippy_state_access access = skippy_state_access::read;
    int32_t layer = -1;
    // canonical backing storage tensor of the persistent state
    const ggml_tensor * storage = nullptr;
    // for writes: the graph node that establishes the ordered state update;
    // null for reads, which declare ownership rather than a read-event order
    const ggml_tensor * effect_node = nullptr;
    /** Per-owner write order: zero-based, contiguous; -1 only for read declarations. */
    int64_t write_ordinal = -1;
    skippy_state_residency residency = skippy_state_residency::layer_local;
};

// Resolve a possibly-view tensor to its canonical backing storage.
const ggml_tensor * skippy_state_canonical_storage(const ggml_tensor * tensor);

/** Validate exact-graph state owners, mutation destinations and per-owner write order.
 * Known CPY, SET_ROWS and in-place SCALE mutations of owned storage must be registered,
 * including zero-sized writes. Destination views must fit their canonical allocation.
 * SET_ROWS index payload bounds and universal read/write hazards are not certified here.
 */
bool skippy_validate_state_effects(
        const std::vector<skippy_state_effect> & effects,
        const ggml_cgraph * graph,
        std::string & error);

// Infer ownership for external buffered storage mutated by recognized GGML
// operations. Existing semantic annotations win; inferred owners are assigned
// to the enclosing block from exact executable order.
bool skippy_infer_state_effects(
        const std::vector<skippy_state_effect> & explicit_effects,
        const ggml_cgraph * graph,
        const std::vector<skippy_block_boundary> & boundaries,
        std::vector<skippy_state_effect> & effects,
        std::string & error);
