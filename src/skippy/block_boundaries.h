#pragma once

// Stage-independent block boundary records for the executable graph.
//
// Model graph builders register a canonical entry and exit tensor for every
// transformer or recurrent block, keyed by the block's layer ordinal. The
// records are owned by the exact graph result that built them and carry no
// stage range, endpoint ownership, family policy, tensor-name rule, or split
// decision. Boundaries are validated structurally, never inferred from node
// names.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

enum class skippy_block_boundary_kind : int32_t {
    entry = 0,
    exit  = 1,
};

struct skippy_block_boundary {
    skippy_block_boundary_kind kind = skippy_block_boundary_kind::entry;
    int32_t layer = -1;
    const ggml_tensor * tensor = nullptr;
};

// Validate the boundary records registered while building one graph profile.
//
// Fails closed: returns false and sets `error` when any declared layer in
// [layer_start, layer_end) is missing a record, any record is duplicated,
// records are unordered or interleaved, an entry/exit pair is mismatched, a
// record lies outside the declared range, or a canonical tensor is null.
// Consecutive blocks must chain: the entry tensor of layer N+1 must be the
// exit tensor of layer N, so a registered frontier cannot silently diverge
// from the values that actually cross it.
bool skippy_validate_block_boundaries(
        const std::vector<skippy_block_boundary> & boundaries,
        int32_t layer_start,
        int32_t layer_end,
        std::string & error);
