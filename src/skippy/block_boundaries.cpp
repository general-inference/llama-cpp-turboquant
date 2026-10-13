#include "block_boundaries.h"

#include "ggml.h"

#include <cinttypes>
#include <cstdio>

static std::string boundary_error(const char * fmt, int32_t layer) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), fmt, layer);
    return buf;
}

bool skippy_validate_block_boundaries(
        const std::vector<skippy_block_boundary> & boundaries,
        int32_t layer_start,
        int32_t layer_end,
        std::string & error) {
    if (layer_start < 0 || layer_end < layer_start) {
        error = "invalid declared layer range";
        return false;
    }

    const size_t expected = 2 * static_cast<size_t>(layer_end - layer_start);
    if (boundaries.size() != expected) {
        error = "boundary record count does not match declared layer range";
        return false;
    }

    const ggml_tensor * prev_exit = nullptr;
    size_t index = 0;
    for (int32_t layer = layer_start; layer < layer_end; ++layer) {
        const skippy_block_boundary & entry = boundaries[index++];
        const skippy_block_boundary & exit  = boundaries[index++];

        if (entry.kind != skippy_block_boundary_kind::entry) {
            error = boundary_error("layer %" PRId32 " record is not an ordered entry", layer);
            return false;
        }
        if (exit.kind != skippy_block_boundary_kind::exit) {
            error = boundary_error("layer %" PRId32 " record is not an ordered exit", layer);
            return false;
        }
        if (entry.layer != layer || exit.layer != layer) {
            error = boundary_error("layer %" PRId32 " boundary pair is missing, duplicated, or unordered", layer);
            return false;
        }
        if (entry.tensor == nullptr) {
            error = boundary_error("layer %" PRId32 " entry tensor is null", layer);
            return false;
        }
        if (exit.tensor == nullptr) {
            error = boundary_error("layer %" PRId32 " exit tensor is null", layer);
            return false;
        }
        if (prev_exit != nullptr && entry.tensor != prev_exit) {
            error = boundary_error("layer %" PRId32 " entry does not chain from the previous exit", layer);
            return false;
        }
        prev_exit = exit.tensor;
    }

    return true;
}
