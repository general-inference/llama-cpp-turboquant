#ifndef SKIPPY_ACTIVATION_H
#define SKIPPY_ACTIVATION_H

/**
 * @file skippy/activation.h
 * @brief Describes activation-frame payloads exchanged between stages.
 *
 * The descriptor carries a typed directory for every tensor crossing a stage
 * frontier. Parts are matched by planner identity rather than model family.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SKIPPY_ACTIVATION_FRAME_VERSION 2
#define SKIPPY_ACTIVATION_BOUNDARY_DESC_VERSION 2
#define SKIPPY_ACTIVATION_IDENTITY_BYTES 32
#define SKIPPY_ACTIVATION_MAX_DIMS 4
#define SKIPPY_ACTIVATION_MAX_PARTS 16

#define SKIPPY_ACTIVATION_PART_OPTIONAL (UINT32_C(1) << 0)

/** @brief One typed tensor in a stage-frontier frame. */
struct skippy_activation_part_desc {
    uint8_t identity[SKIPPY_ACTIVATION_IDENTITY_BYTES];
    uint32_t ggml_type;
    uint32_t rank;
    int32_t token_axis;
    uint32_t flags;
    int64_t dimensions[SKIPPY_ACTIVATION_MAX_DIMS];
    uint64_t byte_strides[SKIPPY_ACTIVATION_MAX_DIMS];
    uint64_t payload_offset;
    uint64_t payload_bytes;
};

/** @brief Stable, graph-observed multipart contract at one stage edge. */
struct skippy_activation_boundary_desc {
    uint32_t version;
    uint32_t part_count;
    uint8_t frontier_identity[SKIPPY_ACTIVATION_IDENTITY_BYTES];
    struct skippy_activation_part_desc parts[SKIPPY_ACTIVATION_MAX_PARTS];
};

/** @brief Metadata for an activation payload passed between staged runtimes. */
struct skippy_activation_desc {
    uint32_t version;
    int32_t producer_stage_index;
    int32_t layer_start;
    int32_t layer_end;
    uint32_t token_count;
    uint32_t sequence_count;
    uint32_t part_count;
    uint32_t reserved;
    uint64_t payload_bytes;
    uint8_t frontier_identity[SKIPPY_ACTIVATION_IDENTITY_BYTES];
    struct skippy_activation_part_desc parts[SKIPPY_ACTIVATION_MAX_PARTS];
};

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_ACTIVATION_H
