#pragma once

#include "common.cuh"

#if !defined(GGML_USE_MUSA)

struct ggml_cuda_cachegen_tile {
    uint32_t payload_offset;
    uint32_t prefix_offset;
    uint32_t token_offset;
    uint32_t rows;
};

cudaError_t ggml_cuda_cachegen_decode_launch(
        const uint8_t * payload,
        const ggml_cuda_cachegen_tile * tiles,
        const uint32_t * prefixes,
        const uint32_t * cells,
        uint8_t * dst,
        uint32_t channels,
        uint32_t tile_count,
        uint64_t token_stride,
        uint64_t channel_stride,
        uint32_t element_bytes,
        cudaStream_t stream);

#endif // !defined(GGML_USE_MUSA)
