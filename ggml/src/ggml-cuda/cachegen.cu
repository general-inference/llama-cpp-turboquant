#include "cachegen.cuh"

#if !defined(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
#define GGML_CACHEGEN_ACTIVE_MASK() __ballot(1)
#else
#define GGML_CACHEGEN_ACTIVE_MASK() __activemask()
#endif

static __device__ __forceinline__ uint32_t cachegen_read_bit(
        const uint8_t * stream, uint32_t stream_bytes, uint32_t & bit) {
    const uint32_t byte_index = bit >> 3;
    const uint32_t value      = byte_index < stream_bytes ? stream[byte_index] : 0;
    const uint32_t result     = (value >> (7 - (bit & 7))) & 1;
    ++bit;
    return result;
}

static __device__ __forceinline__ uint16_t cachegen_scaled_count(
        uint32_t value, uint32_t low, uint64_t span) {
    const uint64_t delta     = uint64_t(value) - uint64_t(low) + 1;
    const uint64_t numerator = delta * 65536ull - 1;
    uint32_t count = min(uint32_t(float(delta) * (65536.0f / float(span))), 65535u);
    uint64_t product = uint64_t(count) * span;
    while (product > numerator) {
        --count;
        product -= span;
    }
    while (count < 65535u && product + span <= numerator) {
        ++count;
        product += span;
    }
    return uint16_t(count);
}

static __device__ __forceinline__ uint32_t cachegen_find_symbol(
        const uint16_t * cdf, uint16_t count) {
    uint32_t first = 0;
    uint32_t last  = 33;
    while (first < last) {
        const uint32_t middle = (first + last) >> 1;
        if (cdf[middle] <= count) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    return first - 1;
}

static __global__ void cachegen_decode(
        const uint8_t * payload,
        const ggml_cuda_cachegen_tile * tiles,
        const uint32_t * prefixes,
        const uint32_t * cells,
        uint8_t * dst,
        uint32_t channels,
        uint32_t tile_count,
        uint64_t token_stride,
        uint64_t channel_stride,
        uint32_t element_bytes) {
    const uint32_t channel    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tile_index = blockIdx.y;
    if (channel >= channels || tile_index >= tile_count) {
        return;
    }

    const ggml_cuda_cachegen_tile tile = tiles[tile_index];
    const uint8_t * segment            = payload + tile.payload_offset;
    const uint32_t bins                = segment[4];
    const bool packed                  = segment[3] == '2';
    const float * maxes                = reinterpret_cast<const float *>(segment + 16);
    const uint16_t * cdf = nullptr;
    const uint8_t * stream = nullptr;
    uint32_t stream_bytes = 0;
    if (packed) {
        stream = segment + 16 + tile.rows * sizeof(float);
        stream_bytes = uint32_t(segment[12]) | uint32_t(segment[13]) << 8 |
            uint32_t(segment[14]) << 16 | uint32_t(segment[15]) << 24;
    } else {
        const uint16_t * cdfs = reinterpret_cast<const uint16_t *>(segment + 16 + tile.rows * sizeof(float));
        cdf = cdfs + channel * 33;
        const uint32_t prefix_index = tile.prefix_offset + channel;
        const uint32_t stream_start = prefixes[prefix_index];
        stream_bytes = prefixes[prefix_index + 1] - stream_start;
        const uint32_t lengths_offset = 16 + tile.rows * sizeof(float) + channels * 33 * sizeof(uint16_t);
        const uint32_t streams_offset = lengths_offset + channels * sizeof(uint16_t);
        stream = segment + streams_offset + stream_start;
    }

    uint32_t bit   = 0;
    uint32_t value = 0;
    if (!packed) {
        for (uint32_t i = 0; i < 32; ++i) {
            value = (value << 1) | cachegen_read_bit(stream, stream_bytes, bit);
        }
    }
    uint32_t low    = 0;
    uint32_t high   = 0xffffffffu;
    const float center = float(bins / 2 - 1);
    for (uint32_t row = 0; row < tile.rows; ++row) {
        const uint64_t span = uint64_t(high) - uint64_t(low) + 1;
        uint32_t symbol;
        if (packed) {
            const uint32_t bits_per_symbol = segment[5];
            const uint64_t symbol_bit = (uint64_t(row) * channels + channel) * bits_per_symbol;
            const uint64_t byte_index = symbol_bit >> 3;
            const uint32_t shift = symbol_bit & 7;
            uint32_t word = stream[byte_index];
            if (shift + bits_per_symbol > 8) {
                word |= uint32_t(stream[byte_index + 1]) << 8;
            }
            symbol = (word >> shift) & ((1u << bits_per_symbol) - 1);
        } else {
            const uint16_t count = cachegen_scaled_count(value, low, span);
            symbol = cachegen_find_symbol(cdf, count);
        }
        if (symbol >= 32) {
            return;
        }

        const float normalized = (float(symbol) - center) / center;
        const half decoded     = __float2half(normalized * maxes[row]);
        const float decoded_f       = __half2float(decoded);
        const uint64_t row_offset   = uint64_t(cells[tile.token_offset + row]) * token_stride;
        if (element_bytes == 34) {
            const uint32_t lane        = channel & 31;
            const uint64_t destination = row_offset + uint64_t(channel >> 5) * channel_stride;
            float amax = fabsf(decoded_f);
#pragma unroll
            for (uint32_t mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(GGML_CACHEGEN_ACTIVE_MASK(), amax, mask, 32));
            }
            const float scale = amax / 127.0f;
            if (lane == 0) {
                *reinterpret_cast<half *>(dst + destination) = __float2half(scale);
            }
            const int quantized = int(roundf(decoded_f * (scale == 0.0f ? 0.0f : 1.0f / scale)));
            *(reinterpret_cast<int8_t *>(dst + destination + 2) + lane) = int8_t(max(-127, min(127, quantized)));
        } else if (element_bytes == 18) {
            const uint32_t lane        = channel & 31;
            const uint64_t destination = row_offset + uint64_t(channel >> 5) * channel_stride;
            float amax = fabsf(decoded_f);
#pragma unroll
            for (uint32_t mask = 16; mask > 0; mask >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(GGML_CACHEGEN_ACTIVE_MASK(), amax, mask, 32));
            }
            uint32_t winner = fabsf(decoded_f) == amax ? lane : 32;
#pragma unroll
            for (uint32_t mask = 16; mask > 0; mask >>= 1) {
                winner = min(winner, __shfl_xor_sync(GGML_CACHEGEN_ACTIVE_MASK(), winner, mask, 32));
            }
            const float signed_max = __shfl_sync(GGML_CACHEGEN_ACTIVE_MASK(), decoded_f, winner, 32);
            const float scale      = signed_max / -8.0f;
            if (lane == 0) {
                *reinterpret_cast<half *>(dst + destination) = __float2half(scale);
            }
            const float high_value = __shfl_down_sync(GGML_CACHEGEN_ACTIVE_MASK(), decoded_f, 16, 32);
            if (lane < 16) {
                const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
                const int low = max(0, min(15, int(decoded_f * inverse + 8.5f)));
                const int high = max(0, min(15, int(high_value * inverse + 8.5f)));
                *(dst + destination + 2 + lane) = uint8_t(low | (high << 4));
            }
        } else if (element_bytes == sizeof(float)) {
            const uint64_t destination = row_offset + uint64_t(channel) * channel_stride;
            *reinterpret_cast<float *>(dst + destination) = __half2float(decoded);
        } else {
            const uint64_t destination = row_offset + uint64_t(channel) * channel_stride;
            *reinterpret_cast<half *>(dst + destination) = decoded;
        }

        if (row + 1 == tile.rows) {
            break;
        }
        if (packed) {
            continue;
        }
        const uint64_t cdf_low  = cdf[symbol];
        const uint64_t cdf_high = symbol == 31 ? 65536ull : cdf[symbol + 1];
        high = low - 1 + uint32_t((span * cdf_high) >> 16);
        low  = low + uint32_t((span * cdf_low) >> 16);
        while (true) {
            if (low >= 0x80000000u || high < 0x80000000u) {
                low <<= 1;
                high  = (high << 1) | 1;
                value = (value << 1) | cachegen_read_bit(stream, stream_bytes, bit);
            } else if (low >= 0x40000000u && high < 0xc0000000u) {
                low  = (low << 1) & 0x7fffffffu;
                high = (high << 1) | 0x80000001u;
                value -= 0x40000000u;
                value = (value << 1) | cachegen_read_bit(stream, stream_bytes, bit);
            } else {
                break;
            }
        }
    }
}

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
        cudaStream_t stream) {
    constexpr uint32_t block_size = 64;
    const dim3 block(block_size, 1, 1);
    const dim3 grid((channels + block_size - 1) / block_size, tile_count, 1);
    cachegen_decode<<<grid, block, 0, stream>>>(
        payload, tiles, prefixes, cells, dst, channels, tile_count, token_stride, channel_stride, element_bytes);
    return cudaGetLastError();
}

#undef GGML_CACHEGEN_ACTIVE_MASK

#endif // !defined(GGML_USE_MUSA)
