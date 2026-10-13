#include "common.h"

struct cachegen_tile {
    uint payload_offset;
    uint prefix_offset;
    uint token_offset;
    uint rows;
};

struct cachegen_params {
    uint  channels;
    uint  tile_count;
    uint  element_bytes;
    ulong token_stride;
    ulong channel_stride;
};

static inline uint cachegen_read_bit(const device uchar * stream, uint stream_bytes, thread uint & bit) {
    const uint byte_index = bit >> 3;
    const uint value      = byte_index < stream_bytes ? stream[byte_index] : 0;
    const uint result     = (value >> (7 - (bit & 7))) & 1;
    ++bit;
    return result;
}

static inline ushort cachegen_scaled_count(uint value, uint low, ulong span) {
    const ulong delta     = ulong(value) - ulong(low) + 1;
    const ulong numerator = delta * 65536ul - 1;
    uint count = min(uint(float(delta) * (65536.0f / float(span))), 65535u);
    ulong product = ulong(count) * span;
    while (product > numerator) {
        --count;
        product -= span;
    }
    while (count < 65535u && product + span <= numerator) {
        ++count;
        product += span;
    }
    return ushort(count);
}

static inline uint cachegen_find_symbol(const device ushort * cdf, ushort count) {
    uint first = 0;
    uint last  = 33;
    while (first < last) {
        const uint middle = (first + last) >> 1;
        if (cdf[middle] <= count) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    return first - 1;
}

kernel void kernel_cachegen_decode(const device uchar *         payload [[buffer(0)]],
                                   const device cachegen_tile * tiles [[buffer(1)]],
                                   const device uint *          prefixes [[buffer(2)]],
                                   const device uint *          cells [[buffer(3)]],
                                   device uchar *               dst [[buffer(4)]],
                                   constant cachegen_params &   params [[buffer(5)]],
                                   uint2                        gid [[thread_position_in_grid]],
                                   uint                         tid [[thread_index_in_threadgroup]]) {
    threadgroup float quant_values[32];
    const uint channel    = gid.x;
    const uint tile_index = gid.y;
    if (channel >= params.channels || tile_index >= params.tile_count) {
        return;
    }

    const cachegen_tile   tile    = tiles[tile_index];
    const device uchar *  segment = payload + tile.payload_offset;
    const uint            bins    = segment[4];
    const bool            packed  = segment[3] == '2';
    const device float *  maxes   = reinterpret_cast<const device float *>(segment + 16);
    const device ushort * cdf     = nullptr;
    const device uchar *  stream  = nullptr;
    uint                  stream_bytes = 0;
    if (packed) {
        stream       = segment + 16 + tile.rows * sizeof(float);
        stream_bytes = uint(segment[12]) | uint(segment[13]) << 8 |
                       uint(segment[14]) << 16 | uint(segment[15]) << 24;
    } else {
        const device ushort * cdfs =
            reinterpret_cast<const device ushort *>(segment + 16 + tile.rows * sizeof(float));
        cdf = cdfs + channel * 33;
        const uint prefix_index   = tile.prefix_offset + channel;
        const uint stream_start   = prefixes[prefix_index];
        stream_bytes              = prefixes[prefix_index + 1] - stream_start;
        const uint lengths_offset = 16 + tile.rows * sizeof(float) + params.channels * 33 * sizeof(ushort);
        const uint streams_offset = lengths_offset + params.channels * sizeof(ushort);
        stream = segment + streams_offset + stream_start;
    }

    uint bit   = 0;
    uint value = 0;
    if (!packed) {
        for (uint i = 0; i < 32; ++i) {
            value = (value << 1) | cachegen_read_bit(stream, stream_bytes, bit);
        }
    }
    uint        low    = 0;
    uint        high   = 0xffffffffu;
    const float center = float(bins / 2 - 1);
    for (uint row = 0; row < tile.rows; ++row) {
        const ulong span = ulong(high) - ulong(low) + 1;
        uint symbol;
        if (packed) {
            const uint bits_per_symbol = segment[5];
            const ulong symbol_bit = (ulong(row) * params.channels + channel) * bits_per_symbol;
            const ulong byte_index = symbol_bit >> 3;
            const uint shift = symbol_bit & 7;
            uint word = stream[byte_index];
            if (shift + bits_per_symbol > 8) {
                word |= uint(stream[byte_index + 1]) << 8;
            }
            symbol = (word >> shift) & ((1u << bits_per_symbol) - 1);
        } else {
            const ushort count = cachegen_scaled_count(value, low, span);
            symbol = cachegen_find_symbol(cdf, count);
        }
        if (symbol >= 32) {
            return;
        }

        const float normalized = (float(symbol) - center) / center;
        const half  decoded    = half(normalized * maxes[row]);
        const float decoded_f  = float(decoded);
        const ulong row_offset = ulong(cells[tile.token_offset + row]) * params.token_stride;
        if (params.element_bytes == 34) {
            const uint  lane        = channel & 31;
            const ulong destination = row_offset + ulong(channel >> 5) * params.channel_stride;
            const float amax        = simd_max(abs(decoded_f));
            const float scale       = amax / 127.0f;
            if (lane == 0) {
                *reinterpret_cast<device half *>(dst + destination) = half(scale);
            }
            const int quantized = int(round(decoded_f * (scale == 0.0f ? 0.0f : 1.0f / scale)));
            *(reinterpret_cast<device char *>(dst + destination + 2) + lane) =
                char(clamp(quantized, -127, 127));
        } else if (params.element_bytes == 18) {
            const uint  lane        = channel & 31;
            const ulong destination = row_offset + ulong(channel >> 5) * params.channel_stride;
            const float amax        = simd_max(abs(decoded_f));
            const uint  winner      = simd_min(select(32u, lane, abs(decoded_f) == amax));
            const float signed_max  = simd_sum(select(0.0f, decoded_f, lane == winner));
            const float scale       = signed_max / -8.0f;
            quant_values[tid] = decoded_f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                *reinterpret_cast<device half *>(dst + destination) = half(scale);
                const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
                for (uint index = 0; index < 16; ++index) {
                    const int low = clamp(
                        int(trunc(quant_values[index] * inverse + 8.5f)), 0, 15);
                    const int high = clamp(
                        int(trunc(quant_values[index + 16] * inverse + 8.5f)), 0, 15);
                    *(dst + destination + 2 + index) = uchar(low | (high << 4));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else if (params.element_bytes == sizeof(float)) {
            const ulong destination = row_offset + ulong(channel) * params.channel_stride;
            *reinterpret_cast<device float *>(dst + destination) = float(decoded);
        } else {
            const ulong destination = row_offset + ulong(channel) * params.channel_stride;
            *reinterpret_cast<device half *>(dst + destination) = decoded;
        }

        if (row + 1 == tile.rows) {
            break;
        }
        if (packed) {
            continue;
        }
        const ulong cdf_low  = cdf[symbol];
        const ulong cdf_high = symbol == 31 ? 65536ul : cdf[symbol + 1];
        high                 = low - 1 + uint((span * cdf_high) >> 16);
        low                  = low + uint((span * cdf_low) >> 16);
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
