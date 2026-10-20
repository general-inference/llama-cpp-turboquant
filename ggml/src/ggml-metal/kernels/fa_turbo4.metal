#include "common.h"
#include "dequantize.h"
#include "turbo4.h"
#include "fa_common.metal"

#define FA_TYPES \
    half, half4, simdgroup_half8x8, \
    half, half4x4, simdgroup_half8x8, \
    half, half4x4, simdgroup_half8x8, \
    float, simdgroup_float8x8, \
    float, float2, simdgroup_float8x8, \
    float, float4, simdgroup_float8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 128, 128>) flash_attn_turbo4_t;
template [[host_name("kernel_flash_attn_ext_kq8_0_vturbo4_dk128_dv128")]] kernel flash_attn_turbo4_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_turbo4_0, 8, dequantize_turbo4_0, 128, 128>;
