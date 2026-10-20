#include "common.h"
#include "dequantize.h"
#include "turbo4.h"

kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const int64_t group_idx = tgpig * ntg + tiitg;
    const int64_t n_groups = args.n_elements / 128;
    if (group_idx >= n_groups) return;

    const device float * in = src + group_idx * 128;
    device float * out = dst + group_idx * 128;

    half4 v[32];
    const bool is_inverse = (args.direction == 1);

    for (int i = 0; i < 32; i++) {
        float4 f = float4(in[i*4], in[i*4+1], in[i*4+2], in[i*4+3]);
        half4 s = is_inverse ? turbo_wht_signs2_h4[i] : turbo_wht_signs1_h4[i];
        v[i] = half4(f) * s;
    }

    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i], b = v[partner];
                v[i]       = a + b;
                v[partner] = a - b;
            }
        }
    }

    const half4 inv_sqrt = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        half4 s = is_inverse ? turbo_wht_signs1_h4[i] : turbo_wht_signs2_h4[i];
        float4 f = float4(v[i] * inv_sqrt * s);
        out[i*4]   = f.x;
        out[i*4+1] = f.y;
        out[i*4+2] = f.z;
        out[i*4+3] = f.w;
    }
}

kernel void kernel_turbo_wht_inverse_simd(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (group >= args.n_elements/128) {
        return;
    }
    const uint64_t offset = uint64_t(group)*128 + uint64_t(lane)*4;
    half4 x = half4(float4(src[offset], src[offset + 1], src[offset + 2], src[offset + 3])) * turbo_wht_signs2_h4[lane];
    x = half4(x.x + x.y, x.x - x.y, x.z + x.w, x.z - x.w);
    x = half4(x.x + x.z, x.y + x.w, x.x - x.z, x.y - x.w);
    for (ushort step = 1; step < 32; step *= 2) {
        const half4 partner = half4(
            simd_shuffle_xor(float(x.x), step),
            simd_shuffle_xor(float(x.y), step),
            simd_shuffle_xor(float(x.z), step),
            simd_shuffle_xor(float(x.w), step));
        x = (lane & step) == 0 ? x + partner : partner - x;
    }
    const float4 out = float4(x * half4(0.08838834764831845h) * turbo_wht_signs1_h4[lane]);
    dst[offset] = out.x;
    dst[offset + 1] = out.y;
    dst[offset + 2] = out.z;
    dst[offset + 3] = out.w;
}

kernel void kernel_turbo_wht_inverse_f32(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const uint64_t group = uint64_t(tgpig)*ntg + tiitg;
    if (group >= args.n_elements/128) {
        return;
    }
    device const float * in = src + group*128;
    device       float * out = dst + group*128;
    float4 x[32];
    for (int i = 0; i < 32; ++i) {
        x[i] = float4(in[i*4], in[i*4 + 1], in[i*4 + 2], in[i*4 + 3]) * float4(turbo_wht_signs2_h4[i]);
    }
    for (int i = 0; i < 32; ++i) {
        const float4 a = x[i];
        x[i] = float4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    for (int i = 0; i < 32; ++i) {
        const float4 a = x[i];
        x[i] = float4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    for (int stride = 1; stride < 32; stride *= 2) {
        for (int i = 0; i < 32; ++i) {
            if (i % (2*stride) < stride) {
                const float4 a = x[i];
                const float4 b = x[i + stride];
                x[i] = a + b;
                x[i + stride] = a - b;
            }
        }
    }
    for (int i = 0; i < 32; ++i) {
        const float4 f = x[i] * 0.08838834764831845f * float4(turbo_wht_signs1_h4[i]);
        out[i*4] = f.x;
        out[i*4 + 1] = f.y;
        out[i*4 + 2] = f.z;
        out[i*4 + 3] = f.w;
    }
}

kernel void kernel_turbo_wht_inverse_simd_f32(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    if (group >= args.n_elements/128) {
        return;
    }
    const uint64_t offset = uint64_t(group)*128 + uint64_t(lane)*4;
    float4 x = float4(src[offset], src[offset + 1], src[offset + 2], src[offset + 3]) * float4(turbo_wht_signs2_h4[lane]);
    x = float4(x.x + x.y, x.x - x.y, x.z + x.w, x.z - x.w);
    x = float4(x.x + x.z, x.y + x.w, x.x - x.z, x.y - x.w);
    for (ushort step = 1; step < 32; step *= 2) {
        const float4 partner = float4(
            simd_shuffle_xor(x.x, step),
            simd_shuffle_xor(x.y, step),
            simd_shuffle_xor(x.z, step),
            simd_shuffle_xor(x.w, step));
        x = (lane & step) == 0 ? x + partner : partner - x;
    }
    const float4 out = x * 0.08838834764831845f * float4(turbo_wht_signs1_h4[lane]);
    dst[offset] = out.x;
    dst[offset + 1] = out.y;
    dst[offset + 2] = out.z;
    dst[offset + 3] = out.w;
}

template<typename TI>
kernel void kernel_set_rows_turbo4(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo4_0 * dst_row = (      device block_turbo4_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float           * src_row = (const device float         *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    const int n_blocks = args.nk0;

    for (int blk_idx = tiitg%tptg.x; blk_idx < n_blocks; blk_idx += tptg.x) {
        const device float * blk_src = src_row + QK_TURBO4 * blk_idx;
        device block_turbo4_0 & blk = dst_row[blk_idx];

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) norm_sq += blk_src[j] * blk_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        float normalized[128];
        for (int j = 0; j < 128; j++) {
            normalized[j] = blk_src[j] * inv_norm;
            x[j] = normalized[j];
        }

        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        for (int j = 0; j < QK_TURBO4 / 2; j++) blk.qs[j] = 0;

        float recon_norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) {
            float val = x[j];
            uint8_t idx;
            if      (val < turbo_mid_4bit[ 0]) idx = 0;
            else if (val < turbo_mid_4bit[ 1]) idx = 1;
            else if (val < turbo_mid_4bit[ 2]) idx = 2;
            else if (val < turbo_mid_4bit[ 3]) idx = 3;
            else if (val < turbo_mid_4bit[ 4]) idx = 4;
            else if (val < turbo_mid_4bit[ 5]) idx = 5;
            else if (val < turbo_mid_4bit[ 6]) idx = 6;
            else if (val < turbo_mid_4bit[ 7]) idx = 7;
            else if (val < turbo_mid_4bit[ 8]) idx = 8;
            else if (val < turbo_mid_4bit[ 9]) idx = 9;
            else if (val < turbo_mid_4bit[10]) idx = 10;
            else if (val < turbo_mid_4bit[11]) idx = 11;
            else if (val < turbo_mid_4bit[12]) idx = 12;
            else if (val < turbo_mid_4bit[13]) idx = 13;
            else if (val < turbo_mid_4bit[14]) idx = 14;
            else                               idx = 15;

            blk.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);

            float c = turbo_centroids_4bit[idx];
            recon_norm_sq += c * c;
        }

        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
    }
}

template<typename TI>
kernel void kernel_set_rows_turbo4_simd(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        ushort               lane[[thread_index_in_simdgroup]]) {
    const TI row = ((device const TI *) src1)[0];
    device block_turbo4_0 & blk = ((device block_turbo4_0 *) ((device char *) dst + row*args.nb1))[tgpig.x];
    device const float * src = (device const float *) src0 + 128*tgpig.x;
    threadgroup uint8_t indices[128];

    float grp_norm = 0.0f;
    float inv_norm = 0.0f;
    if (lane == 0) {
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) norm_sq += src[j] * src[j];
        grp_norm = sqrt(norm_sq);
        inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    }
    inv_norm = simd_shuffle(inv_norm, 0);

    float4 x;
    for (int j = 0; j < 4; ++j) {
        x[j] = src[4*lane + j] * inv_norm;
        x[j] *= turbo_wht_signs1[4*lane + j];
    }
    x = float4(x.x + x.y, x.x - x.y, x.z + x.w, x.z - x.w);
    x = float4(x.x + x.z, x.y + x.w, x.x - x.z, x.y - x.w);
    for (ushort h = 1; h < 32; h *= 2) {
        const float4 other = float4(simd_shuffle_xor(x.x, h), simd_shuffle_xor(x.y, h),
                                   simd_shuffle_xor(x.z, h), simd_shuffle_xor(x.w, h));
        x = (lane & h) == 0 ? x + other : other - x;
    }
    x *= 0.08838834764831845f;
    uint8_t code[4];
    for (int j = 0; j < 4; ++j) {
        const float val = x[j]*turbo_wht_signs2[4*lane + j];
        uint8_t idx;
        if      (val < turbo_mid_4bit[ 0]) idx = 0;
        else if (val < turbo_mid_4bit[ 1]) idx = 1;
        else if (val < turbo_mid_4bit[ 2]) idx = 2;
        else if (val < turbo_mid_4bit[ 3]) idx = 3;
        else if (val < turbo_mid_4bit[ 4]) idx = 4;
        else if (val < turbo_mid_4bit[ 5]) idx = 5;
        else if (val < turbo_mid_4bit[ 6]) idx = 6;
        else if (val < turbo_mid_4bit[ 7]) idx = 7;
        else if (val < turbo_mid_4bit[ 8]) idx = 8;
        else if (val < turbo_mid_4bit[ 9]) idx = 9;
        else if (val < turbo_mid_4bit[10]) idx = 10;
        else if (val < turbo_mid_4bit[11]) idx = 11;
        else if (val < turbo_mid_4bit[12]) idx = 12;
        else if (val < turbo_mid_4bit[13]) idx = 13;
        else if (val < turbo_mid_4bit[14]) idx = 14;
        else                               idx = 15;
        code[j] = idx;
        indices[4*lane + j] = idx;
    }
    blk.qs[2*lane]     = code[0] | (code[1] << 4);
    blk.qs[2*lane + 1] = code[2] | (code[3] << 4);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float recon_norm_sq = 0.0f;
        for (int j = 0; j < 128; ++j) {
            float c = turbo_centroids_4bit[indices[j]];
            recon_norm_sq += c * c;
        }
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
    }
}

typedef decltype(kernel_set_rows_turbo4_simd<int64_t>) set_rows_turbo4_simd_t;
template [[host_name("kernel_set_rows_f32_i64_turbo4_simd")]] kernel set_rows_turbo4_simd_t kernel_set_rows_turbo4_simd<int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo4_simd")]] kernel set_rows_turbo4_simd_t kernel_set_rows_turbo4_simd<int32_t>;

typedef decltype(kernel_set_rows_turbo4<int64_t>) set_rows_turbo4_t;
template [[host_name("kernel_set_rows_f32_i64_turbo4")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo4")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int32_t>;
