#pragma once

#include "common.h"
#include "dequantize.h"

constant float turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
constant float turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

constant half4 turbo_wht_signs1_h4[32] = {
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h)
};
constant half4 turbo_wht_signs2_h4[32] = {
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h)
};

static void turbo_fwht_128(thread float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

static void turbo_rotate_forward(thread float * x, constant float * s1, constant float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

constant float turbo_centroids_4bit[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};
constant float turbo_mid_4bit[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f,
    -0.070671f, -0.046174f, -0.022824f,  0.000000f,
     0.022824f,  0.046174f,  0.070671f,  0.097164f,
     0.127026f,  0.162947f,  0.212203f
};

template <typename type4x4>
void dequantize_turbo4_0(device const block_turbo4_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    const int base = il * 16;
    float4x4 reg_f;

    for (int g = 0; g < 4; g++) {
        for (int k = 0; k < 4; k++) {
            const int j = base + g * 4 + k;
            uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            reg_f[g][k] = turbo_centroids_4bit[idx] * norm;
        }
    }
    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_turbo4_0_t4(device const block_turbo4_0 * xb, short il, thread type4 & reg) {
    const float norm = float(xb->norm);
    const device uint8_t * qs = xb->qs + il * 2;
    const uint8_t qb0 = qs[0];
    const uint8_t qb1 = qs[1];

    reg = type4(float4(
        turbo_centroids_4bit[(qb0     ) & 0xF] * norm,
        turbo_centroids_4bit[(qb0 >> 4) & 0xF] * norm,
        turbo_centroids_4bit[(qb1     ) & 0xF] * norm,
        turbo_centroids_4bit[(qb1 >> 4) & 0xF] * norm
    ));
}
