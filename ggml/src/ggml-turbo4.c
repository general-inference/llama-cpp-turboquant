#include "ggml-quants.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

// Turbo4 KV codec extracted from ebae20eb; packed values stay in the rotated domain.
static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};


static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.212203f) return 0;
    if (val < -0.162947f) return 1;
    if (val < -0.127026f) return 2;
    if (val < -0.097164f) return 3;
    if (val < -0.070671f) return 4;
    if (val < -0.046174f) return 5;
    if (val < -0.022824f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.022824f) return 8;
    if (val <  0.046174f) return 9;
    if (val <  0.070671f) return 10;
    if (val <  0.097164f) return 11;
    if (val <  0.127026f) return 12;
    if (val <  0.162947f) return 13;
    if (val <  0.212203f) return 14;
    return 15;
}

static void turbo_cpu_fwht(float * x) {
    const int group_size = 128;
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = 0.08838834764831845f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[QK_TURBO4];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation */
        float rotated[QK_TURBO4];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht(rotated);

        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.241529f, -0.182877f, -0.143016f, -0.111036f,
            -0.083292f, -0.058050f, -0.034299f, -0.011349f,
             0.011349f,  0.034299f,  0.058050f,  0.083292f,
             0.111036f,  0.143016f,  0.182877f,  0.241529f
        };
        uint8_t indices[QK_TURBO4];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Minimize reconstruction error for the selected codeword. */
        float recon_norm_sq = 0.0f;
        float projection = 0.0f;
        for (int i = 0; i < d; i++) {
            const float c = CENTROIDS_4BIT[indices[i]];
            recon_norm_sq += c * c;
            projection += rotated[i] * c;
        }
        float corrected_norm = (recon_norm_sq > 1e-20f) ? norm * (projection / recon_norm_sq) : 0.0f;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);

        /* Pack */

        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    /* Decode in the rotated domain; attention applies the inverse WHT. */
    static const float CENTROIDS_4BIT[16] = {
        -0.241529f, -0.182877f, -0.143016f, -0.111036f,
        -0.083292f, -0.058050f, -0.034299f, -0.011349f,
         0.011349f,  0.034299f,  0.058050f,  0.083292f,
         0.111036f,  0.143016f,  0.182877f,  0.241529f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
