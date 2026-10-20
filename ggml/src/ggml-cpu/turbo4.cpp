#include "ops.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <cmath>

// WHT sign arrays (must match Metal shader turbo_wht_signs1/2)
static const float turbo_wht_s1[128] = {-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
static const float turbo_wht_s2[128] = {1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};

static void ggml_compute_forward_turbo_wht_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const float * src_data = (const float *) src->data;
    float * dst_data = (float *) dst->data;
    const int direction = ggml_get_op_params_i32(dst, 0);
    constexpr int group_size = 128;
    GGML_ASSERT(ggml_get_op_params_i32(dst, 1) == group_size);
    GGML_ASSERT(dst->src[1] == nullptr && src->ne[0] % group_size == 0);
    GGML_ASSERT(direction == 0 || direction == 1);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    const int64_t n_groups = ggml_nelements(src) / group_size;
    const float inv_sqrt = 1.0f / sqrtf(float(group_size));

    // Parallel over groups
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;
    const int64_t grp_start = (n_groups * ith) / nth;
    const int64_t grp_end = (n_groups * (ith + 1)) / nth;

    const float * s_first = (direction == 0) ? turbo_wht_s1 : turbo_wht_s2;
    const float * s_second = (direction == 0) ? turbo_wht_s2 : turbo_wht_s1;

    for (int64_t g = grp_start; g < grp_end; g++) {
        const int64_t base = g * group_size;
        float x[group_size];
        const float * in = src_data + base;
        for (int i = 0; i < group_size; i++) x[i] = in[i] * s_first[i];

        // WHT butterfly (log2(group_size) stages)
        for (int h = 1; h < group_size; h *= 2) {
            for (int i = 0; i < group_size; i += h * 2) {
                for (int j = i; j < i + h; j++) {
                    float a = x[j], b = x[j + h];
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
            }
        }

        // Normalize + second signs
        float * out = dst_data + base;
        for (int i = 0; i < group_size; i++) {
            out[i] = x[i] * inv_sqrt * s_second[i];
        }
    }
}

void ggml_compute_forward_turbo_wht(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32: ggml_compute_forward_turbo_wht_f32(params, dst); break;
        default: GGML_ABORT("fatal error");
    }
}
