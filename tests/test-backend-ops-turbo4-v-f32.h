#pragma once

// One live token makes attention equal the public F32 dequantization of that V row.
// This isolates arithmetic rounding from lossy cache quantization and softmax error.
struct test_flash_attn_ext_turbo4_v_f32 : public test_flash_attn_ext {
    const bool live_last;
    std::vector<float> expected;

    test_flash_attn_ext_turbo4_v_f32(int head_repeat, int64_t kv, bool permuted, bool live_last)
        : test_flash_attn_ext(128, 128, 8, {head_repeat, 1}, kv, 1, true, false, 0, 0,
              GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0,
              permuted ? std::array<int32_t, 4>{0, 2, 1, 3} : std::array<int32_t, 4>{0, 1, 2, 3}),
          live_last(live_last) {}

    std::string vars() override {
        return "q8_turbo4_vf32_" + test_flash_attn_ext::vars() + ",live_last=" + (live_last ? "true" : "false");
    }

    bool run_whole_graph() override { return true; }
    double max_nmse_err() override { return 0.0; }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (strncmp(t->name, "sent_", 5) == 0) {
                init_tensor_uniform(t);
            } else {
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }

        auto * v = ggml_get_tensor(ctx, "v");
        const size_t row_bytes = ggml_row_size(type_V, 128);
        GGML_ASSERT(row_bytes == sizeof(ggml_fp16_t) + 64);
        const int64_t live = live_last ? kv - 1 : 0;
        // All norms are exactly representable as half; the products generally are not.
        static const float norms[8] = {0.0f, 1.0f, 1.0009765625f, 0.03125f, 0.333251953125f, 16.0f, 256.0f, 4096.0f};
        std::vector<uint8_t> packed(row_bytes);
        std::array<float, 128> row;
        expected.resize(size_t(128*nh*nr23[0]));
        for (int64_t h = 0; h < nh; ++h) {
            const ggml_fp16_t norm = ggml_fp32_to_fp16(norms[h]);
            memcpy(packed.data(), &norm, sizeof(norm));
            for (int64_t t = 0; t < kv; ++t) {
                for (int d = 0; d < 128; d += 2) {
                    const uint8_t lo = uint8_t((d + h*3 + t*5)%16);
                    const uint8_t hi = uint8_t((d + 1 + h*3 + t*5)%16);
                    packed[sizeof(norm) + d/2] = lo | (hi << 4);
                }
                ggml_backend_tensor_set(v, packed.data(), t*v->nb[1] + h*v->nb[2], row_bytes);
            }
            // Read the actual packed cache, not a copy of the initializer's intended values.
            ggml_backend_tensor_get(v, packed.data(), live*v->nb[1] + h*v->nb[2], row_bytes);
            ggml_get_type_traits(type_V)->to_float(packed.data(), row.data(), 128);
            for (int64_t r = 0; r < nr23[0]; ++r) {
                std::copy(row.begin(), row.end(), expected.begin() + (h*nr23[0] + r)*128);
            }
        }

        std::vector<ggml_fp16_t> mask_values(size_t(kv), ggml_fp32_to_fp16(-INFINITY));
        mask_values[size_t(live)] = ggml_fp32_to_fp16(0.0f);
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "m"), mask_values.data(), 0, mask_values.size()*sizeof(ggml_fp16_t));
        auto * out = ggml_get_tensor(ctx, "out");
        ggml_backend_tensor_memset(out, 0xff, 0, ggml_nbytes(out));
    }

    double err(const float * a, const float * b, size_t n) override {
        if (n != expected.size()) {
            return INFINITY;
        }
        // Exactly one weight is 1; all others are 0. No nontrivial summation is needed.
        // Four F32 epsilons allow float arithmetic variation while rejecting half rounding
        // (the smallest nonzero centroid-relative half error is more than 80e-6).
        for (size_t i = 0; i < n; ++i) {
            const double ref = expected[i];
            const double tolerance = 4.0*FLT_EPSILON*std::abs(ref);
            for (float actual : {a[i], b[i]}) {
                if (!std::isfinite(actual) || !std::isfinite(ref) || std::abs(double(actual) - ref) > tolerance) {
                    fprintf(stderr, "TQ_T4_V_F32 mismatch element=%zu expected=%.9g actual=%.9g tolerance=%.9g\n",
                            i, ref, double(actual), tolerance);
                    return INFINITY;
                }
            }
        }
        fprintf(stderr, "TQ_T4_V_F32 oracle=public_to_float elements=%zu passed\n", n);
        return 0.0;
    }
};
