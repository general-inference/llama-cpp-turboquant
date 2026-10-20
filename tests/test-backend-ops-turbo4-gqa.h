#pragma once

// Included after test_flash_attn_ext in test-backend-ops.cpp.
struct test_flash_attn_ext_turbo4_gqa : public test_flash_attn_ext {
    enum class mask_pattern { NONE, PARTIAL, ALL };

    const mask_pattern pattern;
    const bool opposed_queries;
    const int head_repeat;

    test_flash_attn_ext_turbo4_gqa(int64_t kv, bool permuted, mask_pattern pattern, bool opposed_queries = false, int head_repeat = 2)
        : test_flash_attn_ext(128, 128, 8, {head_repeat, 1}, kv, 1, pattern != mask_pattern::NONE, false, 0, 0,
              GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0,
              permuted ? std::array<int32_t, 4>{0, 2, 1, 3} : std::array<int32_t, 4>{0, 1, 2, 3}),
          pattern(pattern), opposed_queries(opposed_queries), head_repeat(head_repeat) {
        GGML_ASSERT(head_repeat == 2 || head_repeat == 4);
    }

    std::string vars() override {
        const char * mask_name = pattern == mask_pattern::NONE ? "none" :
                                pattern == mask_pattern::PARTIAL ? "partial" : "all";
        const std::string prefix = head_repeat == 4 ? "q8_turbo4_paired_gqa4_" : "q8_turbo4_paired_";
        return prefix + test_flash_attn_ext::vars() + ",mask_pattern=" + mask_name +
               ",opposed_queries=" + (opposed_queries ? "true" : "false");
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "FLASH_ATTN_EXT";
    }

    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * grouped = test_flash_attn_ext::build_graph(ctx);
        ggml_set_name(grouped, head_repeat == 4 ? "gqa4" : "gqa2");

        ggml_tensor * k_copy = ggml_new_tensor_4d(ctx, type_K, hsk, kv, head_repeat*nh, 1);
        ggml_tensor * v_copy = ggml_new_tensor_4d(ctx, type_V, hsv, kv, head_repeat*nh, 1);
        ggml_set_name(k_copy, "k_gqa1");
        ggml_set_name(v_copy, "v_gqa1");

        // GQA1 uses the same query and duplicated packed KV, bypassing paired-head kernels.
        ggml_tensor * separate = ggml_flash_attn_ext(ctx, grouped->src[0], k_copy, v_copy, grouped->src[3],
                                                   1.0f/sqrtf(hsk), 0, 0);
        ggml_flash_attn_ext_set_prec(separate, GGML_PREC_F32);
        ggml_set_name(separate, "gqa1");

        ggml_tensor * out = ggml_concat(ctx, grouped, separate, 1);
        ggml_set_name(out, "gqa_comparison");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (strncmp(t->name, "sent_", 5) == 0) {
                init_tensor_uniform(t);
            } else {
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }

        initialize_queries(ggml_get_tensor(ctx, "q"));
        initialize_cache(ggml_get_tensor(ctx, "k"), ggml_get_tensor(ctx, "k_gqa1"), true);
        initialize_cache(ggml_get_tensor(ctx, "v"), ggml_get_tensor(ctx, "v_gqa1"), false);
        if (mask) {
            initialize_mask(ggml_get_tensor(ctx, "m"));
        }
        // All-masked cases must write zeros rather than inherit the initial buffer contents.
        for (const char * name : {head_repeat == 4 ? "gqa4" : "gqa2", "gqa1", "gqa_comparison"}) {
            ggml_tensor * t = ggml_get_tensor(ctx, name);
            ggml_backend_tensor_memset(t, 0xff, 0, ggml_nbytes(t));
        }
    }

    double err(const float * a, const float * b, size_t n) override {
        const size_t head_size = size_t(hsv);
        const size_t half_size = head_size*head_repeat*nh;
        if (n != 2*half_size) {
            return INFINITY;
        }

        if (pattern == mask_pattern::ALL) {
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || a[i] != 0.0f || b[i] != 0.0f) {
                    return INFINITY;
                }
            }
            return 0.0;
        }

        double result = 0.0;
        for (size_t i = 0; i < half_size; i += head_size) {
            // Compare each head separately so another head cannot hide a failure.
            result = std::max(result, finite_nmse(b + i, a + i, head_size));
            result = std::max(result, finite_nmse(b + half_size + i, a + half_size + i, head_size));
            result = std::max(result, finite_nmse(a + half_size + i, a + i, head_size));
            result = std::max(result, finite_nmse(b + half_size + i, b + i, head_size));
        }
        return result;
    }

private:
    static double finite_nmse(const float * reference, const float * actual, size_t n) {
        double error = 0.0;
        double energy = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
                return INFINITY;
            }
            const double delta = double(reference[i]) - actual[i];
            error += delta*delta;
            energy += double(reference[i])*reference[i];
        }
        return energy == 0.0 ? (error == 0.0 ? 0.0 : INFINITY) : error/energy;
    }

    void initialize_queries(ggml_tensor * q) const {
        std::array<float, 128> row;
        for (int64_t h = 0; h < head_repeat*nh; ++h) {
            for (int d = 0; d < 128; ++d) {
                if (opposed_queries) {
                    // The second GQA4 pair has distinct scores and less concentrated weights.
                    const float magnitude = head_repeat == 4 && h % 4 >= 2 ? 0.125f : 1.5f;
                    row[d] = (h % 2 == 0 ? magnitude : -magnitude)*(d % 2 == 0 ? 1.0f : -1.0f);
                } else {
                    row[d] = float((d*5 + h*13) % 29 - 14)/16.0f;
                }
            }
            ggml_backend_tensor_set(q, row.data(), h*q->nb[2], sizeof(row));
        }
    }

    void initialize_cache(ggml_tensor * cache, ggml_tensor * copy, bool is_key) const {
        std::array<float, 128> row;
        std::vector<uint8_t> packed(ggml_row_size(cache->type, 128));
        for (int64_t h = 0; h < nh; ++h) {
            for (int64_t t = 0; t < kv; ++t) {
                for (int d = 0; d < 128; ++d) {
                    if (is_key && opposed_queries) {
                        row[d] = (t % 2 == 0 ? 0.5f : -0.5f)*(d % 2 == 0 ? 1.0f : -1.0f);
                    } else if (is_key) {
                        row[d] = float((d*7 + t*3 + h*11) % 37 - 18)/32.0f;
                    } else {
                        row[d] = float((d*11 + t*17 + h*3) % 41 - 20)/32.0f;
                    }
                }
                const size_t written = ggml_quantize_chunk(cache->type, row.data(), packed.data(), 0, 1, 128, nullptr);
                GGML_ASSERT(written == packed.size());
                ggml_backend_tensor_set(cache, packed.data(), t*cache->nb[1] + h*cache->nb[2], packed.size());
                for (int repeat = 0; repeat < head_repeat; ++repeat) {
                    ggml_backend_tensor_set(copy, packed.data(), t*copy->nb[1] + (head_repeat*h + repeat)*copy->nb[2], packed.size());
                }
            }
        }
    }

    void initialize_mask(ggml_tensor * m) const {
        std::vector<float> values(kv, 0.0f);
        for (int64_t t = 0; t < kv; ++t) {
            // Include whole skipped tiles, isolated masked positions, and a live padded tail.
            if (pattern == mask_pattern::ALL || (t != kv - 1 && ((t/32) % 3 == 0 || t % 11 == 5))) {
                values[t] = -INFINITY;
            } else {
                values[t] = -float(t % 7)/8.0f;
            }
        }
        std::vector<ggml_fp16_t> packed(kv);
        ggml_fp32_to_fp16_row(values.data(), packed.data(), kv);
        ggml_backend_tensor_set(m, packed.data(), 0, packed.size()*sizeof(ggml_fp16_t));
    }
};
