#pragma once

struct test_flash_attn_ext_f16_mixed : public test_flash_attn_ext {
    using test_flash_attn_ext::test_flash_attn_ext;
    bool support_ok = true;

    std::string vars() override {
        return "f16_mixed_ne_" + test_flash_attn_ext::vars();
    }

    void initialize_tensors(ggml_context * ctx) override {
        test_flash_attn_ext::initialize_tensors(ctx);
        auto * q = ggml_get_tensor(ctx, "q");
        auto * buffer = q->view_src ? q->view_src->buffer : q->buffer;
        auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
        support_ok = true;
        if (!dev || strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "MTL") != 0) {
            return;
        }
        struct probe {
            ggml_type k;
            ggml_type v;
            int64_t dk;
            int64_t dv;
        };
        const probe unsupported[] = {
            {GGML_TYPE_F16, GGML_TYPE_Q8_0,      64,  64},
            {GGML_TYPE_F16, GGML_TYPE_Q8_0,     256, 256},
            {GGML_TYPE_F16, GGML_TYPE_TURBO4_0, 256, 256},
            {GGML_TYPE_F16, GGML_TYPE_TURBO4_0, 128, 256},
            {GGML_TYPE_F16, GGML_TYPE_TURBO3_0, 128, 128},
            {GGML_TYPE_F16, GGML_TYPE_TURBO2_0, 128, 128},
            {GGML_TYPE_Q8_0, GGML_TYPE_F16,     128, 128},
            {GGML_TYPE_F32, GGML_TYPE_Q8_0,     128, 128},
        };
        ggml_init_params params = {40*ggml_tensor_overhead(), nullptr, true};
        ggml_context_ptr probe_ctx(ggml_init(params));
        GGML_ASSERT(probe_ctx);
        int rejected = 0;
        for (const auto & p : unsupported) {
            auto * query = ::ggml_new_tensor_4d(probe_ctx.get(), GGML_TYPE_F32, p.dk, 1, 32, 1);
            auto * key = ::ggml_new_tensor_4d(probe_ctx.get(), p.k, p.dk, 128, 8, 1);
            auto * value = ::ggml_new_tensor_4d(probe_ctx.get(), p.v, p.dv, 128, 8, 1);
            auto * op = ggml_flash_attn_ext(probe_ctx.get(), query, key, value, nullptr, 1.0f/sqrtf(p.dk), 0, 0);
            if (ggml_backend_dev_supports_op(dev, op)) {
                fprintf(stderr, "F16 mixed attention incorrectly accepts K=%s V=%s dk=%lld dv=%lld\n",
                        ggml_type_name(p.k), ggml_type_name(p.v), (long long) p.dk, (long long) p.dv);
                support_ok = false;
            } else {
                ++rejected;
            }
        }
        fprintf(stderr, "TQ_F16_MIXED_SUPPORT rejected=%d/8\n", rejected);
    }

    double err(const float * a, const float * b, size_t n) override {
        return support_ok ? test_flash_attn_ext::err(a, b, n) : INFINITY;
    }
};

struct test_flash_attn_ext_f16_mixed_gqa : public test_flash_attn_ext_turbo4_gqa {
    test_flash_attn_ext_f16_mixed_gqa(ggml_type type_v, int64_t kv, bool permuted, mask_pattern pattern,
                                    bool opposed_queries, int head_repeat)
        : test_flash_attn_ext_turbo4_gqa(kv, permuted, pattern, opposed_queries, head_repeat, GGML_TYPE_F16, type_v) {}

    std::string vars() override {
        const char * mask_name = pattern == mask_pattern::NONE ? "none" :
                                pattern == mask_pattern::PARTIAL ? "partial" : "all";
        return "f16_mixed_gqa_" + test_flash_attn_ext::vars() + ",mask_pattern=" + mask_name +
               ",opposed_queries=" + (opposed_queries ? "true" : "false");
    }
};

static void add_f16_mixed_attention_tests(std::vector<std::unique_ptr<test_case>> & cases) {
    for (auto type_v : {GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0}) {
        for (int repeat : {1, 2, 4}) {
            for (auto shape : {std::array<int64_t, 2>{113, 1}, {5120, 1}, {113, 32}, {512, 32}}) {
                for (bool permuted : {false, true}) {
                    for (bool features : {false, true}) {
                        cases.emplace_back(new test_flash_attn_ext_f16_mixed(
                            128, 128, 8, {repeat, 1}, shape[0], shape[1], features, features,
                            features ? 1.0f : 0.0f, features ? 10.0f : 0.0f,
                            GGML_PREC_F32, GGML_TYPE_F16, type_v,
                            permuted ? std::array<int32_t, 4>{0, 2, 1, 3} : std::array<int32_t, 4>{0, 1, 2, 3}));
                    }
                }
            }
        }
        using pattern = test_flash_attn_ext_turbo4_gqa::mask_pattern;
        for (int repeat : {2, 4}) {
            for (int64_t kv : {113, 5120}) {
                for (bool permuted : {false, true}) {
                    for (auto mask : {pattern::NONE, pattern::PARTIAL, pattern::ALL}) {
                        cases.emplace_back(new test_flash_attn_ext_f16_mixed_gqa(type_v, kv, permuted, mask, false, repeat));
                    }
                }
            }
            for (bool permuted : {false, true}) {
                for (auto mask : {pattern::NONE, pattern::PARTIAL}) {
                    cases.emplace_back(new test_flash_attn_ext_f16_mixed_gqa(type_v, 113, permuted, mask, true, repeat));
                }
            }
        }
    }
}
