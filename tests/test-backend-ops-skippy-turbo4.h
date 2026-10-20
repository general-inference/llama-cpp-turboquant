#pragma once

// Fixed seeded rows are written through each tensor's actual cache strides.
struct test_skippy_turbo4_attention : public test_flash_attn_ext {
    using test_flash_attn_ext::test_flash_attn_ext;
    bool support_ok = true;

    std::string vars() override { return "skippy_turbo4_" + test_flash_attn_ext::vars(); }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (strncmp(t->name, "sent_", 5) == 0) {
                init_tensor_uniform(t);
            } else {
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }
        for (const char * name : {"q", "k", "v"}) {
            auto * t = ggml_get_tensor(ctx, name);
            std::vector<float> row(t->ne[0]);
            std::vector<uint8_t> packed(ggml_row_size(t->type, t->ne[0]));
            for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                        for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                            const uint32_t x = uint32_t(i0)*2246822519u + uint32_t(i1)*3266489917u +
                                uint32_t(i2)*668265263u + uint32_t(i3)*374761393u + uint32_t(name[0]);
                            row[i0] = float(int((x ^ (x >> 16)) % 1025) - 512)/512.0f;
                        }
                        const size_t size = ggml_quantize_chunk(t->type, row.data(), packed.data(), 0, 1, t->ne[0], nullptr);
                        GGML_ASSERT(size == packed.size());
                        ggml_backend_tensor_set(t, packed.data(), i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], size);
                    }
                }
            }
        }
        if (auto * m = ggml_get_tensor(ctx, "m")) {
            if (n_kv_max > 0) {
                GGML_ASSERT(m->ne[0] == 4*n_kv_max);
                std::vector<ggml_fp16_t> row(m->ne[0]);
                for (int64_t q = 0; q < m->ne[1]; ++q) {
                    for (int64_t i = 0; i < m->ne[0]; ++i) {
                        row[i] = ggml_fp32_to_fp16(i % 4 == q % 4 ? 0.0f : -INFINITY);
                    }
                    ggml_backend_tensor_set(m, row.data(), q*m->nb[1], row.size()*sizeof(ggml_fp16_t));
                }
            } else {
                init_tensor_kq_mask(m);
            }
        }
        if (auto * s = ggml_get_tensor(ctx, "s")) {
            std::vector<float> values(ggml_nelements(s));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = float(int(i % 9) - 4);
            }
            ggml_backend_tensor_set(s, values.data(), 0, values.size()*sizeof(float));
        }
        if (type_K == GGML_TYPE_F16) {
            auto * q = ggml_get_tensor(ctx, "q");
            auto * buffer = q->view_src ? q->view_src->buffer : q->buffer;
            check_f16_support(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)));
        }
    }

    double err(const float * a, const float * b, size_t n) override {
        return support_ok ? test_flash_attn_ext::err(a, b, n) : INFINITY;
    }

private:
    void check_f16_support(ggml_backend_dev_t dev) {
        support_ok = true;
        GGML_ASSERT(dev);
        const char * backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        const bool metal = strcmp(backend, "MTL") == 0;
        ggml_init_params params = {32*ggml_tensor_overhead(), nullptr, true};
        ggml_context_ptr ctx(ggml_init(params));
        GGML_ASSERT(ctx);
        const auto attention = [&](ggml_type key, int64_t dk, int64_t dv) {
            auto * q = ::ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dk, 1, 32, 1);
            auto * k = ::ggml_new_tensor_4d(ctx.get(), key, dk, 128, 8, 1);
            auto * v = ::ggml_new_tensor_4d(ctx.get(), GGML_TYPE_TURBO4_0, dv, 128, 8, 1);
            return ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 1.0f/sqrtf(dk), 0, 0);
        };
        int rejected = 0;
        for (auto key : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_0}) {
            if (ggml_backend_dev_supports_op(dev, attention(key, 128, 128))) {
                support_ok = false;
            } else {
                ++rejected;
            }
        }
        if (metal) {
            for (auto shape : {std::array<int64_t, 2>{256, 256}, {128, 256}, {256, 128}}) {
                if (ggml_backend_dev_supports_op(dev, attention(GGML_TYPE_F16, shape[0], shape[1]))) {
                    support_ok = false;
                } else {
                    ++rejected;
                }
            }
        }
        auto * valid = attention(GGML_TYPE_F16, 128, 128);
        int rejected_backends = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * other = ggml_backend_dev_get(i);
            const char * name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(other));
            if (strcmp(name, "CPU") == 0 || strcmp(name, "MTL") == 0) {
                continue;
            }
            if (ggml_backend_dev_supports_op(other, valid)) {
                support_ok = false;
            } else {
                ++rejected_backends;
            }
        }
        fprintf(stderr, "SKIPPY_F16_TURBO4_SUPPORT backend=%s rejected=%d/%d other_backends=%d ok=%d\n",
                backend, rejected, metal ? 7 : 4, rejected_backends, int(support_ok));
    }
};

static void add_skippy_turbo4_attention_tests(std::vector<std::unique_ptr<test_case>> & cases,
                                            ggml_type key_type = GGML_TYPE_Q8_0) {
    for (auto k : {key_type}) {
        for (int64_t gqa : {2, 4}) {
            for (auto permutation : {std::array<int32_t, 4>{0, 1, 2, 3}, std::array<int32_t, 4>{0, 2, 1, 3}}) {
                for (int64_t queries : {1, 2, 19, 32}) {
                    for (int64_t kv : {113, 512, 5632}) {
                        for (bool features : {false, true}) {
                            cases.emplace_back(new test_skippy_turbo4_attention(128, 128, 8, {gqa, 1}, kv, queries,
                                true, features, features ? 4.0f : 0.0f, features ? 10.0f : 0.0f,
                                GGML_PREC_F32, k, GGML_TYPE_TURBO4_0, permutation));
                        }
                    }
                }
                cases.emplace_back(new test_skippy_turbo4_attention(128, 128, 8, {gqa, 1}, 512, 512,
                    true, false, 0, 0, GGML_PREC_F32, k, GGML_TYPE_TURBO4_0, permutation));
                for (int64_t queries : {1, 32}) {
                    cases.emplace_back(new test_skippy_turbo4_attention(128, 128, 8, {gqa, 1}, 512, queries,
                        true, false, 0, 0, GGML_PREC_F32, k, GGML_TYPE_TURBO4_0, permutation, true, false, 128));
                }
            }
        }
    }
}

static void add_skippy_f16_turbo4_attention_tests(std::vector<std::unique_ptr<test_case>> & cases) {
    add_skippy_turbo4_attention_tests(cases, GGML_TYPE_F16);
    for (int gqa : {2, 4}) {
        for (int64_t kv : {113, 512, 5632}) {
            for (bool permuted : {false, true}) {
                for (auto pattern : {test_flash_attn_ext_turbo4_gqa::mask_pattern::NONE,
                                     test_flash_attn_ext_turbo4_gqa::mask_pattern::PARTIAL,
                                     test_flash_attn_ext_turbo4_gqa::mask_pattern::ALL}) {
                    for (bool opposed : {false, true}) {
                        cases.emplace_back(new test_flash_attn_ext_turbo4_gqa(kv, permuted, pattern, opposed, gqa, GGML_TYPE_F16));
                    }
                }
            }
        }
    }
}
