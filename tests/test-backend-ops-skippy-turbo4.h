#pragma once

// Fixed seeded rows are written through each tensor's actual cache strides.
struct test_skippy_turbo4_attention : public test_flash_attn_ext {
    using test_flash_attn_ext::test_flash_attn_ext;

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
    }
};

static void add_skippy_turbo4_attention_tests(std::vector<std::unique_ptr<test_case>> & cases) {
    for (auto k : {GGML_TYPE_Q8_0}) {
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
