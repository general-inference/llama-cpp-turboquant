#pragma once

// One selected token makes each output exactly the decoded packed value.
struct test_turbo4_centroids : public test_case {
    static constexpr int64_t width = 128;
    static constexpr int64_t kv_heads = 8;
    const ggml_type key_type;
    const int gqa;
    const int64_t kv;
    const bool permuted;
    const int norm_set;
    ggml_tensor * values = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<uint8_t> packed_before;
    bool target_metal = false;
    bool target_supported = false;

    test_turbo4_centroids(ggml_type key_type, int gqa, int64_t kv, bool permuted, int norm_set)
        : key_type(key_type), gqa(gqa), kv(kv), permuted(permuted), norm_set(norm_set) {}

    std::string vars() override { return "t4_centroid_" + VARS_TO_STR5(key_type, gqa, kv, permuted, norm_set); }
    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "FLASH_ATTN_EXT";
    }
    bool run_whole_graph() override { return true; }
    double max_nmse_err() override { return 0.0; }
    std::vector<ggml_tensor *> fusion_test_nodes() override {
        auto nodes = sentinels;
        nodes.push_back(output);
        return nodes;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto * keys = tensor(ctx, key_type, kv, kv_heads);
        values = tensor(ctx, GGML_TYPE_TURBO4_0, kv, kv_heads);
        ggml_set_name(keys, "centroid_k");
        ggml_set_name(values, "centroid_v");
        auto * decode = attention(ctx, keys, 1, "centroid_decode");
        auto * prefill = attention(ctx, keys, 32, "centroid_prefill");
        output = ggml_concat(ctx, decode, prefill, 2);
        ggml_set_name(output, "centroid_comparison");
        return output;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (strncmp(t->name, "sent_", 5) == 0) {
                init_tensor_uniform(t);
            } else {
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }
        auto * buffer = values->view_src ? values->view_src->buffer : values->buffer;
        auto * buft = ggml_backend_buffer_get_type(buffer);
        auto * device = ggml_backend_buft_get_device(buft);
        if (!device && buft == ggml_backend_cpu_buffer_type()) {
            device = ggml_backend_dev_by_name("CPU");
        }
        GGML_ASSERT(device);
        const char * backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        target_metal = strcmp(backend, "MTL") == 0;
        target_supported = target_metal || strcmp(backend, "CPU") == 0;
        GGML_ASSERT(ggml_row_size(GGML_TYPE_TURBO4_0, width) == 66);
        for (int64_t h = 0; h < kv_heads; ++h) {
            for (int64_t t = 0; t < kv; ++t) {
                std::array<uint8_t, 66> packed = {};
                const ggml_fp16_t norm = norm_bits(h);
                memcpy(packed.data(), &norm, sizeof(norm));
                for (int d = 0; d < width; ++d) {
                    packed[2 + d/2] |= uint8_t(code(h, t, d) << (4*(d%2)));
                }
                ggml_backend_tensor_set(values, packed.data(), h*values->nb[2] + t*values->nb[1], packed.size());
            }
        }
        for (const char * name : {"centroid_mask_1", "centroid_mask_32"}) {
            auto * mask = ggml_get_tensor(ctx, name);
            std::vector<ggml_fp16_t> row(kv, ggml_fp32_to_fp16(-INFINITY));
            row[kv - 1] = ggml_fp32_to_fp16(0.0f);
            for (int64_t q = 0; q < mask->ne[1]; ++q) {
                ggml_backend_tensor_set(mask, row.data(), q*mask->nb[1], row.size()*sizeof(ggml_fp16_t));
            }
        }
        packed_before.resize(ggml_nbytes(values));
        ggml_backend_tensor_get(values, packed_before.data(), 0, packed_before.size());
        uint64_t hash = UINT64_C(14695981039346656037);
        for (uint8_t byte : packed_before) {
            hash = (hash ^ byte)*UINT64_C(1099511628211);
        }
        for (const char * name : {"centroid_decode", "centroid_prefill", "centroid_comparison"}) {
            auto * t = ggml_get_tensor(ctx, name);
            ggml_backend_tensor_memset(t, 0xff, 0, ggml_nbytes(t));
        }
        GGML_ASSERT(canonical_half(1, ggml_fp16_t(0x3c01)) == -0.18310546875f);
        GGML_ASSERT(old_half(1, ggml_fp16_t(0x3c01)) == -0.1829833984375f);
        fprintf(stderr, "TQ_T4_CENTROID_INPUT backend=%s key=%s gqa=%d kv=%lld permuted=%d norm_set=%d bytes=%zu fnv1a64=%016llx\n",
                backend, ggml_type_name(key_type), gqa, (long long) kv, int(permuted), norm_set,
                packed_before.size(), (unsigned long long) hash);
    }

    double err(const float * a, const float * b, size_t n) override {
        if (n == sentinel_size) {
            return memcmp(a, b, n*sizeof(float)) == 0 ? 0.0 : INFINITY;
        }
        if (!target_supported || n != size_t(33*width*kv_heads*gqa)) {
            return INFINITY;
        }
        std::vector<uint8_t> packed_after(packed_before.size());
        ggml_backend_tensor_get(values, packed_after.data(), 0, packed_after.size());
        if (packed_after != packed_before) {
            fprintf(stderr, "TQ_T4_CENTROID_PACKED_CHANGED\n");
            return INFINITY;
        }
        size_t mismatches[2] = {}, boundary[2] = {}, old_matches[2] = {};
        size_t cpu_mismatches = 0;
        for (int q = 0; q < 33; ++q) {
            const int phase = q == 0 ? 0 : 1;
            for (int h = 0; h < kv_heads*gqa; ++h) {
                const auto norm = norm_bits(h/gqa);
                for (int d = 0; d < width; ++d) {
                    const int c = code(h/gqa, kv - 1, d);
                    const size_t i = (size_t(q)*kv_heads*gqa + h)*width + d;
                    const float full = centroids()[c]*ggml_fp16_to_fp32(norm);
                    const float half = canonical_half(c, norm);
                    const float expected = target_metal ? half : full;
                    if (!std::isfinite(b[i]) || b[i] != full) {
                        ++cpu_mismatches;
                    }
                    if (!std::isfinite(a[i]) || a[i] != expected) {
                        ++mismatches[phase];
                        const bool edge = (norm > 0 && norm < 0x0400) || (half != 0.0f && fabsf(half) < 0x1p-14f);
                        boundary[phase] += edge;
                        old_matches[phase] += target_metal && a[i] == old_half(c, norm);
                        if (mismatches[phase] == 1) {
                            fprintf(stderr, "TQ_T4_CENTROID_MISMATCH phase=%s head=%d dim=%d norm_bits=%04x code=%d actual=%.9g expected=%.9g boundary=%d\n",
                                    phase == 0 ? "decode" : "prefill", h, d, unsigned(norm), c, a[i], expected, int(edge));
                        }
                    }
                }
            }
        }
        fprintf(stderr, "TQ_T4_CENTROID_ORACLE decode=%zu prefill=%zu boundary_decode=%zu boundary_prefill=%zu old_lut_decode=%zu old_lut_prefill=%zu cpu_float=%zu packed_unchanged=1\n",
                mismatches[0], mismatches[1], boundary[0], boundary[1], old_matches[0], old_matches[1], cpu_mismatches);
        return mismatches[0] == 0 && mismatches[1] == 0 && cpu_mismatches == 0 ? 0.0 : INFINITY;
    }

private:
    ggml_tensor * tensor(ggml_context * ctx, ggml_type type, int64_t rows, int64_t heads) const {
        if (!permuted) {
            return ggml_new_tensor_4d(ctx, type, width, rows, heads, 1);
        }
        return ggml_permute(ctx, ggml_new_tensor_4d(ctx, type, width, heads, rows, 1), 0, 2, 1, 3);
    }

    ggml_tensor * attention(ggml_context * ctx, ggml_tensor * keys, int64_t queries, const char * name) const {
        auto * q = tensor(ctx, GGML_TYPE_F32, queries, kv_heads*gqa);
        auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, queries, 1, 1);
        ggml_set_name(mask, queries == 1 ? "centroid_mask_1" : "centroid_mask_32");
        auto * result = ggml_flash_attn_ext(ctx, q, keys, values, mask, 1.0f/sqrtf(width), 0.0f, 0.0f);
        ggml_prec_set_acc(result, GGML_PREC_F32);
        ggml_set_name(result, name);
        return result;
    }

    static int code(int64_t head, int64_t token, int d) { return int((d + 3*head + 5*token)%16); }

    ggml_fp16_t norm_bits(int64_t head) const {
        // The second set isolates half-subnormal boundaries without weakening exact comparisons.
        static const ggml_fp16_t norms[16] = {
            0x0000, 0x3c00, 0x3c01, 0x3bff, 0x3555, 0x3c02, 0x4201, 0x7bff,
            0x0001, 0x0002, 0x03ff, 0x0400, 0x0401, 0x1000, 0x1401, 0x2e66,
        };
        GGML_ASSERT(norm_set >= 0 && norm_set < 2 && head >= 0 && head < kv_heads);
        return norms[norm_set*kv_heads + head];
    }

    static const float * centroids() {
        static const float values[16] = {
            -0.241529f, -0.182877f, -0.143016f, -0.111036f,
            -0.083292f, -0.058050f, -0.034299f, -0.011349f,
             0.011349f,  0.034299f,  0.058050f,  0.083292f,
             0.111036f,  0.143016f,  0.182877f,  0.241529f,
        };
        return values;
    }

    static float canonical_half(int code, ggml_fp16_t norm) {
        return ggml_fp16_to_fp32(ggml_fp32_to_fp16(centroids()[code]*ggml_fp16_to_fp32(norm)));
    }

    static float old_half(int code, ggml_fp16_t norm) {
        const float rounded = ggml_fp16_to_fp32(ggml_fp32_to_fp16(centroids()[code]));
        return ggml_fp16_to_fp32(ggml_fp32_to_fp16(rounded*ggml_fp16_to_fp32(norm)));
    }
};

static void add_turbo4_centroid_tests(std::vector<std::unique_ptr<test_case>> & cases) {
    for (auto key : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        for (int gqa : {1, 2, 4}) {
            for (int64_t kv : {113, 512}) {
                for (bool permuted : {false, true}) {
                    for (int norm_set : {0, 1}) {
                        cases.emplace_back(new test_turbo4_centroids(key, gqa, kv, permuted, norm_set));
                    }
                }
            }
        }
    }
}
