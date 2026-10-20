#pragma once

// Synthetic hot-buffer attention diagnostics, not model activations or a quality benchmark.
// Included after test_flash_attn_ext in test-backend-ops.cpp.
struct test_flash_attn_ext_decode_diagnostic : public test_flash_attn_ext {
    static constexpr int64_t cache_capacity = 8192;
    const int64_t valid_kv;

    test_flash_attn_ext_decode_diagnostic(int head_repeat, ggml_type type_k, ggml_type type_v,
                                         int64_t valid_kv, int64_t extent)
        : test_flash_attn_ext(128, 128, 8, {head_repeat, 1}, extent, 1, true, false, 0, 0,
              GGML_PREC_F32, type_k, type_v, {0, 2, 1, 3}, true), valid_kv(valid_kv) {
        GGML_ASSERT(valid_kv > 0 && valid_kv <= extent && extent <= cache_capacity);
        GGML_ASSERT(extent == GGML_PAD(valid_kv, 256));
    }

    std::string vars() override {
        return "attention_decode_diag_" + test_flash_attn_ext::vars() + ",valid_kv=" +
               std::to_string(valid_kv) + ",capacity=" + std::to_string(cache_capacity) +
               ",extent_basis=source_inferred,generator=mix32_v1";
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Match llama_kv_cache::get_k/get_v followed by build_attn_mha's permutation.
        ggml_tensor * q_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, nh*nr23[0], 1, 1);
        ggml_tensor * q = ggml_permute(ctx, q_storage, 0, 2, 1, 3);
        ggml_set_name(q, "q");
        ggml_tensor * k = create_cache(ctx, type_K, "k");
        ggml_tensor * v = create_cache(ctx, type_V, "v");
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, 1, 1, 1);
        ggml_set_name(m, "m");

        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(hsk), 0, 0);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_flash_attn_ext_set_n_kv_max(out, 0);
        ggml_set_name(out, "out");
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
        initialize_rows(ggml_get_tensor(ctx, "q"), 0x243f6a88u);
        initialize_rows(ggml_get_tensor(ctx, "k"), 0x85a308d3u);
        initialize_rows(ggml_get_tensor(ctx, "v"), 0x13198a2eu);
        initialize_mask(ggml_get_tensor(ctx, "m"));
    }

private:
    struct input_identity {
        uint64_t encoded = UINT64_C(14695981039346656037);
        uint64_t source_f32 = UINT64_C(14695981039346656037);
        size_t encoded_bytes = 0;
        size_t source_bytes = 0;

        static void update(uint64_t & hash, const void * data, size_t size) {
            const auto * bytes = static_cast<const uint8_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                hash = (hash ^ bytes[i])*UINT64_C(1099511628211);
            }
        }

        void add(const void * packed, size_t packed_size, const float * source, size_t source_size) {
            update(encoded, packed, packed_size);
            update(source_f32, source, source_size);
            encoded_bytes += packed_size;
            source_bytes += source_size;
        }
    };

    ggml_tensor * create_cache(ggml_context * ctx, ggml_type type, const char * name) {
        ggml_tensor * storage = ggml_new_tensor_4d(ctx, type, hsk, nh, cache_capacity, 1);
        ggml_tensor * view = ggml_view_4d(ctx, storage, hsk, nh, kv, 1,
                                         storage->nb[1], storage->nb[2], storage->nb[3], 0);
        ggml_tensor * cache = ggml_permute(ctx, view, 0, 2, 1, 3);
        ggml_set_name(cache, name);
        GGML_ASSERT(cache->nb[1] == ggml_row_size(type, hsk*nh));
        GGML_ASSERT(cache->nb[2] == ggml_row_size(type, hsk));
        GGML_ASSERT(cache->nb[3] == ggml_row_size(type, hsk*nh*cache_capacity));
        return cache;
    }

    static float source_value(uint32_t salt, int64_t head, int64_t token, int64_t dim) {
        // Fixed integer mixing and dyadic values avoid library-dependent random distributions.
        uint32_t x = salt ^ uint32_t(head)*0x9e3779b9u ^ uint32_t(token)*0x85ebca6bu ^ uint32_t(dim)*0xc2b2ae35u;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return float(int32_t(x & 2047u) - 1024)/1024.0f;
    }

    void initialize_rows(ggml_tensor * t, uint32_t salt) {
        GGML_ASSERT(t->ne[0] == 128 && t->ne[3] == 1);
        std::array<float, 128> row;
        std::vector<uint8_t> packed(ggml_row_size(t->type, t->ne[0]));
        input_identity identity;
        for (int64_t head = 0; head < t->ne[2]; ++head) {
            for (int64_t token = 0; token < t->ne[1]; ++token) {
                for (int64_t dim = 0; dim < t->ne[0]; ++dim) {
                    row[dim] = source_value(salt, head, token, dim);
                }
                if (t->type == GGML_TYPE_F32) {
                    GGML_ASSERT(packed.size() == sizeof(row));
                    memcpy(packed.data(), row.data(), sizeof(row));
                } else {
                    const size_t written = ggml_quantize_chunk(t->type, row.data(), packed.data(), 0, 1, 128, nullptr);
                    GGML_ASSERT(written == packed.size());
                }
                ggml_backend_tensor_set(t, packed.data(), token*t->nb[1] + head*t->nb[2], packed.size());
                identity.add(packed.data(), packed.size(), row.data(), sizeof(row));
            }
        }
        print_identity(t, identity);
    }

    void initialize_mask(ggml_tensor * m) {
        std::vector<float> row(kv, -INFINITY);
        std::fill(row.begin(), row.begin() + valid_kv, 0.0f);
        std::vector<ggml_fp16_t> packed(kv);
        ggml_fp32_to_fp16_row(row.data(), packed.data(), kv);
        const size_t packed_bytes = packed.size()*sizeof(ggml_fp16_t);
        ggml_backend_tensor_set(m, packed.data(), 0, packed_bytes);
        input_identity identity;
        identity.add(packed.data(), packed_bytes, row.data(), row.size()*sizeof(float));
        print_identity(m, identity);
    }

    void print_identity(const ggml_tensor * t, const input_identity & identity) {
        // Hash host upload payloads in logical [head][token][dim] order, not device readback or unused capacity.
        // FNV-1a is noncryptographic; native byte order and source pin remain part of the record.
        fprintf(stderr, "ATTENTION_DECODE_INPUT case=%s tensor=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64
                "] nb=[%zu,%zu,%zu,%zu] order=head_token_dim hash=fnv1a64_noncryptographic uploaded_bytes=%zu"
                " uploaded_checksum=%016" PRIx64 " source_f32_bytes=%zu source_f32_checksum=%016" PRIx64 "\n",
                vars().c_str(), t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                t->nb[0], t->nb[1], t->nb[2], t->nb[3], identity.encoded_bytes, identity.encoded,
                identity.source_bytes, identity.source_f32);
    }
};

static void add_attention_decode_diagnostics(std::vector<std::unique_ptr<test_case>> & cases) {
    for (int head_repeat : {2, 4}) {
        for (ggml_type type_v : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0}) {
            const ggml_type type_k = type_v == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
            // These decode endpoints are inferred from 256-cell cache padding, not an observed graph trace.
            for (const auto & depth : {std::pair<int64_t, int64_t>{5356, 5376}, {5612, 5632}}) {
                cases.emplace_back(new test_flash_attn_ext_decode_diagnostic(head_repeat, type_k, type_v, depth.first, depth.second));
            }
        }
    }
    // Keep the original 16 cases and their ordering unchanged.
    for (int head_repeat : {2, 4}) {
        for (const auto & depth : {std::pair<int64_t, int64_t>{5356, 5376}, {5612, 5632}}) {
            cases.emplace_back(new test_flash_attn_ext_decode_diagnostic(
                head_repeat, GGML_TYPE_F16, GGML_TYPE_TURBO4_0, depth.first, depth.second));
        }
    }
}
