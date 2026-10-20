#pragma once

struct test_set_rows_turbo4_decode : public test_case {
    static constexpr int64_t width = 8*128;
    const ggml_type type_idx;
    const int64_t cache_rows;
    const int64_t row;
    const int pattern;

    test_set_rows_turbo4_decode(ggml_type type_idx, int64_t cache_rows, int64_t row, int pattern)
        : type_idx(type_idx), cache_rows(cache_rows), row(row), pattern(pattern) {
        GGML_ASSERT(type_idx == GGML_TYPE_I32 || type_idx == GGML_TYPE_I64);
        GGML_ASSERT(cache_rows > 1 && row >= 0 && row < cache_rows);
        GGML_ASSERT(pattern >= 0 && pattern <= 2);
    }

    std::string vars() override {
        return "t4_write_decode_" + VARS_TO_STR4(type_idx, cache_rows, row, pattern);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "SET_ROWS_TURBO4";
    }

    double max_nmse_err() override { return 0.05; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, cache_rows);
        auto * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        auto * index = ggml_new_tensor_1d(ctx, type_idx, 1);
        ggml_set_name(cache, "write_cache");
        ggml_set_name(source, "write_source");
        ggml_set_name(index, "write_index");
        auto * written = ggml_set_rows(ctx, cache, source, index);
        ggml_set_name(written, "write_result");
        // The comparison harness reads and dequantizes the full cache on the CPU.
        // Metal does not support a TURBO4-to-F32 CPY node.
        return written;
    }

    size_t op_size(ggml_tensor * t) override {
        GGML_UNUSED(t);
        // Logical bytes touched by one write, not the full cache allocation.
        return width*sizeof(float) + ggml_type_size(type_idx) + ggml_row_size(GGML_TYPE_TURBO4_0, width);
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (strncmp(t->name, "sent_", 5) == 0) {
                init_tensor_uniform(t);
            } else {
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }

        std::array<float, width> values = {};
        for (int h = 0; h < 8; ++h) {
            for (int d = 0; d < 128; ++d) {
                if (pattern == 1) {
                    values[h*128 + d] = d == (h*17)%128 ? (h%2 == 0 ? 1.0f : -1.0f)*(1.0f + h/8.0f) : 0.0f;
                } else if (pattern == 2) {
                    const uint32_t x = uint32_t(h)*2654435761u + uint32_t(d)*2246822519u;
                    values[h*128 + d] = float(int((x ^ (x >> 16))%1025) - 512)/512.0f;
                }
            }
        }
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "write_source"), values.data(), 0, sizeof(values));
        const int32_t index32 = int32_t(row);
        const void * index_data = type_idx == GGML_TYPE_I32 ? static_cast<const void *>(&index32) : static_cast<const void *>(&row);
        const size_t index_bytes = ggml_type_size(type_idx);
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "write_index"), index_data, 0, index_bytes);

        auto * cache = ggml_get_tensor(ctx, "write_cache");
        const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, width);
        GGML_ASSERT(row_bytes == 528);
        // A skipped write leaves NaN norms in every target block.
        ggml_backend_tensor_memset(cache, 0xff, row*cache->nb[1], row_bytes);
        fprintf(stderr, "TQ_T4_WRITE_INPUT index_type=%s cache_rows=%lld row=%lld pattern=%d input_bytes=%zu blocks=8 packed_row_bytes=%zu source_fnv1a64=%016llx index_fnv1a64=%016llx\n",
                ggml_type_name(type_idx), (long long) cache_rows, (long long) row, pattern, sizeof(values), row_bytes,
                (unsigned long long) fingerprint(values.data(), sizeof(values)),
                (unsigned long long) fingerprint(index_data, index_bytes));
    }

    double err(const float * a, const float * b, size_t n) override {
        const size_t expected = size_t(width)*cache_rows;
        if (n == sentinel_size && n != expected) {
            return memcmp(a, b, n*sizeof(float)) == 0 ? 0.0 : INFINITY;
        }
        if (n != expected) {
            return INFINITY;
        }
        const size_t begin = size_t(row)*width;
        const size_t end = begin + width;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
                return INFINITY;
            }
            if ((pattern == 0 || i < begin || i >= end) && (a[i] != 0.0f || b[i] != 0.0f)) {
                return INFINITY;
            }
        }
        if (pattern == 0) {
            return 0.0;
        }
        double worst = 0.0;
        for (size_t head = begin; head < end; head += 128) {
            double error = 0.0, actual_energy = 0.0, reference_energy = 0.0;
            for (size_t i = head; i < head + 128; ++i) {
                const double delta = double(a[i]) - b[i];
                error += delta*delta;
                actual_energy += double(a[i])*a[i];
                reference_energy += double(b[i])*b[i];
            }
            if (actual_energy == 0.0 || reference_energy == 0.0) {
                return INFINITY;
            }
            worst = std::max(worst, error/reference_energy);
        }
        return worst;
    }

private:
    static uint64_t fingerprint(const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        uint64_t hash = 14695981039346656037ULL;
        for (size_t i = 0; i < size; ++i) {
            hash = (hash ^ bytes[i])*1099511628211ULL;
        }
        return hash;
    }
};

static void add_turbo4_write_decode_tests(std::vector<std::unique_ptr<test_case>> & cases) {
    for (auto type_idx : {GGML_TYPE_I32, GGML_TYPE_I64}) {
        for (int64_t row : {0, 56, 112}) {
            for (int pattern : {0, 1, 2}) {
                cases.emplace_back(new test_set_rows_turbo4_decode(type_idx, 113, row, pattern));
            }
        }
        cases.emplace_back(new test_set_rows_turbo4_decode(type_idx, 8192, 4096, 2));
    }
}
