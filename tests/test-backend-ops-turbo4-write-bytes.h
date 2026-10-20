#pragma once

struct test_set_rows_turbo4_bytes : public test_case {
    static constexpr int64_t width = 8*128;
    static constexpr int64_t rows = 113;
    const ggml_type type_idx;
    const int64_t row;
    const int pattern;
    ggml_tensor * fast = nullptr;
    ggml_tensor * fallback = nullptr;

    test_set_rows_turbo4_bytes(ggml_type type_idx, int64_t row, int pattern)
        : type_idx(type_idx), row(row), pattern(pattern) {
        GGML_ASSERT(type_idx == GGML_TYPE_I32 || type_idx == GGML_TYPE_I64);
        GGML_ASSERT(row >= 0 && row < rows && pattern >= 0 && pattern < 10);
    }

    std::string vars() override { return "t4_write_bytes_" + VARS_TO_STR3(type_idx, row, pattern); }
    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "SET_ROWS_TURBO4";
    }
    bool run_whole_graph() override { return true; }
    double max_nmse_err() override { return 0.05; }

    std::vector<ggml_tensor *> fusion_test_nodes() override {
        auto nodes = sentinels;
        nodes.push_back(fast);
        nodes.push_back(fallback);
        return nodes;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto * src_fast = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        auto * src_fallback = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 2);
        auto * idx_fast = ggml_new_tensor_1d(ctx, type_idx, 1);
        auto * idx_fallback = ggml_new_tensor_1d(ctx, type_idx, 2);
        ggml_set_name(src_fast, "byte_src_fast");
        ggml_set_name(src_fallback, "byte_src_fallback");
        ggml_set_name(idx_fast, "byte_idx_fast");
        ggml_set_name(idx_fallback, "byte_idx_fallback");
        fast = ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, rows), src_fast, idx_fast);
        fallback = ggml_set_rows(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO4_0, width, rows), src_fallback, idx_fallback);
        ggml_set_name(fast, "byte_fast");
        ggml_set_name(fallback, "byte_fallback");
        // Keep both independent writes in the graph without a GPU dequantization node.
        if (mode == MODE_TEST) {
            ggml_build_forward_expand(gf, fallback);
        }
        return fast;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (strncmp(t->name, "sent_", 5) == 0) {
                init_tensor_uniform(t);
            } else {
                ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
            }
        }
        const auto values = input_values();
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "byte_src_fast"), values.data(), 0, sizeof(values));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "byte_src_fallback"), values.data(), 0, sizeof(values));
        std::array<float, width> dummy;
        for (size_t i = 0; i < dummy.size(); ++i) {
            dummy[i] = float(int(i%31) - 15)/32.0f;
        }
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "byte_src_fallback"), dummy.data(), sizeof(values), sizeof(dummy));
        const int64_t indices64[2] = {row, dummy_row()};
        const int32_t indices32[2] = {int32_t(row), int32_t(dummy_row())};
        const void * indices = type_idx == GGML_TYPE_I32 ? static_cast<const void *>(indices32) : static_cast<const void *>(indices64);
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "byte_idx_fast"), indices, 0, ggml_type_size(type_idx));
        ggml_backend_tensor_set(ggml_get_tensor(ctx, "byte_idx_fallback"), indices, 0, 2*ggml_type_size(type_idx));
        const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, width);
        GGML_ASSERT(row_bytes == 528);
        ggml_backend_tensor_memset(fast, 0xff, row*fast->nb[1], row_bytes);
        ggml_backend_tensor_memset(fallback, 0xff, row*fallback->nb[1], row_bytes);
        ggml_backend_tensor_memset(fallback, 0xff, dummy_row()*fallback->nb[1], row_bytes);
    }

    double err(const float * a, const float * b, size_t n) override {
        if (n == sentinel_size) {
            return memcmp(a, b, n*sizeof(float)) == 0 ? 0.0 : INFINITY;
        }
        if (n != size_t(width*rows) || !packed_equal()) {
            return INFINITY;
        }
        double worst = 0;
        for (size_t start = 0; start < n; start += 128) {
            double error = 0, energy = 0;
            for (size_t i = start; i < start + 128; ++i) {
                if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
                    return INFINITY;
                }
                const double delta = double(a[i]) - b[i];
                error += delta*delta;
                energy += double(b[i])*b[i];
            }
            worst = std::max(worst, energy == 0 ? (error == 0 ? 0.0 : INFINITY) : error/energy);
        }
        return worst;
    }

private:
    int64_t dummy_row() const { return (row + 1)%rows; }

    bool packed_equal() const {
        const size_t row_bytes = ggml_row_size(GGML_TYPE_TURBO4_0, width);
        std::vector<uint8_t> a(ggml_nbytes(fast)), b(ggml_nbytes(fallback));
        ggml_backend_tensor_get(fast, a.data(), 0, a.size());
        ggml_backend_tensor_get(fallback, b.data(), 0, b.size());
        for (int64_t r = 0; r < rows; ++r) {
            for (size_t j = 0; j < row_bytes; ++j) {
                const uint8_t av = a[r*row_bytes + j], bv = b[r*row_bytes + j];
                if ((r == row && av != bv) || (r != row && av != 0) ||
                    (r != row && r != dummy_row() && bv != 0)) {
                    fprintf(stderr, "TQ_T4_WRITE_BYTES mismatch row=%lld byte=%zu fast=%u fallback=%u\n",
                            (long long) r, j, unsigned(av), unsigned(bv));
                    return false;
                }
            }
        }
        return true;
    }

    std::array<float, width> input_values() const {
        std::array<float, width> values = {};
        for (int h = 0; h < 8; ++h) {
            if (pattern >= 4) {
                midpoint_input(values.data() + h*128, h, (pattern - 4)/3, (pattern - 4)%3);
                continue;
            }
            for (int d = 0; d < 128; ++d) {
                if (pattern == 1) {
                    values[h*128 + d] = d == (h*17)%128 ? (h%2 == 0 ? 1.0f : -1.0f)*(1.0f + h/8.0f) : 0.0f;
                } else if (pattern == 2) {
                    const uint32_t x = uint32_t(h)*2654435761u + uint32_t(d)*2246822519u;
                    values[h*128 + d] = float(int((x ^ (x >> 16))%1025) - 512)/512.0f;
                } else if (pattern == 3) {
                    const float amplitude = h%3 == 0 ? 0.5e-10f : h%3 == 1 ? 1.0e-10f : 2.0e-10f;
                    values[h*128 + d] = d == (h*17)%128 ? (h%2 == 0 ? amplitude : -amplitude) : 0.0f;
                }
            }
        }
        return values;
    }

    static void midpoint_input(float * dst, int head, int batch, int side) {
        // Fixed codec signs; use a direct inverse matrix, not the production rotation helper.
        static const int s1[128] = {
            -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
            -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
            -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
            -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
        };
        static const int s2[128] = {
            1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
            1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
            1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
            1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
        };
        static const float mid[15] = {
            -0.212203f, -0.162947f, -0.127026f, -0.097164f, -0.070671f, -0.046174f, -0.022824f, 0,
             0.022824f,  0.046174f,  0.070671f,  0.097164f,  0.127026f,  0.162947f,  0.212203f
        };
        float threshold = mid[(8*batch + head)%15];
        // Normalization can move these probes across a boundary; packed equality stays exact.
        if (side != 1) {
            threshold += side == 0 ? -2.0e-6f : 2.0e-6f;
        }
        const int axis = (head*17)%128;
        const int other = (axis + 1)%128;
        const double remainder = std::sqrt(1.0 - double(threshold)*threshold);
        for (int d = 0; d < 128; ++d) {
            auto sign = [](unsigned bits) {
                int result = 1;
                while (bits) {
                    result = -result;
                    bits &= bits - 1;
                }
                return result;
            };
            dst[d] = float(s1[d]*(sign(d & axis)*s2[axis]*double(threshold) +
                                  sign(d & other)*s2[other]*remainder)/std::sqrt(128.0));
        }
    }
};

static void add_turbo4_write_byte_tests(std::vector<std::unique_ptr<test_case>> & cases) {
    for (auto type_idx : {GGML_TYPE_I32, GGML_TYPE_I64}) {
        for (int64_t row : {0, 56, 112}) {
            for (int pattern = 0; pattern < 10; ++pattern) {
                cases.emplace_back(new test_set_rows_turbo4_bytes(type_idx, row, pattern));
            }
        }
    }
}
