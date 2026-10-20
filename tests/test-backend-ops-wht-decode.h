#pragma once

struct test_turbo_wht_decode : public test_case {
    const int64_t groups;
    const int mode; // Forward, inverse, or round trip.
    const int pattern;

    test_turbo_wht_decode(int64_t groups, int mode, int pattern)
        : groups(groups), mode(mode), pattern(pattern) {}

    std::string vars() override {
        return "wht_decode_" + VARS_TO_STR3(groups, mode, pattern);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "TURBO_WHT";
    }

    double max_nmse_err() override { return 1e-5; }
    bool run_whole_graph() override { return mode == 2; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, groups);
        ggml_set_param(input);
        ggml_set_name(input, "wht_input");
        auto * out = ggml_turbo_wht(ctx, input, mode == 1 ? 1 : 0, 128, nullptr);
        if (mode == 2) {
            ggml_set_name(out, "wht_forward");
            out = ggml_turbo_wht(ctx, out, 1, 128, nullptr);
        }
        ggml_set_name(out, "wht_output");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (strncmp(t->name, "sent_", 5) == 0) {
                init_tensor_uniform(t);
            } else {
                ggml_backend_tensor_memset(t, 0xff, 0, ggml_nbytes(t));
            }
        }
        std::vector<float> values(size_t(groups)*128);
        for (int64_t g = 0; g < groups; ++g) {
            for (int d = 0; d < 128; ++d) {
                float v = 0;
                if (pattern == 1) {
                    v = d == g%128 ? (g%2 == 0 ? 1.0f : -1.0f) : 0.0f;
                } else if (pattern == 2) {
                    v = ((d + g)%2 == 0 ? 1.0f : -1.0f)*(1.0f + float(g%8)/8.0f);
                } else if (pattern == 3) {
                    const uint32_t x = uint32_t(g)*2654435761u + uint32_t(d)*2246822519u;
                    v = float(int((x ^ (x >> 16))%1025) - 512)/512.0f;
                }
                values[size_t(g)*128 + d] = v;
            }
        }
        auto * input = ggml_get_tensor(ctx, "wht_input");
        ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
        uint64_t hash = 14695981039346656037ULL;
        const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
        for (size_t i = 0; i < values.size()*sizeof(float); ++i) {
            hash = (hash ^ bytes[i])*1099511628211ULL;
        }
        fprintf(stderr, "TQ_WHT_INPUT groups=%lld mode=%d pattern=%d bytes=%zu fnv1a64=%016llx\n",
                (long long) groups, mode, pattern, values.size()*sizeof(float), (unsigned long long) hash);
    }

    double err(const float * a, const float * b, size_t n) override {
        if (n == sentinel_size && n != size_t(groups)*128) {
            return memcmp(a, b, n*sizeof(float)) == 0 ? 0.0 : INFINITY;
        }
        if (n != size_t(groups)*128) {
            return INFINITY;
        }
        double worst = 0;
        for (size_t start = 0; start < n; start += 128) {
            double error = 0, energy = 0;
            for (size_t i = start; i < start + 128; ++i) {
                if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
                    (pattern == 0 && (a[i] != 0 || b[i] != 0))) {
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
};

static void add_turbo_wht_decode_tests(std::vector<std::unique_ptr<test_case>> & cases) {
    for (int64_t groups : {1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257}) {
        for (int mode : {0, 1, 2}) {
            for (int pattern : {0, 1, 2, 3}) {
                cases.emplace_back(new test_turbo_wht_decode(groups, mode, pattern));
            }
        }
    }
    for (int64_t groups : {1024, 16384}) {
        for (int pattern : {0, 1, 2, 3}) {
            cases.emplace_back(new test_turbo_wht_decode(groups, 1, pattern));
        }
    }
}
