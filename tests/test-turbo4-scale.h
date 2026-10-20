#pragma once

static std::array<double, 128> turbo4_scale_direction(const std::array<uint8_t, 66> & packed) {
    static constexpr float centroids[16] = {
        -0.241529f, -0.182877f, -0.143016f, -0.111036f,
        -0.083292f, -0.058050f, -0.034299f, -0.011349f,
         0.011349f,  0.034299f,  0.058050f,  0.083292f,
         0.111036f,  0.143016f,  0.182877f,  0.241529f,
    };
    std::array<double, 128> direction = {};
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 128; ++j) {
            unsigned bits = unsigned(i & j), parity = 0;
            while (bits) { parity ^= bits & 1; bits >>= 1; }
            const int code = (packed[2 + j/2] >> (4*(j%2))) & 15;
            direction[i] += (parity ? -1.0 : 1.0)*turbo_cpu_s2[j]*centroids[code];
        }
        direction[i] *= turbo_cpu_s1[i]/std::sqrt(128.0);
    }
    return direction;
}

static double turbo4_scale_error(const std::array<float, 128> & input,
                                const std::array<double, 128> & direction, double scale) {
    double error = 0;
    for (int i = 0; i < 128; ++i) {
        const double delta = input[i] - scale*direction[i];
        error += delta*delta;
    }
    return error;
}

static int test_turbo4_scale() {
    int failures = 0, cases = 0, improved = 0;
    for (float amplitude : {0.0f, 1e-12f, 1e-6f, 0.125f, 1.0f, 256.0f}) {
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (int seed = 0; seed < 4; ++seed) {
                std::array<float, 128> input = {};
                for (int i = 0; i < 128; ++i) {
                    const uint32_t bits = uint32_t(i + 128*seed)*2654435761u;
                    if (pattern == 0) {
                        input[i] = amplitude*float(int((bits ^ (bits >> 16))%1025) - 512)/512.0f;
                    } else if (pattern == 1) {
                        input[i] = i == 17*seed ? amplitude : 0.0f;
                    } else if (pattern == 2) {
                        input[i] = amplitude;
                    } else {
                        input[i] = amplitude*((i + seed)%2 ? -1.0f : 1.0f);
                    }
                }
                std::array<uint8_t, 66> packed = {};
                ggml_quantize_chunk(GGML_TYPE_TURBO4_0, input.data(), packed.data(), 0, 1, 128, nullptr);
                const auto direction = turbo4_scale_direction(packed);
                double energy = 0, length = 0, projection = 0;
                for (int i = 0; i < 128; ++i) {
                    energy += double(input[i])*input[i];
                    length += direction[i]*direction[i];
                    projection += double(input[i])*direction[i];
                }
                ggml_fp16_t stored;
                memcpy(&stored, packed.data(), sizeof(stored));
                const double scale = ggml_fp16_to_fp32(stored);
                const double old_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(std::sqrt(energy/length))));
                const double error = turbo4_scale_error(input, direction, scale);
                const double old_error = turbo4_scale_error(input, direction, old_scale);
                const double tolerance = energy*1e-10;
                bool ok = std::isfinite(scale) && scale >= 0 && std::isfinite(error) && error <= old_error + tolerance;
                if (scale == 0) {
                    for (size_t i = 2; i < packed.size(); ++i) {
                        ok &= packed[i] == 0;
                    }
                }
                if (energy > 1e-20) {
                    // The direct double matrix is independent of the production float butterfly.
                    const double ideal = projection/length;
                    const double half_ulp = scale >= 0x1p-14 ? std::ldexp(1.0, std::ilogb(scale) - 10) : 0x1p-24;
                    ok &= std::abs(scale - ideal) <= half_ulp;
                    if (stored > 0) {
                        ok &= error <= turbo4_scale_error(input, direction, ggml_fp16_to_fp32(stored - 1)) + tolerance;
                    }
                    if (stored < 0x7bff) {
                        ok &= error <= turbo4_scale_error(input, direction, ggml_fp16_to_fp32(stored + 1)) + tolerance;
                    }
                } else {
                    ok &= scale == 0;
                }
                improved += error + energy*1e-6 < old_error;
                failures += !ok;
                ++cases;
                if (!ok) {
                    fprintf(stderr, "TQ_T4_SCALE_FAIL amplitude=%.9g pattern=%d seed=%d scale=%.9g mse=%.17g old_mse=%.17g\n",
                            amplitude, pattern, seed, scale, error, old_error);
                }
            }
        }
    }
    failures += cases != 96 || improved < 16;
    printf("Turbo4 least-squares scale: %s (cases=%d improved=%d failures=%d)\n",
           failures ? "FAIL" : "OK", cases, improved, failures);
    return failures;
}
