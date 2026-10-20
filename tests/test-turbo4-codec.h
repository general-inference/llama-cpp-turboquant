#pragma once

#include <array>
#include <cstring>

static constexpr float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static constexpr float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};



static int test_turbo4_codec() {
    const auto type = GGML_TYPE_TURBO4_0;
    const auto * traits = ggml_get_type_traits(type);
    const auto * cpu = ggml_get_type_traits_cpu(type);
    int failures = 0;
    for (int blocks : {1, 3, 17}) {
        std::vector<float> input(blocks*128), decoded(input.size()), rhs(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            const uint32_t x = uint32_t(i)*2654435761u;
            input[i] = float(int((x ^ (x >> 16)) % 1025) - 512)/512.0f;
            rhs[i] = float(int(i % 31) - 15)/16.0f;
        }
        std::vector<uint8_t> packed(ggml_row_size(type, input.size())), split(packed.size());
        ggml_quantize_chunk(type, input.data(), packed.data(), 0, 1, input.size(), nullptr);
        for (int b = 0; b < blocks; ++b) {
            ggml_quantize_chunk(type, input.data() + b*128, split.data() + b*66, 0, 1, 128, nullptr);
        }
        failures += packed != split;
        traits->to_float(packed.data(), decoded.data(), decoded.size());
        double error = 0, energy = 0;
        for (int b = 0; b < blocks; ++b) {
            for (int i = 0; i < 128; ++i) {
                double inverse = 0;
                for (int j = 0; j < 128; ++j) {
                    unsigned bits = unsigned(i & j), parity = 0;
                    while (bits) { parity ^= bits & 1; bits >>= 1; }
                    inverse += (parity ? -1.0 : 1.0)*turbo_cpu_s2[j]*decoded[b*128 + j];
                }
                inverse *= turbo_cpu_s1[i]/std::sqrt(128.0);
                const double delta = input[b*128 + i] - inverse;
                error += delta*delta;
                energy += double(input[b*128 + i])*input[b*128 + i];
            }
        }
        const bool reconstruction_ok = std::isfinite(error) && error/energy < 0.03;
        failures += !reconstruction_ok;
        float dot = 0;
        cpu->vec_dot(input.size(), &dot, 0, packed.data(), 0, rhs.data(), 0, 1);
        double reference = 0, absolute = 0;
        for (size_t i = 0; i < input.size(); ++i) {
            const double product = double(decoded[i])*rhs[i];
            reference += product;
            absolute += std::abs(product);
        }
        failures += !std::isfinite(dot) || std::abs(double(dot) - reference) > 2e-5*absolute;
    }
    std::array<float, 128> zero = {}, decoded;
    std::array<uint8_t, 66> packed;
    ggml_quantize_chunk(type, zero.data(), packed.data(), 0, 1, 128, nullptr);
    traits->to_float(packed.data(), decoded.data(), 128);
    failures += !std::all_of(decoded.begin(), decoded.end(), [](float x) { return x == 0; });
    failures += !ggml_validate_row_data(type, packed.data(), packed.size());
    for (float invalid : {-1.0f, INFINITY, NAN}) {
        const ggml_fp16_t norm = ggml_fp32_to_fp16(invalid);
        std::memcpy(packed.data(), &norm, sizeof(norm));
        failures += ggml_validate_row_data(type, packed.data(), packed.size());
    }
    printf("Turbo4 rotated codec: %s (%d failures)\n", failures ? "FAIL" : "OK", failures);
    return failures;
}
