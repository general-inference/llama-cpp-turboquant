#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int D = 128;
constexpr int HK = 8;
constexpr int HQ = 16;
using values = std::vector<float>;

// Fixed codec signs. The oracle multiplies an explicit matrix, not the production WHT.
constexpr int S1[D] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
constexpr int S2[D] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

const std::array<double, D*D> & rotation_matrix() {
    static const auto matrix = [] {
        std::array<double, D*D> result{};
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                unsigned bits = unsigned(i & j);
                int sign = 1;
                while (bits != 0) {
                    sign = -sign;
                    bits &= bits - 1;
                }
                result[i*D + j] = double(S2[i]*sign*S1[j])/std::sqrt(double(D));
            }
        }
        return result;
    }();
    return matrix;
}

values rotate(const values & source, bool inverse = false) {
    values result(source.size());
    const auto & matrix = rotation_matrix();
    for (size_t base = 0; base < source.size(); base += D) {
        for (int i = 0; i < D; ++i) {
            double sum = 0;
            for (int j = 0; j < D; ++j) {
                sum += matrix[inverse ? j*D + i : i*D + j]*source[base + j];
            }
            result[base + i] = float(sum);
        }
    }
    return result;
}

int cache_row(int logical, int kv) {
    // 37 is coprime to both fixture lengths, so every cache row is written once.
    return (37*logical + 19) % kv;
}

values make_source(int heads, int tokens, int seed) {
    values result(D*heads*tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const float amplitude = 0.25f + 0.125f*float((h + t + seed) % 7);
            for (int d = 0; d < D; ++d) {
                const int index = d + D*(h + heads*t);
                float x = amplitude*(std::sin(float(index + 17*seed)*0.03125f) +
                                     0.3f*std::cos(float(3*d + 5*h + 11*t + seed)));
                if (d == (13*h + 7*t + seed) % D) {
                    x += (h % 2 == 0 ? 3.0f : -3.0f)*amplitude;
                }
                result[index] = x;
            }
        }
    }
    return result;
}

values scatter(const values & source, int kv) {
    values result(source.size());
    for (int t = 0; t < kv; ++t) {
        std::copy_n(source.data() + t*HK*D, HK*D, result.data() + cache_row(t, kv)*HK*D);
    }
    return result;
}

// Reconstruct the documented nearest-centroid/norm contract without using the codec writer.
values quantized_reference(const values & source, ggml_type type, int kv) {
    const std::vector<float> centroids = type == GGML_TYPE_TURBO3_0 ? std::vector<float>{
        -0.190207f, -0.118786f, -0.066822f, -0.021663f,
         0.021663f,  0.066822f,  0.118786f,  0.190207f,
    } : std::vector<float>{
        -0.241529f, -0.182877f, -0.143016f, -0.111036f,
        -0.083292f, -0.058050f, -0.034299f, -0.011349f,
         0.011349f,  0.034299f,  0.058050f,  0.083292f,
         0.111036f,  0.143016f,  0.182877f,  0.241529f,
    };
    const values rotated = rotate(source);
    values result(source.size());
    for (size_t base = 0; base < source.size(); base += D) {
        double norm_sq = 0;
        for (int i = 0; i < D; ++i) {
            norm_sq += double(source[base + i])*source[base + i];
        }
        const double norm = std::sqrt(norm_sq);
        double reconstruction_sq = 0;
        for (int i = 0; i < D; ++i) {
            const double normalized = norm > 1e-10 ? rotated[base + i]/norm : 0;
            const auto nearest = std::min_element(centroids.begin(), centroids.end(),
                [normalized](float a, float b) { return std::abs(a - normalized) < std::abs(b - normalized); });
            result[base + i] = *nearest;
            reconstruction_sq += double(*nearest)*(*nearest);
        }
        const float scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(norm/std::sqrt(reconstruction_sq))));
        for (int i = 0; i < D; ++i) {
            result[base + i] *= scale;
        }
    }
    return scatter(result, kv);
}

values attention(const values & q, const values & k, const values & v, int nq, int kv) {
    values result(D*HQ*nq);
    for (int t = 0; t < nq; ++t) {
        for (int h = 0; h < HQ; ++h) {
            const int kh = h/(HQ/HK);
            std::vector<double> scores(kv);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int logical = 0; logical <= kv - nq + t; ++logical) {
                const int row = cache_row(logical, kv);
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += double(q[d + D*(h + HQ*t)])*k[d + D*(kh + HK*row)];
                }
                scores[logical] = dot/std::sqrt(double(D));
                maximum = std::max(maximum, scores[logical]);
            }
            double denominator = 0;
            for (int logical = 0; logical <= kv - nq + t; ++logical) {
                scores[logical] = std::exp(scores[logical] - maximum);
                denominator += scores[logical];
            }
            for (int d = 0; d < D; ++d) {
                double sum = 0;
                for (int logical = 0; logical <= kv - nq + t; ++logical) {
                    sum += scores[logical]*v[d + D*(kh + HK*cache_row(logical, kv))];
                }
                result[d + D*(h + HQ*t)] = float(sum/denominator);
            }
        }
    }
    return result;
}

values read_f32(ggml_tensor * tensor) {
    values result(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size()*sizeof(float));
    return result;
}

values read_cache(ggml_tensor * tensor) {
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    const int64_t kv = tensor->ne[1];
    values result(size_t(D*HK*kv));
    const auto * traits = ggml_get_type_traits(tensor->type);
    GGML_ASSERT(traits->to_float != nullptr);
    for (int row = 0; row < kv; ++row) {
        traits->to_float(bytes.data() + row*tensor->nb[1], result.data() + row*D*HK, D*HK);
    }
    return result;
}

bool finite(const values & data) {
    return std::all_of(data.begin(), data.end(), [](float v) { return std::isfinite(v); });
}

double nmse(const values & actual, const values & expected) {
    if (actual.size() != expected.size() || !finite(actual) || !finite(expected)) {
        return std::numeric_limits<double>::infinity();
    }
    double difference = 0;
    double energy = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i];
        difference += delta*delta;
        energy += double(expected[i])*expected[i];
    }
    return energy > 0 ? difference/energy : (difference == 0 ? 0 : std::numeric_limits<double>::infinity());
}

bool check_error(const char * name, const values & actual, const values & expected, double limit) {
    const double error = nmse(actual, expected);
    const bool ok = std::isfinite(error) && error <= limit;
    std::printf("  check=%s nmse=%.9g limit=%.9g result=%s\n", name, error, limit, ok ? "PASS" : "FAIL");
    return ok;
}

bool check_norm(const char * name, const values & actual, const values & source) {
    double maximum = 0;
    for (size_t base = 0; base < source.size(); base += D) {
        double a = 0;
        double b = 0;
        for (int i = 0; i < D; ++i) {
            a += double(actual[base + i])*actual[base + i];
            b += double(source[base + i])*source[base + i];
        }
        maximum = std::max(maximum, std::abs(std::sqrt(a/b) - 1));
    }
    // A normal binary16 corrected norm has relative rounding error at most 2^-11.
    const bool ok = finite(actual) && maximum <= 1e-3;
    std::printf("  check=%s max_relative_norm_error=%.9g limit=0.001 result=%s\n", name, maximum, ok ? "PASS" : "FAIL");
    return ok;
}

struct graph_tensors {
    ggml_cgraph * graph;
    ggml_tensor * q;
    ggml_tensor * k_source;
    ggml_tensor * v_source;
    ggml_tensor * rows;
    ggml_tensor * scale;
    ggml_tensor * mask;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * rotated_q;
    ggml_tensor * attended;
    ggml_tensor * output;
};

graph_tensors make_graph(ggml_context * ctx, ggml_type type, int nq, int kv) {
    graph_tensors t{};
    t.graph = ggml_new_graph_custom(ctx, 64, false);
    t.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HQ, nq);
    t.k_source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D*HK, kv);
    t.v_source = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D*HK, kv);
    t.rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, kv);
    t.scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
    t.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nq);
    const auto write_cache = [&](ggml_tensor * source) {
        ggml_tensor * cache = ggml_new_tensor_2d(ctx, type, D*HK, kv);
        ggml_tensor * written = ggml_set_rows(ctx, cache, source, t.rows);
        const int32_t group_size = D;
        std::memcpy(written->op_params, &group_size, sizeof(group_size));
        return written;
    };
    t.k = write_cache(t.k_source);
    t.v = write_cache(t.v_source);
    const auto attention_view = [&](ggml_tensor * cache) {
        ggml_tensor * view = ggml_view_3d(ctx, cache, D, HK, kv,
            ggml_row_size(type, D), ggml_row_size(type, D*HK), 0);
        return ggml_permute(ctx, view, 0, 2, 1, 3);
    };
    t.rotated_q = ggml_turbo_wht(ctx, t.q, 0, D, t.scale);
    ggml_tensor * q = ggml_permute(ctx, t.rotated_q, 0, 2, 1, 3);
    t.attended = ggml_flash_attn_ext(ctx, q, attention_view(t.k), attention_view(t.v),
        t.mask, 1.0f/std::sqrt(float(D)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(t.attended, GGML_PREC_F32);
    t.output = ggml_turbo_wht(ctx, t.attended, 1, D, t.scale);
    ggml_build_forward_expand(t.graph, t.output);
    return t;
}

void initialize(const graph_tensors & t, const values & q, const values & k, const values & v, int nq, int kv) {
    ggml_backend_tensor_set(t.q, q.data(), 0, q.size()*sizeof(float));
    ggml_backend_tensor_set(t.k_source, k.data(), 0, k.size()*sizeof(float));
    ggml_backend_tensor_set(t.v_source, v.data(), 0, v.size()*sizeof(float));
    std::vector<int64_t> rows(kv);
    for (int i = 0; i < kv; ++i) {
        rows[i] = cache_row(i, kv);
    }
    ggml_backend_tensor_set(t.rows, rows.data(), 0, rows.size()*sizeof(int64_t));
    std::array<float, D> ones{};
    ones.fill(1.0f);
    ggml_backend_tensor_set(t.scale, ones.data(), 0, sizeof(ones));
    std::vector<ggml_fp16_t> mask(kv*nq);
    for (int query = 0; query < nq; ++query) {
        for (int logical = 0; logical < kv; ++logical) {
            const float value = logical <= kv - nq + query ? 0.0f : -std::numeric_limits<float>::infinity();
            mask[cache_row(logical, kv) + kv*query] = ggml_fp32_to_fp16(value);
        }
    }
    ggml_backend_tensor_set(t.mask, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
}

bool run_case(ggml_backend_t backend, ggml_type type, int nq, int kv) {
    std::printf("case type=%s nq=%d head=128 q_heads=16 kv_heads=8 kv=%d\n", ggml_type_name(type), nq, kv);
    const ggml_init_params params{ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        std::printf("  FAIL: context allocation\n");
        return false;
    }
    const graph_tensors t = make_graph(ctx.get(), type, nq, kv);
    for (int i = 0; i < ggml_graph_n_nodes(t.graph); ++i) {
        ggml_tensor * node = ggml_graph_node(t.graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            std::printf("  FAIL: unsupported node=%d op=%s\n", i, ggml_op_desc(node));
            return false;
        }
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        std::printf("  FAIL: tensor allocation\n");
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);
    const values q = make_source(HQ, nq, 1);
    const values k = make_source(HK, kv, 2);
    const values v = make_source(HK, kv, 3);
    initialize(t, q, k, v, nq, kv);
    if (ggml_backend_graph_compute(backend, t.graph) != GGML_STATUS_SUCCESS) {
        std::printf("  FAIL: graph execution\n");
        return false;
    }
    ggml_backend_synchronize(backend);
    const values actual_q = read_f32(t.rotated_q);
    const values actual_k = read_cache(t.k);
    const values actual_v = read_cache(t.v);
    const values actual_attention = read_f32(t.attended);
    const values actual_output = read_f32(t.output);
    const values expected_q = rotate(q);
    const values quantized_attention = attention(expected_q, actual_k, actual_v, nq, kv);
    const values quantized_output = rotate(quantized_attention, true);
    bool ok = true;
    // Keep the existing WHT, SET_ROWS and attention test tolerances, respectively.
    ok &= check_error("query_rotation", actual_q, expected_q, 1e-5);
    ok &= check_error("key_write", actual_k, quantized_reference(k, type, kv), 0.05);
    ok &= check_error("value_write", actual_v, quantized_reference(v, type, kv), 0.05);
    ok &= check_norm("key_norm", actual_k, scatter(k, kv));
    ok &= check_norm("value_norm", actual_v, scatter(v, kv));
    ok &= check_error("attention_actual_cache", actual_attention, attention(actual_q, actual_k, actual_v, nq, kv), 5e-4);
    ok &= check_error("inverse_rotation", actual_output, rotate(actual_attention, true), 1e-5);
    ok &= check_error("composed_actual_cache", actual_output, quantized_output, 5e-4);
    const values original_output = attention(q, scatter(k, kv), scatter(v, kv), nq, kv);
    std::printf("  diagnostic=quantization_only output_nmse=%.9g key_nmse=%.9g value_nmse=%.9g quality_gate=none\n",
        nmse(quantized_output, original_output), nmse(rotate(actual_k, true), scatter(k, kv)),
        nmse(rotate(actual_v, true), scatter(v, kv)));
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const char * backend_name = "CPU";
    if (argc == 3 && std::strcmp(argv[1], "-b") == 0) {
        backend_name = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [-b CPU|MTL0|CUDA0]\n", argv[0]);
        return 2;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name(backend_name, nullptr));
    if (!backend) {
        std::fprintf(stderr, "FAIL: selected backend %s unavailable\n", backend_name);
        std::printf("0/8 cases passed\n");
        return 1;
    }
    std::printf("Backend %s\n", ggml_backend_name(backend.get()));
    int passed = 0;
    // 256 admits CUDA GQA packing and VEC selection; 113 keeps the unaligned cases.
    for (int kv : {113, 256}) {
        for (ggml_type type : {GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0}) {
            for (int nq : {1, 32}) {
                passed += run_case(backend.get(), type, nq, kv) ? 1 : 0;
            }
        }
    }
    std::printf("%d/8 cases passed\n", passed);
    return passed == 8 ? 0 : 1;
}
