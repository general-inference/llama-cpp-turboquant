#include "llama.h"
#include "llama-model-loader.h"
#include "skippy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

struct progress_state {
    float last = 0.0f;
    bool bounded = true;
    bool monotonic = true;
    bool progressed_after_completion = false;
    size_t completion_count = 0;
};

static bool observe_progress(float progress, void * user_data) {
    auto * state = static_cast<progress_state *>(user_data);
    constexpr float tolerance = 1e-6f;

    state->bounded &= std::isfinite(progress) && progress >= -tolerance && progress <= 1.0f + tolerance;
    state->monotonic &= progress + tolerance >= state->last;
    if (progress == 1.0f) {
        state->completion_count++;
    } else if (state->completion_count > 0) {
        state->progressed_after_completion = true;
    }
    state->last = progress;
    return true;
}

static llama_model_loader make_loader(const char * model_path, llama_load_mode load_mode = LLAMA_LOAD_MODE_NONE) {
    std::vector<std::string> splits;
    return llama_model_loader(
            nullptr,
            nullptr,
            nullptr,
            model_path,
            splits,
            false,
            nullptr,
            load_mode,
            false,
            true,
            false,
            nullptr,
            nullptr);
}

static bool check_exact_resident_selection(const char * model_path) {
    llama_model_loader loader = make_loader(model_path);
    std::unordered_set<std::string> resident_tensor_names;
    for (const auto & [name, _] : loader.weights_map) {
        resident_tensor_names.insert(name);
    }
    if (resident_tensor_names.empty()) {
        std::fprintf(stderr, "model has no resident tensor names\n");
        return false;
    }

    llama_model_params params = llama_model_default_params();
    llama_model_loader_stage_selection selection;
    selection.layer_start = 0;
    selection.layer_end = 1;
    selection.resident_tensor_names = resident_tensor_names;
    llama_model * model = llama_model_load_from_file_with_selection(model_path, params, &selection);
    if (model == nullptr) {
        std::fprintf(stderr, "exact resident tensor selection rejected complete closure\n");
        return false;
    }
    llama_model_free(model);

    resident_tensor_names.insert("skippy.missing.admitted.tensor");
    selection.resident_tensor_names = resident_tensor_names;
    model = llama_model_load_from_file_with_selection(model_path, params, &selection);
    if (model != nullptr) {
        llama_model_free(model);
        std::fprintf(stderr, "exact resident tensor selection accepted an absent tensor\n");
        return false;
    }
    return true;
}

static ggml_context_ptr make_accounting_context(const ggml_tensor * weight) {
    const size_t tensor_count = 2;
    ggml_init_params params = {
        /*.mem_size   =*/ tensor_count * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (ctx == nullptr) {
        return nullptr;
    }

    ggml_tensor * duplicate = ggml_dup_tensor(ctx.get(), weight);
    ggml_set_name(duplicate, ggml_get_name(weight));
    ggml_tensor * tied = ggml_dup_tensor(ctx.get(), weight);
    ggml_set_name(tied, ggml_get_name(weight));
    return ctx;
}

static ggml_context_ptr make_synthetic_context(const ggml_tensor * weight) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (ctx != nullptr) {
        ggml_tensor * synthetic = ggml_dup_tensor(ctx.get(), weight);
        ggml_set_name(synthetic, "skippy.synthetic.after-completion");
    }
    return ctx;
}

static std::vector<ggml_backend_buffer_t> make_buffer_list(ggml_backend_buffer_t buffer) {
    return { buffer };
}

static bool check_context_load_lifecycle(const char * model_path) {
    llama_model_loader loader = make_loader(model_path, LLAMA_LOAD_MODE_MMAP);
    if (loader.weights_map.empty()) {
        std::fprintf(stderr, "model has no weights\n");
        return false;
    }

    const ggml_tensor * weight = loader.weights_map.begin()->second.tensor;
    ggml_context_ptr accounting_ctx = make_accounting_context(weight);
    ggml_context_ptr synthetic_ctx = make_synthetic_context(weight);
    if (accounting_ctx == nullptr || synthetic_ctx == nullptr) {
        std::fprintf(stderr, "failed to allocate accounting contexts\n");
        return false;
    }

    ggml_context * accounting_ctx_raw = accounting_ctx.get();
    ggml_context * synthetic_ctx_raw = synthetic_ctx.get();
    const auto cpu_buft = ggml_backend_cpu_buffer_type();
    loader.ctx_map.emplace(llama_model_loader::ctx_key{cpu_buft, /*lazy=*/false, /*resident=*/true}, std::move(accounting_ctx));
    loader.init_mappings(false);

    const size_t weight_size = ggml_nbytes(weight);
    const size_t expected = 2 * weight_size;
    if (loader.size_data != expected) {
        std::fprintf(
                stderr,
                "context accounting mismatch: expected %zu bytes, got %zu\n",
                expected,
                loader.size_data);
        return false;
    }

    ggml_backend_buffer_ptr accounting_buffer {
        ggml_backend_alloc_ctx_tensors_from_buft(accounting_ctx_raw, cpu_buft)
    };
    ggml_backend_buffer_ptr synthetic_buffer {
        ggml_backend_alloc_ctx_tensors_from_buft(synthetic_ctx_raw, cpu_buft)
    };
    if (accounting_buffer == nullptr || synthetic_buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate CPU accounting buffers\n");
        return false;
    }

    std::vector<ggml_backend_buffer_t> accounting_buffers = make_buffer_list(accounting_buffer.get());
    std::vector<ggml_backend_buffer_t> synthetic_buffers = make_buffer_list(synthetic_buffer.get());
    progress_state progress;

    if (!loader.load_all_data(accounting_ctx_raw, accounting_buffers, nullptr, observe_progress, &progress)) {
        std::fprintf(stderr, "accounting context load was cancelled\n");
        return false;
    }
    if (loader.size_done != loader.size_data) {
        std::fprintf(
                stderr,
                "accounting context did not complete: loaded %zu of %zu bytes\n",
                loader.size_done,
                loader.size_data);
        return false;
    }
    if (!loader.load_all_data(synthetic_ctx_raw, synthetic_buffers, nullptr, observe_progress, &progress)) {
        std::fprintf(stderr, "synthetic context load was cancelled\n");
        return false;
    }
    // The final 1.0 callback is emitted only after load_all_data tears down the
    // mmap fragments. The post-completion synthetic-only context must not emit
    // it again, so a single completion proves that cleanup ran exactly once.
    if (!progress.bounded || !progress.monotonic || progress.completion_count != 1 ||
            progress.progressed_after_completion) {
        std::fprintf(stderr, "tied-weight progress lifecycle was invalid\n");
        return false;
    }
    return true;
}

static bool check_contextless_fallback(const char * model_path) {
    llama_model_loader loader = make_loader(model_path);
    size_t expected = 0;
    for (const auto & [_, weight] : loader.weights_map) {
        expected += ggml_nbytes(weight.tensor);
    }

    loader.init_mappings(false);
    if (loader.size_data != expected) {
        std::fprintf(
                stderr,
                "contextless accounting mismatch: expected %zu bytes, got %zu\n",
                expected,
                loader.size_data);
        return false;
    }
    return true;
}

static bool check_multilane_context_capacity(const char * model_path) {
    skippy_runtime_config config = skippy_runtime_config_default();
    config.ctx_size = 256;
    config.lane_count = 3;
    config.n_batch = 32;
    config.n_ubatch = 32;

    skippy_model * model = nullptr;
    skippy_error * error = nullptr;
    if (skippy_model_open(model_path, &config, &model, &error) != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "multi-lane model open failed: %s\n", error != nullptr ? error->message : "unknown error");
        skippy_error_free(error);
        return false;
    }

    skippy_session * session = nullptr;
    if (skippy_session_create(model, &session, &error) != SKIPPY_STATUS_OK) {
        std::fprintf(stderr, "multi-lane session create failed: %s\n", error != nullptr ? error->message : "unknown error");
        skippy_error_free(error);
        skippy_model_free(model, nullptr);
        return false;
    }

    const llama_context * context = skippy_session_llama_context(session);
    const bool valid = context != nullptr && llama_n_ctx(context) == 3 * 256;
    if (!valid) {
        std::fprintf(
                stderr,
                "multi-lane context capacity mismatch: expected %d cells, got %u\n",
                3 * 256,
                context != nullptr ? llama_n_ctx(context) : 0);
    }

    skippy_session_free(session, nullptr);
    skippy_model_free(model, nullptr);
    return valid;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL\n", argv[0]);
        return EXIT_FAILURE;
    }

    llama_backend_init();

    if (!check_context_load_lifecycle(argv[1]) ||
            !check_contextless_fallback(argv[1]) ||
            !check_exact_resident_selection(argv[1]) ||
            !check_multilane_context_capacity(argv[1])) {
        llama_backend_free();
        return EXIT_FAILURE;
    }

    progress_state progress;
    llama_model_params params = llama_model_default_params();
    params.load_mode = LLAMA_LOAD_MODE_MMAP;
    params.n_gpu_layers = 99;
    params.progress_callback = observe_progress;
    params.progress_callback_user_data = &progress;
    llama_model_loader accounting_loader = make_loader(argv[1]);
    std::unordered_set<std::string> resident_tensor_names;
    for (const auto & [name, _] : accounting_loader.weights_map) {
        resident_tensor_names.insert(name);
    }

    llama_model_loader_stage_selection selection;
    selection.layer_start = 0;
    selection.layer_end = 1;
    selection.resident_tensor_names = resident_tensor_names;
    llama_model * model = llama_model_load_from_file_with_selection(argv[1], params, &selection);

    const bool loaded = model != nullptr;
    if (model != nullptr) {
        llama_model_free(model);
    }
    llama_backend_free();

    if (!loaded) {
        std::fprintf(stderr, "selected model load failed\n");
        return EXIT_FAILURE;
    }
    if (!progress.bounded) {
        std::fprintf(stderr, "model-load progress escaped [0, 1]\n");
        return EXIT_FAILURE;
    }
    if (!progress.monotonic) {
        std::fprintf(stderr, "model-load progress regressed\n");
        return EXIT_FAILURE;
    }
    if (progress.completion_count != 1) {
        std::fprintf(
                stderr,
                "model-load completion callback count mismatch: expected 1, got %zu\n",
                progress.completion_count);
        return EXIT_FAILURE;
    }
    if (progress.progressed_after_completion) {
        std::fprintf(stderr, "model-load progress continued after completion\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
