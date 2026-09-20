#include "skippy/system_one.h"

#include "skippy/errors.h"
#include "skippy/graph_build_inputs.h"
#include "skippy/runtime_support.h"
#include "skippy/runtime_types.h"

#include "models/models.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

namespace {

enum skippy_status system_one_fail(
        skippy_error ** out_error,
        enum skippy_status status,
        const char * message) {
    skippy_set_error(out_error, status, message);
    return status;
}

class system_one_scope {
public:
    system_one_scope(skippy_model * model, int32_t lane) : model(model), lane(lane) {}

    ~system_one_scope() {
        if (model != nullptr && model->model != nullptr) {
            llama_diffusion_set_sc(model->model, nullptr, 0.0f, 1.0f, false);
            llama_diffusion_set_phase(model->model, 0, 0, 0);
        }
        skippy_release_execution_lane(model, lane);
    }

private:
    skippy_model * model;
    int32_t lane;
};

class system_one_batch_guard {
public:
    explicit system_one_batch_guard(llama_batch value) : batch(value) {}

    ~system_one_batch_guard() { llama_batch_free(batch); }

    system_one_batch_guard(const system_one_batch_guard &) = delete;
    system_one_batch_guard & operator=(const system_one_batch_guard &) = delete;

    llama_batch batch;
};

} // namespace

extern "C" enum skippy_status skippy_system_one_read(
        struct skippy_model * model,
        const llama_token * prompt_tokens,
        size_t prompt_token_count,
        const llama_token * canvas_tokens,
        size_t canvas_token_count,
        const llama_token * label_token_ids,
        size_t label_token_count,
        const struct skippy_system_one_slot * slots,
        size_t slot_count,
        float * out_probabilities,
        size_t output_capacity,
        size_t * out_output_count,
        struct skippy_error ** out_error) {
    if (out_output_count != nullptr) {
        *out_output_count = label_token_count;
    }
    if (model == nullptr || model->model == nullptr || model->ctx == nullptr ||
            prompt_tokens == nullptr || prompt_token_count == 0 ||
            canvas_tokens == nullptr || canvas_token_count == 0 ||
            label_token_ids == nullptr || label_token_count == 0 ||
            slots == nullptr || slot_count == 0 || out_output_count == nullptr) {
        return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "model, prompt, canvas, labels, slots, and out_output_count are required");
    }
    if (out_probabilities == nullptr || output_capacity < label_token_count) {
        return system_one_fail(out_error, SKIPPY_STATUS_BUFFER_TOO_SMALL,
                "system-one probability output buffer is too small");
    }
    if (model->model->arch != LLM_ARCH_DIFFUSION_GEMMA) {
        return system_one_fail(out_error, SKIPPY_STATUS_UNSUPPORTED,
                "system-one reads require a DiffusionGemma model");
    }
    if (skippy_runtime_has_stage_plan(&model->config)) {
        return system_one_fail(out_error, SKIPPY_STATUS_UNSUPPORTED,
                "system-one reads require a complete unsplit model");
    }
    if (model->lane_count != 1) {
        return system_one_fail(out_error, SKIPPY_STATUS_UNSUPPORTED,
                "system-one reads currently require exactly one runtime execution lane");
    }
    if (prompt_token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            canvas_token_count > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            prompt_token_count + canvas_token_count > llama_n_ctx(model->ctx)) {
        return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "system-one prompt and canvas exceed the native context capacity");
    }

    auto * diffusion_model = dynamic_cast<llama_model_diffusion_gemma *>(model->model);
    if (diffusion_model == nullptr || canvas_token_count != diffusion_model->canvas_length) {
        return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "canvas token count does not match diffusion.canvas_length");
    }
    if (canvas_token_count > llama_n_ubatch(model->ctx)) {
        return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                "canvas length exceeds the configured native micro-batch size");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model->model));
    try {
        std::vector<uint8_t> seen_canvas_positions(canvas_token_count, 0);
        std::vector<uint8_t> seen_label_positions(label_token_count, 0);
        for (size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
            const skippy_system_one_slot & slot = slots[slot_index];
            if (slot.canvas_position >= canvas_token_count ||
                    slot.label_token_count == 0 ||
                    slot.label_token_offset > label_token_count ||
                    slot.label_token_count > label_token_count - slot.label_token_offset) {
                return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "system-one slot is outside the canvas or flattened label-token array");
            }
            if (seen_canvas_positions[slot.canvas_position] != 0) {
                return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                        "system-one canvas positions must be unique");
            }
            seen_canvas_positions[slot.canvas_position] = 1;
            for (size_t label_index = 0; label_index < slot.label_token_count; ++label_index) {
                const size_t flattened_index = slot.label_token_offset + label_index;
                if (seen_label_positions[flattened_index] != 0) {
                    return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                            "system-one label ranges must not overlap");
                }
                seen_label_positions[flattened_index] = 1;
                const llama_token token = label_token_ids[flattened_index];
                if (token < 0 || token >= n_vocab) {
                    return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                            "system-one label token is outside the model vocabulary");
                }
            }
        }
        if (std::find(seen_label_positions.begin(), seen_label_positions.end(), 0) != seen_label_positions.end()) {
            return system_one_fail(out_error, SKIPPY_STATUS_INVALID_ARGUMENT,
                    "system-one label ranges must cover the flattened label-token array exactly");
        }

        const int32_t lane = skippy_claim_execution_lane(model);
        if (lane < 0) {
            return system_one_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                    "the system-one execution lane is busy");
        }
        system_one_scope scope(model, lane);

        skippy_synchronize_context_if_pending(model);
        llama_set_causal_attn(model->ctx, false);
        llama_diffusion_set_sc(model->model, nullptr, 0.0f, 1.0f, false);

        const int32_t prompt_count = static_cast<int32_t>(prompt_token_count);
        const int32_t canvas_count = static_cast<int32_t>(canvas_token_count);
        const int32_t ubatch_size = std::max<int32_t>(1, static_cast<int32_t>(llama_n_ubatch(model->ctx)));
        const int32_t batch_capacity = std::max(ubatch_size, canvas_count);
        system_one_batch_guard batch_guard(
                llama_batch_init(batch_capacity, 0, 1));
        llama_batch & batch = batch_guard.batch;
        const skippy_graph_build_inputs graph_inputs = skippy_graph_build_inputs_from_config(&model->config);

        for (int32_t offset = 0; offset < prompt_count; offset += ubatch_size) {
            const int32_t chunk_count = std::min(ubatch_size, prompt_count - offset);
            llama_diffusion_set_phase(model->model, 1, prompt_count, offset);
            batch.n_tokens = chunk_count;
            for (int32_t index = 0; index < chunk_count; ++index) {
                batch.token[index] = prompt_tokens[offset + index];
                batch.pos[index] = offset + index;
                batch.n_seq_id[index] = 1;
                batch.seq_id[index][0] = lane;
                batch.logits[index] = index == chunk_count - 1 ? 1 : 0;
            }
            if (model->ctx->decode(batch, graph_inputs) != 0) {
                return system_one_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                        "DiffusionGemma prompt prefill failed");
            }
        }

        llama_diffusion_set_phase(model->model, 2, prompt_count, 0);
        batch.n_tokens = canvas_count;
        for (int32_t index = 0; index < canvas_count; ++index) {
            batch.token[index] = canvas_tokens[index];
            batch.pos[index] = prompt_count + index;
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = lane;
            batch.logits[index] = seen_canvas_positions[index] != 0 ? 1 : 0;
        }
        if (model->ctx->decode(batch, graph_inputs) != 0) {
            return system_one_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                    "DiffusionGemma canvas read failed");
        }
        skippy_mark_context_work_pending(model);
        model->ctx->synchronize();
        skippy_mark_context_synchronized(model);

        for (size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
            const skippy_system_one_slot & slot = slots[slot_index];
            const float * logits = llama_get_logits_ith(model->ctx, static_cast<int32_t>(slot.canvas_position));
            if (logits == nullptr) {
                return system_one_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                        "DiffusionGemma did not return logits for a requested canvas slot");
            }

            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t label_index = 0; label_index < slot.label_token_count; ++label_index) {
                const llama_token token = label_token_ids[slot.label_token_offset + label_index];
                maximum = std::max(maximum, logits[token]);
            }
            float denominator = 0.0f;
            for (size_t label_index = 0; label_index < slot.label_token_count; ++label_index) {
                const llama_token token = label_token_ids[slot.label_token_offset + label_index];
                const float value = std::exp(logits[token] - maximum);
                out_probabilities[slot.label_token_offset + label_index] = value;
                denominator += value;
            }
            if (!(denominator > 0.0f) || !std::isfinite(denominator)) {
                return system_one_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                        "DiffusionGemma returned a non-finite label distribution");
            }
            for (size_t label_index = 0; label_index < slot.label_token_count; ++label_index) {
                out_probabilities[slot.label_token_offset + label_index] /= denominator;
            }
        }

        return skippy_success(out_error);
    } catch (const std::exception & exception) {
        return system_one_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR, exception.what());
    } catch (...) {
        return system_one_fail(out_error, SKIPPY_STATUS_RUNTIME_ERROR,
                "unexpected native exception during system-one read");
    }
}
