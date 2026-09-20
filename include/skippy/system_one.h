#ifndef SKIPPY_SYSTEM_ONE_H
#define SKIPPY_SYSTEM_ONE_H

/**
 * @file skippy/system_one.h
 * @brief One-pass structured label scoring for diffusion decision models.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct skippy_model;

/**
 * Describes one answer slot in a diffusion canvas.
 *
 * `canvas_position` is zero-based. The label token IDs for this slot occupy
 * `[label_token_offset, label_token_offset + label_token_count)` in the
 * request's flattened label-token array.
 */
struct skippy_system_one_slot {
    uint32_t canvas_position;
    size_t label_token_offset;
    size_t label_token_count;
};

/**
 * @brief Returns the fixed answer-canvas length encoded by a DiffusionGemma model.
 *
 * Callers use this value to size and pad the canvas supplied to
 * `skippy_system_one_read`.
 */
LLAMA_API enum skippy_status skippy_system_one_canvas_length(
        struct skippy_model * model,
        size_t * out_canvas_token_count,
        struct skippy_error ** out_error);

/**
 * @brief Scores structured labels with one DiffusionGemma read.
 *
 * Runs a single zero-self-conditioning DiffusionGemma read and returns a
 * probability distribution over each slot's declared label tokens.
 *
 * Output probabilities use the same flattened layout as `label_token_ids`.
 * The model must be a complete, unsplit DiffusionGemma model with one runtime
 * execution lane. The canvas length must match `diffusion.canvas_length` in
 * the GGUF metadata.
 */
LLAMA_API enum skippy_status skippy_system_one_read(
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
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_SYSTEM_ONE_H
