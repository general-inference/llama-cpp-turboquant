#pragma once

#include <cstdint>

struct ggml_tensor;

// Preserve the admitted token axis when runtime dimensions are ambiguous.
int32_t skippy_activation_token_axis(const ggml_tensor * tensor, uint32_t n_tokens, int32_t prepared_axis);
int32_t skippy_admitted_token_axis(const ggml_tensor * tensor, uint32_t n_tokens, int32_t admitted_axis);

struct llama_model;
struct llm_graph_params;
class llm_graph_result;

// Replace the canonical whole-model graph with the exact admitted semantic
// slice. This is the only runtime graph-filtering authority: model builders
// remain stage-independent and merely register block/state/input semantics.
void skippy_realize_stage_graph(
        const llama_model & model,
        const llm_graph_params & params,
        llm_graph_result & result);
