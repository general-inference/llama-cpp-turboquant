#include "skippy/state_memory.h"
#include "skippy/state.h"
#include "skippy/runtime_types.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-graph.h"
#include "skippy/state_effects.h"
#include <typeinfo>

llama_kv_cache * skippy_memory_kv_adapter(llama_memory_t memory) {
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        return hybrid->get_mem_attn();
    }
    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
        return kv;
    }
    if (auto * dsa = dynamic_cast<llama_kv_cache_dsa *>(memory)) {
        return dsa->get_mla();
    }
    return nullptr;
}

llama_kv_cache_iswa * skippy_memory_iswa_adapter(llama_memory_t memory) {
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return iswa;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        return hybrid->get_mem_attn();
    }
    return nullptr;
}

llama_memory_recurrent * skippy_memory_recurrent_adapter(llama_memory_t memory) {
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory)) {
        return recurrent;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        return hybrid->get_mem_recr();
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        return hybrid->get_mem_recr();
    }
    return nullptr;
}

uint32_t skippy_memory_cache_capabilities(llama_memory_t memory) {
    if (!memory) {
        return 0;
    }
    // Exact types are intentional: an unknown derived class may own additional
    // state that a base-class adapter would silently omit. Never infer complete
    // coverage merely from a successful dynamic_cast or a nonempty export.
    const auto & type = typeid(*memory);
    if (type == typeid(llama_kv_cache) || type == typeid(llama_kv_cache_iswa)) {
        return SKIPPY_MEMORY_CACHE_RESIDENT | SKIPPY_MEMORY_CACHE_KV_RECURRENT | SKIPPY_MEMORY_CACHE_FULL_STATE;
    }
    if (type == typeid(llama_memory_recurrent)) {
        return skippy_memory_recurrent_adapter(memory)
                ? SKIPPY_MEMORY_CACHE_KV_RECURRENT | SKIPPY_MEMORY_CACHE_FULL_STATE : SKIPPY_MEMORY_CACHE_FULL_STATE;
    }
    if (type == typeid(llama_memory_hybrid) || type == typeid(llama_memory_hybrid_iswa)) {
        return skippy_memory_recurrent_adapter(memory) &&
                (skippy_memory_kv_adapter(memory) || skippy_memory_iswa_adapter(memory))
                ? SKIPPY_MEMORY_CACHE_KV_RECURRENT | SKIPPY_MEMORY_CACHE_FULL_STATE : SKIPPY_MEMORY_CACHE_FULL_STATE;
    }
    // DSA/MSA memories include an indexer tier. Until a real-model resident
    // save/restore proves that tier survives, use complete state snapshots.
    return SKIPPY_MEMORY_CACHE_FULL_STATE;
}

uint32_t skippy_graph_cache_capabilities(const llm_graph_result & result) {
    if (!result.get_gf()) {
        return 0;
    }
    for (const auto & input : result.inputs) {
        if (input->has_unserialized_context_state()) {
            return 0;
        }
    }
    std::vector<skippy_state_effect> effects;
    std::string error;
    if (!skippy_infer_state_effects(result.get_state_effects(), result.get_gf(),
                result.get_block_boundaries(), effects, error)) {
        return 0;
    }
    // A phase with no persistent effects adds no cache restrictions. It must
    // not be confused with failed graph admission or unexported context state.
    uint32_t supported = SKIPPY_MEMORY_CACHE_RESIDENT | SKIPPY_MEMORY_CACHE_KV_RECURRENT |
            SKIPPY_MEMORY_CACHE_FULL_STATE;
    for (const auto & effect : effects) {
        switch (effect.kind) {
            case skippy_state_kind::kv_key:
            case skippy_state_kind::kv_value:
                break;
            case skippy_state_kind::recurrent_conv:
            case skippy_state_kind::recurrent_ssm:
                supported &= ~SKIPPY_MEMORY_CACHE_RESIDENT;
                break;
            case skippy_state_kind::derived_persistent:
                supported &= SKIPPY_MEMORY_CACHE_FULL_STATE;
                break;
        }
    }
    return supported;
}

extern "C" uint32_t skippy_model_memory_cache_capabilities(const skippy_model * model) {
    return model && model->ctx
            ? model->ctx->admit_skippy_cache_capabilities(
                skippy_memory_cache_capabilities(model->ctx->get_memory()))
            : 0;
}
