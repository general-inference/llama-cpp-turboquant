#pragma once

#include "../llama-memory.h"
#include <cstdint>

class llama_kv_cache;
class llama_kv_cache_iswa;
class llama_memory_recurrent;
class llm_graph_result;

// The export adapters and capability query share this loaded-memory dispatch.
llama_kv_cache * skippy_memory_kv_adapter(llama_memory_t memory);
llama_kv_cache_iswa * skippy_memory_iswa_adapter(llama_memory_t memory);
llama_memory_recurrent * skippy_memory_recurrent_adapter(llama_memory_t memory);
uint32_t skippy_memory_cache_capabilities(llama_memory_t memory);
uint32_t skippy_graph_cache_capabilities(const llm_graph_result & result);
