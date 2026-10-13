#pragma once
#include <cstdio>
#include <filesystem>
#include <set>
#include "../src/llama-model.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#endif

// Extending a file leaves a hole on Linux, but NTFS allocates the whole range
// unless the file is marked sparse first, so each fixture would take its full
// logical size on disk (tens of GiB for the larger ones).
static void mark_graph_contract_fixture_sparse(const std::filesystem::path & path) {
#ifdef _WIN32
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    DWORD returned = 0;
    const bool sparse = file != INVALID_HANDLE_VALUE &&
            DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
    if (file != INVALID_HANDLE_VALUE) { CloseHandle(file); }
    if (!sparse) {
        fprintf(stderr, "warning: cannot mark %s sparse, it will take its full size on disk\n", path.string().c_str());
    }
#else
    (void) path;
#endif
}

// Sparse GGUFs carry synthetic tensor shapes only; they must never run inference.
static int save_graph_contract_fixture(llm_arch arch, const std::string & out, uint32_t layers, uint32_t width, uint32_t mtp_layers) {
    if (out.empty() || arch == LLM_ARCH_UNKNOWN || layers < 2 || width == 0) {
        throw std::runtime_error("graph contract fixture requires architecture, output, layers and width");
    }
    auto metadata = get_gguf_ctx(arch, moe_mandatory(arch) || arch == LLM_ARCH_NEMOTRON_H_MOE || arch == LLM_ARCH_INKLING || arch == LLM_ARCH_GRANITE_MOE, glm_dsa_indexshare_fixture::DEFAULT,
            256, 8, false, layers + mtp_layers, arch == LLM_ARCH_QWEN4EXP ? width / 4 : width);
    llama_model_saver kv(arch, metadata.get());
    if (arch == LLM_ARCH_BAILINGMOE3) {
        kv.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, std::vector<float>(layers, 4.0f));
        kv.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>(layers, 5.0f));
    }
    if (arch == LLM_ARCH_GEMMA4) {
        std::vector<uint32_t> pattern(layers);
        for (uint32_t i = 0; i < layers; ++i) { pattern[i] = i % 2 == 0; }
        kv.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        kv.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS, layers - 24);
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        kv.add_kv(LLM_KV_ADAPTER_COUNT, uint32_t(2));
        kv.add_kv(LLM_KV_ADAPTER_LORA_RANK, uint32_t(8));
        kv.add_kv(LLM_KV_ADAPTER_TOKEN_IDS_ACTIVATE, std::vector<uint32_t>{10, 11});
        kv.add_kv(LLM_KV_ADAPTER_TOKEN_IDS_SUBSTITUTE, std::vector<uint32_t>{12, 13});
    }
    if (arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_JINA_BERT_V2) {
        kv.add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, uint32_t(2));
        kv.add_kv(LLM_KV_CLASSIFIER_OUTPUT_LABELS, std::vector<std::string>{"score"});
    }
    if (arch == LLM_ARCH_QWEN3VL || arch == LLM_ARCH_QWEN3VLMOE) {
        kv.add_kv(LLM_KV_NUM_DEEPSTACK_LAYERS, uint32_t(3));
    }
    if (arch == LLM_ARCH_INKLING) {
        kv.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE, 1.0f);
        kv.add_kv(LLM_KV_INKLING_D_REL, uint32_t(16));
        kv.add_kv(LLM_KV_INKLING_REL_EXTENT, uint32_t(32));
        kv.add_kv(LLM_KV_INKLING_REL_EXTENT_SWA, uint32_t(16));
        kv.add_kv(LLM_KV_INKLING_SHORTCONV_KERNEL, uint32_t(3));
        kv.add_kv(LLM_KV_INKLING_DENSE_BLOCK_COUNT, uint32_t(1));
        kv.add_kv(LLM_KV_INKLING_LOGIT_SCALE_DENOM, 1.0f);
    }
    if (mtp_layers) {
        kv.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, mtp_layers);
        ggml_tensor probe{};
        probe.type = GGML_TYPE_F16;
        probe.ne[0] = 2 * width;
        probe.ne[1] = width;
        probe.ne[2] = probe.ne[3] = 1;
        ggml_format_name(&probe, "blk.%u.nextn.eh_proj.weight", layers);
        gguf_add_tensor(metadata.get(), &probe);
    }
    if (arch == LLM_ARCH_GLM_DSA) {
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(layers);
        for (uint32_t il = 0; il < layers; ++il) {
            indexer_types.push_back(il % 2 == 0 ? 1 : 0);
        }
        kv.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }
    if (arch == LLM_ARCH_ARWKV7 || arch == LLM_ARCH_RWKV7) {
        llama_model_saver kv(arch, metadata.get());
        kv.add_kv(LLM_KV_ATTENTION_DECAY_LORA_RANK, uint32_t(32));
        kv.add_kv(LLM_KV_ATTENTION_ICLR_LORA_RANK, uint32_t(32));
        kv.add_kv(LLM_KV_ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, uint32_t(32));
        kv.add_kv(LLM_KV_ATTENTION_GATE_LORA_RANK, uint32_t(32));
        kv.add_kv(LLM_KV_TOKEN_SHIFT_COUNT, uint32_t(arch == LLM_ARCH_ARWKV7 ? 1 : 2));
        kv.add_kv(LLM_KV_WKV_HEAD_SIZE, uint32_t(64));
        kv.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, width / 64);
        kv.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, width / 64);
    }
    auto params = llama_model_default_params();
    params.no_alloc = true;
    params.load_mtp = true;
    params.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), [](ggml_tensor *, void *) {}, nullptr, params));
    if (!model) { throw std::runtime_error("cannot construct synthetic graph contract model"); }
    const uint32_t runtime_width = arch == LLM_ARCH_DEEPSEEK4
            ? model->hparams.n_embd : model->hparams.n_embd_out();
    if (model->hparams.n_layer() != layers || (model->hparams.router_layer >= 0 ? 0 : model->hparams.n_layer_nextn) != mtp_layers ||
            runtime_width != width) {
        throw std::runtime_error("synthetic runtime dimensions differ from canary registry");
    }
    gguf_context_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), metadata.get());
    std::set<std::string> replaced_qkv;
    if (arch == LLM_ARCH_MIMO2) {
        for (uint32_t head = 0; head < mtp_layers; ++head) {
            for (const char * projection : {"q", "k", "v"}) {
                for (const char * suffix : {"weight", "scale", "input_scale"}) {
                    replaced_qkv.insert("blk." + std::to_string(layers + head) + ".attn_" + projection + "." + suffix);
                }
            }
        }
    }
    for (const auto & item : llama_internal_get_tensor_map(model.get())) {
        if (replaced_qkv.count(item.second->name)) { continue; }
        if (gguf_find_tensor(output.get(), item.second->name) < 0) {
            gguf_add_tensor(output.get(), item.second);
        }
    }
    if (arch == LLM_ARCH_MIMO2 && mtp_layers) {
        ggml_context_ptr fused(ggml_init({ggml_tensor_overhead() * mtp_layers, nullptr, true}));
        for (uint32_t head = 0; head < mtp_layers; ++head) {
            auto * qkv = ggml_new_tensor_2d(fused.get(), GGML_TYPE_F32, width, 3 * width);
            ggml_format_name(qkv, "blk.%u.attn_qkv.weight", layers + head);
            gguf_add_tensor(output.get(), qkv);
        }
    }
    const std::string path = out + "/graph-contract.gguf";
    if (!gguf_write_to_file(output.get(), path.c_str(), true)) {
        throw std::runtime_error("cannot write graph contract fixture");
    }
    uint64_t size = gguf_get_meta_size(output.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(output.get()); ++i) {
        size = std::max(size, uint64_t(gguf_get_meta_size(output.get()) +
                gguf_get_tensor_offset(output.get(), i) + gguf_get_tensor_size(output.get(), i)));
    }
    mark_graph_contract_fixture_sparse(path);
    std::filesystem::resize_file(path, size);
    return 0;
}
