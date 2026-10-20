#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"
#include "skippy.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
             const char *   name_tag) :
    model(model), hparams(hparams), v_trans(v_trans),
    key_type(type_k), value_type(type_v), tensor_name_tag(name_tag),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    if (type_k == GGML_TYPE_TURBO4_0) {
        throw std::runtime_error("Turbo4 is supported only for the V cache");
    }
    if (type_v == GGML_TYPE_TURBO4_0) {
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("Turbo4 V cache does not support tensor split mode");
        }
        if (type_k != GGML_TYPE_Q8_0) {
            throw std::runtime_error("Turbo4 V cache requires Q8_0 K cache");
        }
        if (v_trans) {
            throw std::runtime_error("Turbo4 V cache requires flash attention");
        }
        if (hparams.is_mla()) {
            throw std::runtime_error("Turbo4 V cache requires separate V storage");
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (hparams.has_kv(il) &&
                    (hparams.n_embd_head_v(il) == 0 || hparams.n_embd_head_v(il) % 128 != 0)) {
                throw std::runtime_error("Turbo4 V cache requires V head dimensions divisible by 128");
            }
        }
    }

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(3u*(1 + n_stream)*n_layer*ggml_tensor_overhead()), //Reserve tensor metadata for up to 3 tensors per layer (K, V, and optional K_idx), plus one view per tensor per stream.
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (n_embd_head_v_all == 0) {
            n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
        } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
            n_embd_head_v_all = -1;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            if (type_v == GGML_TYPE_TURBO4_0) {
                const char * backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
                if (strcmp(backend, "CPU") != 0 && strcmp(backend, "MTL") != 0) {
                    throw std::runtime_error("Turbo4 V cache requires CPU or Metal placement");
                }
            }
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        const uint32_t n_embd_k_idx = hparams.n_embd_k_idx(il);
        ggml_tensor * k_idx = n_embd_k_idx > 0
            ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_k_idx, kv_size, n_stream)
            : nullptr;
        if (k_idx) {
            ggml_format_name(k_idx, "cache_k_idx_l%d", il);
            msa_strict_slots = (n_stream == n_seq_max);
        }

        std::vector<ggml_tensor *> k_idx_stream;
        for (uint32_t s = 0; s < n_stream; ++s) {
            k_idx_stream.push_back(k_idx
                ? ggml_view_2d(ctx, k_idx, n_embd_k_idx, kv_size, k_idx->nb[1], s*k_idx->nb[2])
                : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_idx, k_stream, v_stream, k_idx_stream });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k     = size_k_bytes();
        const size_t memory_size_v     = size_v_bytes();
        const size_t memory_size_k_idx = size_k_idx_bytes();
        const size_t memory_size_total = memory_size_k + memory_size_v + memory_size_k_idx;

        constexpr float mib = 1024.0f * 1024.0f;

        const std::string k_log = format(", K (%s): %7.2f MiB", ggml_type_name(type_k), (float) memory_size_k / mib);
        const std::string v_log = format(", V (%s): %7.2f MiB", ggml_type_name(type_v), (float) memory_size_v / mib);

        std::string k_idx_log;
        if (memory_size_k_idx > 0) {
            k_idx_log = format(", K_idx (%s): %7.2f MiB", ggml_type_name(GGML_TYPE_F32), (float) memory_size_k_idx / mib);
        }

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs)%s%s%s\n", __func__,
                (float) memory_size_total / mib, kv_size, (int) layers.size(), n_seq_max, n_stream,
                k_log.c_str(), v_log.c_str(), k_idx_log.c_str());
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        n_rot_k = other->n_rot_k;
        n_rot_v = other->n_rot_v;
    } else {
        // two knobs turn the same thing off: the environment variable, for the whole process, and
        //   the model setting, for the contexts of one model - see llama_model_set_attn_rot
        const char * env_rot_disable = getenv("LLAMA_ATTN_ROT_DISABLE");

        const bool attn_rot_disable = !model.attn_rot || (env_rot_disable != nullptr && atoi(env_rot_disable) != 0);
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (%s)\n", __func__,
                    model.attn_rot ? "LLAMA_ATTN_ROT_DISABLE" : "llama_model_set_attn_rot");
        }

        bool attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        const bool attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            type_v != GGML_TYPE_TURBO4_0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;

        if (attn_rot_k) {
            n_rot_k = 64;

            // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
            while (n_embd_head_k_all % (2*n_rot_k) == 0) {
                n_rot_k *= 2;
            }
        }
        if (attn_rot_v) {
            n_rot_v = 64;
            // using smaller rotation matrices for V seems beneficial
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
            //while (hparams.n_embd_head_v() % (2*n_rot_v) == 0) {
            //    n_rot_v *= 2;
            //}
        }
    }

    LLAMA_LOG_INFO("%s: n_rot_k = %u, n_embd_head_k_all = %d\n", __func__, n_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: n_rot_v = %u, n_embd_head_k_all = %d\n", __func__, n_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (n_rot_k || n_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // empty range - nothing to remove
    if (p0 >= p1) {
        return true;
    }

    // MSA anchors block selection to absolute cache slots (slot == position). Tail trim and full removal preserve this invariant, but removing a prefix
    // or middle range would free slots while later cells survive, desynchronizing the indexer cache. Reject such removals before modifying the cache.
    if (msa_strict_slots) {
        for (llama_seq_id sid = 0; sid < (llama_seq_id) seq_to_stream.size(); ++sid) {
            if (seq_id >= 0 && sid != seq_id) {
                continue;
            }

            const auto & cells = v_cells[seq_to_stream[sid]];

            const llama_pos pmin = cells.seq_pos_min(sid);
            const llama_pos pmax = cells.seq_pos_max(sid);

            if (pmin < 0) {
                continue;   // empty sequence
            }

            const bool overlaps    = p0 <= pmax && p1 > pmin;   // the range removes something
            const bool leaves_tail = p1 <= pmax;                // cells beyond the range survive

            if (overlaps && leaves_tail) {
                LLAMA_LOG_WARN("%s: MSA: partial (non-suffix) removal [%d, %d) for seq %d is not supported "
                        "(block selection is anchored to cache slots) - rejected\n", __func__, p0, p1, sid);
                return false;
            }
        }
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : embd_all ? balloc.split_seq(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    bool do_shift = get_has_shift();
    bool did_compact = false;

    if (optimize) {
        if (lctx != nullptr) {
            lctx->synchronize();
        }

        bool has_indexer_keys = false;
        for (const auto & layer : layers) {
            if (layer.k_idx != nullptr) {
                has_indexer_keys = true;
                break;
            }
        }

        if (!has_indexer_keys) {
            const auto moves = compact_cells();
            did_compact = !moves.empty();
            if (did_compact) {
                copy_compacted_cells(moves);
                LLAMA_LOG_DEBUG("%s: compacted %zu KV cells\n", __func__, moves.size());
            }
        }
    }

    return std::make_unique<llama_kv_cache_context>(
            this,
            lctx,
            do_shift,
            std::move(sc_info),
            did_compact);
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
                if (layer.k_idx_stream[ssrc]) {
                    GGML_ASSERT(layer.k_idx_stream[sdst]);
                    ggml_backend_tensor_copy(layer.k_idx_stream[ssrc], layer.k_idx_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        // MSA block selection assumes slot == logical position (append-only streams).
        if (msa_strict_slots) {
            for (uint32_t ii = 0; ii < n_tokens; ++ii) {
                const llama_pos pos = ubatch.pos[s*n_tokens + ii];

                if (pos < 0 || (uint64_t) pos >= cells.size()) {
                    LLAMA_LOG_WARN("%s: MSA: position %d is outside the cache range [0, %u)\n",
                            __func__, pos, cells.size());
                    return { };
                }

                const uint32_t idx = (uint32_t) pos;

                if (!cells.is_empty(idx)) {
                    LLAMA_LOG_WARN("%s: MSA: required slot %u is already occupied (stream %u)\n",
                            __func__, idx, seq_to_stream[seq_id]);
                    return { };
                }

                // strictly increasing positions, rules out duplicates and, for contiguous requests, is tightened to exact adjacency
                if (!res.idxs[s].empty() && (cont ? idx != res.idxs[s].back() + 1
                                                  : idx <= res.idxs[s].back())) {
                    LLAMA_LOG_WARN("%s: MSA: token positions are not %s within the ubatch\n",
                            __func__, cont ? "contiguous" : "strictly increasing");
                    return { };
                }

                res.idxs[s].push_back(idx);
            }

            continue;
        }

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (msa_strict_slots && (llama_pos) idx != ubatch.pos[i]) {
                LLAMA_LOG_ERROR("%s: MSA slot/position invariant violated: "
                        "writing pos %d into cell %u (stream %u). The indexer cache "
                        "would desync and block selection would silently corrupt. "
                        "This is a bug, please report it with reproduction steps.\n",
                        __func__, ubatch.pos[i], idx, sinfo.strm[s]);
                GGML_ABORT("MSA: slot != pos");
            }

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);

                if (!is_embd) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                    // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                    ext.tok = hparams.ple_image_token_id != 0
                        ? (llama_token) hparams.ple_image_token_id
                        : (llama_token) hparams.ple_eos_token_id;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            // under MSA strict slots this path should be unreachable, since strict MSA placement never selects occupied cells
            GGML_ASSERT(seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1));
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

std::vector<llama_kv_cache::compaction_move> llama_kv_cache::compact_cells() {
    std::vector<compaction_move> moves;

    for (uint32_t s = 0; s < n_stream; ++s) {
        auto & cells = v_cells[s];
        uint32_t dst = 0;

        for (uint32_t src = 0; src < cells.size(); ++src) {
            if (cells.is_empty(src)) {
                continue;
            }
            if (src != dst) {
                moves.push_back({ s, src, dst });
                cells.mv(src, dst);
            }
            ++dst;
        }

        v_heads[s] = dst;
    }

    return moves;
}

static void llama_kv_cache_copy_tensor_bytes(
        ggml_tensor * tensor,
        size_t src_offset,
        size_t dst_offset,
        size_t size) {
    if (tensor == nullptr || size == 0 || src_offset == dst_offset) {
        return;
    }

    std::vector<uint8_t> bytes(size);
    ggml_backend_tensor_get(tensor, bytes.data(), src_offset, size);
    ggml_backend_tensor_set(tensor, bytes.data(), dst_offset, size);
}

void llama_kv_cache::copy_compacted_cells(const std::vector<compaction_move> & moves) const {
    for (const auto & layer : layers) {
        for (const compaction_move & move : moves) {
            if (move.strm >= layer.k_stream.size()) {
                continue;
            }

            auto * k = layer.k_stream[move.strm];
            if (k != nullptr) {
                const size_t k_row_bytes = ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il));
                llama_kv_cache_copy_tensor_bytes(
                        k,
                        static_cast<size_t>(move.src) * k_row_bytes,
                        static_cast<size_t>(move.dst) * k_row_bytes,
                        k_row_bytes);
            }

            if (move.strm >= layer.v_stream.size()) {
                continue;
            }

            auto * v = layer.v_stream[move.strm];
            if (v == nullptr) {
                continue;
            }

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer.il);
            if (!v_trans) {
                const size_t v_row_bytes = ggml_row_size(v->type, n_embd_v_gqa);
                llama_kv_cache_copy_tensor_bytes(
                        v,
                        static_cast<size_t>(move.src) * v_row_bytes,
                        static_cast<size_t>(move.dst) * v_row_bytes,
                        v_row_bytes);
            } else {
                const size_t v_element_bytes = ggml_type_size(v->type);
                const size_t cells_per_stream = v_cells[move.strm].size();
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    llama_kv_cache_copy_tensor_bytes(
                            v,
                            (static_cast<size_t>(move.src) + static_cast<size_t>(j) * cells_per_stream) * v_element_bytes,
                            (static_cast<size_t>(move.dst) + static_cast<size_t>(j) * cells_per_stream) * v_element_bytes,
                            v_element_bytes);
                }
            }
        }
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    // shifting would leave k_idx stale
    for (const auto & layer : layers) {
        if (layer.k_idx) {
            return false;
        }
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_seq_max() const {
    return n_seq_max;
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

ggml_tensor * llama_kv_cache::get_v_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].v;
}

ggml_tensor * llama_kv_cache::graph_k_storage(ggml_context * ctx, int32_t il) const {
    const auto found = map_layer_ids.find(il);
    if (found != map_layer_ids.end()) {
        return layers[found->second].k;
    }
    const std::string name = format("cache_%sk_l%d", tensor_name_tag.c_str(), il);
    if (ggml_tensor * existing = ggml_get_tensor(ctx, name.c_str())) {
        return existing;
    }
    ggml_tensor * result = ggml_new_tensor_3d(
            ctx, key_type, hparams.n_embd_k_gqa(il), get_size(), get_n_stream());
    ggml_set_name(result, name.c_str());
    return result;
}

ggml_tensor * llama_kv_cache::graph_v_storage(ggml_context * ctx, int32_t il) const {
    const auto found = map_layer_ids.find(il);
    if (found != map_layer_ids.end()) {
        return layers[found->second].v;
    }
    if (hparams.is_mla()) {
        return nullptr;
    }
    const std::string name = format("cache_%sv_l%d", tensor_name_tag.c_str(), il);
    if (ggml_tensor * existing = ggml_get_tensor(ctx, name.c_str())) {
        return existing;
    }
    const uint32_t n_embd_v_gqa = !v_trans
            ? hparams.n_embd_v_gqa(il)
            : hparams.n_embd_v_gqa_max();
    ggml_tensor * result = ggml_new_tensor_3d(
            ctx, value_type, n_embd_v_gqa, get_size(), get_n_stream());
    ggml_set_name(result, name.c_str());
    return result;
}

ggml_tensor * llama_kv_cache::graph_k_idx_storage(ggml_context * ctx, int32_t il) const {
    const auto found = map_layer_ids.find(il);
    if (found != map_layer_ids.end()) {
        return layers[found->second].k_idx;
    }
    const uint32_t n_embd_k_idx = hparams.n_embd_k_idx(il);
    if (n_embd_k_idx == 0) {
        return nullptr;
    }
    const std::string name = format("cache_k_idx_l%d", il);
    if (ggml_tensor * existing = ggml_get_tensor(ctx, name.c_str())) {
        return existing;
    }
    ggml_tensor * result = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, n_embd_k_idx, get_size(), get_n_stream());
    ggml_set_name(result, name.c_str());
    return result;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

uint32_t llama_kv_cache::get_stream(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return seq_to_stream[seq_id];
}

uint64_t llama_kv_cache::get_used_cells() const {
    uint64_t used = 0;
    for (const auto & cells : v_cells) {
        used += cells.get_used();
    }
    return used;
}

struct llama_kv_cache_cell_run {
    uint32_t first;
    uint32_t count;
};

static std::vector<llama_kv_cache_cell_run> llama_kv_cache_cell_runs(const std::vector<uint32_t> & cell_idxs) {
    std::vector<llama_kv_cache_cell_run> runs;
    runs.reserve(cell_idxs.size());

    for (uint32_t cell_idx : cell_idxs) {
        if (!runs.empty() && cell_idx == runs.back().first + runs.back().count) {
            ++runs.back().count;
        } else {
            runs.push_back({ cell_idx, 1 });
        }
    }

    return runs;
}

bool llama_kv_cache::stage_export_kv_page(
        llama_seq_id seq_id,
        int32_t layer_start,
        int32_t layer_end,
        uint64_t token_start,
        uint64_t token_count,
        skippy_kv_page_desc * out_desc,
        void * output,
        size_t output_capacity,
        size_t * out_bytes,
        std::string & error) const {
    if (out_desc == nullptr || out_bytes == nullptr) {
        error = "out_desc and out_bytes are required";
        return false;
    }
    if (layer_start < 0 || layer_end <= layer_start) {
        error = "invalid layer range";
        return false;
    }
    if (token_count == 0) {
        error = "token_count must be greater than zero";
        return false;
    }
    if (token_start > static_cast<uint64_t>(std::numeric_limits<llama_pos>::max()) ||
            token_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            token_start + token_count < token_start ||
            token_start + token_count > static_cast<uint64_t>(std::numeric_limits<llama_pos>::max())) {
        error = "token range is too large";
        return false;
    }
    if (seq_to_stream.empty() || v_cells.empty()) {
        error = "KV cache has no stream";
        return false;
    }
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= seq_to_stream.size()) {
        error = "sequence id is mapped to an invalid KV stream";
        return false;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    if (strm >= v_cells.size()) {
        error = "sequence id is mapped to an invalid KV stream";
        return false;
    }
    bool has_selected_layer = false;
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        if (il >= static_cast<uint32_t>(layer_start) &&
                il < static_cast<uint32_t>(layer_end) &&
                layer.k_stream.size() > strm &&
                layer.k_stream[strm] != nullptr) {
            has_selected_layer = true;
            break;
        }
    }
    if (!has_selected_layer) {
        error = "no KV cache layers selected by layer range";
        return false;
    }

    const auto & cells = v_cells[strm];
    if (token_count > static_cast<uint64_t>(cells.size())) {
        error = "token_count exceeds KV cache capacity";
        return false;
    }

    std::vector<uint32_t> cell_idxs(static_cast<size_t>(token_count), std::numeric_limits<uint32_t>::max());
    uint32_t found_cells = 0;
    const llama_pos pos_start = static_cast<llama_pos>(token_start);
    const llama_pos pos_end = static_cast<llama_pos>(token_start + token_count);
    for (uint32_t i = 0; i < cells.size() && found_cells < token_count; ++i) {
        if (!cells.seq_has(i, seq_id) || !cells.pos_in(i, pos_start, pos_end)) {
            continue;
        }
        const size_t token_index = static_cast<size_t>(cells.pos_get(i) - pos_start);
        if (token_index < cell_idxs.size() && cell_idxs[token_index] == std::numeric_limits<uint32_t>::max()) {
            cell_idxs[token_index] = i;
            ++found_cells;
        }
    }
    if (found_cells != token_count) {
        error = "requested token position is not present in the KV cache";
        return false;
    }

    std::vector<const kv_layer *> selected;
    selected.reserve(layers.size());
    uint32_t k_row_bytes = 0;
    uint32_t v_row_bytes = 0;
    uint32_t v_element_bytes = 0;
    uint32_t k_idx_row_bytes = 0;
    uint32_t k_type = GGML_TYPE_COUNT;
    uint32_t v_type = GGML_TYPE_COUNT;
    uint64_t payload_bytes = 0;

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        if (il < static_cast<uint32_t>(layer_start) || il >= static_cast<uint32_t>(layer_end)) {
            continue;
        }
        if (layer.k_stream.size() <= strm || layer.k_stream[strm] == nullptr) {
            continue;
        }
        const uint32_t n_embd_v = hparams.n_embd_v_gqa(il);
        const bool has_v_stream = layer.v_stream.size() > strm && layer.v_stream[strm] != nullptr;
        const bool has_k_idx_stream = layer.k_idx_stream.size() > strm && layer.k_idx_stream[strm] != nullptr;

        const size_t k_row = ggml_row_size(layer.k_stream[strm]->type, hparams.n_embd_k_gqa(il));
        const size_t v_row = has_v_stream && !v_trans ? ggml_row_size(layer.v_stream[strm]->type, n_embd_v) : 0;
        const size_t v_el  = has_v_stream &&  v_trans ? ggml_type_size(layer.v_stream[strm]->type) : 0;
        // MSA/DSA indexer keys are F32 rows of width n_embd_k_idx
        const size_t k_idx_row = has_k_idx_stream ? ggml_row_size(layer.k_idx_stream[strm]->type, hparams.n_embd_k_idx(il)) : 0;
        const uint32_t layer_v_type = has_v_stream ? static_cast<uint32_t>(layer.v_stream[strm]->type) : GGML_TYPE_F32;

        if (k_row > std::numeric_limits<uint32_t>::max() ||
                v_row > std::numeric_limits<uint32_t>::max() ||
                v_el > std::numeric_limits<uint32_t>::max() ||
                k_idx_row > std::numeric_limits<uint32_t>::max()) {
            error = "KV row size exceeds descriptor capacity";
            return false;
        }
        if (selected.empty()) {
            k_row_bytes = static_cast<uint32_t>(k_row);
            v_row_bytes = static_cast<uint32_t>(v_row);
            v_element_bytes = static_cast<uint32_t>(v_el);
            k_idx_row_bytes = static_cast<uint32_t>(k_idx_row);
            k_type = static_cast<uint32_t>(layer.k_stream[strm]->type);
            v_type = layer_v_type;
        } else if (k_row_bytes != k_row ||
                v_row_bytes != v_row ||
                v_element_bytes != v_el ||
                k_idx_row_bytes != k_idx_row ||
                k_type != static_cast<uint32_t>(layer.k_stream[strm]->type) ||
                v_type != layer_v_type) {
            error = "native KV page export requires uniform K/V types and row sizes across selected layers";
            return false;
        }

        const uint64_t v_bytes = has_v_stream ?
            (v_trans ? static_cast<uint64_t>(n_embd_v) * token_count * v_el : token_count * v_row) :
            0;
        const uint64_t k_bytes = token_count * k_row;
        const uint64_t k_idx_bytes = token_count * k_idx_row;
        if (payload_bytes > std::numeric_limits<uint64_t>::max() - k_bytes ||
                payload_bytes + k_bytes > std::numeric_limits<uint64_t>::max() - v_bytes ||
                payload_bytes + k_bytes + v_bytes > std::numeric_limits<uint64_t>::max() - k_idx_bytes) {
            error = "KV page payload is too large";
            return false;
        }
        payload_bytes += k_bytes + v_bytes + k_idx_bytes;
        selected.push_back(&layer);
    }

    if (selected.empty()) {
        error = "no KV cache layers selected by layer range";
        return false;
    }
    if (payload_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        error = "KV page payload exceeds addressable memory";
        return false;
    }

    out_desc->version = 1;
    out_desc->layer_start = layer_start;
    out_desc->layer_end = layer_end;
    out_desc->token_start = token_start;
    out_desc->token_count = token_count;
    out_desc->layer_count = static_cast<uint32_t>(selected.size());
    out_desc->k_type = k_type;
    out_desc->v_type = v_type;
    out_desc->k_row_bytes = k_row_bytes;
    out_desc->v_row_bytes = v_row_bytes;
    out_desc->v_element_bytes = v_element_bytes;
    out_desc->k_idx_row_bytes = k_idx_row_bytes;
    out_desc->payload_bytes = payload_bytes;
    out_desc->flags = v_trans ? SKIPPY_KV_PAGE_FLAG_V_TRANSPOSED : 0;
    if (k_idx_row_bytes > 0) {
        out_desc->flags |= SKIPPY_KV_PAGE_FLAG_HAS_K_IDX;
    }
    *out_bytes = static_cast<size_t>(payload_bytes);

    if (output == nullptr) {
        return true;
    }
    if (output_capacity < static_cast<size_t>(payload_bytes)) {
        error = "output buffer is too small";
        return false;
    }

    const auto cell_runs = llama_kv_cache_cell_runs(cell_idxs);
    char * dst = static_cast<char *>(output);
    for (const auto * layer : selected) {
        auto * k = layer->k_stream[strm];
        for (const auto & run : cell_runs) {
            const size_t run_bytes = static_cast<size_t>(run.count) * k_row_bytes;
            ggml_backend_tensor_get(k, dst, static_cast<size_t>(run.first) * k_row_bytes, run_bytes);
            dst += run_bytes;
        }
    }

    if (!v_trans) {
        for (const auto * layer : selected) {
            if (v_row_bytes == 0) {
                continue;
            }
            auto * v = layer->v_stream[strm];
            for (const auto & run : cell_runs) {
                const size_t run_bytes = static_cast<size_t>(run.count) * v_row_bytes;
                ggml_backend_tensor_get(v, dst, static_cast<size_t>(run.first) * v_row_bytes, run_bytes);
                dst += run_bytes;
            }
        }
    } else {
        for (const auto * layer : selected) {
            if (v_element_bytes == 0) {
                continue;
            }
            auto * v = layer->v_stream[strm];
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer->il);
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & run : cell_runs) {
                    const size_t run_bytes = static_cast<size_t>(run.count) * v_element_bytes;
                    const size_t src_offset = (static_cast<size_t>(run.first) + static_cast<size_t>(j) * cells.size()) * v_element_bytes;
                    ggml_backend_tensor_get(v, dst, src_offset, run_bytes);
                    dst += run_bytes;
                }
            }
        }
    }

    // MSA/DSA indexer keys ride after the V payload, one F32 row per token per layer
    for (const auto * layer : selected) {
        if (k_idx_row_bytes == 0) {
            break;
        }
        auto * k_idx = layer->k_idx_stream[strm];
        for (const auto & run : cell_runs) {
            const size_t run_bytes = static_cast<size_t>(run.count) * k_idx_row_bytes;
            ggml_backend_tensor_get(k_idx, dst, static_cast<size_t>(run.first) * k_idx_row_bytes, run_bytes);
            dst += run_bytes;
        }
    }

    return true;
}

bool llama_kv_cache::stage_import_kv_page(
        llama_seq_id seq_id,
        const skippy_kv_page_desc & desc,
        const void * input,
        size_t input_bytes,
        std::string & error,
        bool validate_only) {
    if (input == nullptr) {
        error = "input payload is required";
        return false;
    }
    if (desc.version != 1) {
        error = "unsupported native KV page descriptor version";
        return false;
    }
    if (desc.layer_start < 0 ||
            desc.layer_end <= desc.layer_start ||
            static_cast<uint32_t>(desc.layer_end) > hparams.n_layer_all ||
            desc.token_count == 0) {
        error = "invalid native KV page descriptor";
        return false;
    }
    if (desc.payload_bytes != input_bytes) {
        error = "native KV page payload size does not match descriptor";
        return false;
    }
    if (desc.token_start > static_cast<uint64_t>(std::numeric_limits<llama_pos>::max()) ||
            desc.token_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            desc.token_start + desc.token_count < desc.token_start ||
            desc.token_start + desc.token_count > static_cast<uint64_t>(std::numeric_limits<llama_pos>::max())) {
        error = "token range is too large";
        return false;
    }
    if ((desc.flags & SKIPPY_KV_PAGE_FLAG_V_TRANSPOSED) != (v_trans ? SKIPPY_KV_PAGE_FLAG_V_TRANSPOSED : 0)) {
        error = "native KV page V transposition does not match runtime";
        return false;
    }
    if (seq_to_stream.empty() || v_cells.empty()) {
        error = "KV cache has no stream";
        return false;
    }
    if (seq_id < 0 || static_cast<size_t>(seq_id) >= seq_to_stream.size() || seq_id >= LLAMA_MAX_SEQ) {
        error = "sequence id is mapped to an invalid KV stream";
        return false;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    if (strm >= v_cells.size()) {
        error = "sequence id is mapped to an invalid KV stream";
        return false;
    }
    if (desc.token_count > static_cast<uint64_t>(v_cells[strm].size())) {
        error = "token_count exceeds KV cache capacity";
        return false;
    }

    std::vector<const kv_layer *> selected;
    selected.reserve(layers.size());
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        if (il < static_cast<uint32_t>(desc.layer_start) || il >= static_cast<uint32_t>(desc.layer_end)) {
            continue;
        }
        if (layer.k_stream.size() <= strm || layer.k_stream[strm] == nullptr) {
            continue;
        }
        const uint32_t n_embd_v = hparams.n_embd_v_gqa(il);
        const bool has_v_stream = layer.v_stream.size() > strm && layer.v_stream[strm] != nullptr;
        const bool has_k_idx_stream = layer.k_idx_stream.size() > strm && layer.k_idx_stream[strm] != nullptr;
        const size_t k_row = ggml_row_size(layer.k_stream[strm]->type, hparams.n_embd_k_gqa(il));
        const size_t v_row = has_v_stream && !v_trans ? ggml_row_size(layer.v_stream[strm]->type, n_embd_v) : 0;
        const size_t v_el  = has_v_stream &&  v_trans ? ggml_type_size(layer.v_stream[strm]->type) : 0;
        const size_t k_idx_row = has_k_idx_stream ? ggml_row_size(layer.k_idx_stream[strm]->type, hparams.n_embd_k_idx(il)) : 0;
        const uint32_t layer_v_type = has_v_stream ? static_cast<uint32_t>(layer.v_stream[strm]->type) : GGML_TYPE_F32;
        if (desc.k_type != static_cast<uint32_t>(layer.k_stream[strm]->type) ||
                desc.v_type != layer_v_type ||
                desc.k_row_bytes != k_row ||
                desc.v_row_bytes != v_row ||
                desc.v_element_bytes != v_el ||
                desc.k_idx_row_bytes != k_idx_row) {
            error = "native KV page descriptor does not match runtime KV cache layout";
            return false;
        }
        selected.push_back(&layer);
    }
    if (selected.empty()) {
        error = "no KV cache layers selected by layer range";
        return false;
    }
    if (selected.size() != desc.layer_count) {
        error = "native KV page layer count does not match runtime";
        return false;
    }

    uint64_t expected_bytes = 0;
    for (const auto * layer : selected) {
        expected_bytes += desc.token_count * desc.k_row_bytes;
        expected_bytes += v_trans ?
            static_cast<uint64_t>(hparams.n_embd_v_gqa(layer->il)) * desc.token_count * desc.v_element_bytes :
            desc.token_count * desc.v_row_bytes;
        expected_bytes += desc.token_count * desc.k_idx_row_bytes;
    }
    if (expected_bytes != desc.payload_bytes) {
        error = "native KV page descriptor payload size is inconsistent";
        return false;
    }
    if (validate_only) {
        return true;
    }

    llama_ubatch ubatch = {};
    auto udata = std::make_shared<llama_ubatch::data_t>();
    const uint32_t n_tokens = static_cast<uint32_t>(desc.token_count);
    const uint32_t n_pos = hparams.n_pos_per_embd();
    if (n_pos != 0 && n_tokens > std::numeric_limits<size_t>::max() / n_pos) {
        error = "token position count exceeds addressable memory";
        return false;
    }
    udata->token.resize(n_tokens);
    udata->pos.resize(static_cast<size_t>(n_tokens) * n_pos);
    udata->n_seq_id.resize(n_tokens, 1);
    udata->seq_id.resize(n_tokens);
    udata->seq_id_unq = { seq_id };
    udata->seq_idx.resize(LLAMA_MAX_SEQ, -1);
    udata->seq_idx[seq_id] = 0;
    udata->output.resize(n_tokens, 0);
    udata->seq_id_data.resize(n_tokens, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_pos pos = static_cast<llama_pos>(desc.token_start + i);
        for (uint32_t j = 0; j < n_pos; ++j) {
            udata->pos[static_cast<size_t>(j) * n_tokens + i] = pos;
        }
        udata->seq_id_data[i] = seq_id;
        udata->seq_id[i] = &udata->seq_id_data[i];
    }
    ubatch.b_equal_seqs = true;
    ubatch.n_tokens = n_tokens;
    ubatch.n_seq_tokens = n_tokens;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = n_pos;
    ubatch.token = udata->token.data();
    ubatch.embd = nullptr;
    ubatch.pos = udata->pos.data();
    ubatch.n_seq_id = udata->n_seq_id.data();
    ubatch.seq_id = udata->seq_id.data();
    ubatch.seq_id_unq = udata->seq_id_unq.data();
    ubatch.seq_idx = udata->seq_idx.data();
    ubatch.output = udata->output.data();
    ubatch.data = std::move(udata);

    auto & stream_cells = v_cells[strm];
    std::vector<uint32_t> replaced_idxs;
    replaced_idxs.reserve(n_tokens);
    const llama_pos pos_start = static_cast<llama_pos>(desc.token_start);
    const llama_pos pos_end = static_cast<llama_pos>(desc.token_start + desc.token_count);
    for (uint32_t i = 0; i < stream_cells.size(); ++i) {
        if (stream_cells.seq_has(i, seq_id) && stream_cells.pos_in(i, pos_start, pos_end)) {
            replaced_idxs.push_back(i);
        }
    }
    const auto replaced_cells = stream_cells.cp(replaced_idxs);
    const uint32_t head_before = v_heads[strm];

    seq_rm(seq_id, static_cast<llama_pos>(desc.token_start), static_cast<llama_pos>(desc.token_start + desc.token_count));
    const slot_info sinfo = find_slot(ubatch, false);
    if (sinfo.empty()) {
        stream_cells.set(replaced_idxs, replaced_cells);
        v_heads[strm] = head_before;
        error = "failed to allocate KV cache cells for imported page";
        return false;
    }
    apply_ubatch(sinfo, ubatch);

    const auto cell_runs = llama_kv_cache_cell_runs(sinfo.idxs[0]);
    const char * src = static_cast<const char *>(input);
    for (const auto * layer : selected) {
        auto * k = layer->k_stream[strm];
        for (const auto & run : cell_runs) {
            const size_t run_bytes = static_cast<size_t>(run.count) * desc.k_row_bytes;
            ggml_backend_tensor_set(k, src, static_cast<size_t>(run.first) * desc.k_row_bytes, run_bytes);
            src += run_bytes;
        }
    }

    auto & cells = v_cells[strm];
    if (!v_trans) {
        for (const auto * layer : selected) {
            if (desc.v_row_bytes == 0) {
                continue;
            }
            auto * v = layer->v_stream[strm];
            for (const auto & run : cell_runs) {
                const size_t run_bytes = static_cast<size_t>(run.count) * desc.v_row_bytes;
                ggml_backend_tensor_set(v, src, static_cast<size_t>(run.first) * desc.v_row_bytes, run_bytes);
                src += run_bytes;
            }
        }
    } else {
        for (const auto * layer : selected) {
            if (desc.v_element_bytes == 0) {
                continue;
            }
            auto * v = layer->v_stream[strm];
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer->il);
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & run : cell_runs) {
                    const size_t run_bytes = static_cast<size_t>(run.count) * desc.v_element_bytes;
                    const size_t dst_offset = (static_cast<size_t>(run.first) + static_cast<size_t>(j) * cells.size()) * desc.v_element_bytes;
                    ggml_backend_tensor_set(v, src, dst_offset, run_bytes);
                    src += run_bytes;
                }
            }
        }
    }

    // MSA/DSA indexer keys ride after the V payload, one F32 row per token per layer
    if (desc.k_idx_row_bytes > 0) {
        for (const auto * layer : selected) {
            auto * k_idx = layer->k_idx_stream[strm];
            for (const auto & run : cell_runs) {
                const size_t run_bytes = static_cast<size_t>(run.count) * desc.k_idx_row_bytes;
                ggml_backend_tensor_set(k_idx, src, static_cast<size_t>(run.first) * desc.k_idx_row_bytes, run_bytes);
                src += run_bytes;
            }
        }
    }

    return src == static_cast<const char *>(input) + input_bytes;
}

static uint16_t skippy_cachegen_read_u16(const uint8_t * bytes) {
    return static_cast<uint16_t>(bytes[0]) | static_cast<uint16_t>(bytes[1]) << 8;
}

static uint32_t skippy_cachegen_read_u32(const uint8_t * bytes) {
    return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
           static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
}

static bool skippy_cachegen_validate_segment(const skippy_cachegen_record_v1 & record,
                                             size_t                            expected_rows,
                                             size_t                            expected_channels,
                                             size_t                            output_element_bytes,
                                             size_t                            output_row_bytes,
                                             std::string &                     error) {
    constexpr size_t header_bytes = 16;
    constexpr size_t cdf_entries  = 33;
    constexpr size_t max_rows     = 256;
    const auto *     payload      = static_cast<const uint8_t *>(record.payload);
    if (record.abi_version != SKIPPY_CACHEGEN_RECORD_V1_ABI_VERSION || record.reserved0 != 0 ||
        record.element_bytes != output_element_bytes || payload == nullptr || record.payload_bytes < header_bytes ||
        record.decoded_bytes == 0 || record.token_count != expected_rows || expected_rows == 0 ||
        expected_rows > max_rows || expected_channels == 0 ||
        expected_channels > std::numeric_limits<uint32_t>::max()) {
        error = "invalid CacheGen record descriptor";
        return false;
    }
    const bool packed = std::memcmp(payload, "LCG2", 4) == 0;
    const uint8_t expected_bits = payload[4] == 16 ? 4 : 5;
    if ((!packed && std::memcmp(payload, "LCG1", 4) != 0) ||
        (packed ? payload[5] != expected_bits : payload[5] != 0) ||
        (payload[4] != 16 && payload[4] != 32) ||
        skippy_cachegen_read_u16(payload + 6) != expected_rows ||
        skippy_cachegen_read_u32(payload + 8) != expected_channels) {
        error = "CacheGen record disagrees with its segment header";
        return false;
    }
    if (expected_rows > std::numeric_limits<size_t>::max() / expected_channels ||
        expected_rows > std::numeric_limits<size_t>::max() / output_row_bytes ||
        record.decoded_bytes != expected_rows * output_row_bytes) {
        error = "CacheGen record decoded geometry is inconsistent";
        return false;
    }
    const size_t max_bytes = expected_rows * sizeof(float);
    if (header_bytes > std::numeric_limits<size_t>::max() - max_bytes ||
        header_bytes + max_bytes > record.payload_bytes) {
        error = "CacheGen segment maximum metadata is truncated";
        return false;
    }
    for (size_t row = 0; row < expected_rows; ++row) {
        const uint32_t bits = skippy_cachegen_read_u32(payload + header_bytes + row * sizeof(float));
        float          value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value) || value < 0.0f) {
            error = "CacheGen segment contains an invalid row maximum";
            return false;
        }
    }
    const size_t stream_bytes = skippy_cachegen_read_u32(payload + 12);
    if (packed) {
        const size_t values = expected_rows * expected_channels;
        if (values > (std::numeric_limits<size_t>::max() - 7) / expected_bits) {
            error = "CacheGen packed segment length overflows";
            return false;
        }
        const size_t expected_stream_bytes = (values * expected_bits + 7) / 8;
        if (stream_bytes != expected_stream_bytes ||
            stream_bytes != record.payload_bytes - (header_bytes + max_bytes)) {
            error = "CacheGen packed segment length is inconsistent";
            return false;
        }
        return true;
    }
    if (expected_channels > std::numeric_limits<size_t>::max() / (cdf_entries * sizeof(uint16_t)) ||
        expected_channels > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
        error = "CacheGen segment metadata overflows";
        return false;
    }
    const size_t cdf_bytes    = expected_channels * cdf_entries * sizeof(uint16_t);
    const size_t length_bytes = expected_channels * sizeof(uint16_t);
    if (header_bytes > std::numeric_limits<size_t>::max() - max_bytes ||
        header_bytes + max_bytes > std::numeric_limits<size_t>::max() - cdf_bytes ||
        header_bytes + max_bytes + cdf_bytes > std::numeric_limits<size_t>::max() - length_bytes) {
        error = "CacheGen segment metadata range overflows";
        return false;
    }
    const size_t cdf_offset    = header_bytes + max_bytes;
    const size_t length_offset = cdf_offset + cdf_bytes;
    const size_t stream_offset = length_offset + length_bytes;
    if (stream_offset > record.payload_bytes || stream_bytes != record.payload_bytes - stream_offset) {
        error = "CacheGen segment stream length is inconsistent";
        return false;
    }
    for (size_t channel = 0; channel < expected_channels; ++channel) {
        const uint8_t * cdf      = payload + cdf_offset + channel * cdf_entries * sizeof(uint16_t);
        uint16_t        previous = skippy_cachegen_read_u16(cdf);
        if (previous != 0) {
            error = "CacheGen segment has an invalid CDF endpoint";
            return false;
        }
        for (size_t entry = 1; entry < cdf_entries; ++entry) {
            const uint16_t current = skippy_cachegen_read_u16(cdf + entry * sizeof(uint16_t));
            if (current <= previous) {
                error = "CacheGen segment CDF is not strictly increasing";
                return false;
            }
            previous = current;
        }
        if (previous != std::numeric_limits<uint16_t>::max()) {
            error = "CacheGen segment has an invalid CDF endpoint";
            return false;
        }
    }
    size_t summed_stream_bytes = 0;
    for (size_t channel = 0; channel < expected_channels; ++channel) {
        const size_t length = skippy_cachegen_read_u16(payload + length_offset + channel * sizeof(uint16_t));
        if (length == 0 || length > max_rows || summed_stream_bytes > stream_bytes ||
            length > stream_bytes - summed_stream_bytes) {
            error = "CacheGen segment has an invalid channel stream length";
            return false;
        }
        summed_stream_bytes += length;
    }
    if (summed_stream_bytes != stream_bytes) {
        error = "CacheGen channel lengths do not cover the segment stream";
        return false;
    }
    return true;
}

struct skippy_cachegen_job_storage {
    ggml_tensor *                           dst = nullptr;
    std::vector<ggml_backend_cachegen_tile> tiles;
    uint32_t                                channels       = 0;
    uint64_t                                token_stride   = 0;
    uint64_t                                channel_stride = 0;
};

bool llama_kv_cache::stage_import_cachegen_kv_page(llama_seq_id                      seq_id,
                                                   const skippy_kv_page_desc &       desc,
                                                   const skippy_cachegen_record_v1 * records,
                                                   size_t                            record_count,
                                                   size_t                            output_base,
                                                   std::string &                     error,
                                                   bool &                            unsupported,
                                                   bool &                            invalid_argument,
                                                   bool                              validate_only) {
    unsupported                 = false;
    invalid_argument            = true;
    uint8_t validation_sentinel = 0;
    if (!stage_import_kv_page(seq_id, desc, &validation_sentinel, desc.payload_bytes, error, true)) {
        return false;
    }
    const auto cachegen_type_supported = [](uint32_t type) {
        return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0;
    };
    if (records == nullptr || record_count == 0 || !cachegen_type_supported(desc.k_type) ||
        !cachegen_type_supported(desc.v_type)) {
        error = "CacheGen import requires non-empty supported K/V records";
        return false;
    }

    const uint32_t                strm = seq_to_stream[seq_id];
    std::vector<const kv_layer *> selected;
    for (const auto & layer : layers) {
        if (layer.il >= static_cast<uint32_t>(desc.layer_start) && layer.il < static_cast<uint32_t>(desc.layer_end) &&
            layer.k_stream.size() > strm && layer.k_stream[strm] != nullptr) {
            selected.push_back(&layer);
        }
    }

    size_t                                   record_index   = 0;
    size_t                                   logical_offset = output_base;
    std::vector<skippy_cachegen_job_storage> storage;
    storage.reserve(selected.size() * 2);
    auto consume_tiles = [&](ggml_tensor * dst, size_t row_bytes, bool transposed) -> bool {
        const size_t element_bytes = ggml_type_size(dst->type);
        const size_t block_values  = ggml_blck_size(dst->type);
        const bool   quantized     = dst->type == GGML_TYPE_Q8_0 || dst->type == GGML_TYPE_Q4_0;
        if (element_bytes == 0 || block_values == 0 || row_bytes == 0 || row_bytes % element_bytes != 0 ||
            (!quantized && element_bytes != sizeof(uint16_t) && element_bytes != sizeof(float)) ||
            (quantized && (transposed || block_values != 32))) {
            error = "CacheGen destination element geometry is invalid";
            return false;
        }
        const size_t channels = row_bytes / element_bytes * block_values;
        if (channels == 0 || channels > UINT32_MAX) {
            error = "CacheGen destination channel count is invalid";
            return false;
        }
        skippy_cachegen_job_storage entry;
        entry.dst             = dst;
        entry.token_stride    = transposed ? element_bytes : row_bytes;
        entry.channel_stride  = transposed ? static_cast<uint64_t>(v_cells[strm].size()) * element_bytes : element_bytes;
        entry.channels        = static_cast<uint32_t>(channels);
        size_t token_offset   = 0;
        while (token_offset < desc.token_count) {
            if (record_index >= record_count) {
                error = "CacheGen records do not cover the native KV page";
                return false;
            }
            const auto &   record = records[record_index++];
            const size_t   rows   = static_cast<size_t>(record.token_count);
            uint32_t expected_kind = SKIPPY_CACHEGEN_RECORD_F16;
            switch (dst->type) {
                case GGML_TYPE_F16:
                    expected_kind = transposed ? SKIPPY_CACHEGEN_RECORD_F16_TRANSPOSED : SKIPPY_CACHEGEN_RECORD_F16;
                    break;
                case GGML_TYPE_F32:
                    expected_kind = transposed ? SKIPPY_CACHEGEN_RECORD_F32_TRANSPOSED : SKIPPY_CACHEGEN_RECORD_F32;
                    break;
                case GGML_TYPE_Q8_0:
                    expected_kind = SKIPPY_CACHEGEN_RECORD_Q8_0;
                    break;
                case GGML_TYPE_Q4_0:
                    expected_kind = SKIPPY_CACHEGEN_RECORD_Q4_0;
                    break;
                default:
                    GGML_ABORT("unreachable CacheGen destination type");
            }
            if (record.kind != expected_kind || rows == 0 || rows > desc.token_count - token_offset ||
                record.output_offset != logical_offset + (transposed ? 0 : token_offset * row_bytes) ||
                record.token_start != (transposed ? token_offset : 0) ||
                record.total_tokens != (transposed ? desc.token_count : 0) ||
                !skippy_cachegen_validate_segment(record, rows, channels, element_bytes, row_bytes, error)) {
                if (error.empty()) {
                    error = "CacheGen record order or destination geometry is invalid";
                }
                return false;
            }
            entry.tiles.push_back({ record.payload, record.payload_bytes, static_cast<uint32_t>(token_offset),
                                    static_cast<uint32_t>(rows) });
            token_offset += rows;
        }
        storage.push_back(std::move(entry));
        logical_offset += static_cast<size_t>(desc.token_count) * row_bytes;
        return true;
    };

    for (const auto * layer : selected) {
        if (!consume_tiles(layer->k_stream[strm], desc.k_row_bytes, false)) {
            return false;
        }
    }
    // MLA caches (DeepSeek/GLM indexers) hold no V stream: their page carries an empty V
    // payload and no V records, so V tiles must not be consumed. The mixed-stream check below
    // is unreachable today (stage_import_kv_page rejects a V mismatch first) but stays fail-closed.
    const bool has_v_payload = desc.v_row_bytes != 0 || desc.v_element_bytes != 0;
    for (const auto * layer : selected) {
        const bool layer_has_v = layer->v_stream.size() > strm && layer->v_stream[strm] != nullptr;
        if (layer_has_v != has_v_payload) {
            error = "CacheGen page V streams disagree with its V geometry";
            return false;
        }
        if (!has_v_payload) {
            continue;
        }
        const size_t v_row_bytes =
            v_trans ? static_cast<size_t>(hparams.n_embd_v_gqa(layer->il)) * desc.v_element_bytes : desc.v_row_bytes;
        if (!consume_tiles(layer->v_stream[strm], v_row_bytes, v_trans)) {
            return false;
        }
    }

    const skippy_cachegen_record_v1 * exact = nullptr;
    if (desc.k_idx_row_bytes != 0 && (desc.token_count > std::numeric_limits<size_t>::max() / desc.k_idx_row_bytes ||
                                      static_cast<size_t>(desc.token_count) * desc.k_idx_row_bytes >
                                          std::numeric_limits<size_t>::max() / selected.size())) {
        error = "CacheGen exact indexer geometry overflows";
        return false;
    }
    const size_t exact_bytes = static_cast<size_t>(desc.token_count) * desc.k_idx_row_bytes * selected.size();
    if (exact_bytes > 0) {
        if (record_index >= record_count) {
            error = "CacheGen page is missing exact indexer state";
            return false;
        }
        exact = &records[record_index++];
        if (exact->abi_version != SKIPPY_CACHEGEN_RECORD_V1_ABI_VERSION ||
            exact->kind != SKIPPY_CACHEGEN_RECORD_EXACT || exact->element_bytes != 1 || exact->reserved0 != 0 ||
            exact->output_offset != logical_offset || exact->decoded_bytes != exact_bytes ||
            exact->token_count != desc.token_count || exact->token_start != 0 || exact->total_tokens != 0 ||
            exact->payload == nullptr || exact->payload_bytes != exact_bytes) {
            error = "CacheGen exact indexer record is invalid";
            return false;
        }
        logical_offset += exact_bytes;
    }
    if (record_index != record_count || logical_offset != output_base + desc.payload_bytes) {
        error = "CacheGen records do not exactly cover the native KV page";
        return false;
    }

    ggml_backend_cachegen_decode_t decode = nullptr;
    for (const auto & entry : storage) {
        if (entry.dst == nullptr || entry.dst->buffer == nullptr) {
            error = "CacheGen destination tensor has no backend buffer";
            return false;
        }
        const auto buft      = ggml_backend_buffer_get_type(entry.dst->buffer);
        const auto device    = ggml_backend_buft_get_device(buft);
        const auto reg       = device == nullptr ? nullptr : ggml_backend_dev_backend_reg(device);
        auto       candidate = reg == nullptr ?
                                   nullptr :
                                   reinterpret_cast<ggml_backend_cachegen_decode_t>(
                                       ggml_backend_reg_get_proc_address(reg, "ggml_backend_cachegen_decode"));
        if (candidate == nullptr || (decode != nullptr && candidate != decode)) {
            invalid_argument = false;
            unsupported      = true;
            error            = "resident KV backend does not provide typed CacheGen decode";
            return false;
        }
        decode = candidate;
    }
    invalid_argument = false;
    if (validate_only) {
        return true;
    }

    auto & stream_cells = v_cells[strm];
    for (uint32_t i = 0; i < stream_cells.size(); ++i) {
        if (stream_cells.seq_has(i, seq_id)) {
            invalid_argument = true;
            error            = "CacheGen page import requires an empty session KV sequence";
            return false;
        }
    }

    llama_ubatch   ubatch   = {};
    auto           udata    = std::make_shared<llama_ubatch::data_t>();
    const uint32_t n_tokens = static_cast<uint32_t>(desc.token_count);
    const uint32_t n_pos    = hparams.n_pos_per_embd();
    if (n_pos != 0 && n_tokens > std::numeric_limits<size_t>::max() / n_pos) {
        invalid_argument = true;
        error            = "CacheGen page position geometry overflows";
        return false;
    }
    udata->token.resize(n_tokens);
    udata->pos.resize(static_cast<size_t>(n_tokens) * n_pos);
    udata->n_seq_id.resize(n_tokens, 1);
    udata->seq_id.resize(n_tokens);
    udata->seq_id_unq = { seq_id };
    udata->seq_idx.resize(LLAMA_MAX_SEQ, -1);
    udata->seq_idx[seq_id] = 0;
    udata->output.resize(n_tokens, 0);
    udata->seq_id_data.resize(n_tokens, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_pos pos = static_cast<llama_pos>(desc.token_start + i);
        for (uint32_t j = 0; j < n_pos; ++j) {
            udata->pos[static_cast<size_t>(j) * n_tokens + i] = pos;
        }
        udata->seq_id_data[i] = seq_id;
        udata->seq_id[i]      = &udata->seq_id_data[i];
    }
    ubatch.b_equal_seqs = true;
    ubatch.n_tokens     = n_tokens;
    ubatch.n_seq_tokens = n_tokens;
    ubatch.n_seqs       = 1;
    ubatch.n_seqs_unq   = 1;
    ubatch.n_pos        = n_pos;
    ubatch.token        = udata->token.data();
    ubatch.pos          = udata->pos.data();
    ubatch.n_seq_id     = udata->n_seq_id.data();
    ubatch.seq_id       = udata->seq_id.data();
    ubatch.seq_id_unq   = udata->seq_id_unq.data();
    ubatch.seq_idx      = udata->seq_idx.data();
    ubatch.output       = udata->output.data();
    ubatch.data         = std::move(udata);

    const uint32_t  head_before = v_heads[strm];
    const slot_info sinfo       = find_slot(ubatch, false);
    if (sinfo.empty()) {
        error = "failed to allocate KV cache cells for CacheGen page";
        return false;
    }
    apply_ubatch(sinfo, ubatch);

    std::vector<ggml_backend_cachegen_job> jobs;
    jobs.reserve(storage.size());
    for (const auto & entry : storage) {
        jobs.push_back({ entry.dst, entry.tiles.data(), entry.tiles.size(), sinfo.idxs[0].data(), sinfo.idxs[0].size(),
                         entry.channels, entry.token_stride, entry.channel_stride });
    }
    char backend_error[256] = {};
    if (!decode(jobs.data(), jobs.size(), backend_error, sizeof(backend_error))) {
        seq_rm(seq_id, static_cast<llama_pos>(desc.token_start),
               static_cast<llama_pos>(desc.token_start + desc.token_count));
        v_heads[strm] = head_before;
        error         = backend_error[0] == '\0' ? "CacheGen backend decode failed" : backend_error;
        return false;
    }

    if (exact != nullptr) {
        const char * src       = static_cast<const char *>(exact->payload);
        const auto   cell_runs = llama_kv_cache_cell_runs(sinfo.idxs[0]);
        for (const auto * layer : selected) {
            auto * k_idx = layer->k_idx_stream[strm];
            for (const auto & run : cell_runs) {
                const size_t run_bytes = static_cast<size_t>(run.count) * desc.k_idx_row_bytes;
                ggml_backend_tensor_set(k_idx, src, static_cast<size_t>(run.first) * desc.k_idx_row_bytes, run_bytes);
                src += run_bytes;
            }
        }
    }
    return true;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

uint32_t llama_kv_cache::get_n_kv_pos_contiguous(const slot_info & sinfo, const llama_ubatch & ubatch) const {
    if (sinfo.n_stream() != 1 || ubatch.n_seqs_unq != 1 || ubatch.n_tokens == 0 ||
        ubatch.pos == nullptr || ubatch.n_seq_id == nullptr ||
        ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr) {
        return 0;
    }

    const llama_seq_id seq_id = ubatch.seq_id[0][0];
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return 0;
    }

    const uint32_t stream = seq_to_stream[seq_id];
    if (sinfo.strm[0] < 0 || (uint32_t) sinfo.strm[0] != stream || stream >= v_cells.size()) {
        return 0;
    }

    const auto & cells = v_cells[stream];
    const llama_pos pos_max = cells.seq_pos_max(seq_id);

    if (pos_max < 0 || pos_max >= (llama_pos) cells.size()) {
        return 0;
    }

    // the banded op aligns Q to the tail of K: the ubatch must be that monotonic tail, else dense bias
    if ((uint32_t) pos_max + 1 < ubatch.n_tokens) {
        return 0;
    }
    const llama_pos pos_start = pos_max + 1 - ubatch.n_tokens;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] != pos_start + (llama_pos) i ||
            ubatch.n_seq_id[i] < 1 || ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != seq_id) {
            return 0;
        }
    }

    if (!cells.seq_pos_is_cell_contiguous(seq_id)) {
        return 0;
    }

    return pos_max + 1;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    auto * k = graph_k_storage(ctx, il);

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    ggml_tensor * result = ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
    return result;
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    auto * v = graph_v_storage(ctx, il);

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        ggml_tensor * result = ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
        return result;
    }

    // note: v->nb[1] > v->nb[2]
    ggml_tensor * result = ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
    return result;
}

ggml_tensor * llama_kv_cache::get_k_idx(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    auto * k_idx = graph_k_idx_storage(ctx, il);
    GGML_ASSERT(k_idx);

    const uint64_t kv_size = get_size();
    const int64_t  n_idx   = k_idx->ne[0];                 // 128
    const uint32_t ns      = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k_idx,
            n_idx, 1, n_kv, ns,
            ggml_row_size(k_idx->type, n_idx),             // nb1 (single head)
            ggml_row_size(k_idx->type, n_idx),             // nb2 (per cell)
            ggml_row_size(k_idx->type, n_idx*kv_size),     // nb3 (per stream)
            ggml_row_size(k_idx->type, n_idx*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    ggml_tensor * k = graph_k_storage(ctx, il);

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    auto * v = graph_v_storage(ctx, il);

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        ggml_tensor * result = ggml_set_rows(ctx, v, v_cur, v_idxs);
        if (v->type == GGML_TYPE_TURBO4_0) {
            GGML_ASSERT(n_embd_head % 128 == 0);
            const int32_t wht_group = 128;
            memcpy(result->op_params, &wht_group, sizeof(wht_group));
        }
        return result;
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);
    ggml_set_name(k_idxs, "attn_inp_k_idxs");

    return k_idxs;
}

ggml_tensor * llama_kv_cache::cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);
    ggml_tensor * k_idx = graph_k_idx_storage(ctx, il);
    GGML_ASSERT(k_idx && "cpy_k_idx on a layer with no indexer cache");

    const int64_t n_embd_head = k_idx_cur->ne[0];          // 128
    const int64_t n_head      = k_idx_cur->ne[1];          // 1
    const int64_t n_tokens    = k_idx_cur->ne[2];
    const int64_t n_embd_gqa  = n_embd_head*n_head;        // 128

    GGML_ASSERT(ggml_row_size(k_idx_cur->type, n_embd_head) == k_idx_cur->nb[1]);
    k_idx_cur = ggml_view_2d(ctx, k_idx_cur, n_embd_gqa, n_tokens, k_idx_cur->nb[2], 0);

    const int64_t n_stream = k_idx->ne[2];
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        k_idx = ggml_reshape_2d(ctx, k_idx, n_embd_gqa, kv_size*n_stream);
    }
    return ggml_set_rows(ctx, k_idx, k_idx_cur, k_idxs);   // same k_idxs as the K store
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);
    ggml_set_name(v_idxs, "attn_inp_v_idxs");

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (n_rot_k) {
        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot_k, n_rot_k);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (n_rot_v) {
        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot_v, n_rot_v);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    // see llama_non_causal_type
                    const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
                    if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    // see llama_non_causal_type
    // only the SWA cache (or the SWA layers of a single cache) become non-causal
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    const int64_t n_kv = dst->ne[0];
    GGML_ASSERT(dst->ne[1] == n_tokens);

    // [n_kv, n_tokens] in GLOBAL token order (stream-major, same as the KQ mask)
    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        const llama_pos p1 = ubatch->pos[i];

        for (int64_t j = 0; j < n_kv; ++j) {
            // use the ACTUAL absolute position in the KV cell; physical slot order is not monotonic
            int32_t rel = (int32_t) extent; // zero-bias column
            if (!cells.is_empty(j)) {
                const llama_pos d = p1 - cells.pos_get(j);
                if (d >= 0 && d < (llama_pos) extent) {
                    rel = (int32_t) d;
                }
            }

            data[i*n_kv + j] = (int32_t) (i*(extent + 1)) + rel;
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
    //       of this very ubatch as well, which is what we want
    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
    // share the same temporal pos

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    // same for a mixed ubatch
    const bool by_order = !ubatch.token || ubatch.is_mixed();

    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (by_order) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (by_order) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

size_t llama_kv_cache::size_k_idx_bytes() const {
    size_t size_k_idx_bytes = 0;

    for (const auto & layer : layers) {
        if (layer.k_idx) {
            size_k_idx_bytes += ggml_nbytes(layer.k_idx);
        }
    }

    return size_k_idx_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift && k_shift->buffer) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot && k_rot->buffer) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE) {
        return;
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE) {
        return;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // a whole-context restore replaces every stream, so the cache is emptied once here
    // clear() resets all streams at once, so doing it per stream below would keep only the last one
    if (seq_id == -1) {
        clear(true);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
            if (sinfos_in && !(*sinfos_in)[s].empty()) {
                throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
            }
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, sinfos_in ? &(*sinfos_in)[s] : nullptr);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            state_clear(seq_id, strm, sinfo);
            throw std::runtime_error("failed to restore kv cache");
        }

        if (sinfos_out) {
            (*sinfos_out)[s] = sinfo;
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&n_rot_k, sizeof(n_rot_k));
    io.write(&n_rot_v, sizeof(n_rot_v));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (size_k_idx_bytes() > 0) {
        const uint32_t has_k_idx_u32 = 1;
        io.write(&has_k_idx_u32, sizeof(has_k_idx_u32));

        for (const auto & layer : layers) {
            const uint32_t layer_has_k_idx = layer.k_idx ? 1 : 0;
            io.write(&layer_has_k_idx, sizeof(layer_has_k_idx));

            if (!layer_has_k_idx) {
                continue;
            }

            GGML_ASSERT(layer.k_idx_stream[cr.strm]);

            const int32_t k_idx_type_i = (int32_t) layer.k_idx->type;
            io.write(&k_idx_type_i, sizeof(k_idx_type_i));

            const uint64_t k_idx_size_row = ggml_row_size(layer.k_idx->type, layer.k_idx->ne[0]);
            io.write(&k_idx_size_row, sizeof(k_idx_size_row));

            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size   = range_size * k_idx_size_row;
                const size_t offset     = range.first * k_idx_size_row;

                io.write_tensor(layer.k_idx_stream[cr.strm], offset, buf_size);
            }
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        // the ext as it was saved, to put back after apply_ubatch()
        std::vector<llama_kv_cell_ext> exts;
        if (has_cell_ext()) {
            exts.resize(cell_count);
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;

                exts[i] = ext;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // the layout is cell indices, so it means the same in both caches only while their streams line up
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.strm[0] = strm;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[0][i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
        // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
        // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
        // is what the whole-context path below already does.
        for (uint32_t i = 0; i < (uint32_t) exts.size(); ++i) {
            cells.ext_set(sinfo.idxs[0][i], exts[i]);
        }

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count)) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    // batch the scatter reads per contiguous run of destination indices
    // from inclusive, to exclusive - same convention as cell_ranges_t
    // contiguous cells yield a single run covering the whole block
    struct cell_run { uint32_t from; uint32_t to; };
    std::vector<cell_run> runs;
    if (cell_count > 0) {
        const auto & idxs = sinfo.idxs[0];
        uint32_t i0 = 0;
        while (i0 < cell_count) {
            uint32_t i1 = i0 + 1;
            while (i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    uint32_t v_trans;
    uint32_t n_layer;
    uint32_t n_rot_k_ref;
    uint32_t n_rot_v_ref;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));
    io.read(&n_rot_k_ref, sizeof(n_rot_k_ref));
    io.read(&n_rot_v_ref, sizeof(n_rot_v_ref));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    if (n_rot_k_ref != n_rot_k) {
        LLAMA_LOG_ERROR("%s: incompatible key rotation (%u instead of %u)\n", __func__, n_rot_k_ref, n_rot_k);
        return false;
    }

    if (n_rot_v_ref != n_rot_v) {
        LLAMA_LOG_ERROR("%s: incompatible value rotation (%u instead of %u)\n", __func__, n_rot_v_ref, n_rot_v);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        for (const auto & r : runs) {
            io.read_tensor(k, (size_t) r.from * k_size_row, (size_t) (r.to - r.from) * k_size_row);
        }
    }

    if (size_k_idx_bytes() > 0) {
        uint32_t has_k_idx_u32 = 0;
        io.read(&has_k_idx_u32, sizeof(has_k_idx_u32));

        if (has_k_idx_u32 != 1) {
            LLAMA_LOG_ERROR("%s: missing k_idx data in KV cache state\n", __func__);
            return false;
        }

        for (const auto & layer : layers) {
            uint32_t layer_has_k_idx = 0;
            io.read(&layer_has_k_idx, sizeof(layer_has_k_idx));

            const uint32_t expected_layer_has_k_idx = layer.k_idx ? 1 : 0;

            if (layer_has_k_idx != expected_layer_has_k_idx) {
                LLAMA_LOG_ERROR(
                    "%s: mismatched k_idx state for layer: got %u, expected %u\n",
                    __func__, layer_has_k_idx, expected_layer_has_k_idx);
                return false;
            }

            if (!layer_has_k_idx) {
                continue;
            }

            GGML_ASSERT(layer.k_idx_stream[strm]);

            int32_t k_idx_type_i = -1;
            io.read(&k_idx_type_i, sizeof(k_idx_type_i));

            if (k_idx_type_i != (int32_t) layer.k_idx->type) {
                LLAMA_LOG_ERROR(
                    "%s: mismatched k_idx type: got %d, expected %d\n",
                    __func__, k_idx_type_i, (int32_t) layer.k_idx->type);
                return false;
            }

            uint64_t k_idx_size_row = 0;
            io.read(&k_idx_size_row, sizeof(k_idx_size_row));

            const uint64_t expected_k_idx_size_row = ggml_row_size(layer.k_idx->type, layer.k_idx->ne[0]);

            if (k_idx_size_row != expected_k_idx_size_row) {
                LLAMA_LOG_ERROR(
                    "%s: mismatched k_idx row size: got %zu, expected %zu\n",
                    __func__, (size_t) k_idx_size_row, (size_t) expected_k_idx_size_row);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    io.read_tensor(layer.k_idx_stream[strm], sinfo.head() * k_idx_size_row, cell_count * k_idx_size_row);
                } else {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        io.read_tensor(layer.k_idx_stream[strm], sinfo.idxs[0][i] * k_idx_size_row, k_idx_size_row);
                    }
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            for (const auto & r : runs) {
                io.read_tensor(v, (size_t) r.from * v_size_row, (size_t) (r.to - r.from) * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & r : runs) {
                    const size_t dst_offset = ((size_t) r.from + j * cells.size()) * v_size_el;
                    io.read_tensor(v, dst_offset, (size_t) (r.to - r.from) * v_size_el);
                }
            }
        }
    }

    return true;
}

void llama_kv_cache::state_clear(llama_seq_id seq_id) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t strm = seq_to_stream[seq_id];

    const auto & cells = v_cells[strm];

    slot_info sinfo;
    sinfo.s0 = strm;
    sinfo.s1 = strm;
    sinfo.resize(1);
    sinfo.strm[0] = strm;

    // a cell that another sequence still uses keeps its data
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_has(i, seq_id) && cells.seq_count(i) == 1) {
            sinfo.idxs[0].push_back(i);
        }
    }

    state_clear(seq_id, strm, sinfo);
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
void llama_kv_cache::state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    // zero the K/V data of the failed restore attempt - the attention can still read the data of free cells
    if (sinfo.empty() || sinfo.size() == 0) {
        return;
    }

    const auto & cells = v_cells[strm];

    const uint32_t cell_count = sinfo.size();

    const bool is_contiguous = sinfo.is_contiguous();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);

        if (is_contiguous) {
            llama_clear_tensor_data(k, sinfo.head() * k_size_row, cell_count * k_size_row);
        } else {
            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_clear_tensor_data(k, sinfo.idxs[0][i] * k_size_row, k_size_row);
            }
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        auto * v = layer.v_stream[strm];
        if (!v) {
            continue;
        }

        if (!v_trans) {
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);

            if (is_contiguous) {
                llama_clear_tensor_data(v, sinfo.head() * v_size_row, cell_count * v_size_row);
            } else {
                for (uint32_t i = 0; i < cell_count; ++i) {
                    llama_clear_tensor_data(v, sinfo.idxs[0][i] * v_size_row, v_size_row);
                }
            }
        } else {
            const size_t v_size_el = ggml_type_size(v->type);

            if (is_contiguous) {
                const uint32_t h = sinfo.head();

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    llama_clear_tensor_data(v, (h + j * cells.size()) * v_size_el, cell_count * v_size_el);
                }
            } else {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        llama_clear_tensor_data(v, (sinfo.idxs[0][i] + j * cells.size()) * v_size_el, v_size_el);
                    }
                }
            }
        }
    }
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info,
        bool did_compact) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty() && !did_compact) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_context::get_n_kv_pos_contiguous() const {
    // Full-cache and update contexts do not carry a concrete ubatch/slot pair.
    if (ubatches.empty() || sinfos.empty() || i_cur >= ubatches.size() || i_cur >= sinfos.size()) {
        // reserve context: report the whole cache as position-contiguous so the worst-case graph
        // is the banded path; reserving the dense fallback is unallocatable at large n_ctx
        if (kv != nullptr && lctx == nullptr && kv->get_n_stream() == 1) {
            return n_kv;
        }
        return 0;
    }

    const uint32_t result = kv->get_n_kv_pos_contiguous(sinfos[i_cur], ubatches[i_cur]);
    return result <= (uint32_t) n_kv ? result : 0;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_storage(int32_t il) const {
    return kv->get_k_storage(il);
}

ggml_tensor * llama_kv_cache_context::get_v_storage(int32_t il) const {
    return kv->get_v_storage(il);
}

ggml_tensor * llama_kv_cache_context::get_k_idx(ggml_context * ctx, int32_t il) const {
    return kv->get_k_idx(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_idx(ggml_context * ctx, ggml_tensor * k_idx_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k_idx(ctx, k_idx_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_pos_rel_flat(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    kv->set_input_pos_rel_flat(dst, ubatch, extent);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
