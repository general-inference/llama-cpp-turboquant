// Inkling (PRIVATE arch): hybrid iSWA attention + per-layer packed shortconv state; see INKLING_DESIGN.md.

#include "models.h"

#include "../llama-kv-cache-iswa.h"
#include "../llama-kv-cache.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-recurrent.h"

#include <algorithm>
#include <unordered_map>

void llama_model_inkling::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func, false);

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD; // visible iff pos_q - pos_k < n_swa (includes self)
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer_all);

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_recr_impl[il] = 1;
    }

    ml.get_key(LLM_KV_INKLING_D_REL,            hparams.inkling_d_rel);
    ml.get_key(LLM_KV_INKLING_REL_EXTENT,       hparams.inkling_rel_extent);
    ml.get_key(LLM_KV_INKLING_REL_EXTENT_SWA,   hparams.inkling_rel_extent_swa);
    ml.get_key(LLM_KV_INKLING_SHORTCONV_KERNEL, hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_INKLING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);

    float logit_scale_denom = 0.0f;
    ml.get_key(LLM_KV_INKLING_LOGIT_SCALE_DENOM, logit_scale_denom);
    GGML_ASSERT(logit_scale_denom != 0.0f);
    hparams.f_logit_scale = 1.0f / logit_scale_denom;

    ml.get_key(LLM_KV_INKLING_LOG_SCALING_N_FLOOR, hparams.inkling_log_n_floor,      false);
    ml.get_key(LLM_KV_INKLING_LOG_SCALING_ALPHA,   hparams.inkling_log_alpha,        false);
    ml.get_key(LLM_KV_INKLING_UNPADDED_VOCAB_SIZE, hparams.inkling_unpadded_n_vocab, false);

    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
    GGML_ASSERT(hparams.inkling_d_rel > 0);
    GGML_ASSERT(hparams.inkling_rel_extent > 0 && hparams.inkling_rel_extent_swa > 0);

    // uniform state per cell: 4 packed streams [k | v | attn | mlp] of last K-1 columns, k/v sized for the widest layer
    const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;
    hparams.n_embd_r_impl = d_conv * (hparams.n_embd_k_gqa_max() + hparams.n_embd_v_gqa_max() + 2*hparams.n_embd);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = hparams.n_layer_nextn > 0 && ml.get_weight("blk.0.attn_norm.weight") == nullptr;
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = hparams.n_layer_nextn > 0 && ml.get_weight(mtp_probe.c_str()) == nullptr;
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    const int64_t head_dim = hparams.n_embd_head_k();
    const int64_t d_rel    = hparams.inkling_d_rel;
    const int64_t K        = hparams.n_shortconv_l_cache;
    const int64_t n_ff_exp = hparams.n_ff_exp();
    const int64_t n_shexp  = hparams.n_expert_shared;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab}, mtp_only ? TENSOR_NOT_REQUIRED : 0);
    tok_norm    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, mtp_only ? TENSOR_NOT_REQUIRED : 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"), {n_embd}, mtp_only ? TENSOR_NOT_REQUIRED : 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"), {n_embd, n_vocab}, mtp_only ? TENSOR_NOT_REQUIRED : 0);

    auto load_block = [&](int i, int flags, bool force_dense) {
        auto & layer = layers[i];

        const int64_t n_head_kv_i = hparams.n_head_kv(i);
        const int64_t kvw         = n_head_kv_i * head_dim;
        const int64_t rel_extent  = hparams.is_swa(i) ? hparams.inkling_rel_extent_swa : hparams.inkling_rel_extent;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_head*head_dim}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, kvw}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, kvw}, flags);
        layer.wr = create_tensor(tn(LLM_TENSOR_ATTN_R,   "weight", i), {n_embd, n_head*d_rel}, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head*head_dim, n_embd}, flags);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {head_dim}, flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {head_dim}, flags);

        // stored in checkpoint orientation [d_rel, E] -> gguf ne = [E, d_rel]
        layer.attn_rel_proj = create_tensor(tn(LLM_TENSOR_ATTN_REL_PROJ, "weight", i), {rel_extent, d_rel}, flags);

        layer.shortconv_k    = create_tensor(tn(LLM_TENSOR_SHORTCONV_K,    "weight", i), {K, kvw}, flags);
        layer.shortconv_v    = create_tensor(tn(LLM_TENSOR_SHORTCONV_V,    "weight", i), {K, kvw}, flags);
        layer.shortconv_attn = create_tensor(tn(LLM_TENSOR_SHORTCONV_ATTN, "weight", i), {K, n_embd}, flags);
        layer.shortconv_mlp  = create_tensor(tn(LLM_TENSOR_SHORTCONV_MLP,  "weight", i), {K, n_embd}, flags);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM,   "weight", i), {n_embd}, flags);
        layer.ffn_gscale = create_tensor(tn(LLM_TENSOR_FFN_GSCALE, "weight", i), {1}, flags);

        if (force_dense || i < (int) hparams.n_layer_dense_lead) {
            const int64_t n_ff_i = hparams.n_ff(i);

            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff_i}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff_i}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_i, n_embd}, flags);
        } else {
            GGML_ASSERT(n_expert > 0 && n_expert_used > 0 && n_shexp > 0);

            // gate holds n_expert + n_shexp rows (incl. shared-expert sink logits)
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert + n_shexp}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, flags);

            // shared experts stacked as an n_shexp bank, registered MUL_MAT_ID so the loader picks a mul_mat_id-capable buffer
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXPS, "weight", i), {n_embd, n_ff_exp, n_shexp}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXPS,   "weight", i), {n_embd, n_ff_exp, n_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXPS, "weight", i), {n_ff_exp, n_embd, n_shexp}, flags);
        }
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block(i, trunk_flags, false);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block(i, mtp_flags, true);
        auto & layer = layers[i];
        layer.nextn.eh_proj = create_tensor(
            tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {2*n_embd, n_embd}, mtp_flags);
        layer.nextn.enorm = create_tensor(
            tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), {n_embd}, mtp_flags);
        layer.nextn.hnorm = create_tensor(
            tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), {n_embd}, mtp_flags);
    }
}

class llm_graph_input_inkling : public llm_graph_input_i {
public:
    llm_graph_input_inkling(
            const llama_hparams & hparams,
            const llama_memory_hybrid_iswa_context * mctx) :
        hparams(hparams),
        mctx(mctx) {}
    virtual ~llm_graph_input_inkling() = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (tau) {
            GGML_ASSERT(ggml_backend_buffer_is_host(tau->buffer));
            float * data = (float *) tau->data;

            const float n_floor = (float) hparams.inkling_log_n_floor;
            const float alpha   = hparams.inkling_log_alpha;

            for (int64_t i = 0; i < (int64_t) ubatch->n_tokens; ++i) {
                const float eff = (float) (ubatch->pos[i] + 1) / n_floor;
                data[i] = 1.0f + alpha*logf(std::max(eff, 1.0f));
            }
        }

        if (rel_idx) {
            mctx->get_attn()->get_base()->set_input_pos_rel_flat(rel_idx, ubatch, hparams.inkling_rel_extent);
        }

        if (rel_idx_swa) {
            mctx->get_attn()->get_swa()->set_input_pos_rel_flat(rel_idx_swa, ubatch, hparams.inkling_rel_extent_swa);
        }

        if (vocab_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(vocab_mask->buffer));
            float * data = (float *) vocab_mask->data;

            const int64_t n_vocab    = vocab_mask->ne[0];
            const int64_t n_unpadded = hparams.inkling_unpadded_n_vocab;

            for (int64_t id = 0; id < n_vocab; ++id) {
                data[id] = id < n_unpadded ? 0.0f : -INFINITY;
            }
        }

        if (shexp_idx) {
            GGML_ASSERT(ggml_backend_buffer_is_host(shexp_idx->buffer));
            int32_t * data = (int32_t *) shexp_idx->data;

            const int64_t n_shexp  = shexp_idx->ne[0];
            const int64_t n_tokens = shexp_idx->ne[1];

            for (int64_t j = 0; j < n_tokens; ++j) {
                for (int64_t s = 0; s < n_shexp; ++s) {
                    data[j*n_shexp + s] = (int32_t) s;
                }
            }
        }
    }

    ggml_tensor * tau         = nullptr; // F32 [1, 1, n_tokens]
    ggml_tensor * rel_idx     = nullptr; // I32 [n_kv_base, n_tokens]
    ggml_tensor * rel_idx_swa = nullptr; // I32 [n_kv_swa, n_tokens]
    ggml_tensor * vocab_mask  = nullptr; // F32 [n_vocab]
    ggml_tensor * shexp_idx   = nullptr; // I32 [n_shexp, n_tokens], constant 0..n_shexp-1

    const llama_hparams hparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

std::unique_ptr<llm_graph_context> llama_model_inkling::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_inkling::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const bool is_mtp = params.gtype == LLM_GRAPH_TYPE_DECODER_MTP;
    const int mtp_il = n_layer + cparams.nextn_layer_offset;
    GGML_ASSERT(!is_mtp || (cparams.nextn_layer_offset >= 0 &&
        cparams.nextn_layer_offset < (int) hparams.n_layer_nextn));
    const int scan_start = is_mtp ? mtp_il : 0;
    const int scan_end   = is_mtp ? mtp_il + 1 : n_layer;

    const int64_t head_dim = hparams.n_embd_head_k();
    const int64_t d_rel    = hparams.inkling_d_rel;
    const int64_t d_conv   = hparams.n_shortconv_l_cache - 1;
    const int64_t n_embd_r = hparams.n_embd_r();
    const int64_t kw_max   = hparams.n_embd_k_gqa_max();
    const int64_t vw_max   = hparams.n_embd_v_gqa_max();

    // packed conv-state stream offsets within one cell: [k | v | attn | mlp]
    const int64_t off_k    = 0;
    const int64_t off_v    = d_conv*kw_max;
    const int64_t off_attn = d_conv*(kw_max + vw_max);
    const int64_t off_mlp  = d_conv*(kw_max + vw_max + n_embd);

    const auto * mctx_hyb  = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);
    const auto * mctx_recr = mctx_hyb->get_recr();
    const auto * mctx_attn = mctx_hyb->get_attn();

    const uint32_t kv_head = mctx_recr->get_head();

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    bool has_global = false;
    bool needs_rel_idx_local  = false;
    bool needs_rel_idx_global = false;

    const auto use_banded_flash = [&](int il) {
        GGML_UNUSED(il);
        return false;
    };

    for (int il = scan_start; il < scan_end; ++il) {
        if (hparams.is_swa(il)) {
            needs_rel_idx_local |= !use_banded_flash(il);
        } else {
            has_global = true;
            needs_rel_idx_global |= !use_banded_flash(il);
        }
    }

    auto inp = std::make_unique<llm_graph_input_inkling>(hparams, mctx_hyb);

    if (hparams.inkling_log_n_floor > 0 && has_global) {
        inp->tau = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(inp->tau);
        ggml_set_name(inp->tau, "inkling_tau");
    }

    if (needs_rel_idx_global) {
        const int64_t n_kv = mctx_attn->get_base()->get_n_kv();
        inp->rel_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
        ggml_set_input(inp->rel_idx);
        ggml_set_name(inp->rel_idx, "inkling_rel_idx");
    }

    if (needs_rel_idx_local) {
        const int64_t n_kv_swa = mctx_attn->get_swa()->get_n_kv();
        inp->rel_idx_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv_swa, n_tokens);
        ggml_set_input(inp->rel_idx_swa);
        ggml_set_name(inp->rel_idx_swa, "inkling_rel_idx_swa");
    }

    const int64_t n_vocab = model.vocab.n_tokens();
    if (!cparams.embeddings &&
        hparams.inkling_unpadded_n_vocab > 0 &&
        (int64_t) hparams.inkling_unpadded_n_vocab < n_vocab) {
        inp->vocab_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_vocab);
        ggml_set_input(inp->vocab_mask);
        ggml_set_name(inp->vocab_mask, "inkling_vocab_mask");
    }

    // shared experts go through mul_mat_id: 2D views into a repacked/quantized 3D bank are invalid
    if (!is_mtp && hparams.n_expert_shared > 0 && (uint32_t) scan_end > hparams.n_layer_dense_lead) {
        inp->shexp_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, hparams.n_expert_shared, n_tokens);
        ggml_set_input(inp->shexp_idx);
        ggml_set_name(inp->shexp_idx, "inkling_shexp_idx");
    }

    ggml_tensor * tau         = inp->tau;
    ggml_tensor * rel_idx     = inp->rel_idx;
    ggml_tensor * rel_idx_swa = inp->rel_idx_swa;
    ggml_tensor * vocab_mask  = inp->vocab_mask;
    ggml_tensor * shexp_idx   = inp->shexp_idx;

    request_input(tau,         "inkling.tau");
    request_input(rel_idx,     "inkling.relative_indices");
    request_input(rel_idx_swa, "inkling.swa.relative_indices");
    request_input(vocab_mask,  "inkling.vocab_mask");
    request_input(shexp_idx,   "inkling.shared_expert_indices");

    res->add_input(std::move(inp));

    auto * inp_hybrid = build_inp_mem_hybrid_iswa();

    // shared by the 4 stream sub-views; build_rs must run exactly once per layer (it zero-inits fresh states)
    ggml_tensor * conv_rs_cur = nullptr;

    // sconv(x) = x + causal_depthwise_conv1d(x); rolling state = last K-1 inputs
    auto build_sconv = [&](ggml_tensor * x2d, ggml_tensor * kernel, int64_t off, int il) -> ggml_tensor * {
        const int64_t w = x2d->ne[0];

        ggml_tensor * x3 = ggml_reshape_3d(ctx0, x2d, w, n_seq_tokens, n_seqs);
        ggml_tensor * xt = ggml_transpose(ctx0, x3); // time-major for the conv

        ggml_tensor * conv_state = mctx_recr->get_r_l(il);
        ggml_tensor * conv_rs    = conv_rs_cur; // {n_embd_r, n_seqs}
        GGML_ASSERT(conv_rs != nullptr);

        const size_t sz = ggml_element_size(conv_rs);

        // this stream's slice of the packed state
        ggml_tensor * state = ggml_view_3d(ctx0, conv_rs, d_conv, w, n_seqs,
                d_conv*sz, conv_rs->nb[1], off*sz);

        ggml_tensor * sx = ggml_concat(ctx0, state, xt, 0); // {d_conv + n_seq_tokens, w, n_seqs}

        // write the last d_conv time columns back into the cache
        ggml_tensor * new_state = ggml_view_3d(ctx0, sx, d_conv, w, n_seqs,
                sx->nb[1], sx->nb[2], (sx->ne[0] - d_conv)*sx->nb[0]);
        ggml_tensor * state_dst = ggml_view_3d(ctx0, conv_state, d_conv, w, n_seqs,
                d_conv*sz, n_embd_r*sz, (kv_head*n_embd_r + off)*sz);
        ggml_tensor * conv_write = ggml_cpy(ctx0, new_state, state_dst);
        ggml_build_forward_expand(gf, conv_write);
        state_write(skippy_state_kind::recurrent_conv, conv_state, conv_write, il);

        ggml_tensor * conv_out = ggml_ssm_conv(ctx0, sx, kernel); // {w, n_seq_tokens, n_seqs}

        ggml_tensor * y = ggml_add(ctx0, x3, conv_out); // built-in residual, no activation

        return ggml_reshape_2d(ctx0, y, w, n_seq_tokens*n_seqs);
    };

    auto build_attn_block = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];

        const bool    is_swa     = hparams.is_swa(il);
        const int64_t n_head_kv  = hparams.n_head_kv(il);
        const int64_t rel_extent = is_swa ? hparams.inkling_rel_extent_swa : hparams.inkling_rel_extent;

        ggml_tensor * q = build_lora_mm(layer.wq, cur);
        ggml_tensor * k = build_lora_mm(layer.wk, cur);
        ggml_tensor * v = build_lora_mm(layer.wv, cur);
        ggml_tensor * r = build_lora_mm(layer.wr, cur);
        cb(q, "inkling_attn_q", il);
        cb(k, "inkling_attn_k", il);
        cb(v, "inkling_attn_v", il);
        cb(r, "inkling_attn_r", il);

        // k/v short convs on the flat projections, before the head reshape
        k = build_sconv(k, layer.shortconv_k, off_k, il);
        v = build_sconv(v, layer.shortconv_v, off_v, il);
        cb(k, "inkling_attn_k_sconv", il);
        cb(v, "inkling_attn_v_sconv", il);

        q = ggml_reshape_3d(ctx0, q, head_dim, n_head,    n_tokens);
        k = ggml_reshape_3d(ctx0, k, head_dim, n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, head_dim, n_head_kv, n_tokens);

        q = build_norm(q, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        k = build_norm(k, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(q, "inkling_attn_q_norm", il);
        cb(k, "inkling_attn_k_norm", il);

        // log-N tau on global layers only, after q_norm
        if (tau && !is_swa) {
            q = ggml_mul(ctx0, q, tau);
        }

        // relative position bias
        ggml_tensor * r2 = ggml_reshape_2d(ctx0, r, d_rel, n_head*n_tokens);

        // proj stored [E, d_rel]; transpose so ggml_mul_mat contracts over d_rel
        ggml_tensor * proj = ggml_cont(ctx0, ggml_transpose(ctx0, layer.attn_rel_proj)); // {d_rel, E}

        ggml_tensor * rel = ggml_mul_mat(ctx0, proj, r2); // {E, n_head*n_tokens}
        ggml_prec_set_acc(rel, GGML_PREC_F32);
        rel = ggml_reshape_3d(ctx0, rel, rel_extent, n_head, n_tokens);

        if (tau && !is_swa) {
            rel = ggml_mul(ctx0, rel, tau);
        }
        cb(rel, "inkling_rel_logits", il);

        auto * inp_attn = inp_hybrid->get_attn();
        const int64_t n_stream = (is_swa ? inp_attn->get_kq_mask_swa() : inp_attn->get_kq_mask())->ne[3];
        GGML_ASSERT(n_tokens % n_stream == 0);

            // soft_max_ext scales kq + kq_b jointly: fold 1/head_dim into q to keep the bias unscaled
            q = ggml_scale(ctx0, q, 1.0f/float(head_dim));

            // zero column at index E is gathered by out-of-band / empty-cell indices
            rel = ggml_pad(ctx0, rel, 1, 0, 0, 0); // {E+1, n_head, n_tokens}
            rel = ggml_cont(ctx0, ggml_permute(ctx0, rel, 1, 0, 2, 3)); // {n_head, E+1, n_tokens}
            rel = ggml_reshape_2d(ctx0, rel, n_head, (rel_extent + 1)*n_tokens);

            ggml_tensor * idx = is_swa ? rel_idx_swa : rel_idx; // {n_kv, n_tokens}
            GGML_ASSERT(idx != nullptr);
            const int64_t n_kv = idx->ne[0];

            ggml_tensor * idx1 = ggml_reshape_1d(ctx0, idx, n_kv*n_tokens);

            ggml_tensor * kq_b = ggml_get_rows(ctx0, rel, idx1); // {n_head, n_kv*n_tokens}
            kq_b = ggml_reshape_3d(ctx0, kq_b, n_head, n_kv, n_tokens);
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3)); // {n_kv, n_tokens, n_head}
            cb(kq_b, "inkling_kq_b", il);

            // streamed kq is [n_kv, n_tokens/n_stream, n_head, n_stream], tokens stream-major: view kq_b to match (same trick as the KQ mask)
            if (n_stream > 1) {
                kq_b = ggml_view_4d(ctx0, kq_b, n_kv, n_tokens/n_stream, n_head, n_stream,
                        kq_b->nb[1],
                        kq_b->nb[2],
                        (n_tokens/n_stream)*kq_b->nb[1],
                        0);
            }

            cur = build_attn(inp_attn,
                    layer.wo, NULL, NULL,
                    q, k, v, kq_b, nullptr, nullptr, 1.0f, il);
        cb(cur, "inkling_attn_o", il);

        return cur;
    };

    auto build_dense_ffn = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_mul(ctx0, cur, model.layers[il].ffn_gscale);
        cb(cur, "inkling_dense_ffn_out", il);
        return cur;
    };

    // custom MoE routing (not expressible via build_moe_ffn): select by top-k(sigmoid(logits) + bias), weight by softmax(logsigmoid(raw logits)) * scales
    auto build_moe = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];

        const int64_t n_shexp = hparams.n_expert_shared;

        ggml_tensor * logits = build_lora_mm(
            layer.ffn_gate_inp, cur, nullptr, GGML_PREC_F32); // {n_expert + n_shexp, n_tokens}
        cb(logits, "inkling_moe_logits", il);

        const size_t lsz = ggml_element_size(logits);

        ggml_tensor * routed = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, n_expert, n_tokens, logits->nb[1], 0));
        ggml_tensor * shared_logits = ggml_view_2d(ctx0, logits, n_shexp, n_tokens, logits->nb[1], n_expert*lsz);

        // bias affects selection only, not the weights
        ggml_tensor * scores = ggml_sigmoid(ctx0, routed);
        scores = ggml_add(ctx0, scores, layer.ffn_exp_probs_b);
        cb(scores, "inkling_moe_scores", il);

        ggml_tensor * selected = ggml_argsort_top_k(ctx0, scores, n_expert_used); // I32 {n_expert_used, n_tokens}
        cb(selected, "inkling_moe_topk", il);

        // weights use the raw top-k logits, not the biased scores
        ggml_tensor * routed3     = ggml_reshape_3d(ctx0, routed, 1, n_expert, n_tokens);
        ggml_tensor * topk_logits = ggml_get_rows(ctx0, routed3, selected); // {1, n_expert_used, n_tokens}
        topk_logits = ggml_reshape_2d(ctx0, topk_logits, n_expert_used, n_tokens);

        ggml_tensor * all_logits = ggml_concat(ctx0, topk_logits, shared_logits, 0); // {n_expert_used + n_shexp, n_tokens}

        // logsigmoid(x) = -softplus(-x)
        ggml_tensor * w = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, all_logits)));
        w = ggml_soft_max(ctx0, w);
        w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
        w = ggml_mul(ctx0, w, layer.ffn_gscale); // gate global_scale (F32 [1])
        cb(w, "inkling_moe_weights", il);

        const size_t wsz = ggml_element_size(w);

        ggml_tensor * weights = ggml_cont(ctx0, ggml_view_2d(ctx0, w, n_expert_used, n_tokens, w->nb[1], 0));
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);

        ggml_tensor * xr   = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        ggml_tensor * gate = build_lora_mm_id(layer.ffn_gate_exps, xr, selected); // {n_ff_exp, n_expert_used, n_tokens}
        ggml_tensor * up   = build_lora_mm_id(layer.ffn_up_exps,   xr, selected);
        ggml_tensor * h    = ggml_swiglu_split(ctx0, gate, up);

        ggml_tensor * experts = build_lora_mm_id(layer.ffn_down_exps, h, selected); // {n_embd, n_expert_used, n_tokens}
        experts = ggml_mul(ctx0, experts, weights);

        ggml_tensor * moe_out = nullptr;
        for (int64_t i = 0; i < n_expert_used; ++i) {
            ggml_tensor * e = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx0, moe_out, e) : e;
        }

        // shared experts: mul_mat_id with constant ids (never 2D-view a quantized/repacked weight)
        GGML_ASSERT(shexp_idx != nullptr);
        ggml_tensor * gs = build_lora_mm_id(layer.ffn_gate_shexp, xr, shexp_idx); // {n_ff_exp, n_shexp, n_tokens}
        ggml_tensor * us = build_lora_mm_id(layer.ffn_up_shexp,   xr, shexp_idx);
        ggml_tensor * hs = ggml_swiglu_split(ctx0, gs, us);

        // gammas (last n_shexp weight rows) must scale hs BEFORE the down-proj to match reference rounding in bf16/quant
        ggml_tensor * gammas = ggml_cont(ctx0, ggml_view_2d(ctx0, w, n_shexp, n_tokens, w->nb[1], n_expert_used*wsz));
        hs = ggml_mul(ctx0, hs, ggml_reshape_3d(ctx0, gammas, 1, n_shexp, n_tokens));
        ggml_tensor * ds = build_lora_mm_id(layer.ffn_down_shexp, hs, shexp_idx); // {n_embd, n_shexp, n_tokens}

        for (int64_t s = 0; s < n_shexp; ++s) {
            ggml_tensor * e = ggml_view_2d(ctx0, ds, n_embd, n_tokens, ds->nb[2], s*ds->nb[1]);
            moe_out = ggml_add(ctx0, moe_out, e);
        }
        cb(moe_out, "inkling_moe_out", il);

        return moe_out;
    };

    auto build_block = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        conv_rs_cur = build_rs(
                inp_hybrid->get_recr(),
                mctx_recr->get_r_l(il),
                n_embd_r,
                n_seqs,
                skippy_state_kind::recurrent_conv,
                il);

        // h += attn_sconv(attn(attn_norm(h)))
        ggml_tensor * attn_in  = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(attn_in, "inkling_attn_norm", il);
        ggml_tensor * attn_out = build_attn_block(attn_in, il);
        attn_out = build_sconv(attn_out, model.layers[il].shortconv_attn, off_attn, il);
        cb(attn_out, "inkling_attn_sconv", il);

        cur = ggml_add(ctx0, cur, attn_out);

        // h += mlp_sconv(mlp(mlp_norm(h)))
        ggml_tensor * ffn_in  = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(ffn_in, "inkling_ffn_norm", il);
        ggml_tensor * ffn_out = is_mtp || il < (int) hparams.n_layer_dense_lead ?
            build_dense_ffn(ffn_in, il) : build_moe(ffn_in, il);
        ffn_out = build_sconv(ffn_out, model.layers[il].shortconv_mlp, off_mlp, il);
        cb(ffn_out, "inkling_ffn_sconv", il);

        cur = ggml_add(ctx0, cur, ffn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        return cur;
    };

    const llama_model * shared_model = &model;
    if (is_mtp && cparams.ctx_other != nullptr) {
        const llama_model * target_model = llama_get_model(cparams.ctx_other);
        if (target_model != nullptr && target_model->tok_embd != nullptr &&
                target_model->tok_norm != nullptr && target_model->output != nullptr) {
            shared_model = target_model;
        }
    }

    if (is_mtp) {
        const auto & layer = model.layers[mtp_il];
        GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm);
        GGML_ASSERT(shared_model->tok_embd && shared_model->tok_norm && shared_model->output);

        auto mtp_inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd, hparams.n_embd);
        mtp_inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(mtp_inp->tokens);
        mtp_inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_input(mtp_inp->embd);
        ggml_set_name(mtp_inp->embd, "inkling_mtp_embd_input");
        mtp_inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
        ggml_set_input(mtp_inp->h);
        ggml_set_name(mtp_inp->h, "inkling_mtp_h_input");

        ggml_tensor * tok_embd = nullptr;
        if (ubatch.token) {
            tok_embd = ggml_get_rows(ctx0, shared_model->tok_embd, mtp_inp->tokens);
            tok_embd = build_norm(tok_embd, shared_model->tok_norm, nullptr, LLM_NORM_RMS, -1);
        } else {
            tok_embd = mtp_inp->embd;
        }
        tok_embd = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, mtp_il);
        ggml_tensor * hidden = build_norm(mtp_inp->h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, mtp_il);
        ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, hidden, tok_embd, 0));
        cb(cur, "inkling_mtp_input", mtp_il);
        res->add_input(std::move(mtp_inp));
        ggml_build_forward_expand(gf, cur);
        cur = build_block(cur, mtp_il);

        ggml_tensor * inp_out_ids = build_inp_out_ids();
        ggml_tensor * h_nextn = cur;
        if (cparams.embeddings_nextn_masked && inp_out_ids) {
            h_nextn = ggml_get_rows(ctx0, h_nextn, inp_out_ids);
        }
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;

        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        cur = build_lora_mm(
            shared_model->output, cur, nullptr,
            shared_model->output->type == GGML_TYPE_F32 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
        if (vocab_mask) {
            cur = ggml_add(ctx0, cur, vocab_mask);
        }
        cb(cur, "result_output", -1);
        res->t_logits = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    // mtmd embd rows arrive pre-normalized; embed_norm applies to text token lookups only
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, NULL, LLM_NORM_RMS, -1);
    }
    cb(cur, ubatch.token ? "inkling_embd_norm" : "inkling_mm_embd", -1);
    ggml_build_forward_expand(gf, cur);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = cur;
        cur = build_block(cur, il);
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Conv states need every layer to see all tokens. Intermediate stages hand off the full
    // activation sequence, and the final stage trims outputs only after its last local layer.
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm_all", -1);
    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = cur;
        if (cparams.embeddings_nextn_masked && inp_out_ids) {
            h_nextn = ggml_get_rows(ctx0, h_nextn, inp_out_ids);
        }
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
    }
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (!cparams.embeddings) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        cur = build_lora_mm(
            model.output, cur, nullptr,
            model.output->type == GGML_TYPE_F32 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);

        // padded vocab rows get -inf so samplers never emit a padded id
        if (vocab_mask) {
            cur = ggml_add(ctx0, cur, vocab_mask);
        }
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}
