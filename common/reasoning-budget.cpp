#include "reasoning-budget.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<std::vector<llama_token>> seqs;
    std::vector<llama_token> recent;
    size_t max_length = 0;

    token_matcher(const std::vector<std::vector<llama_token>> & seqs) : seqs(collect(seqs)) {
        for (const auto & seq : this->seqs) {
            max_length = std::max(max_length, seq.size());
        }
    }

    static std::vector<std::vector<llama_token>> collect(const std::vector<std::vector<llama_token>> & seqs) {
        std::vector<std::vector<llama_token>> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        recent.push_back(token);
        if (recent.size() > max_length) {
            recent.erase(recent.begin());
        }
        int32_t match = -1;
        size_t match_length = 0;
        for (size_t i = 0; i < seqs.size(); ++i) {
            const auto & seq = seqs[i];
            if (seq.size() >= match_length && seq.size() <= recent.size() &&
                    std::equal(seq.rbegin(), seq.rend(), recent.rbegin())) {
                match = static_cast<int32_t>(i);
                match_length = seq.size();
            }
        }
        if (match >= 0) {
            recent.clear();
        }
        return match;
    }

    void reset() { recent.clear(); }
};

static std::string reasoning_budget_token_to_piece(const llama_vocab * vocab, llama_token token) {
    std::string piece;
    piece.resize(piece.capacity());
    int32_t length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false);
    if (length < 0) {
        piece.resize(static_cast<size_t>(-length));
        length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, false);
    }
    if (length < 0) {
        return {};
    }
    piece.resize(static_cast<size_t>(length));
    return piece;
}

static bool reasoning_budget_utf8_is_complete(const std::string & value) {
    if (value.empty()) {
        return true;
    }
    for (int offset = 1; offset <= std::min(4, static_cast<int>(value.size())); ++offset) {
        const unsigned char byte = value[value.size() - offset];
        if ((byte & 0xc0) != 0x80) {
            const int expected = byte >= 0xf0 ? 4 : byte >= 0xe0 ? 3 : byte >= 0xc0 ? 2 : 1;
            return offset >= expected;
        }
    }
    return false;
}

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    std::vector<llama_token> forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = reasoning_budget_token_to_piece(ctx->vocab, token);
                utf8_complete = reasoning_budget_utf8_is_complete(piece);
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                if (utf8_complete) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->end_matcher.reset();
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->forced_tokens.size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    if (ctx->force_pos >= ctx->forced_tokens.size()) {
        return;
    }

    const llama_token forced = ctx->forced_tokens[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<std::vector<llama_token>> & start_seqs,
        const std::vector<std::vector<llama_token>> & end_seqs, const std::vector<llama_token> & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<std::vector<llama_token>> & start_seqs,
        const std::vector<std::vector<llama_token>> & end_seqs,
        const std::vector<llama_token>              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ token_matcher(start_seqs),
            /* .end_matcher   = */ token_matcher(end_seqs),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<std::vector<llama_token>> & start_seqs,
        const std::vector<std::vector<llama_token>> & end_seqs,
        const std::vector<llama_token>              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const std::vector<llama_token> * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    return true;
}
