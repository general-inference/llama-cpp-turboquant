#include "ngram-cache.h"
#include "common.h"
#include "log.h"
#include "../include/skippy.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <thread>
#include <algorithm>

struct skippy_ngram_cache {
    uint16_t ngram_min;
    uint16_t ngram_max;
    std::vector<llama_token> history;
    common_ngram_cache context;
    common_ngram_cache dynamic;
    common_ngram_cache static_cache;
};

static bool skippy_ngram_cache_valid_config(uint16_t ngram_min, uint16_t ngram_max) {
    return ngram_min >= LLAMA_NGRAM_MIN && ngram_min <= ngram_max && ngram_max <= LLAMA_NGRAM_MAX;
}

static void skippy_ngram_cache_clear_request_state(skippy_ngram_cache * cache) {
    cache->history.clear();
    cache->context.clear();
    cache->dynamic.clear();
    cache->static_cache.clear();
}

static enum skippy_status skippy_ngram_cache_update(
        skippy_ngram_cache * cache,
        const llama_token * token_ids,
        size_t token_count,
        bool reset) {
    if (cache == nullptr || (token_count > 0 && token_ids == nullptr)) {
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (token_count == 0) {
        if (reset) {
            skippy_ngram_cache_clear_request_state(cache);
        }
        return SKIPPY_STATUS_OK;
    }
    if (token_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            cache->history.size() > static_cast<size_t>(std::numeric_limits<int>::max()) - token_count) {
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    if (reset) {
        skippy_ngram_cache_clear_request_state(cache);
        cache->history.assign(token_ids, token_ids + token_count);
        common_ngram_cache_update(
                cache->context,
                cache->ngram_min,
                cache->ngram_max,
                cache->history,
                static_cast<int>(token_count),
                false);
        return SKIPPY_STATUS_OK;
    }
    cache->history.insert(cache->history.end(), token_ids, token_ids + token_count);
    common_ngram_cache_update(
            cache->context,
            cache->ngram_min,
            cache->ngram_max,
            cache->history,
            static_cast<int>(token_count),
            false);
    return SKIPPY_STATUS_OK;
}

extern "C" SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_create(
        uint16_t ngram_min,
        uint16_t ngram_max,
        struct skippy_ngram_cache ** out_cache,
        struct skippy_error ** /* out_error */) {
    if (out_cache == nullptr || !skippy_ngram_cache_valid_config(ngram_min, ngram_max)) {
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    // common_ngram_cache_update() reads ggml_time_ms(), which on Windows divides
    // by a timer frequency that only ggml_time_init() sets. This cache can run
    // before any model open or device query has initialized ggml, so initialize
    // the timer here: it is once-only on Windows and a no-op elsewhere.
    ggml_time_init();
    *out_cache = new skippy_ngram_cache{ ngram_min, ngram_max, {}, {}, {}, {} };
    return SKIPPY_STATUS_OK;
}

extern "C" SKIPPY_COMMON_API void skippy_ngram_cache_free(struct skippy_ngram_cache * cache) {
    delete cache;
}

extern "C" SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_reset(
        struct skippy_ngram_cache * cache,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** /* out_error */) {
    return skippy_ngram_cache_update(cache, token_ids, token_count, true);
}

extern "C" SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_append(
        struct skippy_ngram_cache * cache,
        const llama_token * token_ids,
        size_t token_count,
        struct skippy_error ** /* out_error */) {
    return skippy_ngram_cache_update(cache, token_ids, token_count, false);
}

extern "C" SKIPPY_COMMON_API enum skippy_status skippy_ngram_cache_draft(
        struct skippy_ngram_cache * cache,
        const llama_token * continuation_prefix,
        size_t continuation_prefix_count,
        uint16_t max_draft_tokens,
        llama_token * output_tokens,
        size_t output_token_capacity,
        size_t * out_token_count,
        struct skippy_error ** /* out_error */) {
    if (out_token_count == nullptr || cache == nullptr ||
            (continuation_prefix_count > 0 && continuation_prefix == nullptr)) {
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }
    *out_token_count = 0;
    if (max_draft_tokens == 0 || cache->history.empty()) {
        return SKIPPY_STATUS_OK;
    }

    std::vector<llama_token> input = cache->history;
    std::vector<llama_token> draft;
    if (continuation_prefix_count == 0) {
        draft.push_back(input.back());
    } else {
        input.insert(input.end(), continuation_prefix, continuation_prefix + continuation_prefix_count);
        draft.push_back(continuation_prefix[continuation_prefix_count - 1]);
    }
    common_ngram_cache_draft(
            input,
            draft,
            max_draft_tokens,
            cache->ngram_min,
            cache->ngram_max,
            cache->context,
            cache->dynamic,
            cache->static_cache);
    const size_t drafted_count = draft.size() - 1;
    *out_token_count = drafted_count;
    if (drafted_count == 0) {
        return SKIPPY_STATUS_OK;
    }
    if (output_tokens == nullptr || output_token_capacity < drafted_count) {
        return SKIPPY_STATUS_BUFFER_TOO_SMALL;
    }
    std::copy(draft.begin() + 1, draft.end(), output_tokens);
    return SKIPPY_STATUS_OK;
}

void common_ngram_cache_update(common_ngram_cache & ngram_cache, int ngram_min, int ngram_max,
                              std::vector<llama_token> & inp, int nnew, bool print_progress) {
    const int64_t t_start_ms = ggml_time_ms();
    const int64_t inp_size = inp.size();

    const int64_t n_todo = inp_size * (ngram_max - ngram_min + 1);
    int64_t n_done = 0;

    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            common_ngram ngram(&inp[ngram_start], ngram_size);
            const llama_token token = inp[i];

            common_ngram_cache::iterator part_it = ngram_cache.find(ngram);
            if (part_it == ngram_cache.end()) {
                common_ngram_cache_part part;
                part.emplace(token, 1);
                ngram_cache.emplace(ngram, part);
            } else {
                common_ngram_cache_part::iterator token_count_it = part_it->second.find(token);
                if (token_count_it == part_it->second.end()) {
                    part_it->second.emplace(token, 1);
                } else {
                    token_count_it->second++;
                }
            }
            ++n_done;

            if (print_progress && n_done % 10000000 == 0) {
                const int64_t t_now_ms = ggml_time_ms();
                const int64_t eta_ms   = (inp_size*(ngram_max-ngram_min+1) - n_done) * (t_now_ms - t_start_ms) / n_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

                fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, n_done, n_todo, eta_min, eta_s);
            }
        }
    }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static llama_token get_token(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, const size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int    draft_min_sample_size_lax[LLAMA_NGRAM_MAX] = { 2,  2,  1,  1};
constexpr int        draft_min_percent_lax[LLAMA_NGRAM_MAX] = {66, 50, 50, 50};
constexpr int draft_min_sample_size_strict[LLAMA_NGRAM_MAX] = { 4,  3,  2,  2};
constexpr int     draft_min_percent_strict[LLAMA_NGRAM_MAX] = {75, 66, 66, 66};

// Helper function that tries to draft a token from only the static ngram cache:
static llama_token try_draft(common_ngram_cache & nc_static, const common_ngram ngram_static) {
    common_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
    if (part_static_it == nc_static.end()) {
        return LLAMA_TOKEN_NULL;
    }
    const common_ngram_cache_part part_static = part_static_it->second;

    int max_count_static  = 0;
    int sum_count_static  = 0;
    llama_token max_token = LLAMA_TOKEN_NULL;

    for (std::pair<llama_token, int> token_count_static : part_static) {
        const llama_token token = token_count_static.first;
        const int32_t count_static  = token_count_static.second;

        if (count_static > max_count_static) {
            max_token        = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    if (sum_count_static < draft_min_sample_size_lax[LLAMA_NGRAM_STATIC-1]) {
        return LLAMA_TOKEN_NULL;
    }
    if (100*max_count_static < draft_min_percent_lax[LLAMA_NGRAM_STATIC-1]*sum_count_static) {
        return LLAMA_TOKEN_NULL;
    }
    return max_token;
}

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static llama_token try_draft(
    common_ngram_cache & nc_primary, const std::vector<common_ngram> & ngrams_primary, common_ngram_cache_part & part_static,
    const int * min_sample_size, const int * min_percent, int ngram_min) {

    llama_token drafted_token = LLAMA_TOKEN_NULL;

    for (int i = ngrams_primary.size()-1; i >= 0 && drafted_token == LLAMA_TOKEN_NULL; --i) {
        const common_ngram ngram_primary = ngrams_primary[i];
        const int threshold_index = ngram_min + i - 1;

        common_ngram_cache::iterator part_primary_it = nc_primary.find(ngram_primary);
        if (part_primary_it == nc_primary.end()) {
            continue;
        }
        const common_ngram_cache_part part_primary = part_primary_it->second;

        int max_count_primary = 0;
        int max_count_static  = 0;
        int sum_count_primary = 0;
        llama_token max_token = LLAMA_TOKEN_NULL;

        for (std::pair<llama_token, int> token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;

            common_ngram_cache_part::iterator token_count_static_it = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static  = token_count_static_it != part_static.end() ? 100*token_count_static_it->second : 1;

            if (count_primary*count_static > max_count_primary*max_count_static) {
                max_token         = token;
                max_count_primary = count_primary;
                max_count_static  = count_static;
            }
            sum_count_primary += count_primary;
        }

        if (sum_count_primary < min_sample_size[threshold_index]) {
            continue;
        }
        if (100*max_count_primary < min_percent[threshold_index]*sum_count_primary) {
            continue;;
        }
        drafted_token = max_token;
    }

    return drafted_token;
}

void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, common_ngram_cache & nc_static
) {
    GGML_ASSERT(draft.size() == 1);
    const int inp_size = inp.size();

    if (inp_size < LLAMA_NGRAM_STATIC) {
        return;
    }

    while ((int) draft.size()-1 < n_draft) {
        llama_token drafted_token = LLAMA_TOKEN_NULL;

        const int ngram_start_static = inp_size-LLAMA_NGRAM_STATIC + draft.size()-1;
        common_ngram ngram_static;
        for (int j = ngram_start_static; j < ngram_start_static + LLAMA_NGRAM_STATIC; ++j) {
            ngram_static.tokens[j-ngram_start_static] = get_token(inp, draft, j);
        }
        common_ngram_cache::iterator part_static_it = nc_static.find(ngram_static);
        common_ngram_cache_part part_static;
        if (part_static_it != nc_static.end()) {
            part_static = part_static_it->second;
        }

        // cd = context + dynamic
        std::vector<common_ngram> ngrams_cd;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max; ++ngram_size_cd) {
            const int ngram_start_cd = inp_size-ngram_size_cd + draft.size()-1;
            common_ngram ngram_cd;
            for (int j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                ngram_cd.tokens[j-ngram_start_cd] = get_token(inp, draft, j);
            }
            ngrams_cd.push_back(ngram_cd);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_context, ngrams_cd, part_static, draft_min_sample_size_lax, draft_min_percent_lax, ngram_min);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_dynamic, ngrams_cd, part_static, draft_min_sample_size_strict, draft_min_percent_strict, ngram_min);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_static, ngram_static);
        }

        if (drafted_token == LLAMA_TOKEN_NULL) {
            break;
        }

        LOG_DBG(" - draft candidate: token=%d\n", drafted_token);
        draft.push_back(drafted_token);
    }
}

void common_ngram_cache_save(common_ngram_cache & ngram_cache, const std::string & filename) {
    std::ofstream file_out(filename, std::ios::binary);
    for (std::pair<common_ngram, common_ngram_cache_part> item : ngram_cache) {
        const common_ngram      ngram        = item.first;
        common_ngram_cache_part token_counts = item.second;
        GGML_ASSERT(!token_counts.empty());
        const int32_t ntokens = token_counts.size();
        GGML_ASSERT(ntokens > 0);

        file_out.write(reinterpret_cast<const char *>(&ngram),   sizeof(common_ngram));
        file_out.write(reinterpret_cast<const char *>(&ntokens), sizeof(int32_t));
        for (std::pair<llama_token, int32_t> item2 : token_counts) {
            const llama_token token = item2.first;
            const int32_t     count = item2.second;
            GGML_ASSERT(count > 0);

            file_out.write(reinterpret_cast<const char *>(&token), sizeof(llama_token));
            file_out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    }
}

common_ngram_cache common_ngram_cache_load(const std::string & filename) {
    std::ifstream hashmap_file(filename, std::ios::binary);
    if (!hashmap_file) {
        throw std::ifstream::failure("Unable to open file " + filename);
    }
    common_ngram_cache ngram_cache;

    common_ngram ngram;
    int32_t     ntokens;
    llama_token token;
    int32_t     count;

    char * ngramc   = reinterpret_cast<char*>(&ngram);
    char * ntokensc = reinterpret_cast<char*>(&ntokens);
    char * tokenc   = reinterpret_cast<char*>(&token);
    char * countc   = reinterpret_cast<char*>(&count);
    while(hashmap_file.read(ngramc, sizeof(common_ngram))) {
        GGML_ASSERT(!hashmap_file.eof());
        GGML_ASSERT(hashmap_file.read(ntokensc, sizeof(int32_t)));
        GGML_ASSERT(ntokens > 0);
        common_ngram_cache_part token_counts;

        for (int i = 0; i < ntokens; ++i) {
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(tokenc, sizeof(llama_token)));
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(countc, sizeof(int32_t)));
            GGML_ASSERT(count > 0);
            token_counts.emplace(token, count);
        }

        ngram_cache.emplace(ngram, token_counts);
    }
    GGML_ASSERT(hashmap_file.eof());

    return ngram_cache;
}

void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add) {
    for (std::pair<common_ngram, common_ngram_cache_part> ngram_part : ngram_cache_add) {
        const common_ngram      ngram = ngram_part.first;
        common_ngram_cache_part  part = ngram_part.second;

        common_ngram_cache::iterator part_merged_it = ngram_cache_target.find(ngram);
        if (part_merged_it == ngram_cache_target.end()) {
            ngram_cache_target.emplace(ngram, part);
            continue;
        }

        for (std::pair<llama_token, int32_t> token_count : part) {
            const llama_token token = token_count.first;
            const int32_t     count = token_count.second;
            GGML_ASSERT(count > 0);

            common_ngram_cache_part::iterator token_count_merged_it = part_merged_it->second.find(token);
            if (token_count_merged_it == part_merged_it->second.end()) {
                part_merged_it->second.emplace(token, count);
                continue;
            }

            token_count_merged_it->second += count;
        }
    }
}
