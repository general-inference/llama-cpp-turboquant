#include "skippy.h"

#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// CacheGen KV-page import for V-less (MLA) caches.
//
// A page exported from an MLA cache carries no V payload, and the CacheGen
// encoder emits no V records for it. The importer must therefore skip V tiles
// instead of consuming the null V stream, which is what a page from an MLA
// cache used to dereference on import.
//
// Scope: ggml_backend_cachegen_decode is implemented by the Metal and CUDA
// backends only, so a CPU-resident cache can be validated but not decoded. The
// cases below pin the validation walk - page and record geometry, the V-less
// skip, and record coverage - and stop at the backend rejection, which is only
// reached once that walk has succeeded. test-skippy-cachegen-metal covers the
// decode itself.
//
// The mixed-stream guard in stage_import_cachegen_kv_page (a page whose V
// geometry disagrees with the selected layers' V streams) is unreachable for a
// descriptor mismatch: stage_import_kv_page(..., validate_only = true) already
// rejects that before a record is consumed. The V geometry cases below pin that
// ordering; the guard stays as the fail-closed invariant.

namespace {

// Rows and channels declared by the LCG1 segment below; k_row_bytes = channels
// * sizeof(float). The segment is the encoder's own F32 payload, reused here so
// the importer validates a real record instead of a synthetic descriptor.
constexpr uint32_t segment_rows     = 17;
constexpr uint32_t segment_channels = 8;
constexpr uint32_t k_row_bytes      = segment_channels * sizeof(float);
constexpr uint32_t page_token_count = segment_rows;
constexpr uint32_t kv_capacity      = 64;

const uint8_t encoded_f32_segment[] = {
    0x4c, 0x43, 0x47, 0x31, 0x20, 0x00, 0x11, 0x00, 0x08, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00,
    0x00, 0xa0, 0x66, 0x3e, 0x00, 0x80, 0xcd, 0x3e, 0x00, 0x60, 0x0e, 0x3f, 0x00, 0x40, 0x2e, 0x3f,
    0x00, 0x40, 0x44, 0x3f, 0x00, 0x00, 0x4f, 0x3f, 0x00, 0xe0, 0x4d, 0x3f, 0x00, 0x00, 0x41, 0x3f,
    0x00, 0x20, 0x29, 0x3f, 0x00, 0xa0, 0x09, 0x3f, 0x00, 0xe0, 0xcd, 0x3e, 0x00, 0x20, 0x75, 0x3e,
    0x00, 0x80, 0x71, 0x3d, 0x00, 0xc0, 0x36, 0x3e, 0x00, 0x80, 0xb1, 0x3e, 0x00, 0xa0, 0xfa, 0x3e,
    0x00, 0x40, 0x1d, 0x3f, 0x00, 0x00, 0x1b, 0x1e, 0x1c, 0x1e, 0x2a, 0x2d, 0x2b, 0x2d, 0x39, 0x3c,
    0x3a, 0x3c, 0x3b, 0x3c, 0x3c, 0x3c, 0x3d, 0x3c, 0x3e, 0x3c, 0x3f, 0x3c, 0x40, 0x3c, 0x4e, 0x4b,
    0x4f, 0x4b, 0x50, 0x4b, 0x51, 0x4b, 0x52, 0x4b, 0x53, 0x4b, 0x54, 0x4b, 0x55, 0x4b, 0x63, 0x5a,
    0x64, 0x5a, 0x65, 0x5a, 0x73, 0x69, 0x74, 0x69, 0x82, 0x78, 0x91, 0x87, 0x9f, 0x96, 0xad, 0xa5,
    0xc8, 0xc3, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0x1b, 0x1e, 0x29, 0x2d, 0x2a, 0x2d, 0x2b, 0x2d,
    0x39, 0x3c, 0x3a, 0x3c, 0x3b, 0x3c, 0x3c, 0x3c, 0x3d, 0x3c, 0x3e, 0x3c, 0x3f, 0x3c, 0x40, 0x3c,
    0x41, 0x3c, 0x42, 0x3c, 0x50, 0x4b, 0x51, 0x4b, 0x52, 0x4b, 0x53, 0x4b, 0x54, 0x4b, 0x55, 0x4b,
    0x56, 0x4b, 0x57, 0x4b, 0x65, 0x5a, 0x66, 0x5a, 0x74, 0x69, 0x75, 0x69, 0x83, 0x78, 0x92, 0x87,
    0xba, 0xb4, 0xc8, 0xc3, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0x1b, 0x1e, 0x29, 0x2d, 0x2a, 0x2d,
    0x38, 0x3c, 0x39, 0x3c, 0x3a, 0x3c, 0x3b, 0x3c, 0x3c, 0x3c, 0x3d, 0x3c, 0x3e, 0x3c, 0x3f, 0x3c,
    0x40, 0x3c, 0x41, 0x3c, 0x42, 0x3c, 0x43, 0x3c, 0x44, 0x3c, 0x45, 0x3c, 0x53, 0x4b, 0x54, 0x4b,
    0x55, 0x4b, 0x56, 0x4b, 0x57, 0x4b, 0x58, 0x4b, 0x66, 0x5a, 0x67, 0x5a, 0x75, 0x69, 0x83, 0x78,
    0x92, 0x87, 0xad, 0xa5, 0xd5, 0xd2, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0x1b, 0x1e, 0x29, 0x2d,
    0x37, 0x3c, 0x38, 0x3c, 0x39, 0x3c, 0x3a, 0x3c, 0x3b, 0x3c, 0x3c, 0x3c, 0x3d, 0x3c, 0x3e, 0x3c,
    0x3f, 0x3c, 0x40, 0x3c, 0x41, 0x3c, 0x42, 0x3c, 0x43, 0x3c, 0x44, 0x3c, 0x45, 0x3c, 0x46, 0x3c,
    0x47, 0x3c, 0x48, 0x3c, 0x56, 0x4b, 0x57, 0x4b, 0x58, 0x4b, 0x59, 0x4b, 0x74, 0x69, 0x75, 0x69,
    0x83, 0x78, 0x92, 0x87, 0xa0, 0x96, 0xd5, 0xd2, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0x1b, 0x1e,
    0x29, 0x2d, 0x37, 0x3c, 0x38, 0x3c, 0x39, 0x3c, 0x3a, 0x3c, 0x3b, 0x3c, 0x3c, 0x3c, 0x3d, 0x3c,
    0x3e, 0x3c, 0x3f, 0x3c, 0x40, 0x3c, 0x41, 0x3c, 0x42, 0x3c, 0x43, 0x3c, 0x44, 0x3c, 0x45, 0x3c,
    0x46, 0x3c, 0x47, 0x3c, 0x48, 0x3c, 0x49, 0x3c, 0x4a, 0x3c, 0x65, 0x5a, 0x66, 0x5a, 0x67, 0x5a,
    0x68, 0x5a, 0x76, 0x69, 0x84, 0x78, 0xa0, 0x96, 0xd5, 0xd2, 0xfe, 0xff, 0xff, 0xff, 0x00, 0x00,
    0x28, 0x2d, 0x36, 0x3c, 0x37, 0x3c, 0x38, 0x3c, 0x39, 0x3c, 0x3a, 0x3c, 0x3b, 0x3c, 0x3c, 0x3c,
    0x3d, 0x3c, 0x3e, 0x3c, 0x3f, 0x3c, 0x40, 0x3c, 0x41, 0x3c, 0x42, 0x3c, 0x43, 0x3c, 0x44, 0x3c,
    0x45, 0x3c, 0x46, 0x3c, 0x47, 0x3c, 0x48, 0x3c, 0x56, 0x4b, 0x57, 0x4b, 0x58, 0x4b, 0x59, 0x4b,
    0x5a, 0x4b, 0x68, 0x5a, 0x69, 0x5a, 0x77, 0x69, 0x93, 0x87, 0xc8, 0xc3, 0xfe, 0xff, 0xff, 0xff,
    0x00, 0x00, 0x28, 0x2d, 0x36, 0x3c, 0x37, 0x3c, 0x38, 0x3c, 0x39, 0x3c, 0x3a, 0x3c, 0x3b, 0x3c,
    0x3c, 0x3c, 0x3d, 0x3c, 0x3e, 0x3c, 0x3f, 0x3c, 0x40, 0x3c, 0x41, 0x3c, 0x42, 0x3c, 0x43, 0x3c,
    0x44, 0x3c, 0x45, 0x3c, 0x46, 0x3c, 0x54, 0x4b, 0x55, 0x4b, 0x56, 0x4b, 0x57, 0x4b, 0x58, 0x4b,
    0x59, 0x4b, 0x5a, 0x4b, 0x5b, 0x4b, 0x5c, 0x4b, 0x77, 0x69, 0x78, 0x69, 0xae, 0xa5, 0xfe, 0xff,
    0xff, 0xff, 0x00, 0x00, 0x35, 0x3c, 0x36, 0x3c, 0x37, 0x3c, 0x38, 0x3c, 0x39, 0x3c, 0x3a, 0x3c,
    0x3b, 0x3c, 0x3c, 0x3c, 0x3d, 0x3c, 0x3e, 0x3c, 0x3f, 0x3c, 0x40, 0x3c, 0x41, 0x3c, 0x42, 0x3c,
    0x43, 0x3c, 0x44, 0x3c, 0x52, 0x4b, 0x53, 0x4b, 0x54, 0x4b, 0x55, 0x4b, 0x56, 0x4b, 0x57, 0x4b,
    0x58, 0x4b, 0x59, 0x4b, 0x5a, 0x4b, 0x5b, 0x4b, 0x5c, 0x4b, 0x6a, 0x5a, 0x6b, 0x5a, 0x79, 0x69,
    0xfe, 0xff, 0xff, 0xff, 0x08, 0x00, 0x07, 0x00, 0x08, 0x00, 0x07, 0x00, 0x07, 0x00, 0x06, 0x00,
    0x06, 0x00, 0x04, 0x00, 0x41, 0x04, 0x03, 0xa5, 0xc8, 0x5a, 0x02, 0xc0, 0x41, 0x06, 0x23, 0xa4,
    0x7a, 0x6e, 0xeb, 0x41, 0x0a, 0x0f, 0x21, 0xb2, 0xd0, 0xef, 0x00, 0x41, 0x7e, 0x7c, 0xc9, 0x9a,
    0xa4, 0x08, 0x47, 0xb5, 0x66, 0xc1, 0x5d, 0x23, 0x70, 0x51, 0x1b, 0x08, 0x21, 0x46, 0xbb, 0x5b,
    0xc1, 0xcd, 0xf2, 0x3a, 0x80, 0xff, 0x2f, 0xba, 0xac,
};

const char * backend_rejection = "resident KV backend does not provide typed CacheGen decode";
const char * layout_rejection  = "native KV page descriptor does not match runtime KV cache layout";
const char * coverage_rejection = "CacheGen records do not cover the native KV page";

uint32_t read_u16(const uint8_t * bytes) {
    return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8;
}

uint32_t read_u32(const uint8_t * bytes) {
    return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
           static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
}

std::unique_ptr<llama_model> make_model() {
    return std::unique_ptr<llama_model>(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
}

llama_hparams make_hparams(bool mla) {
    llama_hparams hparams = {};
    hparams.n_layer_all        = 1;
    hparams.n_embd_head_k_full = segment_channels;
    hparams.n_embd_head_v_full = segment_channels;
    hparams.n_embd_head_k_swa  = segment_channels;
    hparams.n_embd_head_v_swa  = segment_channels;
    if (mla) {
        // is_mla() requires both MLA head sizes, which is what drops the V stream.
        hparams.n_embd_head_k_mla_impl = segment_channels;
        hparams.n_embd_head_v_mla_impl = segment_channels;
    }
    hparams.n_head_arr[0]    = 1;
    hparams.n_head_kv_arr[0] = 1;
    return hparams;
}

std::unique_ptr<llama_kv_cache> make_cache(const llama_model & model, const llama_hparams & hparams) {
    return std::unique_ptr<llama_kv_cache>(new llama_kv_cache(
            model,
            hparams,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            true,
            false,
            true,
            kv_capacity,
            1,
            1,
            0,
            LLAMA_SWA_TYPE_NONE,
            nullptr,
            [](int32_t il) { return il == 0; },
            {},
            {}));
}

// Writes n_tokens positions into cells 0..n_tokens-1 so the page exports as a
// single contiguous range, which is what stage_export_kv_page requires.
bool populate(llama_kv_cache & kv, uint32_t n_tokens) {
    std::vector<llama_pos>     pos(n_tokens);
    std::vector<llama_seq_id>  seq_ids(n_tokens, 0);
    std::vector<llama_seq_id *> seq_id(n_tokens);
    std::vector<int32_t>       n_seq_id(n_tokens, 1);
    llama_seq_id               unique_seq = 0;
    std::vector<int32_t>       seq_idx(LLAMA_MAX_SEQ, -1);
    seq_idx[0] = 0;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pos[i]      = static_cast<llama_pos>(i);
        seq_id[i]   = &seq_ids[i];
    }
    llama_ubatch ubatch = {};
    ubatch.b_equal_seqs = 1;
    ubatch.n_tokens     = n_tokens;
    ubatch.n_seq_tokens = n_tokens;
    ubatch.n_seqs       = 1;
    ubatch.n_seqs_unq   = 1;
    ubatch.n_pos        = 1;
    ubatch.pos          = pos.data();
    ubatch.n_seq_id     = n_seq_id.data();
    ubatch.seq_id       = seq_id.data();
    ubatch.seq_id_unq   = &unique_seq;
    ubatch.seq_idx      = seq_idx.data();

    llama_kv_cache::slot_info slot = {};
    slot.s0   = 0;
    slot.s1   = 0;
    slot.strm = { 0 };
    slot.idxs.resize(1);
    slot.idxs[0].resize(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        slot.idxs[0][i] = i;
    }
    kv.apply_ubatch(slot, ubatch);
    return kv.seq_pos_min(0) == 0 && kv.seq_pos_max(0) == static_cast<llama_pos>(n_tokens) - 1;
}

bool export_page(llama_kv_cache & kv, uint32_t tokens, skippy_kv_page_desc & desc,
        std::vector<uint8_t> & payload, std::string & error) {
    size_t bytes = 0;
    if (!kv.stage_export_kv_page(0, 0, 1, 0, tokens, &desc, nullptr, 0, &bytes, error)) {
        return false;
    }
    payload.resize(bytes);
    return kv.stage_export_kv_page(0, 0, 1, 0, tokens, &desc, payload.data(), payload.size(), &bytes, error) &&
           bytes == payload.size();
}

struct import_result {
    bool        ok               = false;
    bool        unsupported      = false;
    bool        invalid_argument = false;
    std::string error;
};

import_result import_page(
        llama_kv_cache &                  kv,
        const skippy_kv_page_desc &       desc,
        const skippy_cachegen_record_v1 * records,
        size_t                            record_count,
        bool                              validate_only) {
    import_result result;
    result.ok = kv.stage_import_cachegen_kv_page(
            0, desc, records, record_count, 0, result.error, result.unsupported, result.invalid_argument, validate_only);
    return result;
}

int fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    return 1;
}

}  // namespace

int main() {
    std::unique_ptr<llama_model> model = make_model();
    if (!model) {
        return fail("failed to create test model");
    }

    const llama_hparams mla_hparams = make_hparams(true);
    std::unique_ptr<llama_kv_cache> mla_kv = make_cache(*model, mla_hparams);
    if (!populate(*mla_kv, page_token_count)) {
        return fail("failed to populate the V-less KV cache");
    }

    skippy_kv_page_desc    desc = {};
    std::vector<uint8_t>   payload;
    std::string            error;
    if (!export_page(*mla_kv, page_token_count, desc, payload, error)) {
        std::fprintf(stderr, "K-only page export failed: %s\n", error.c_str());
        return 1;
    }
    // Case 1: a page from a V-less cache is K-only, and carries the exporter's
    // transposed-V flag because the runtime cache itself is transposed.
    if (desc.layer_count != 1 || desc.k_type != GGML_TYPE_F32 || desc.k_row_bytes != k_row_bytes ||
            desc.v_row_bytes != 0 || desc.v_element_bytes != 0 || desc.k_idx_row_bytes != 0 ||
            (desc.flags & SKIPPY_KV_PAGE_FLAG_V_TRANSPOSED) == 0 ||
            desc.payload_bytes != payload.size() || payload.size() != page_token_count * k_row_bytes) {
        return fail("unexpected K-only page descriptor for a V-less cache");
    }

    // Case 2: the LCG1 fixture covers exactly one page of K records.
    if (read_u16(encoded_f32_segment + 6) != segment_rows ||
            read_u32(encoded_f32_segment + 8) != segment_channels) {
        return fail("CacheGen segment fixture no longer matches the page geometry");
    }
    skippy_cachegen_record_v1 record = {};
    record.abi_version   = SKIPPY_CACHEGEN_RECORD_V1_ABI_VERSION;
    record.kind          = SKIPPY_CACHEGEN_RECORD_F32;
    record.element_bytes = sizeof(float);
    record.output_offset = 0;
    record.decoded_bytes = static_cast<uint64_t>(segment_rows) * k_row_bytes;
    record.token_count   = segment_rows;
    record.token_start   = 0;
    record.total_tokens  = 0;
    record.payload       = encoded_f32_segment;
    record.payload_bytes = sizeof(encoded_f32_segment);

    // Case 3: a K-only page with a complete record set passes descriptor,
    // record, coverage and V-stream validation and stops only at the backend.
    // Before the V-less skip this walk dereferenced the null V stream and
    // crashed, so reaching the backend rejection is the regression assertion.
    {
        std::unique_ptr<llama_kv_cache> target = make_cache(*model, mla_hparams);
        const import_result result = import_page(*target, desc, &record, 1, false);
        if (result.ok || !result.unsupported || result.invalid_argument || result.error != backend_rejection) {
            std::fprintf(stderr, "K-only CacheGen page import did not stop at the CPU backend: %s\n",
                    result.error.c_str());
            return 1;
        }
    }

    // Case 4: the record set must cover the page exactly - a page holding one
    // more token than the records describe is rejected instead of leaving the
    // last rows unwritten.
    {
        std::unique_ptr<llama_kv_cache> longer = make_cache(*model, mla_hparams);
        if (!populate(*longer, page_token_count + 1)) {
            return fail("failed to populate the longer V-less KV cache");
        }
        skippy_kv_page_desc  longer_desc = {};
        std::vector<uint8_t> longer_payload;
        std::string          longer_error;
        if (!export_page(*longer, page_token_count + 1, longer_desc, longer_payload, longer_error)) {
            std::fprintf(stderr, "longer K-only page export failed: %s\n", longer_error.c_str());
            return 1;
        }
        if (longer_desc.payload_bytes != (page_token_count + 1) * k_row_bytes) {
            return fail("unexpected payload size for the longer K-only page");
        }
        std::unique_ptr<llama_kv_cache> target = make_cache(*model, mla_hparams);
        const import_result result = import_page(*target, longer_desc, &record, 1, false);
        if (result.ok || !result.invalid_argument || result.error != coverage_rejection) {
            std::fprintf(stderr, "under-covering K-only page was not rejected: %s\n", result.error.c_str());
            return 1;
        }
    }

    // Case 5: a V-less cache rejects a descriptor that declares V geometry. The
    // layout comparison in stage_import_kv_page fires before the mixed-stream
    // guard, which is why that guard is defence in depth rather than live logic.
    {
        skippy_kv_page_desc v_geometry_page = desc;
        v_geometry_page.v_element_bytes     = sizeof(float);
        std::unique_ptr<llama_kv_cache> target = make_cache(*model, mla_hparams);
        const import_result result = import_page(*target, v_geometry_page, &record, 1, true);
        if (result.ok || !result.invalid_argument || result.error != layout_rejection) {
            std::fprintf(stderr, "V geometry on a V-less cache was not rejected: %s\n", result.error.c_str());
            return 1;
        }
    }

    // Case 6: the mirror image - a page with no V payload is rejected by a cache
    // that does hold a V stream, so the guard never has to decide the mixture.
    {
        const llama_hparams dense_hparams = make_hparams(false);
        std::unique_ptr<llama_kv_cache> dense = make_cache(*model, dense_hparams);
        if (!populate(*dense, page_token_count)) {
            return fail("failed to populate the V-bearing KV cache");
        }
        skippy_kv_page_desc    dense_desc = {};
        std::vector<uint8_t>   dense_payload;
        std::string            dense_error;
        if (!export_page(*dense, page_token_count, dense_desc, dense_payload, dense_error)) {
            std::fprintf(stderr, "V-bearing page export failed: %s\n", dense_error.c_str());
            return 1;
        }
        if (dense_desc.v_element_bytes != sizeof(float) || dense_desc.v_row_bytes != 0) {
            return fail("unexpected page descriptor for a V-bearing cache");
        }
        std::unique_ptr<llama_kv_cache> target = make_cache(*model, dense_hparams);
        const import_result result = import_page(*target, desc, &record, 1, true);
        if (result.ok || !result.invalid_argument || result.error != layout_rejection) {
            std::fprintf(stderr, "K-only page was not rejected by a V-bearing cache: %s\n", result.error.c_str());
            return 1;
        }
    }

    return 0;
}
