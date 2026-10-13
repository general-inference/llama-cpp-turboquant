#ifndef SKIPPY_STAGE_PLAN_H
#define SKIPPY_STAGE_PLAN_H

/**
 * @file skippy/stage_plan.h
 * @brief Metadata-only construction and inspection of guarded stage plans.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SKIPPY_STAGE_PLANNER_CONFIG_V1_ABI_VERSION 1
#define SKIPPY_STAGE_PLANNER_TENSOR_V1_ABI_VERSION 1
#define SKIPPY_STAGE_PLANNER_PROFILE_V1_ABI_VERSION 1
#define SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION 1
#define SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION 1
#define SKIPPY_STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION 1
#define SKIPPY_STAGE_PLAN_STATE_DESC_V1_ABI_VERSION 1
#define SKIPPY_STAGE_PLAN_MAX_DIMS 4

struct skippy_stage_planner;
struct skippy_stage_plan;

/** @brief Offset and length of a UTF-8 string owned by a stage plan. */
struct skippy_stage_plan_string_ref_v1 {
    uint64_t offset;
    uint64_t length;
};

/** @brief Exact package tensor binding used by metadata planning. */
struct skippy_stage_planner_tensor_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * tensor_id;
    const char * native_name;
    int32_t ggml_type;
    uint32_t dimension_count;
    int64_t dimensions[SKIPPY_STAGE_PLAN_MAX_DIMS];
    uint32_t split_no;
    uint32_t reserved;
    uint64_t data_offset;
    uint64_t stored_length;
};

/** @brief One requested whole-model execution profile. */
struct skippy_stage_planner_profile_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * profile_id;
    uint32_t n_tokens;
    uint32_t n_sequences;
    uint32_t n_outputs;
    uint32_t n_recurrent_rollback_sequences;
};

/** @brief Versioned inputs for a metadata-only stage planner.
 *
 * Graph configuration and backend identities are copied by the planner and
 * become part of every guarded profile and plan identity.
 */
struct skippy_stage_planner_config_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * package_id;
    const char * const * shard_paths;
    size_t shard_count;
    const struct skippy_stage_planner_tensor_v1 * tensors;
    size_t tensor_count;
    const struct skippy_stage_planner_profile_v1 * profiles;
    size_t profile_count;
    const char * graph_configuration_id;
    const char * backend_id;
};

/** @brief Kind of semantic value enumerated for an execution profile. */
enum skippy_stage_plan_value_kind {
    SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_IMPORT = 0,
    SKIPPY_STAGE_PLAN_VALUE_ACTIVATION_EXPORT = 1,
    SKIPPY_STAGE_PLAN_VALUE_REQUEST_INPUT = 2,
};

/** @brief Kind of persistent state accessed by a stage. */
enum skippy_stage_plan_state_kind {
    SKIPPY_STAGE_PLAN_STATE_KV_KEY = 0,
    SKIPPY_STAGE_PLAN_STATE_KV_VALUE = 1,
    SKIPPY_STAGE_PLAN_STATE_RECURRENT_CONV = 2,
    SKIPPY_STAGE_PLAN_STATE_RECURRENT_SSM = 3,
};

/** @brief Access performed against persistent state. */
enum skippy_stage_plan_state_access {
    SKIPPY_STAGE_PLAN_STATE_READ = 0,
    SKIPPY_STAGE_PLAN_STATE_WRITE = 1,
};

/** @brief Residency policy for persistent state. */
enum skippy_stage_plan_state_residency {
    SKIPPY_STAGE_PLAN_STATE_LAYER_LOCAL = 0,
    SKIPPY_STAGE_PLAN_STATE_PER_STAGE = 1,
};

/** @brief Pointer-free summary of one realized guarded stage plan. */
struct skippy_stage_plan_desc_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    struct skippy_stage_plan_string_ref_v1 package_id;
    struct skippy_stage_plan_string_ref_v1 plan_id;
    int32_t layer_count;
    int32_t layer_start;
    int32_t layer_end;
    uint32_t reserved;
    uint64_t profile_count;
    uint64_t resident_tensor_count;
    /** @brief Opaque native execution dependency contract; carry unchanged to runtime config. */
    struct skippy_stage_plan_string_ref_v1 execution_contract;
};

/** @brief Pointer-free guard and identity summary for one graph profile. */
struct skippy_stage_plan_profile_desc_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    struct skippy_stage_plan_string_ref_v1 profile_id;
    struct skippy_stage_plan_string_ref_v1 graph_identity;
    struct skippy_stage_plan_string_ref_v1 profile_identity;
    struct skippy_stage_plan_string_ref_v1 slice_identity;
    struct skippy_stage_plan_string_ref_v1 source_snapshot_identity;
    struct skippy_stage_plan_string_ref_v1 graph_configuration_id;
    struct skippy_stage_plan_string_ref_v1 backend_id;
    uint32_t n_tokens;
    uint32_t n_sequences;
    uint32_t n_outputs;
    uint32_t n_recurrent_rollback_sequences;
    uint64_t activation_import_count;
    uint64_t activation_export_count;
    uint64_t request_input_count;
    uint64_t state_effect_count;
};

/** @brief Pointer-free semantic identity for a resident or frontier value. */
struct skippy_stage_plan_value_desc_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    struct skippy_stage_plan_string_ref_v1 identity;
    struct skippy_stage_plan_string_ref_v1 binding;
};

/** @brief Pointer-free persistent-state effect descriptor. */
struct skippy_stage_plan_state_desc_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    struct skippy_stage_plan_string_ref_v1 identity;
    int32_t kind;
    int32_t access;
    int32_t layer;
    int32_t residency;
    int64_t write_ordinal;
};

/** @brief Creates a planner from explicit ordered shards and exact package bindings.
 *
 * All pointed-to inputs are borrowed only for this synchronous call. On
 * success `out_planner` owns an independent metadata-only model. On failure it
 * is set to null and no partial planner is returned.
 */
LLAMA_API enum skippy_status skippy_stage_planner_create_v1(
        const struct skippy_stage_planner_config_v1 * config,
        struct skippy_stage_planner ** out_planner,
        struct skippy_error ** out_error);

/** @brief Releases a metadata-only stage planner. */
LLAMA_API void skippy_stage_planner_free(struct skippy_stage_planner * planner);

/** @brief Realizes one guarded layer range across every configured profile.
 *
 * The returned immutable plan remains valid independently of the planner and
 * must be released with `skippy_stage_plan_free`. On failure `out_plan` is set
 * to null.
 */
LLAMA_API enum skippy_status skippy_stage_planner_realize_v1(
        const struct skippy_stage_planner * planner,
        int32_t layer_start,
        int32_t layer_end,
        struct skippy_stage_plan ** out_plan,
        struct skippy_error ** out_error);

/** @brief Releases an immutable realized stage plan. */
LLAMA_API void skippy_stage_plan_free(struct skippy_stage_plan * plan);

/** @brief Copies the root descriptor for a realized plan. */
LLAMA_API enum skippy_status skippy_stage_plan_describe_v1(
        const struct skippy_stage_plan * plan,
        struct skippy_stage_plan_desc_v1 * out_desc,
        struct skippy_error ** out_error);

/** @brief Copies one profile descriptor by deterministic profile index. */
LLAMA_API enum skippy_status skippy_stage_plan_profile_at_v1(
        const struct skippy_stage_plan * plan,
        size_t index,
        struct skippy_stage_plan_profile_desc_v1 * out_desc,
        struct skippy_error ** out_error);

/** @brief Copies one resident package tensor identity by sorted index. */
LLAMA_API enum skippy_status skippy_stage_plan_resident_tensor_at_v1(
        const struct skippy_stage_plan * plan,
        size_t index,
        struct skippy_stage_plan_value_desc_v1 * out_desc,
        struct skippy_error ** out_error);

/** @brief Copies one activation-frontier or request-input identity. */
LLAMA_API enum skippy_status skippy_stage_plan_value_at_v1(
        const struct skippy_stage_plan * plan,
        size_t profile_index,
        enum skippy_stage_plan_value_kind kind,
        size_t index,
        struct skippy_stage_plan_value_desc_v1 * out_desc,
        struct skippy_error ** out_error);

/** @brief Copies one persistent-state effect by profile and deterministic index. */
LLAMA_API enum skippy_status skippy_stage_plan_state_at_v1(
        const struct skippy_stage_plan * plan,
        size_t profile_index,
        size_t index,
        struct skippy_stage_plan_state_desc_v1 * out_desc,
        struct skippy_error ** out_error);

/** @brief Resolves a plan-owned string reference.
 *
 * On success `out_data` points into immutable plan storage and remains valid
 * until the plan is freed. The bytes are not NUL-terminated; `out_length`
 * receives their exact length.
 */
LLAMA_API enum skippy_status skippy_stage_plan_string_v1(
        const struct skippy_stage_plan * plan,
        struct skippy_stage_plan_string_ref_v1 reference,
        const char ** out_data,
        size_t * out_length,
        struct skippy_error ** out_error);

/** @brief Validates that realized plans form one complete compatible chain. */
LLAMA_API enum skippy_status skippy_stage_plan_validate_chain_v1(
        const struct skippy_stage_plan * const * plans,
        size_t plan_count,
        struct skippy_error ** out_error);

#ifdef __cplusplus
}
#endif

#endif // SKIPPY_STAGE_PLAN_H
