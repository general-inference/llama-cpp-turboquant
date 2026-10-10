llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-moe-cache.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            skippy/activation.cpp
            skippy/activation_layout.cpp
            skippy/execution-batch.cpp
            skippy/execution-single.cpp
            skippy/model_load_config.cpp
            skippy/model_lifecycle.cpp
            skippy/model_loading.cpp
            skippy/model_source.cpp
            skippy/session.cpp
            skippy/verification.cpp
            skippy/workloads.cpp
            skippy/abi.cpp
            skippy/block_boundaries.cpp
            skippy/devices.cpp
            skippy/glm_dsa.cpp
            skippy/graph_dependencies.cpp
            skippy/graph_identity.cpp
            skippy/executable_graph.cpp
            skippy/model_package.cpp
            skippy/package_tensor_binding.cpp
            ../common/reasoning-budget.cpp
            skippy/sampling.cpp
            skippy/sampling_vocab.cpp
            skippy/stage_slice_plan.cpp
            skippy/stage_plan.cpp
            skippy/stage_realization.cpp
            skippy/stage_execution_contract.cpp
            skippy/state.cpp
            skippy/state_memory.cpp
            skippy/state_effects.cpp
            skippy/speculative_decoding.cpp
            skippy/tokenization.cpp
            skippy/events.cpp
            skippy/system_one.cpp
            skippy/laya.cpp
            ../tools/laya/laya.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h ../common/reasoning-budget.cpp
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

# Keep model sources out of the unity build as well. The Skippy stage
# rewriter (Clang LibTooling) needs a per-source compilation database entry
# for every model builder, and unity composites collapse those entries.
set_source_files_properties(${LLAMA_MODELS_SOURCES}
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ../common ../tools/laya ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump
# Upstream added a models.h PCH for the llama target. The Skippy stage
# rewriter (Clang LibTooling) consumes this target's compilation database
# entries with its own clang, which cannot read a PCH produced by the
# configured compiler, so the llama target must stay PCH-free.
# target_precompile_headers (llama PRIVATE models/models.h)

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_STAGE_BUILD_TESTS)
    target_compile_definitions(llama PRIVATE LLAMA_STAGE_BUILD_TESTS)

    add_executable(skippy-hardware-application-probe
        skippy/tests/hardware_application_probe.cpp)
    target_include_directories(skippy-hardware-application-probe PRIVATE .)
    target_compile_features(skippy-hardware-application-probe PRIVATE cxx_std_17)
    target_link_libraries(skippy-hardware-application-probe PRIVATE llama)
    add_test(
        NAME skippy_hardware_application_probe
        COMMAND skippy-hardware-application-probe)

    target_compile_definitions(llama PRIVATE SKIPPY_RUNTIME_EVENTS_TEST_HOOKS)

    add_executable(skippy-runtime-events-test
        skippy/tests/runtime_events.cpp)
    target_include_directories(skippy-runtime-events-test PRIVATE .)
    target_compile_features(skippy-runtime-events-test PRIVATE cxx_std_17)
    target_compile_definitions(skippy-runtime-events-test PRIVATE
        LLAMA_STAGE_BUILD_TESTS
        SKIPPY_RUNTIME_EVENTS_TEST_HOOKS)
    target_link_libraries(skippy-runtime-events-test PRIVATE llama)
    foreach(SKIPPY_EVENT_TEST_CASE
            register_and_dispatch
            malformed_reporter_rejected
            replacement_waits_for_previous_callback
            clear_is_quiescent
            callback_lifecycle_is_nonblocking
            struct_size_extension
            concurrent_dispatch
            model_load_events_v2_dispatch
            kv_events_dispatch
            device_events_dispatch
            diagnostic_events_dispatch
            unload_events_dispatch
            families_independent)
        add_test(
            NAME skippy_runtime_event_${SKIPPY_EVENT_TEST_CASE}
            COMMAND skippy-runtime-events-test ${SKIPPY_EVENT_TEST_CASE})
    endforeach()

    set(SKIPPY_MODEL_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/skippy/test-models")
    file(MAKE_DIRECTORY "${SKIPPY_MODEL_FIXTURE_DIR}")

    add_executable(skippy-model-fixture-generator
        ../tests/test-llama-archs.cpp)
    target_compile_features(skippy-model-fixture-generator PRIVATE cxx_std_17)
    target_include_directories(skippy-model-fixture-generator PRIVATE .)
    target_link_libraries(skippy-model-fixture-generator PRIVATE llama-common)
    set(MTP_WIDTH_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mtp-width-models")
    file(MAKE_DIRECTORY "${MTP_WIDTH_FIXTURE_DIR}")
    add_test(NAME skippy_generate_mtp_width COMMAND skippy-model-fixture-generator
        --arch deepseek4 --seed 42 --deepseek4-mtp --out "${MTP_WIDTH_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_mtp_width PROPERTIES FIXTURES_SETUP mtp-width-model)
    add_test(NAME skippy_mtp_width COMMAND skippy-model-fixture-generator
        --test-mtp-model "${MTP_WIDTH_FIXTURE_DIR}/deepseek4-mtp-moe.gguf")
    set_tests_properties(skippy_mtp_width PROPERTIES FIXTURES_REQUIRED mtp-width-model)


    add_executable(skippy-profile-residency skippy/tests/profile_residency.cpp)
    target_include_directories(skippy-profile-residency PRIVATE .)
    target_compile_definitions(skippy-profile-residency PRIVATE LLAMA_STAGE_BUILD_TESTS)
    target_compile_features(skippy-profile-residency PRIVATE cxx_std_17)
    target_link_libraries(skippy-profile-residency PRIVATE llama)
    add_test(NAME skippy_profile_residency COMMAND skippy-profile-residency
        "${MTP_WIDTH_FIXTURE_DIR}/deepseek4-mtp-moe.gguf")
    set_tests_properties(skippy_profile_residency PROPERTIES FIXTURES_REQUIRED mtp-width-model)
    set(DENSE_MTP_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/dense-mtp-models")
    file(MAKE_DIRECTORY "${DENSE_MTP_FIXTURE_DIR}")
    add_test(NAME skippy_generate_dense_mtp COMMAND skippy-model-fixture-generator
        --arch qwen35 --seed 42 --dense-mtp --out "${DENSE_MTP_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_dense_mtp PROPERTIES FIXTURES_SETUP dense-mtp-model)
    add_test(NAME skippy_live_graph_mtp COMMAND skippy-profile-residency "${DENSE_MTP_FIXTURE_DIR}/dense-mtp.gguf")
    set_tests_properties(skippy_live_graph_mtp PROPERTIES FIXTURES_REQUIRED dense-mtp-model)
    set(GLM_DSA_MTP_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/glm-dsa-mtp-models")
    file(MAKE_DIRECTORY "${GLM_DSA_MTP_FIXTURE_DIR}")
    add_test(NAME skippy_generate_glm_dsa_mtp COMMAND skippy-model-fixture-generator
        --arch glm-dsa --seed 42 --dense-mtp --out "${GLM_DSA_MTP_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_glm_dsa_mtp PROPERTIES FIXTURES_SETUP glm-dsa-mtp-model)
    add_test(NAME skippy_glm_dsa_mtp COMMAND skippy-profile-residency
        "${GLM_DSA_MTP_FIXTURE_DIR}/dense-mtp.gguf")
    set_tests_properties(skippy_glm_dsa_mtp PROPERTIES FIXTURES_REQUIRED glm-dsa-mtp-model)
    set(QWEN3MOE_MTP_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/qwen3moe-mtp-models")
    file(MAKE_DIRECTORY "${QWEN3MOE_MTP_FIXTURE_DIR}")
    add_test(NAME skippy_generate_qwen3moe_mtp COMMAND skippy-model-fixture-generator
        --arch qwen3moe --seed 42 --dense-mtp --out "${QWEN3MOE_MTP_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_qwen3moe_mtp PROPERTIES FIXTURES_SETUP qwen3moe-mtp-model)
    add_test(NAME skippy_qwen3moe_mtp COMMAND skippy-profile-residency
        "${QWEN3MOE_MTP_FIXTURE_DIR}/dense-mtp.gguf")
    set_tests_properties(skippy_qwen3moe_mtp PROPERTIES FIXTURES_REQUIRED qwen3moe-mtp-model)
    set(MULTIDEPTH_MTP_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/multidepth-mtp-models")
    file(MAKE_DIRECTORY "${MULTIDEPTH_MTP_FIXTURE_DIR}")
    add_test(NAME skippy_generate_multidepth_mtp COMMAND skippy-model-fixture-generator
        --arch hy_v3 --seed 42 --dense-mtp --dense-mtp-depth 3 --out "${MULTIDEPTH_MTP_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_multidepth_mtp PROPERTIES FIXTURES_SETUP multidepth-mtp-model)
    add_test(NAME skippy_live_graph_mtp_multidepth COMMAND skippy-profile-residency
        "${MULTIDEPTH_MTP_FIXTURE_DIR}/dense-mtp.gguf")
    set_tests_properties(skippy_live_graph_mtp_multidepth PROPERTIES
        FIXTURES_REQUIRED multidepth-mtp-model)

    add_executable(skippy-model-loader-accounting
        ../tests/test-skippy-model-loader-accounting.cpp)
    target_include_directories(skippy-model-loader-accounting PRIVATE .)
    target_compile_features(skippy-model-loader-accounting PRIVATE cxx_std_17)
    target_link_libraries(skippy-model-loader-accounting PRIVATE llama-common)

    add_test(
        NAME skippy_runtime_event_native_model_load_production
        COMMAND skippy-runtime-events-test native_model_load_production
            "${SKIPPY_MODEL_FIXTURE_DIR}/gemma-dense.gguf")
    set_tests_properties(skippy_runtime_event_native_model_load_production PROPERTIES
        FIXTURES_REQUIRED skippy-model-loader-fixtures)

    add_executable(skippy-noalloc-graph-planning
        skippy/tests/noalloc_graph_planning.cpp)
    target_include_directories(skippy-noalloc-graph-planning PRIVATE .)
    target_compile_features(skippy-noalloc-graph-planning PRIVATE cxx_std_17)
    target_link_libraries(skippy-noalloc-graph-planning PRIVATE llama)

    add_executable(skippy-renamed-multishard-planning
        skippy/tests/renamed_multishard_planning.cpp)
    target_include_directories(skippy-renamed-multishard-planning PRIVATE .)
    target_compile_definitions(skippy-renamed-multishard-planning PRIVATE LLAMA_STAGE_BUILD_TESTS)
    target_compile_features(skippy-renamed-multishard-planning PRIVATE cxx_std_17)
    target_link_libraries(skippy-renamed-multishard-planning PRIVATE llama)
    add_dependencies(skippy-noalloc-graph-planning skippy-renamed-multishard-planning)

    add_executable(skippy-graph-contract-models skippy/tests/graph_contract_models.cpp)
    target_compile_features(skippy-graph-contract-models PRIVATE cxx_std_17)
    target_link_libraries(skippy-graph-contract-models PRIVATE llama)
    add_dependencies(skippy-noalloc-graph-planning skippy-graph-contract-models)
    include(skippy/tests/graph_contract_cases.cmake)

    foreach(LFM_FAMILY lfm2 lfm2-vl)
        add_test(NAME skippy_live_graph_${LFM_FAMILY}_cross_sequence_geometry
            COMMAND skippy-profile-residency
                "${SKIPPY_MODEL_FIXTURE_DIR}/contract-${LFM_FAMILY}/graph-contract.gguf"
                cross-sequence-geometry)
        set_tests_properties(skippy_live_graph_${LFM_FAMILY}_cross_sequence_geometry PROPERTIES
            FIXTURES_REQUIRED contract-${LFM_FAMILY} TIMEOUT 900)
    endforeach()

    set(GLM_DSA_SYNTHETIC_FIXTURE_DIR "${SKIPPY_MODEL_FIXTURE_DIR}/glm-dsa-synthetic")
    file(MAKE_DIRECTORY "${GLM_DSA_SYNTHETIC_FIXTURE_DIR}")
    add_test(NAME skippy_generate_glm_dsa_synthetic COMMAND skippy-model-fixture-generator
        --arch glm-dsa --seed 42 --glm-dsa-indexshare full-shared-shared
        --out "${GLM_DSA_SYNTHETIC_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_glm_dsa_synthetic PROPERTIES
        FIXTURES_SETUP glm-dsa-synthetic)
    add_test(NAME skippy_glm_dsa_synthetic_live COMMAND skippy-renamed-multishard-planning
        "${GLM_DSA_SYNTHETIC_FIXTURE_DIR}/glm-dsa-moe.gguf"
        "${GLM_DSA_SYNTHETIC_FIXTURE_DIR}/replay" 0)
    set_tests_properties(skippy_glm_dsa_synthetic_live PROPERTIES
        FIXTURES_REQUIRED glm-dsa-synthetic TIMEOUT 900)
    foreach(glm_dsa_invalid_layout shared-first conflicting-metadata)
        add_test(NAME skippy_glm_dsa_reject_${glm_dsa_invalid_layout} COMMAND skippy-model-fixture-generator
            --arch glm-dsa --seed 42 --glm-dsa-indexshare ${glm_dsa_invalid_layout}
            --expect-invalid --out "${GLM_DSA_SYNTHETIC_FIXTURE_DIR}")
    endforeach()

    add_executable(skippy-input-capabilities skippy/tests/input_capabilities.cpp)
    target_compile_features(skippy-input-capabilities PRIVATE cxx_std_17)
    target_link_libraries(skippy-input-capabilities PRIVATE llama)
    add_dependencies(skippy-noalloc-graph-planning skippy-input-capabilities)
    foreach(arch glm4 glm4moe)
        set(text_fixture_dir "${SKIPPY_MODEL_FIXTURE_DIR}/text-only-${arch}")
        file(MAKE_DIRECTORY "${text_fixture_dir}")
        add_test(NAME skippy_generate_text_only_${arch}
            COMMAND skippy-model-fixture-generator --arch ${arch} --seed 1 --text-only --out "${text_fixture_dir}")
        set_tests_properties(skippy_generate_text_only_${arch} PROPERTIES FIXTURES_SETUP text-only-${arch})
        if(arch STREQUAL "glm4moe")
            set(model_kind moe)
        else()
            set(model_kind dense)
        endif()
        add_test(NAME skippy_input_capabilities_${arch}
            COMMAND skippy-input-capabilities "${text_fixture_dir}/${arch}-${model_kind}.gguf" 1)
        set_tests_properties(skippy_input_capabilities_${arch} PROPERTIES FIXTURES_REQUIRED text-only-${arch})
        set(embedding_fixture_dir "${SKIPPY_MODEL_FIXTURE_DIR}/embedding-${arch}")
        file(MAKE_DIRECTORY "${embedding_fixture_dir}")
        add_test(NAME skippy_generate_embedding_${arch}
            COMMAND skippy-model-fixture-generator --arch ${arch} --seed 1 --out "${embedding_fixture_dir}")
        set_tests_properties(skippy_generate_embedding_${arch} PROPERTIES FIXTURES_SETUP embedding-${arch})
        add_test(NAME skippy_input_capabilities_embedding_${arch}
            COMMAND skippy-input-capabilities "${embedding_fixture_dir}/${arch}-${model_kind}.gguf" 2)
        set_tests_properties(skippy_input_capabilities_embedding_${arch} PROPERTIES FIXTURES_REQUIRED embedding-${arch})
    endforeach()
    add_test(NAME skippy_input_capabilities_embeddings
        COMMAND skippy-input-capabilities "${SKIPPY_MODEL_FIXTURE_DIR}/gemma-dense.gguf" 2)
    set_tests_properties(skippy_input_capabilities_embeddings PROPERTIES FIXTURES_REQUIRED skippy-model-loader-fixtures)

    add_executable(skippy-stage-plan-header-c
        skippy/tests/stage_plan_header.c)
    target_compile_features(skippy-stage-plan-header-c PRIVATE c_std_11)
    target_include_directories(skippy-stage-plan-header-c PRIVATE ../include ../ggml/include)

    add_executable(skippy-stage-plan-header-cpp
        skippy/tests/stage_plan_header.cpp)
    target_compile_features(skippy-stage-plan-header-cpp PRIVATE cxx_std_17)
    target_include_directories(skippy-stage-plan-header-cpp PRIVATE ../include ../ggml/include)

    add_test(NAME skippy_stage_plan_header_c COMMAND skippy-stage-plan-header-c)
    add_test(NAME skippy_stage_plan_header_cpp COMMAND skippy-stage-plan-header-cpp)

    add_executable(skippy-stage-slice-plan
        skippy/tests/stage_slice_plan.cpp)
    target_include_directories(skippy-stage-slice-plan PRIVATE .)
    target_compile_features(skippy-stage-slice-plan PRIVATE cxx_std_17)
    target_link_libraries(skippy-stage-slice-plan PRIVATE llama)
    add_dependencies(skippy-stage-slice-plan skippy-stage-plan-header-c skippy-stage-plan-header-cpp)

    add_test(
        NAME skippy_stage_slice_plan
        COMMAND skippy-stage-slice-plan)

    add_executable(skippy-graph-build-inputs
        skippy/tests/graph_build_inputs.cpp)
    target_include_directories(skippy-graph-build-inputs PRIVATE .)
    target_compile_definitions(skippy-graph-build-inputs PRIVATE LLAMA_STAGE_BUILD_TESTS)
    target_compile_features(skippy-graph-build-inputs PRIVATE cxx_std_17)
    target_link_libraries(skippy-graph-build-inputs PRIVATE llama)
    add_dependencies(skippy-stage-slice-plan skippy-graph-build-inputs skippy-profile-residency)
    add_test(NAME skippy_decode_input_width COMMAND skippy-graph-build-inputs --input-width)

    add_test(
        NAME skippy_graph_build_inputs
        COMMAND skippy-graph-build-inputs
            "${SKIPPY_MODEL_FIXTURE_DIR}/gemma-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/dream-dense.gguf")
    set_tests_properties(skippy_graph_build_inputs PROPERTIES
        FIXTURES_REQUIRED "skippy-model-loader-fixtures;skippy-graph-build-input-fixtures")


    add_test(
        NAME skippy_generate_graph_build_input_fixture
        COMMAND skippy-model-fixture-generator
            --arch dream
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_graph_build_input_fixture PROPERTIES
        FIXTURES_SETUP skippy-graph-build-input-fixtures)

    add_test(
        NAME skippy_generate_gemma_fixture
        COMMAND skippy-model-fixture-generator
            --arch gemma
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_gemma_fixture PROPERTIES
        FIXTURES_SETUP skippy-model-loader-fixtures)

    add_executable(skippy-sampling-history skippy/tests/sampling_history.cpp)
    target_include_directories(skippy-sampling-history PRIVATE .)
    target_compile_features(skippy-sampling-history PRIVATE cxx_std_17)
    target_link_libraries(skippy-sampling-history PRIVATE llama)
    function(add_skippy_sampling_history_tests family fixture state)
        foreach(mode grammar penalty)
            add_test(NAME skippy_sampling_history_${family}_${mode}
                COMMAND skippy-sampling-history
                    "${SKIPPY_MODEL_FIXTURE_DIR}/${family}-dense.gguf" ${mode} ${state})
            set_tests_properties(skippy_sampling_history_${family}_${mode} PROPERTIES
                FIXTURES_REQUIRED ${fixture})
        endforeach()
    endfunction()
    add_skippy_sampling_history_tests(gemma skippy-model-loader-fixtures dense)
    add_skippy_sampling_history_tests(qwen35 skippy-noalloc-qwen35-fixture recurrent)
    add_skippy_sampling_history_tests(granitehybrid skippy-noalloc-granite-hybrid-fixture recurrent)

    function(add_skippy_sampling_history_state_tests family fixture state)
        add_test(NAME skippy_sampling_history_${family}_state
            COMMAND skippy-sampling-history
                "${SKIPPY_MODEL_FIXTURE_DIR}/${family}-dense.gguf" state ${state})
        set_tests_properties(skippy_sampling_history_${family}_state PROPERTIES
            FIXTURES_REQUIRED ${fixture})
    endfunction()
    add_skippy_sampling_history_state_tests(granitehybrid skippy-noalloc-granite-hybrid-fixture recurrent)

    add_test(
        NAME skippy_generate_qwen3_fixture
        COMMAND skippy-model-fixture-generator
            --arch qwen3
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_qwen3_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-qwen3-fixture)

    add_test(
        NAME skippy_generate_qwen2moe_fixture
        COMMAND skippy-model-fixture-generator
            --arch qwen2moe
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_qwen2moe_fixture PROPERTIES
        FIXTURES_SETUP skippy-model-loader-fixtures)

    add_test(
        NAME skippy_generate_granite_hybrid_fixture
        COMMAND skippy-model-fixture-generator
            --arch granitehybrid
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_granite_hybrid_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-granite-hybrid-fixture)

    add_test(
        NAME skippy_generate_qwen35_fixture
        COMMAND skippy-model-fixture-generator
            --arch qwen35
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_qwen35_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-qwen35-fixture)

    add_test(
        NAME skippy_renamed_multishard_planning_qwen35
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/qwen35-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-qwen35"
            1)
    set_tests_properties(skippy_renamed_multishard_planning_qwen35 PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-qwen35-fixture)

    add_test(
        NAME skippy_model_loader_accounting_dense
        COMMAND skippy-model-loader-accounting
            "${SKIPPY_MODEL_FIXTURE_DIR}/gemma-dense.gguf")
    set_tests_properties(skippy_model_loader_accounting_dense PROPERTIES
        FIXTURES_REQUIRED skippy-model-loader-fixtures)

    add_test(
        NAME skippy_model_loader_accounting_moe
        COMMAND skippy-model-loader-accounting
            "${SKIPPY_MODEL_FIXTURE_DIR}/qwen2moe-moe.gguf")
    set_tests_properties(skippy_model_loader_accounting_moe PROPERTIES
        FIXTURES_REQUIRED skippy-model-loader-fixtures)

    add_test(
        NAME skippy_renamed_multishard_planning_qwen2moe
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/qwen2moe-moe.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-qwen2moe"
            0)
    set_tests_properties(skippy_renamed_multishard_planning_qwen2moe PROPERTIES
        FIXTURES_REQUIRED skippy-model-loader-fixtures)

    add_test(
        NAME skippy_noalloc_graph_planning_dense
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/gemma-dense.gguf"
            0
            "kv,kv")
    set_tests_properties(skippy_noalloc_graph_planning_dense PROPERTIES
        FIXTURES_REQUIRED skippy-model-loader-fixtures)

    add_test(
        NAME skippy_noalloc_graph_planning_qwen3
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/qwen3-dense.gguf"
            0
            "kv,kv"
            0
            "token-kv")
    set_tests_properties(skippy_noalloc_graph_planning_qwen3 PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-qwen3-fixture)

    add_test(
        NAME skippy_renamed_multishard_planning_qwen3
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/qwen3-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-qwen3"
            0)
    set_tests_properties(skippy_renamed_multishard_planning_qwen3 PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-qwen3-fixture)

    add_test(
        NAME skippy_generate_gemma2_fixture
        COMMAND skippy-model-fixture-generator
            --arch gemma2
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_gemma2_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-gemma2-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_iswa
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/gemma2-dense.gguf"
            0
            "kv,kv")
    set_tests_properties(skippy_noalloc_graph_planning_iswa PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-gemma2-fixture)

    add_test(
        NAME skippy_renamed_multishard_planning_gemma2
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/gemma2-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-gemma2"
            0)
    set_tests_properties(skippy_renamed_multishard_planning_gemma2 PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-gemma2-fixture)

    add_test(
        NAME skippy_generate_minimax_m3_fixture
        COMMAND skippy-model-fixture-generator
            --arch minimax-m3
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_minimax_m3_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-minimax-m3-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_msa
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/minimax-m3-moe.gguf"
            0
            "kv,kv")
    set_tests_properties(skippy_noalloc_graph_planning_msa PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-minimax-m3-fixture)

    add_test(
        NAME skippy_renamed_multishard_planning_minimax_m3
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/minimax-m3-moe.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-minimax-m3"
            0)
    set_tests_properties(skippy_renamed_multishard_planning_minimax_m3 PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-minimax-m3-fixture)

    add_test(
        NAME skippy_generate_kimi_linear_fixture
        COMMAND skippy-model-fixture-generator
            --arch kimi-linear
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_kimi_linear_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-kimi-linear-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_kimi_linear
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/kimi-linear-moe.gguf"
            1
            "k,kda")
    set_tests_properties(skippy_noalloc_graph_planning_kimi_linear PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-kimi-linear-fixture)

    add_test(
        NAME skippy_live_graph_kimi_linear_unfused
        COMMAND skippy-profile-residency
            "${SKIPPY_MODEL_FIXTURE_DIR}/kimi-linear-moe.gguf"
            unfused-gdn)
    set_tests_properties(skippy_live_graph_kimi_linear_unfused PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-kimi-linear-fixture)

    add_test(
        NAME skippy_generate_deepseek4_fixture
        COMMAND skippy-model-fixture-generator
            --arch deepseek4
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_deepseek4_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-deepseek4-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_deepseek4_hyperconnection
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/deepseek4-moe.gguf"
            0
            "none,none,none,none"
            0
            "unchecked"
            2048)
    set_tests_properties(skippy_noalloc_graph_planning_deepseek4_hyperconnection PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-deepseek4-fixture)

    add_test(
        NAME skippy_stage_fused_ops_auto_deepseek4
        COMMAND skippy-graph-build-inputs
            --stage-fused-ops
            "${SKIPPY_MODEL_FIXTURE_DIR}/deepseek4-moe.gguf")
    set_tests_properties(skippy_stage_fused_ops_auto_deepseek4 PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-deepseek4-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_granite_hybrid
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/granitehybrid-dense.gguf"
            1
            "kv,recurrent")
    set_tests_properties(skippy_noalloc_graph_planning_granite_hybrid PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-granite-hybrid-fixture)

    add_test(
        NAME skippy_renamed_multishard_planning_dense
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/gemma-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-gemma"
            0)
    set_tests_properties(skippy_renamed_multishard_planning_dense PROPERTIES
        FIXTURES_REQUIRED skippy-model-loader-fixtures)

    add_test(
        NAME skippy_renamed_multishard_planning_granite_hybrid
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/granitehybrid-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-granite-hybrid"
            1)
    set_tests_properties(skippy_renamed_multishard_planning_granite_hybrid PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-granite-hybrid-fixture)

    add_test(
        NAME skippy_generate_nemotron_h_fixture
        COMMAND skippy-model-fixture-generator
            --arch nemotron_h
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_nemotron_h_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-nemotron-h-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_recurrent_rollback
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/nemotron_h-dense.gguf"
            1
            "kv,recurrent,none"
            2)
    set_tests_properties(skippy_noalloc_graph_planning_recurrent_rollback PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-nemotron-h-fixture)

    add_test(
        NAME skippy_generate_mamba2_fixture
        COMMAND skippy-model-fixture-generator
            --arch mamba2
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_mamba2_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-mamba2-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_recurrent_source
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/mamba2-dense.gguf"
            1
            "recurrent,recurrent")
    set_tests_properties(skippy_noalloc_graph_planning_recurrent_source PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-mamba2-fixture)

    add_test(
        NAME skippy_renamed_multishard_planning_mamba2
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/mamba2-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-mamba2"
            1)
    set_tests_properties(skippy_renamed_multishard_planning_mamba2 PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-mamba2-fixture)

    add_test(
        NAME skippy_generate_mamba_fixture
        COMMAND skippy-model-fixture-generator
            --arch mamba
            --seed 1
            --out "${SKIPPY_MODEL_FIXTURE_DIR}")
    set_tests_properties(skippy_generate_mamba_fixture PROPERTIES
        FIXTURES_SETUP skippy-noalloc-mamba-fixture)

    add_test(
        NAME skippy_noalloc_graph_planning_mamba_state_ownership
        COMMAND skippy-noalloc-graph-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/mamba-dense.gguf"
            1
            "recurrent,recurrent")
    set_tests_properties(skippy_noalloc_graph_planning_mamba_state_ownership PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-mamba-fixture)

    add_test(
        NAME skippy_renamed_multishard_planning_mamba
        COMMAND skippy-renamed-multishard-planning
            "${SKIPPY_MODEL_FIXTURE_DIR}/mamba-dense.gguf"
            "${SKIPPY_MODEL_FIXTURE_DIR}/renamed-mamba"
            1)
    set_tests_properties(skippy_renamed_multishard_planning_mamba PROPERTIES
        FIXTURES_REQUIRED skippy-noalloc-mamba-fixture)

endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
if (APPLE)
    find_library(SKIPPY_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(llama PRIVATE ${SKIPPY_ACCELERATE_FRAMEWORK})
endif()
