# laya

# Self-contained laya model (encoder + typed decision head) that loads its own
# GGUF and builds the inference graph with ggml directly, following the
# tools/mtmd pattern.
# The runtime (laya.cpp) is compiled into libllama so the Skippy ABI can
# expose it; this interface target only carries the header path.
add_library(laya INTERFACE)

target_link_libraries     (laya INTERFACE llama)
target_include_directories(laya INTERFACE .)

# Gate the CLI on LLAMA_BUILD_TOOLS so that library-only builds skip it.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-laya-cli)
    add_executable(${TARGET} laya-cli.cpp)
    set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-laya-cli)
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries  (${TARGET} PRIVATE llama-common laya Threads::Threads)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()
