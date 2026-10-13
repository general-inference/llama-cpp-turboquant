#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

template<typename T>
bool skippy_retire_verify_checkpoint_exact(
        std::vector<T> & checkpoints,
        uint64_t token_start,
        uint64_t token_count) {
    const auto it = std::find_if(
            checkpoints.begin(), checkpoints.end(),
            [token_start, token_count](const T & checkpoint) {
                return checkpoint.valid &&
                       static_cast<uint64_t>(checkpoint.token_start) == token_start &&
                       checkpoint.token_count == token_count;
            });
    if (it == checkpoints.end()) {
        return false;
    }
    checkpoints.erase(it);
    return true;
}
