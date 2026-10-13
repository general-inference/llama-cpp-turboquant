#include "../src/skippy-verify-checkpoint.h"

#include <cstddef>
#include <vector>

struct checkpoint {
    bool valid;
    int token_start;
    size_t token_count;
};

int main() {
    std::vector<checkpoint> checkpoints;
    for (int window = 0; window < 129; ++window) {
        checkpoints.push_back({true, window * 4, 4});
        if (!skippy_retire_verify_checkpoint_exact(checkpoints, window * 4, 4)) return 1;
        if (!checkpoints.empty()) return 2;
    }

    checkpoints = {{true, 10, 4}, {true, 14, 4}, {true, 18, 4}};
    if (!skippy_retire_verify_checkpoint_exact(checkpoints, 14, 4)) return 3;
    if (checkpoints.size() != 2 || checkpoints[0].token_start != 10 || checkpoints[1].token_start != 18) return 4;
    if (skippy_retire_verify_checkpoint_exact(checkpoints, 14, 4)) return 5;
    if (skippy_retire_verify_checkpoint_exact(checkpoints, 10, 3)) return 6;
    return 0;
}
