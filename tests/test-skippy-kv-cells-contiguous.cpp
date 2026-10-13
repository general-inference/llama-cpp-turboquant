#include "../src/llama-kv-cells.h"

int main() {
    llama_kv_cells cells;
    cells.resize(8);
    for (uint32_t i = 0; i < 4; ++i) {
        cells.pos_set(i, static_cast<llama_pos>(i));
        cells.seq_add(i, 0);
    }
    if (!cells.seq_pos_is_cell_contiguous(0)) return 1;

    cells.mv(1, 4);
    if (cells.seq_pos_is_cell_contiguous(0)) return 2;
    cells.mv(4, 1);
    if (!cells.seq_pos_is_cell_contiguous(0)) return 3;

    if (!cells.seq_rm(2, 0)) return 4;
    if (cells.seq_pos_is_cell_contiguous(0)) return 5;
    return 0;
}
