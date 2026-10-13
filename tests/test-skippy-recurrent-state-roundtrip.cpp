#include "../src/llama-batch.h"
#include "../src/llama-io.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

class buffer_writer final : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t old_size = data.size();
        data.resize(old_size + size);
        ggml_backend_tensor_get(tensor, data.data() + old_size, offset, size);
    }

    size_t n_bytes() override {
        return data.size();
    }

    std::vector<uint8_t> data;
};

class buffer_reader final : public llama_io_read_i {
public:
    explicit buffer_reader(const std::vector<uint8_t> & data) : data(data) {}

    void read(void * dst, size_t size) override {
        require(size);
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        require(size);
        ggml_backend_tensor_set(tensor, data.data() + offset, tensor_offset, size);
        offset += size;
    }

    size_t n_bytes() override {
        return offset;
    }

private:
    void require(size_t size) const {
        if (size > data.size() - offset) {
            throw std::runtime_error("unexpectedly reached end of recurrent-state buffer");
        }
    }

    const std::vector<uint8_t> & data;
    size_t offset = 0;
};

static llama_memory_recurrent make_memory(const llama_model & model) {
    return llama_memory_recurrent(
            model,
            GGML_TYPE_F32,
            GGML_TYPE_F32,
            false,
            4,
            2,
            0,
            {});
}

int main() {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_MAMBA, llama_model_default_params()));
    if (!model) {
        std::fprintf(stderr, "failed to create test model\n");
        return 1;
    }

    model->hparams.n_layer_all = 1;
    model->hparams.n_embd_r_impl = 4;
    model->hparams.ssm_d_state = 2;
    model->hparams.ssm_d_inner = 2;

    auto source = make_memory(*model);
    llama_pos source_pos = 7;
    llama_seq_id source_seq_id = 0;
    llama_seq_id * source_seq_ids = &source_seq_id;
    int32_t source_n_seq_id = 1;
    llama_ubatch source_batch = {};
    source_batch.b_equal_seqs = 1;
    source_batch.n_tokens = 1;
    source_batch.n_seq_tokens = 1;
    source_batch.n_seqs = 1;
    source_batch.n_pos = 1;
    source_batch.pos = &source_pos;
    source_batch.n_seq_id = &source_n_seq_id;
    source_batch.seq_id = &source_seq_ids;
    if (!source.find_slot(source_batch)) {
        std::fprintf(stderr, "failed to allocate source recurrent cell\n");
        return 2;
    }

    const std::vector<float> source_r = { 1.0f, 2.0f, 3.0f, 4.0f };
    const std::vector<float> source_s = { 5.0f, 6.0f, 7.0f, 8.0f };
    ggml_backend_tensor_set(source.r_l[0], source_r.data(), 0, source_r.size() * sizeof(float));
    ggml_backend_tensor_set(source.s_l[0], source_s.data(), 0, source_s.size() * sizeof(float));

    buffer_writer writer;
    source.state_write(writer, source_seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    uint32_t cell_count = 0;
    llama_pos serialized_pos = -1;
    uint32_t serialized_n_seq_id = 1;
    std::memcpy(&cell_count, writer.data.data(), sizeof(cell_count));
    std::memcpy(&serialized_pos, writer.data.data() + sizeof(cell_count), sizeof(serialized_pos));
    std::memcpy(
            &serialized_n_seq_id,
            writer.data.data() + sizeof(cell_count) + sizeof(serialized_pos),
            sizeof(serialized_n_seq_id));
    if (cell_count != 1 || serialized_pos != source_pos || serialized_n_seq_id != 0) {
        std::fprintf(stderr, "unexpected single-sequence recurrent-state metadata\n");
        return 3;
    }

    auto destination = make_memory(*model);
    constexpr llama_seq_id destination_seq_id = 1;
    buffer_reader reader(writer.data);
    destination.state_read(reader, destination_seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    if (reader.n_bytes() != writer.n_bytes()) {
        std::fprintf(stderr, "recurrent-state round-trip byte count mismatch\n");
        return 4;
    }
    if (destination.seq_pos_min(destination_seq_id) != source_pos ||
            destination.seq_pos_max(destination_seq_id) != source_pos) {
        std::fprintf(stderr, "recurrent-state position was not preserved across sequence ids\n");
        return 5;
    }
    if (destination.seq_pos_min(source_seq_id) != -1) {
        std::fprintf(stderr, "recurrent state retained the source sequence id\n");
        return 6;
    }

    std::vector<float> restored_r(source_r.size());
    std::vector<float> restored_s(source_s.size());
    ggml_backend_tensor_get(destination.r_l[0], restored_r.data(), 0, restored_r.size() * sizeof(float));
    ggml_backend_tensor_get(destination.s_l[0], restored_s.data(), 0, restored_s.size() * sizeof(float));
    if (restored_r != source_r || restored_s != source_s) {
        std::fprintf(stderr, "recurrent tensor data changed during cross-sequence restore\n");
        return 7;
    }

    return 0;
}
