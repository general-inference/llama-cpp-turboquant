#include "skippy/devices.h"
#include "skippy/errors.h"

#include <mutex>

static enum skippy_backend_device_type skippy_backend_device_type_from_ggml(
        enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return SKIPPY_BACKEND_DEVICE_TYPE_CPU;
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return SKIPPY_BACKEND_DEVICE_TYPE_GPU;
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return SKIPPY_BACKEND_DEVICE_TYPE_IGPU;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return SKIPPY_BACKEND_DEVICE_TYPE_ACCEL;
        case GGML_BACKEND_DEVICE_TYPE_META:
            return SKIPPY_BACKEND_DEVICE_TYPE_META;
    }
    return SKIPPY_BACKEND_DEVICE_TYPE_ACCEL;
}

static uint64_t skippy_backend_device_caps_from_ggml(const ggml_backend_dev_caps & caps) {
    uint64_t out = 0;
    if (caps.async) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_ASYNC;
    }
    if (caps.host_buffer) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_HOST_BUFFER;
    }
    if (caps.buffer_from_host_ptr) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_BUFFER_FROM_HOST_PTR;
    }
    if (caps.events) {
        out |= SKIPPY_BACKEND_DEVICE_CAP_EVENTS;
    }
    return out;
}


static void skippy_silent_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static void skippy_load_backends_for_device_query() {
    static std::once_flag once;
    std::call_once(once, []() {
        ggml_log_callback previous_callback = nullptr;
        void * previous_user_data = nullptr;
        llama_log_get(&previous_callback, &previous_user_data);

        llama_log_set(skippy_silent_log_callback, nullptr);
        llama_backend_init();
        if (!ggml_backend_reg_count()) {
            ggml_backend_load_all();
        }
        llama_log_set(previous_callback, previous_user_data);
    });
}

enum skippy_status skippy_backend_device_count(
        size_t * out_count,
        struct skippy_error ** out_error) {
    if (out_count == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_count is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_load_backends_for_device_query();
    *out_count = ggml_backend_dev_count();
    return skippy_success(out_error);
}

enum skippy_status skippy_backend_device_at(
        size_t index,
        struct skippy_backend_device * out_device,
        struct skippy_error ** out_error) {
    if (out_device == nullptr) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "out_device is required");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_load_backends_for_device_query();
    const size_t device_count = ggml_backend_dev_count();
    if (index >= device_count) {
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, "backend device index is out of range");
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);

    out_device->version = 1;
    out_device->name = props.name;
    out_device->description = props.description;
    out_device->device_id = props.device_id;
    out_device->memory_free = static_cast<uint64_t>(props.memory_free);
    out_device->memory_total = static_cast<uint64_t>(props.memory_total);
    out_device->type = skippy_backend_device_type_from_ggml(props.type);
    out_device->caps = skippy_backend_device_caps_from_ggml(props.caps);
    return skippy_success(out_error);
}

