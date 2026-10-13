#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>

inline bool skippy_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "no") != 0;
}

inline bool skippy_env_disabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") == 0 ||
           std::strcmp(value, "false") == 0 ||
           std::strcmp(value, "FALSE") == 0 ||
           std::strcmp(value, "off") == 0 ||
           std::strcmp(value, "OFF") == 0 ||
           std::strcmp(value, "no") == 0 ||
           std::strcmp(value, "NO") == 0;
}

inline uint32_t skippy_env_u32(const char * name, uint32_t fallback, uint32_t min_value, uint32_t max_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return fallback;
    }

    return std::min<uint32_t>(max_value, std::max<uint32_t>(min_value, static_cast<uint32_t>(parsed)));
}
