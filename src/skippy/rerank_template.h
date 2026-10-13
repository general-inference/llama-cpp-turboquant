#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

// Substitute only markers from the model-authored template. User-supplied
// query/document bytes are appended verbatim and must never be parsed again.
inline std::string skippy_render_rerank_template(
        std::string_view prompt_template,
        std::string_view query,
        std::string_view document) {
    constexpr std::string_view query_marker = "{query}";
    constexpr std::string_view document_marker = "{document}";
    std::string rendered;
    std::size_t cursor = 0;
    while (cursor < prompt_template.size()) {
        const std::size_t query_pos = prompt_template.find(query_marker, cursor);
        const std::size_t document_pos = prompt_template.find(document_marker, cursor);
        const std::size_t next = std::min(query_pos, document_pos);
        if (next == std::string_view::npos) {
            rendered.append(prompt_template.data() + cursor, prompt_template.size() - cursor);
            break;
        }
        rendered.append(prompt_template.data() + cursor, next - cursor);
        if (next == query_pos) {
            rendered.append(query.data(), query.size());
            cursor = next + query_marker.size();
        } else {
            rendered.append(document.data(), document.size());
            cursor = next + document_marker.size();
        }
    }
    return rendered;
}
