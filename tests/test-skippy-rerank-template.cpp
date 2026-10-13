#include "../src/skippy/rerank_template.h"

#include <cstdio>
#include <string_view>

static bool expect_rendered(
        std::string_view prompt_template,
        std::string_view query,
        std::string_view document,
        std::string_view expected) {
    const auto rendered = skippy_render_rerank_template(prompt_template, query, document);
    if (std::string_view(rendered) == expected) {
        return true;
    }
    std::fprintf(stderr, "rerank template rendered unexpected text: %s\n", rendered.c_str());
    return false;
}

int main() {
    if (!expect_rendered(
                "Q={query}|D={document}|Q={query}",
                "literal {document}",
                "literal {query}",
                "Q=literal {document}|D=literal {query}|Q=literal {document}")) {
        return 1;
    }
    if (!expect_rendered("{query}{document}", "a", "b", "ab")) {
        return 2;
    }
    if (!expect_rendered("no markers", "a", "b", "no markers")) {
        return 3;
    }
    return 0;
}
