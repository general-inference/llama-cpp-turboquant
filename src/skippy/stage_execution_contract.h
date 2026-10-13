#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct skippy_graph_build_inputs;

// Validate the selected decoder or auxiliary profile separately from the
// complete shared resident set. Throws before graph execution on disagreement.
void skippy_validate_execution_dependencies(const skippy_graph_build_inputs & inputs,
                                            bool                              auxiliary,
                                            int32_t                           offset,
                                            const std::vector<std::string> &  parameters,
                                            bool                              embedding_input = false);
