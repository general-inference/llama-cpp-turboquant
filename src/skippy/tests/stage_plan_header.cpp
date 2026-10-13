#include "skippy/stage_plan.h"

#include <cstddef>
#include <type_traits>

int main() {
    static_assert(std::is_standard_layout<skippy_stage_planner_config_v1>::value, "C ABI layout required");
    static_assert(std::is_standard_layout<skippy_stage_plan_profile_desc_v1>::value, "C ABI layout required");
    static_assert(std::is_trivial<skippy_stage_plan_state_desc_v1>::value, "C ABI triviality required");
    static_assert(offsetof(skippy_stage_planner_config_v1, graph_configuration_id) >
            offsetof(skippy_stage_planner_config_v1, profile_count), "identity fields must be appended");
    return SKIPPY_STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION == 1 ? 0 : 1;
}
