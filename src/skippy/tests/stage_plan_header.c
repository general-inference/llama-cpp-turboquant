#include "skippy/stage_plan.h"

int main(void) {
    struct skippy_stage_plan_desc_v1 descriptor = {0};
    struct skippy_stage_planner_config_v1 config = {0};
    config.graph_configuration_id = "skippy.test.graph.v1";
    config.backend_id = "skippy.test.backend";
    return sizeof(descriptor) > 0 && config.graph_configuration_id != 0 &&
            config.backend_id != 0 && SKIPPY_STAGE_PLAN_DESC_V1_ABI_VERSION == 1 ? 0 : 1;
}
