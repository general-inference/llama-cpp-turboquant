#ifndef SKIPPY_H
#define SKIPPY_H

/**
 * @file skippy.h
 * @brief Umbrella include for the capability-oriented Skippy C API.
 *
 * Include a narrower header when a consumer only needs one capability. This
 * file exists for consumers that want the complete staged-runtime surface.
 */

// Public umbrella for the capability-oriented Skippy C API.
#include "skippy/common.h"
#include "skippy/events.h"
#include "skippy/devices.h"
#include "skippy/runtime.h"
#include "skippy/model_source.h"
#include "skippy/stage_plan.h"
#include "skippy/activation.h"
#include "skippy/sampling.h"
#include "skippy/speculative_decoding.h"
#include "skippy/execution.h"
#include "skippy/state.h"
#include "skippy/tokenization.h"
#include "skippy/model_package.h"
#include "skippy/signals.h"
#include "skippy/workloads.h"
#include "skippy/system_one.h"

#endif // SKIPPY_H
