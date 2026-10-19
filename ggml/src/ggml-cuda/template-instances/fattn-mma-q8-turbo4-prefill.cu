#include "../fattn-mma-turbo.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
DECL_FATTN_MMA_TURBO_CASE(128, 128, 32, 2, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
#endif
