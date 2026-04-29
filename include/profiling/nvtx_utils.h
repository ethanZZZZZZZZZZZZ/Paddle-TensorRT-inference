#pragma once

#ifdef EDGE_ENABLE_NVTX
#include <nvToolsExt.h>

namespace edge {

class NvtxScopedRange {
public:
    explicit NvtxScopedRange(const char* name) {
        nvtxRangePushA(name);
    }

    ~NvtxScopedRange() {
        nvtxRangePop();
    }

    NvtxScopedRange(const NvtxScopedRange&) = delete;
    NvtxScopedRange& operator=(const NvtxScopedRange&) = delete;
};

}  // namespace edge

#define EDGE_PROFILE_CONCAT_INNER(a, b) a##b
#define EDGE_PROFILE_CONCAT(a, b) EDGE_PROFILE_CONCAT_INNER(a, b)
#define PROFILE_RANGE(stage_name) \
    ::edge::NvtxScopedRange EDGE_PROFILE_CONCAT(edge_nvtx_range_, __LINE__)(stage_name)

#else

#define PROFILE_RANGE(stage_name) \
    do {                          \
        (void)(stage_name);       \
    } while (false)

#endif
