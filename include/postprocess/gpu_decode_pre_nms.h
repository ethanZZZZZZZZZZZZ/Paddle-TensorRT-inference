#pragma once

#ifdef EDGE_ENABLE_CUDA

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/config.h"
#include "common/types.h"

namespace edge {

class GpuDecodePreNMS {
public:
    explicit GpuDecodePreNMS(PostprocessConfig config, bool use_pinned_memory = false);

    bool Run(
        const TensorBuffer& model_output,
        const std::vector<FrameMeta>& frame_metas,
        const std::vector<PreprocessMeta>& preprocess_metas,
        int input_width,
        int input_height,
        cudaStream_t stream,
        std::vector<Detection>& decoded,
        double* gpu_decode_pre_nms_ms = nullptr) const;

    bool RunWithGpuNms(
        const TensorBuffer& model_output,
        const std::vector<FrameMeta>& frame_metas,
        const std::vector<PreprocessMeta>& preprocess_metas,
        int input_width,
        int input_height,
        cudaStream_t stream,
        std::vector<Detection>& final_detections,
        double* gpu_decode_pre_nms_ms = nullptr,
        double* gpu_nms_ms = nullptr) const;

    std::string Name() const;

private:
    PostprocessConfig config_;
    bool use_pinned_memory_ = false;
};

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA
