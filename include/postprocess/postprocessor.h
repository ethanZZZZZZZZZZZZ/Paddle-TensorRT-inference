#pragma once

#include <string>
#include <vector>

#include "common/config.h"
#include "common/types.h"

namespace edge {

struct PostprocessTiming {
    double decode_ms = 0.0;
    double cpu_decode_ms = 0.0;
    double gpu_decode_pre_nms_ms = 0.0;
    double nms_ms = 0.0;
    double gpu_nms_ms = 0.0;
    double trt_plugin_ms = 0.0;
    double total_ms = 0.0;
};

class CPUPostprocessor {
public:
    explicit CPUPostprocessor(PostprocessConfig config);

    bool Run(
        const TensorBuffer& model_output,
        const std::vector<FrameMeta>& frame_metas,
        const std::vector<PreprocessMeta>& preprocess_metas,
        std::vector<Detection>& final_detections,
        PostprocessTiming* timing = nullptr) const;

    std::string Name() const;

    bool DecodePreNms(
        const TensorBuffer& model_output,
        const std::vector<FrameMeta>& frame_metas,
        const std::vector<PreprocessMeta>& preprocess_metas,
        std::vector<Detection>& decoded) const;

    static Detection MapBoxToOriginal(
        const Detection& input_space_detection,
        const FrameMeta& frame_meta,
        const PreprocessMeta& preprocess_meta);

    static std::vector<Detection> Nms(std::vector<Detection> detections, float nms_threshold, int top_k);

private:
    PostprocessConfig config_;
};

}  // namespace edge
