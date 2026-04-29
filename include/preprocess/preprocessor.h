#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "common/types.h"

namespace edge {

class CPUPreprocessor {
public:
    CPUPreprocessor(int input_width, int input_height);

    bool Run(const VideoFrame& frame, TensorBuffer& output, PreprocessMeta& meta) const;
    bool RunBatch(
        const std::vector<VideoFrame>& frames,
        TensorBuffer& output,
        std::vector<PreprocessMeta>& metas) const;
    std::string Name() const;

private:
    bool FillOne(const VideoFrame& frame, std::size_t batch_index, TensorBuffer& output, PreprocessMeta& meta) const;

    int input_width_ = 0;
    int input_height_ = 0;
};

}  // namespace edge
