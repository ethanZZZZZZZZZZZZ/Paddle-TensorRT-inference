#include "inference/mock_infer_engine.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include "common/logging.h"

namespace edge {

bool MockInferEngine::Init(const AppConfig& config) {
    input_width_ = config.model.input_width;
    input_height_ = config.model.input_height;
    num_classes_ = std::max(1, config.model.num_classes);
    mock_num_boxes_ = std::max(1, config.model.mock_num_boxes);

    EDGE_LOG_INFO("MockInferEngine initialized with input="
                  << input_width_ << "x" << input_height_
                  << ", num_classes=" << num_classes_
                  << ", mock_num_boxes=" << mock_num_boxes_);
    return true;
}

bool MockInferEngine::Infer(const TensorBuffer& input, TensorBuffer& output) {
    if (input.shape.size() != 4 || input.shape[0] <= 0 || input.shape[2] <= 0 || input.shape[3] <= 0) {
        EDGE_LOG_ERROR("MockInferEngine expects input shape [batch, 3, H, W], got "
                       << ShapeToString(input.shape));
        return false;
    }

    const int batch = static_cast<int>(input.shape[0]);
    constexpr int values_per_box = 6;
    output.shape = {batch, mock_num_boxes_, values_per_box};
    output.ClearExternalHostData();
    output.host_data.assign(
        static_cast<size_t>(batch) * static_cast<size_t>(mock_num_boxes_) * values_per_box,
        0.0F);

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < mock_num_boxes_; ++i) {
            const float offset = static_cast<float>(i * 24);
            const float box_w = std::min(160.0F, static_cast<float>(input_width_) * 0.25F);
            const float box_h = std::min(120.0F, static_cast<float>(input_height_) * 0.20F);
            const float x1 = static_cast<float>(input_width_) * 0.20F + offset;
            const float y1 = static_cast<float>(input_height_) * 0.30F + offset;
            const float x2 = std::min(static_cast<float>(input_width_ - 1), x1 + box_w);
            const float y2 = std::min(static_cast<float>(input_height_ - 1), y1 + box_h);
            const float score = std::max(0.05F, 0.90F - static_cast<float>(i) * 0.15F);
            const float class_id = static_cast<float>(i % num_classes_);
            const size_t base =
                (static_cast<size_t>(b) * static_cast<size_t>(mock_num_boxes_) + static_cast<size_t>(i)) *
                values_per_box;
            output.host_data[base + 0] = x1;
            output.host_data[base + 1] = y1;
            output.host_data[base + 2] = x2;
            output.host_data[base + 3] = y2;
            output.host_data[base + 4] = score;
            output.host_data[base + 5] = class_id;
        }
    }

    EDGE_LOG_INFO("MockInferEngine produced tensor shape=" << ShapeToString(output.shape));
    return true;
}

std::string MockInferEngine::Name() const {
    return "MockInferEngine";
}

}  // namespace edge
