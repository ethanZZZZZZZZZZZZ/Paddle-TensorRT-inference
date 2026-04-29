#pragma once

#include "inference/infer_engine.h"

namespace edge {

class MockInferEngine final : public InferEngine {
public:
    bool Init(const AppConfig& config) override;
    bool Infer(const TensorBuffer& input, TensorBuffer& output) override;
    std::string Name() const override;

private:
    int input_width_ = 640;
    int input_height_ = 640;
    int num_classes_ = 3;
    int mock_num_boxes_ = 3;
};

}  // namespace edge
