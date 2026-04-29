#pragma once

#ifdef EDGE_ENABLE_PADDLE

#include <memory>
#include <string>
#include <vector>

#include "inference/infer_engine.h"
#include "paddle_inference_api.h"

namespace edge {

class PaddleInferEngine : public InferEngine {
public:
    bool Init(const AppConfig& config) override;
    bool Infer(const TensorBuffer& input, TensorBuffer& output) override;
    bool Infer(const TensorBuffer& input, InferOutput& output) override;
    std::string Name() const override;

private:
    bool ConfigureModel(const PaddleConfig& paddle_config, paddle_infer::Config& predictor_config) const;
    bool ConfigureTensorRt(const TrtConfig& trt_config, paddle_infer::Config& predictor_config) const;
    bool RunPredictor(const TensorBuffer& input);
    bool CopyInput(const TensorBuffer& input);
    bool CopyOutput(TensorBuffer& output);
    bool ExportOutput(InferOutput& output);

    std::shared_ptr<paddle_infer::Predictor> predictor_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

}  // namespace edge

#endif  // EDGE_ENABLE_PADDLE
