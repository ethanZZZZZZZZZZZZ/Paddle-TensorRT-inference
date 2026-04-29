#pragma once

#include <string>

#include "common/config.h"
#include "common/types.h"

namespace edge {

class InferEngine {
public:
    virtual ~InferEngine() = default;

    virtual bool Init(const AppConfig& config) = 0;
    virtual bool Infer(const TensorBuffer& input, TensorBuffer& output) = 0;
    virtual bool Infer(const TensorBuffer& input, InferOutput& output) {
        output = InferOutput{};
        if (!Infer(input, output.host_tensor)) {
            return false;
        }
        output.has_host_tensor = true;
        return true;
    }
    virtual std::string Name() const = 0;
};

}  // namespace edge
