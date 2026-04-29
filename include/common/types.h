#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef EDGE_ENABLE_OPENCV
#include <opencv2/core.hpp>
#endif

namespace edge {

struct FrameMeta {
    int stream_id = 0;
    int frame_id = 0;
    int width = 0;
    int height = 0;
    int64_t timestamp_ms = 0;
};

struct VideoFrame {
#ifdef EDGE_ENABLE_OPENCV
    cv::Mat image;
#else
    std::vector<uint8_t> image;
#endif
    FrameMeta meta;
    int channels = 3;

    bool Empty() const {
        return image.empty() || meta.width <= 0 || meta.height <= 0 || channels <= 0;
    }
};

struct TensorBuffer {
    std::vector<float> host_data;
    std::shared_ptr<float> external_host_data;
    size_t external_host_elements = 0;
    bool external_host_pinned = false;
    std::vector<int64_t> shape;

    size_t NumElements() const {
        return host_data.empty() && external_host_data ? external_host_elements : host_data.size();
    }

    const float* Data() const {
        return host_data.empty() && external_host_data ? external_host_data.get() : host_data.data();
    }

    float* MutableData() {
        return host_data.empty() && external_host_data ? external_host_data.get() : host_data.data();
    }

    bool IsPinnedHost() const {
        return host_data.empty() && external_host_data && external_host_pinned;
    }

    void ClearExternalHostData() {
        external_host_data.reset();
        external_host_elements = 0;
        external_host_pinned = false;
    }

    void SetExternalHostData(std::shared_ptr<float> data, size_t elements, bool pinned) {
        host_data.clear();
        external_host_data = std::move(data);
        external_host_elements = elements;
        external_host_pinned = pinned;
    }
};

enum class TensorMemoryPlace {
    kUnknown,
    kCPU,
    kGPU
};

struct DeviceTensorView {
    const float* data = nullptr;
    std::vector<int64_t> shape;
    size_t num_elements = 0;
    TensorMemoryPlace place = TensorMemoryPlace::kUnknown;
    std::string producer;

    bool IsGpuFloat() const {
        return data != nullptr && place == TensorMemoryPlace::kGPU && num_elements > 0;
    }
};

struct InferOutput {
    TensorBuffer host_tensor;
    DeviceTensorView device_tensor;
    bool has_host_tensor = false;
    bool has_device_tensor = false;
};

struct PreprocessMeta {
    int original_width = 0;
    int original_height = 0;
    int input_width = 0;
    int input_height = 0;
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
};

struct Detection {
    int stream_id = 0;
    int frame_id = 0;
    int class_id = 0;
    float score = 0.0F;
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
};

inline std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            out += ", ";
        }
    }
    out += "]";
    return out;
}

}  // namespace edge
