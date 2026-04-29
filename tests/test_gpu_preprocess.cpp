#include "preprocess/gpu_preprocessor.h"

#ifdef EDGE_ENABLE_CUDA

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "preprocess/preprocessor.h"

namespace {

edge::VideoFrame MakeFrame() {
    edge::VideoFrame frame;
    frame.meta.stream_id = 0;
    frame.meta.frame_id = 0;
    frame.meta.width = 32;
    frame.meta.height = 16;
    frame.meta.timestamp_ms = 0;
    frame.channels = 3;
    frame.image.create(frame.meta.height, frame.meta.width, CV_8UC3);

    for (int y = 0; y < frame.image.rows; ++y) {
        auto* row = frame.image.ptr<uint8_t>(y);
        for (int x = 0; x < frame.image.cols; ++x) {
            const int base = x * 3;
            row[base + 0] = static_cast<uint8_t>((x * 3 + y) % 256);
            row[base + 1] = static_cast<uint8_t>((x + y * 5) % 256);
            row[base + 2] = static_cast<uint8_t>((x * 7 + y * 2) % 256);
        }
    }
    return frame;
}

float MeanAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    assert(lhs.size() == rhs.size());
    double sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        sum += std::fabs(lhs[i] - rhs[i]);
    }
    return static_cast<float>(sum / static_cast<double>(lhs.size()));
}

void CheckPadding(const edge::TensorBuffer& tensor, int input_w, int input_h, int pad_y) {
    const float expected = 114.0F / 255.0F;
    const int plane = input_w * input_h;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < pad_y; ++y) {
            for (int x = 0; x < input_w; ++x) {
                const size_t index = static_cast<size_t>(c) * static_cast<size_t>(plane) +
                                     static_cast<size_t>(y) * static_cast<size_t>(input_w) +
                                     static_cast<size_t>(x);
                assert(std::fabs(tensor.host_data[index] - expected) < 1e-5F);
            }
        }
    }
}

}  // namespace

int main() {
    constexpr int input_w = 64;
    constexpr int input_h = 64;
    edge::VideoFrame frame = MakeFrame();
    const std::vector<edge::VideoFrame> frames{frame};

    edge::CPUPreprocessor cpu_preprocessor(input_w, input_h);
    edge::TensorBuffer cpu_tensor;
    std::vector<edge::PreprocessMeta> cpu_metas;
    assert(cpu_preprocessor.RunBatch(frames, cpu_tensor, cpu_metas));

    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);

    edge::GpuPreprocessor gpu_preprocessor(input_w, input_h);
    edge::GpuTensorBuffer gpu_tensor;
    edge::TensorBuffer gpu_host_tensor;
    std::vector<edge::PreprocessMeta> gpu_metas;
    double gpu_ms = 0.0;
    double d2h_ms = 0.0;
    assert(gpu_preprocessor.RunBatch(frames, stream, gpu_tensor, gpu_metas, &gpu_ms));
    assert(gpu_preprocessor.CopyToHost(gpu_tensor, stream, gpu_host_tensor, &d2h_ms));
    assert(cudaStreamDestroy(stream) == cudaSuccess);

    assert(cpu_tensor.shape == gpu_host_tensor.shape);
    assert(cpu_metas.size() == gpu_metas.size());
    assert(cpu_metas[0].original_width == gpu_metas[0].original_width);
    assert(cpu_metas[0].original_height == gpu_metas[0].original_height);
    assert(cpu_metas[0].input_width == gpu_metas[0].input_width);
    assert(cpu_metas[0].input_height == gpu_metas[0].input_height);
    assert(std::fabs(cpu_metas[0].scale - gpu_metas[0].scale) < 1e-6F);
    assert(cpu_metas[0].pad_x == gpu_metas[0].pad_x);
    assert(cpu_metas[0].pad_y == gpu_metas[0].pad_y);

    const float mean_abs_diff = MeanAbsDiff(cpu_tensor.host_data, gpu_host_tensor.host_data);
    if (mean_abs_diff > 0.03F) {
        std::cerr << "GPU preprocess diff too large: " << mean_abs_diff << '\n';
        return 1;
    }

    CheckPadding(gpu_host_tensor, input_w, input_h, gpu_metas[0].pad_y);
    return 0;
}

#endif  // EDGE_ENABLE_CUDA
