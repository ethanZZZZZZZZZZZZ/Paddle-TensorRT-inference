#include "preprocess/gpu_preprocessor.h"

#ifdef EDGE_ENABLE_CUDA

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <opencv2/core.hpp>

#include "common/logging.h"
#include "cuda/cuda_runtime_utils.h"
#include "cuda/preprocess_kernel.h"

namespace edge {
namespace {

double ElapsedMs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(us) / 1000.0;
}

bool CheckCuda(cudaError_t status, const std::string& what) {
    if (status == cudaSuccess) {
        return true;
    }
    EDGE_LOG_ERROR(what << " failed: " << cudaGetErrorString(status));
    return false;
}

std::size_t CountElements(const std::vector<int64_t>& shape) {
    std::size_t elements = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        elements *= static_cast<std::size_t>(dim);
    }
    return elements;
}

}  // namespace

GpuTensorBuffer::~GpuTensorBuffer() {
    Reset();
}

GpuTensorBuffer::GpuTensorBuffer(GpuTensorBuffer&& other) noexcept
    : device_data_(other.device_data_),
      shape_(std::move(other.shape_)),
      elements_(other.elements_) {
    other.device_data_ = nullptr;
    other.elements_ = 0;
}

GpuTensorBuffer& GpuTensorBuffer::operator=(GpuTensorBuffer&& other) noexcept {
    if (this != &other) {
        Reset();
        device_data_ = other.device_data_;
        shape_ = std::move(other.shape_);
        elements_ = other.elements_;
        other.device_data_ = nullptr;
        other.elements_ = 0;
    }
    return *this;
}

bool GpuTensorBuffer::Allocate(const std::vector<int64_t>& shape) {
    const std::size_t elements = CountElements(shape);
    if (elements == 0) {
        EDGE_LOG_ERROR("GpuTensorBuffer cannot allocate an empty tensor");
        return false;
    }

    if (elements_ == elements && device_data_ != nullptr) {
        shape_ = shape;
        return true;
    }

    Reset();
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_data_), elements * sizeof(float)),
                   "cudaMalloc GpuTensorBuffer")) {
        return false;
    }
    shape_ = shape;
    elements_ = elements;
    return true;
}

void GpuTensorBuffer::Reset() {
    if (device_data_ != nullptr) {
        cudaFree(device_data_);
        device_data_ = nullptr;
    }
    shape_.clear();
    elements_ = 0;
}

float* GpuTensorBuffer::Data() {
    return device_data_;
}

const float* GpuTensorBuffer::Data() const {
    return device_data_;
}

const std::vector<int64_t>& GpuTensorBuffer::Shape() const {
    return shape_;
}

std::size_t GpuTensorBuffer::Elements() const {
    return elements_;
}

std::size_t GpuTensorBuffer::Bytes() const {
    return elements_ * sizeof(float);
}

GpuPreprocessor::GpuPreprocessor(int input_width, int input_height)
    : input_width_(input_width),
      input_height_(input_height) {}

bool GpuPreprocessor::RunBatch(
    const std::vector<VideoFrame>& frames,
    cudaStream_t stream,
    GpuTensorBuffer& output,
    std::vector<PreprocessMeta>& metas,
    double* gpu_preprocess_ms) const {
    const auto start = std::chrono::steady_clock::now();
    if (frames.empty()) {
        EDGE_LOG_ERROR("GpuPreprocessor received an empty batch");
        return false;
    }
    if (input_width_ <= 0 || input_height_ <= 0) {
        EDGE_LOG_ERROR("Invalid GpuPreprocessor input size");
        return false;
    }

    const std::vector<int64_t> output_shape{
        static_cast<int64_t>(frames.size()), 3, input_height_, input_width_};
    if (!output.Allocate(output_shape)) {
        return false;
    }

    metas.clear();
    metas.reserve(frames.size());
    std::vector<uint8_t*> device_inputs;
    device_inputs.reserve(frames.size());
    auto cleanup_device_inputs = [&device_inputs]() {
        for (uint8_t* ptr : device_inputs) {
            cudaFree(ptr);
        }
        device_inputs.clear();
    };

    for (std::size_t batch_index = 0; batch_index < frames.size(); ++batch_index) {
        const VideoFrame& frame = frames[batch_index];
        if (frame.Empty() || frame.image.empty()) {
            EDGE_LOG_ERROR("GpuPreprocessor received an empty cv::Mat frame");
            cleanup_device_inputs();
            return false;
        }
        if (frame.image.channels() != 3) {
            EDGE_LOG_ERROR("GpuPreprocessor expects BGR cv::Mat with 3 channels");
            cleanup_device_inputs();
            return false;
        }

        PreprocessMeta meta;
        if (!PrepareMeta(frame, meta)) {
            cleanup_device_inputs();
            return false;
        }
        metas.push_back(meta);

        uint8_t* device_input = nullptr;
        const std::size_t input_bytes =
            static_cast<std::size_t>(frame.image.step) * static_cast<std::size_t>(frame.image.rows);
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes),
                       "cudaMalloc input frame")) {
            cleanup_device_inputs();
            return false;
        }
        device_inputs.push_back(device_input);

        if (!CheckCuda(cudaMemcpy2DAsync(
                           device_input,
                           frame.image.step,
                           frame.image.data,
                           frame.image.step,
                           static_cast<std::size_t>(frame.image.cols) * 3U,
                           static_cast<std::size_t>(frame.image.rows),
                           cudaMemcpyHostToDevice,
                           stream),
                       "cudaMemcpy2DAsync input frame")) {
            cleanup_device_inputs();
            return false;
        }

        if (!CheckCuda(LaunchPreprocessKernel(
                           device_input,
                           frame.image.cols,
                           frame.image.rows,
                           static_cast<int>(frame.image.step),
                           output.Data(),
                           input_width_,
                           input_height_,
                           meta.scale,
                           meta.pad_x,
                           meta.pad_y,
                           static_cast<int>(batch_index),
                           static_cast<int>(frames.size()),
                           stream),
                       "LaunchPreprocessKernel")) {
            cleanup_device_inputs();
            return false;
        }

        EDGE_LOG_INFO("GpuPreprocessor sample stream_id=" << frame.meta.stream_id
                                                          << ", frame_id=" << frame.meta.frame_id
                                                          << ", original=" << meta.original_width
                                                          << "x" << meta.original_height
                                                          << ", input=" << meta.input_width
                                                          << "x" << meta.input_height
                                                          << ", scale=" << meta.scale
                                                          << ", pad_x=" << meta.pad_x
                                                          << ", pad_y=" << meta.pad_y);
    }

    if (!CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize gpu preprocess")) {
        cleanup_device_inputs();
        return false;
    }
    cleanup_device_inputs();

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = ElapsedMs(start, end);
    if (gpu_preprocess_ms != nullptr) {
        *gpu_preprocess_ms = elapsed_ms;
    }
    EDGE_LOG_INFO("[Preprocess] gpu_preprocess_latency_ms=" << elapsed_ms
                                                            << ", batch=" << frames.size()
                                                            << ", output_shape="
                                                            << ShapeToString(output.Shape()));
    return true;
}

bool GpuPreprocessor::CopyToHost(
    const GpuTensorBuffer& gpu_tensor,
    cudaStream_t stream,
    TensorBuffer& host_tensor,
    double* d2h_copy_ms,
    bool use_pinned_memory) const {
    const auto start = std::chrono::steady_clock::now();
    if (gpu_tensor.Data() == nullptr || gpu_tensor.Elements() == 0) {
        EDGE_LOG_ERROR("GpuPreprocessor CopyToHost received an empty GPU tensor");
        return false;
    }

    if (use_pinned_memory) {
        if (!AllocatePinnedFloatTensor(gpu_tensor.Shape(), gpu_tensor.Elements(), host_tensor)) {
            return false;
        }
    } else {
        host_tensor.ClearExternalHostData();
        host_tensor.shape = gpu_tensor.Shape();
        host_tensor.host_data.resize(gpu_tensor.Elements());
    }

    if (!CheckCuda(cudaMemcpyAsync(host_tensor.MutableData(),
                                   gpu_tensor.Data(),
                                   gpu_tensor.Bytes(),
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync GPU tensor to host")) {
        return false;
    }
    if (!CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize d2h copy")) {
        return false;
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = ElapsedMs(start, end);
    if (d2h_copy_ms != nullptr) {
        *d2h_copy_ms = elapsed_ms;
    }
    EDGE_LOG_INFO("[Preprocess] d2h_copy_latency_ms=" << elapsed_ms
                                                      << ", pinned_host="
                                                      << (host_tensor.IsPinnedHost() ? "true" : "false")
                                                      << ", output_shape="
                                                      << ShapeToString(host_tensor.shape));
    return true;
}

std::string GpuPreprocessor::Name() const {
    return "GpuPreprocessor";
}

bool GpuPreprocessor::PrepareMeta(const VideoFrame& frame, PreprocessMeta& meta) const {
    if (frame.meta.width <= 0 || frame.meta.height <= 0) {
        EDGE_LOG_ERROR("GpuPreprocessor got invalid frame metadata");
        return false;
    }

    const float scale_x = static_cast<float>(input_width_) / static_cast<float>(frame.meta.width);
    const float scale_y = static_cast<float>(input_height_) / static_cast<float>(frame.meta.height);
    const float scale = std::min(scale_x, scale_y);
    const int resized_w =
        std::max(1, static_cast<int>(std::round(static_cast<float>(frame.meta.width) * scale)));
    const int resized_h =
        std::max(1, static_cast<int>(std::round(static_cast<float>(frame.meta.height) * scale)));

    meta.original_width = frame.meta.width;
    meta.original_height = frame.meta.height;
    meta.input_width = input_width_;
    meta.input_height = input_height_;
    meta.scale = scale;
    meta.pad_x = (input_width_ - resized_w) / 2;
    meta.pad_y = (input_height_ - resized_h) / 2;
    return true;
}

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA
