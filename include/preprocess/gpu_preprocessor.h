#pragma once

#ifdef EDGE_ENABLE_CUDA

#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/types.h"

namespace edge {

class GpuTensorBuffer {
public:
    GpuTensorBuffer() = default;
    ~GpuTensorBuffer();

    GpuTensorBuffer(const GpuTensorBuffer&) = delete;
    GpuTensorBuffer& operator=(const GpuTensorBuffer&) = delete;

    GpuTensorBuffer(GpuTensorBuffer&& other) noexcept;
    GpuTensorBuffer& operator=(GpuTensorBuffer&& other) noexcept;

    bool Allocate(const std::vector<int64_t>& shape);
    void Reset();

    float* Data();
    const float* Data() const;
    const std::vector<int64_t>& Shape() const;
    std::size_t Elements() const;
    std::size_t Bytes() const;

private:
    float* device_data_ = nullptr;
    std::vector<int64_t> shape_;
    std::size_t elements_ = 0;
};

class GpuPreprocessor {
public:
    GpuPreprocessor(int input_width, int input_height);

    bool RunBatch(
        const std::vector<VideoFrame>& frames,
        cudaStream_t stream,
        GpuTensorBuffer& output,
        std::vector<PreprocessMeta>& metas,
        double* gpu_preprocess_ms = nullptr) const;

    bool CopyToHost(
        const GpuTensorBuffer& gpu_tensor,
        cudaStream_t stream,
        TensorBuffer& host_tensor,
        double* d2h_copy_ms = nullptr,
        bool use_pinned_memory = false) const;

    std::string Name() const;

private:
    bool PrepareMeta(const VideoFrame& frame, PreprocessMeta& meta) const;

    int input_width_ = 0;
    int input_height_ = 0;
};

}  // namespace edge

#endif  // EDGE_ENABLE_CUDA
