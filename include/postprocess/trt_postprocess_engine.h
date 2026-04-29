#pragma once

#ifdef EDGE_ENABLE_TENSORRT_PLUGIN

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "common/config.h"
#include "common/types.h"
#include "postprocess/postprocessor.h"

namespace edge {

class TrtPostprocessEngine {
public:
    TrtPostprocessEngine(
        PostprocessConfig config,
        int input_width,
        int input_height,
        bool use_pinned_memory = false);

    bool Run(
        const TensorBuffer& model_output,
        const std::vector<FrameMeta>& frame_metas,
        const std::vector<PreprocessMeta>& preprocess_metas,
        cudaStream_t stream,
        std::vector<Detection>& detections,
        PostprocessTiming* timing = nullptr);

    bool RunDevice(
        const DeviceTensorView& model_output,
        const std::vector<FrameMeta>& frame_metas,
        const std::vector<PreprocessMeta>& preprocess_metas,
        cudaStream_t stream,
        std::vector<Detection>& detections,
        PostprocessTiming* timing = nullptr);

    std::string Name() const;

private:
    template <typename T>
    struct TrtDeleter {
        void operator()(T* ptr) const {
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR < 10
            if (ptr != nullptr) {
                ptr->destroy();
            }
#else
            delete ptr;
#endif
        }
    };

    template <typename T>
    using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

    bool BuildForShape(const std::vector<int64_t>& model_shape, TensorDataType model_dtype);
    bool IsBuiltForShape(const std::vector<int64_t>& model_shape, TensorDataType model_dtype) const;
    bool AllocateDeviceBuffers(
        const std::vector<int64_t>& model_shape,
        TensorDataType model_dtype,
        int batch,
        const std::vector<PreprocessMeta>& preprocess_metas,
        bool allocate_model_output,
        std::vector<float>* host_meta,
        void** device_model_output,
        float** device_meta,
        float** device_detections,
        int** device_counts) const;
    bool EnqueueAndCopy(
        int batch,
        const std::vector<FrameMeta>& frame_metas,
        cudaStream_t stream,
        const void* device_model_output,
        float* device_meta,
        float* device_detections,
        int* device_counts,
        std::vector<Detection>& detections) const;
    bool CopyOutputsToHost(
        int batch,
        const std::vector<FrameMeta>& frame_metas,
        cudaStream_t stream,
        const float* device_detections,
        const int* device_counts,
        std::vector<Detection>& detections) const;

    PostprocessConfig config_;
    int input_width_ = 0;
    int input_height_ = 0;
    bool use_pinned_memory_ = false;
    std::vector<int64_t> built_model_shape_;
    TensorDataType built_model_dtype_ = TensorDataType::kUnknown;

    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
};

}  // namespace edge

#endif  // EDGE_ENABLE_TENSORRT_PLUGIN
