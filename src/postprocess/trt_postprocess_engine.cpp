#include "postprocess/trt_postprocess_engine.h"

#ifdef EDGE_ENABLE_TENSORRT_PLUGIN

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#include "common/logging.h"
#include "cuda/cuda_runtime_utils.h"
#include "yolo_decode_plugin.h"
#include "profiling/nvtx_utils.h"

namespace edge {
namespace {

constexpr int kMetaValues = 7;
constexpr int kDetectionValues = 7;
constexpr const char* kModelInputName = "model_output";
constexpr const char* kMetaInputName = "preprocess_meta";
constexpr const char* kDetectionsOutputName = "detections";
constexpr const char* kCountsOutputName = "detection_count";

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT][Postprocess] " << msg << std::endl;
        }
    }
};

TrtLogger& Logger() {
    static TrtLogger logger;
    return logger;
}

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
    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::size_t ElementBytes(TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::kFloat32:
            return sizeof(float);
        case TensorDataType::kFloat16:
            return 2;
        case TensorDataType::kInt8:
            return 1;
        case TensorDataType::kInt32:
            return sizeof(int);
        default:
            return 0;
    }
}

nvinfer1::DataType ToTrtDataType(TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::kFloat16:
            return nvinfer1::DataType::kHALF;
        case TensorDataType::kInt8:
            return nvinfer1::DataType::kINT8;
        case TensorDataType::kFloat32:
        default:
            return nvinfer1::DataType::kFLOAT;
    }
}

std::string DTypeName(TensorDataType dtype) {
    switch (dtype) {
        case TensorDataType::kFloat32:
            return "fp32";
        case TensorDataType::kFloat16:
            return "fp16";
        case TensorDataType::kInt8:
            return "int8";
        case TensorDataType::kInt32:
            return "int32";
        default:
            return "unknown";
    }
}

nvinfer1::Dims MakeDims(const std::vector<int64_t>& shape) {
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = static_cast<int>(shape[static_cast<std::size_t>(i)]);
    }
    return dims;
}

nvinfer1::Dims MakeMetaDims(int batch) {
    nvinfer1::Dims dims{};
    dims.nbDims = 2;
    dims.d[0] = batch;
    dims.d[1] = kMetaValues;
    return dims;
}

std::string ShapeKey(const std::vector<int64_t>& shape) {
    return ShapeToString(shape);
}

}  // namespace

TrtPostprocessEngine::TrtPostprocessEngine(
    PostprocessConfig config,
    int input_width,
    int input_height,
    bool use_pinned_memory)
    : config_(std::move(config)),
      input_width_(input_width),
      input_height_(input_height),
      use_pinned_memory_(use_pinned_memory) {}

bool TrtPostprocessEngine::Run(
    const TensorBuffer& model_output,
    const std::vector<FrameMeta>& frame_metas,
    const std::vector<PreprocessMeta>& preprocess_metas,
    cudaStream_t stream,
    std::vector<Detection>& detections,
    PostprocessTiming* timing) {
    PROFILE_RANGE("postprocess");
    detections.clear();
    const auto start = std::chrono::steady_clock::now();

    if (model_output.shape.size() != 3) {
        EDGE_LOG_ERROR("TrtPostprocessEngine expects model output rank 3, got "
                       << ShapeToString(model_output.shape));
        return false;
    }
    const int batch = static_cast<int>(model_output.shape[0]);
    if (batch <= 0) {
        return true;
    }
    if (static_cast<int>(frame_metas.size()) < batch ||
        static_cast<int>(preprocess_metas.size()) < batch) {
        EDGE_LOG_ERROR("TrtPostprocessEngine got metadata size smaller than output batch");
        return false;
    }
    const std::size_t expected_elements = CountElements(model_output.shape);
    if (expected_elements == 0 || model_output.NumElements() != expected_elements) {
        EDGE_LOG_ERROR("TrtPostprocessEngine model output size mismatch, expected "
                       << expected_elements << ", got " << model_output.NumElements());
        return false;
    }

    constexpr TensorDataType model_dtype = TensorDataType::kFloat32;
    if (!IsBuiltForShape(model_output.shape, model_dtype) &&
        !BuildForShape(model_output.shape, model_dtype)) {
        return false;
    }

    std::vector<float> host_meta;
    void* device_model_output = nullptr;
    float* device_meta = nullptr;
    float* device_detections = nullptr;
    int* device_counts = nullptr;

    auto cleanup = [&]() {
        cudaFree(device_model_output);
        cudaFree(device_meta);
        cudaFree(device_detections);
        cudaFree(device_counts);
    };

    if (!AllocateDeviceBuffers(model_output.shape,
                               model_dtype,
                               batch,
                               preprocess_metas,
                               true,
                               &host_meta,
                               &device_model_output,
                               &device_meta,
                               &device_detections,
                               &device_counts)) {
        cleanup();
        return false;
    }

    const std::size_t model_bytes = model_output.NumElements() * sizeof(float);
    const std::size_t meta_bytes = host_meta.size() * sizeof(float);

    PinnedHostBuffer<float> pinned_meta;
    const float* host_meta_data = host_meta.data();
    if (use_pinned_memory_) {
        if (!pinned_meta.Allocate(host_meta.size())) {
            EDGE_LOG_ERROR("cudaHostAlloc failed for TensorRT plugin pinned preprocess meta");
            cleanup();
            return false;
        }
        std::copy(host_meta.begin(), host_meta.end(), pinned_meta.Data());
        host_meta_data = pinned_meta.Data();
    }

    if (!CheckCuda(cudaMemcpyAsync(device_model_output,
                                   model_output.Data(),
                                   model_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync TRT plugin model output H2D") ||
        !CheckCuda(cudaMemcpyAsync(device_meta,
                                   host_meta_data,
                                   meta_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync TRT plugin preprocess meta H2D")) {
        cleanup();
        return false;
    }

    if (!EnqueueAndCopy(batch,
                        frame_metas,
                        stream,
                        device_model_output,
                        device_meta,
                        device_detections,
                        device_counts,
                        detections)) {
        cleanup();
        return false;
    }
    cleanup();

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = ElapsedMs(start, end);
    if (timing != nullptr) {
        timing->decode_ms = elapsed_ms;
        timing->cpu_decode_ms = 0.0;
        timing->gpu_decode_pre_nms_ms = 0.0;
        timing->nms_ms = 0.0;
        timing->gpu_nms_ms = 0.0;
        timing->trt_plugin_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
    EDGE_LOG_INFO("[Postprocess] decode_backend=trt_plugin"
                  << ", nms_backend=trt_plugin"
                  << ", zero_copy_model_output=false"
                  << ", pinned_host=" << (use_pinned_memory_ ? "true" : "false")
                  << ", trt_plugin_ms=" << elapsed_ms
                  << ", detection_count=" << detections.size());
    return true;
}

bool TrtPostprocessEngine::RunDevice(
    const DeviceTensorView& model_output,
    const std::vector<FrameMeta>& frame_metas,
    const std::vector<PreprocessMeta>& preprocess_metas,
    cudaStream_t stream,
    std::vector<Detection>& detections,
    PostprocessTiming* timing) {
    PROFILE_RANGE("postprocess");
    detections.clear();
    const auto start = std::chrono::steady_clock::now();

    if (!model_output.IsGpu()) {
        EDGE_LOG_ERROR("TrtPostprocessEngine RunDevice requires a GPU model output view");
        return false;
    }
    if (model_output.dtype != TensorDataType::kFloat32 &&
        model_output.dtype != TensorDataType::kFloat16 &&
        model_output.dtype != TensorDataType::kInt8) {
        EDGE_LOG_ERROR("TrtPostprocessEngine RunDevice unsupported model output dtype="
                       << DTypeName(model_output.dtype));
        return false;
    }
    if (model_output.shape.size() != 3) {
        EDGE_LOG_ERROR("TrtPostprocessEngine expects model output rank 3, got "
                       << ShapeToString(model_output.shape));
        return false;
    }
    const int batch = static_cast<int>(model_output.shape[0]);
    if (batch <= 0) {
        return true;
    }
    if (static_cast<int>(frame_metas.size()) < batch ||
        static_cast<int>(preprocess_metas.size()) < batch) {
        EDGE_LOG_ERROR("TrtPostprocessEngine got metadata size smaller than output batch");
        return false;
    }
    const std::size_t expected_elements = CountElements(model_output.shape);
    if (expected_elements == 0 || model_output.num_elements < expected_elements) {
        EDGE_LOG_ERROR("TrtPostprocessEngine device model output size mismatch, expected "
                       << expected_elements << ", got " << model_output.num_elements);
        return false;
    }

    if (!IsBuiltForShape(model_output.shape, model_output.dtype) &&
        !BuildForShape(model_output.shape, model_output.dtype)) {
        return false;
    }

    std::vector<float> host_meta;
    void* unused_device_model_output = nullptr;
    float* device_meta = nullptr;
    float* device_detections = nullptr;
    int* device_counts = nullptr;

    auto cleanup = [&]() {
        cudaFree(device_meta);
        cudaFree(device_detections);
        cudaFree(device_counts);
    };

    if (!AllocateDeviceBuffers(model_output.shape,
                               model_output.dtype,
                               batch,
                               preprocess_metas,
                               false,
                               &host_meta,
                               &unused_device_model_output,
                               &device_meta,
                               &device_detections,
                               &device_counts)) {
        cleanup();
        return false;
    }

    const std::size_t meta_bytes = host_meta.size() * sizeof(float);
    PinnedHostBuffer<float> pinned_meta;
    const float* host_meta_data = host_meta.data();
    if (use_pinned_memory_) {
        if (!pinned_meta.Allocate(host_meta.size())) {
            EDGE_LOG_ERROR("cudaHostAlloc failed for TensorRT plugin pinned preprocess meta");
            cleanup();
            return false;
        }
        std::copy(host_meta.begin(), host_meta.end(), pinned_meta.Data());
        host_meta_data = pinned_meta.Data();
    }

    if (!CheckCuda(cudaMemcpyAsync(device_meta,
                                   host_meta_data,
                                   meta_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "cudaMemcpyAsync TRT plugin preprocess meta H2D")) {
        cleanup();
        return false;
    }

    if (!EnqueueAndCopy(batch,
                        frame_metas,
                        stream,
                        model_output.data,
                        device_meta,
                        device_detections,
                        device_counts,
                        detections)) {
        cleanup();
        return false;
    }
    cleanup();

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = ElapsedMs(start, end);
    if (timing != nullptr) {
        timing->decode_ms = elapsed_ms;
        timing->cpu_decode_ms = 0.0;
        timing->gpu_decode_pre_nms_ms = 0.0;
        timing->nms_ms = 0.0;
        timing->gpu_nms_ms = 0.0;
        timing->trt_plugin_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
    EDGE_LOG_INFO("[Postprocess] decode_backend=trt_plugin"
                  << ", nms_backend=trt_plugin"
                  << ", zero_copy_model_output=true"
                  << ", producer=" << model_output.producer
                  << ", model_dtype=" << DTypeName(model_output.dtype)
                  << ", pinned_host=" << (use_pinned_memory_ ? "true" : "false")
                  << ", trt_plugin_ms=" << elapsed_ms
                  << ", detection_count=" << detections.size());
    return true;
}

std::string TrtPostprocessEngine::Name() const {
    return "TrtPostprocessEngine";
}

bool TrtPostprocessEngine::BuildForShape(
    const std::vector<int64_t>& model_shape,
    TensorDataType model_dtype) {
    EDGE_LOG_INFO("Building TensorRT postprocess plugin engine for model_output_shape="
                  << ShapeKey(model_shape)
                  << ", model_dtype=" << DTypeName(model_dtype));

    initLibNvInferPlugins(&Logger(), "");

    TrtUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(Logger())};
    if (!builder) {
        EDGE_LOG_ERROR("Failed to create TensorRT builder for postprocess plugin");
        return false;
    }
    const uint32_t network_flags =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(network_flags)};
    TrtUniquePtr<nvinfer1::IBuilderConfig> builder_config{builder->createBuilderConfig()};
    if (!network || !builder_config) {
        EDGE_LOG_ERROR("Failed to create TensorRT network/config for postprocess plugin");
        return false;
    }
    if (model_dtype == TensorDataType::kFloat16) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (model_dtype == TensorDataType::kInt8) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    auto* model_input = network->addInput(
        kModelInputName,
        ToTrtDataType(model_dtype),
        MakeDims(model_shape));
    auto* meta_input = network->addInput(
        kMetaInputName,
        nvinfer1::DataType::kFLOAT,
        MakeMetaDims(static_cast<int>(model_shape[0])));
    if (model_input == nullptr || meta_input == nullptr) {
        EDGE_LOG_ERROR("Failed to add TensorRT postprocess plugin inputs");
        return false;
    }
    if (model_dtype == TensorDataType::kInt8) {
        const float scale = config_.plugin_int8_input_scale > 0.0F
                                ? config_.plugin_int8_input_scale
                                : 1.0F;
        if (!model_input->setDynamicRange(-127.0F * scale, 127.0F * scale)) {
            EDGE_LOG_ERROR("Failed to set INT8 dynamic range for TensorRT postprocess plugin input");
            return false;
        }
    }

    YoloDecodePlugin plugin(
        config_.score_threshold,
        config_.nms_threshold,
        config_.top_k,
        input_width_,
        input_height_,
        config_.plugin_int8_input_scale);
    nvinfer1::ITensor* plugin_inputs[] = {model_input, meta_input};
    auto* plugin_layer = network->addPluginV2(plugin_inputs, 2, plugin);
    if (plugin_layer == nullptr || plugin_layer->getNbOutputs() != 2) {
        EDGE_LOG_ERROR("Failed to add YoloDecodeNMS_TRT plugin layer");
        return false;
    }
    plugin_layer->getOutput(0)->setName(kDetectionsOutputName);
    plugin_layer->getOutput(1)->setName(kCountsOutputName);
    network->markOutput(*plugin_layer->getOutput(0));
    network->markOutput(*plugin_layer->getOutput(1));

    TrtUniquePtr<nvinfer1::IHostMemory> serialized{builder->buildSerializedNetwork(*network, *builder_config)};
    if (!serialized) {
        EDGE_LOG_ERROR("Failed to build serialized TensorRT postprocess plugin engine");
        return false;
    }

    runtime_.reset(nvinfer1::createInferRuntime(Logger()));
    if (!runtime_) {
        EDGE_LOG_ERROR("Failed to create TensorRT runtime for postprocess plugin");
        return false;
    }
    engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));
    if (!engine_) {
        EDGE_LOG_ERROR("Failed to deserialize TensorRT postprocess plugin engine");
        return false;
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        EDGE_LOG_ERROR("Failed to create TensorRT postprocess plugin execution context");
        return false;
    }

    built_model_shape_ = model_shape;
    built_model_dtype_ = model_dtype;
    EDGE_LOG_INFO("TensorRT postprocess plugin engine built successfully"
                  << ", model_dtype=" << DTypeName(model_dtype));
    return true;
}

bool TrtPostprocessEngine::IsBuiltForShape(
    const std::vector<int64_t>& model_shape,
    TensorDataType model_dtype) const {
    return engine_ != nullptr &&
           context_ != nullptr &&
           built_model_shape_ == model_shape &&
           built_model_dtype_ == model_dtype;
}

bool TrtPostprocessEngine::AllocateDeviceBuffers(
    const std::vector<int64_t>& model_shape,
    TensorDataType model_dtype,
    int batch,
    const std::vector<PreprocessMeta>& preprocess_metas,
    bool allocate_model_output,
    std::vector<float>* host_meta,
    void** device_model_output,
    float** device_meta,
    float** device_detections,
    int** device_counts) const {
    if (host_meta == nullptr ||
        device_model_output == nullptr ||
        device_meta == nullptr ||
        device_detections == nullptr ||
        device_counts == nullptr) {
        return false;
    }

    *device_model_output = nullptr;
    *device_meta = nullptr;
    *device_detections = nullptr;
    *device_counts = nullptr;

    host_meta->assign(static_cast<std::size_t>(batch) * kMetaValues, 0.0F);
    for (int b = 0; b < batch; ++b) {
        const auto& meta = preprocess_metas[static_cast<std::size_t>(b)];
        const std::size_t base = static_cast<std::size_t>(b) * kMetaValues;
        (*host_meta)[base + 0] = static_cast<float>(meta.original_width);
        (*host_meta)[base + 1] = static_cast<float>(meta.original_height);
        (*host_meta)[base + 2] = static_cast<float>(meta.input_width);
        (*host_meta)[base + 3] = static_cast<float>(meta.input_height);
        (*host_meta)[base + 4] = meta.scale;
        (*host_meta)[base + 5] = static_cast<float>(meta.pad_x);
        (*host_meta)[base + 6] = static_cast<float>(meta.pad_y);
    }
    const std::size_t model_element_bytes = ElementBytes(model_dtype);
    if (model_element_bytes == 0) {
        EDGE_LOG_ERROR("Unsupported TensorRT plugin model dtype=" << DTypeName(model_dtype));
        return false;
    }
    const std::size_t model_bytes = CountElements(model_shape) * model_element_bytes;
    const std::size_t meta_bytes = host_meta->size() * sizeof(float);
    const std::size_t detection_bytes =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(config_.top_k) *
        kDetectionValues *
        sizeof(float);
    const std::size_t count_bytes = static_cast<std::size_t>(batch) * sizeof(int);

    if (allocate_model_output &&
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(device_model_output), model_bytes),
                   "cudaMalloc TRT plugin model output")) {
        return false;
    }

    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(device_meta), meta_bytes),
                   "cudaMalloc TRT plugin meta") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(device_detections), detection_bytes),
                   "cudaMalloc TRT plugin detections") ||
        !CheckCuda(cudaMalloc(reinterpret_cast<void**>(device_counts), count_bytes),
                   "cudaMalloc TRT plugin counts")) {
        return false;
    }
    return true;
}

bool TrtPostprocessEngine::EnqueueAndCopy(
    int batch,
    const std::vector<FrameMeta>& frame_metas,
    cudaStream_t stream,
    const void* device_model_output,
    float* device_meta,
    float* device_detections,
    int* device_counts,
    std::vector<Detection>& detections) const {
    if (device_model_output == nullptr ||
        device_meta == nullptr ||
        device_detections == nullptr ||
        device_counts == nullptr) {
        EDGE_LOG_ERROR("TrtPostprocessEngine received null TensorRT binding buffer");
        return false;
    }

    bool enqueue_ok = false;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
    enqueue_ok =
        context_->setTensorAddress(kModelInputName, const_cast<void*>(device_model_output)) &&
        context_->setTensorAddress(kMetaInputName, device_meta) &&
        context_->setTensorAddress(kDetectionsOutputName, device_detections) &&
        context_->setTensorAddress(kCountsOutputName, device_counts) &&
        context_->enqueueV3(stream);
#else
    void* bindings[4]{};
    const int model_index = engine_->getBindingIndex(kModelInputName);
    const int meta_index = engine_->getBindingIndex(kMetaInputName);
    const int detections_index = engine_->getBindingIndex(kDetectionsOutputName);
    const int counts_index = engine_->getBindingIndex(kCountsOutputName);
    if (model_index < 0 || meta_index < 0 || detections_index < 0 || counts_index < 0) {
        EDGE_LOG_ERROR("TrtPostprocessEngine failed to resolve TensorRT binding indices");
        return false;
    }
    bindings[model_index] = const_cast<void*>(device_model_output);
    bindings[meta_index] = device_meta;
    bindings[detections_index] = device_detections;
    bindings[counts_index] = device_counts;
    enqueue_ok = context_->enqueueV2(bindings, stream, nullptr);
#endif
    if (!enqueue_ok) {
        EDGE_LOG_ERROR("TrtPostprocessEngine enqueue failed");
        return false;
    }

    return CopyOutputsToHost(batch, frame_metas, stream, device_detections, device_counts, detections);
}

bool TrtPostprocessEngine::CopyOutputsToHost(
    int batch,
    const std::vector<FrameMeta>& frame_metas,
    cudaStream_t stream,
    const float* device_detections,
    const int* device_counts,
    std::vector<Detection>& detections) const {
    const std::size_t count_elements = static_cast<std::size_t>(batch);
    const std::size_t detection_elements =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(config_.top_k) *
        kDetectionValues;
    std::vector<int> host_counts_vector;
    std::vector<float> host_detections_vector;
    PinnedHostBuffer<int> pinned_counts;
    PinnedHostBuffer<float> pinned_detections;
    int* host_counts = nullptr;
    float* host_detections = nullptr;
    if (use_pinned_memory_) {
        if (!pinned_counts.Allocate(count_elements) ||
            !pinned_detections.Allocate(detection_elements)) {
            EDGE_LOG_ERROR("cudaHostAlloc failed for TensorRT plugin pinned output buffers");
            return false;
        }
        host_counts = pinned_counts.Data();
        host_detections = pinned_detections.Data();
    } else {
        host_counts_vector.assign(count_elements, 0);
        host_detections_vector.assign(detection_elements, 0.0F);
        host_counts = host_counts_vector.data();
        host_detections = host_detections_vector.data();
    }
    const std::size_t count_bytes = count_elements * sizeof(int);
    const std::size_t detection_bytes = detection_elements * sizeof(float);

    if (!CheckCuda(cudaMemcpyAsync(host_counts,
                                   device_counts,
                                   count_bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync TRT plugin counts D2H") ||
        !CheckCuda(cudaMemcpyAsync(host_detections,
                                   device_detections,
                                   detection_bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream),
                   "cudaMemcpyAsync TRT plugin detections D2H") ||
        !CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize TRT plugin outputs")) {
        return false;
    }

    detections.reserve(static_cast<std::size_t>(batch) * static_cast<std::size_t>(config_.top_k));
    for (int b = 0; b < batch; ++b) {
        const int count = std::clamp(host_counts[static_cast<std::size_t>(b)], 0, config_.top_k);
        for (int i = 0; i < count; ++i) {
            const std::size_t base =
                (static_cast<std::size_t>(b) * static_cast<std::size_t>(config_.top_k) +
                 static_cast<std::size_t>(i)) *
                kDetectionValues;
            const int batch_index = std::clamp(
                static_cast<int>(std::round(host_detections[base + 6])),
                0,
                batch - 1);
            const auto& frame_meta = frame_metas[static_cast<std::size_t>(batch_index)];
            Detection det;
            det.stream_id = frame_meta.stream_id;
            det.frame_id = frame_meta.frame_id;
            det.x1 = host_detections[base + 0];
            det.y1 = host_detections[base + 1];
            det.x2 = host_detections[base + 2];
            det.y2 = host_detections[base + 3];
            det.score = host_detections[base + 4];
            det.class_id = static_cast<int>(std::round(host_detections[base + 5]));
            detections.push_back(det);
        }
    }
    return true;
}

}  // namespace edge

#endif  // EDGE_ENABLE_TENSORRT_PLUGIN
