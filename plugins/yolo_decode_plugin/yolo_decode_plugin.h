#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

namespace edge {

class YoloDecodePlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    YoloDecodePlugin(
        float score_threshold,
        float nms_threshold,
        int top_k,
        int input_width,
        int input_height,
        float int8_input_scale = 1.0F);
    YoloDecodePlugin(const void* data, std::size_t length);

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int output_index,
        const nvinfer1::DimsExprs* inputs,
        int nb_inputs,
        nvinfer1::IExprBuilder& expr_builder) noexcept override;
    bool supportsFormatCombination(
        int position,
        const nvinfer1::PluginTensorDesc* in_out,
        int nb_inputs,
        int nb_outputs) noexcept override;
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int nb_inputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int nb_outputs) noexcept override;
    std::size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs,
        int nb_inputs,
        const nvinfer1::PluginTensorDesc* outputs,
        int nb_outputs) const noexcept override;
    int enqueue(
        const nvinfer1::PluginTensorDesc* input_desc,
        const nvinfer1::PluginTensorDesc* output_desc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;
    nvinfer1::DataType getOutputDataType(
        int index,
        const nvinfer1::DataType* input_types,
        int nb_inputs) const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    std::size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* plugin_namespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
    bool isOutputBroadcastAcrossBatch(
        int output_index,
        const bool* input_is_broadcasted,
        int nb_inputs) const noexcept override;
    bool canBroadcastInputAcrossBatch(int input_index) const noexcept override;
    void attachToContext(
        cudnnContext* cudnn_context,
        cublasContext* cublas_context,
        nvinfer1::IGpuAllocator* gpu_allocator) noexcept override;
    void detachFromContext() noexcept override;
#endif

private:
    float score_threshold_ = 0.25F;
    float nms_threshold_ = 0.45F;
    int top_k_ = 100;
    int input_width_ = 640;
    int input_height_ = 640;
    float int8_input_scale_ = 1.0F;
    std::string namespace_;
};

class YoloDecodePluginCreator final : public nvinfer1::IPluginCreator {
public:
    YoloDecodePluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* field_collection) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serial_data,
        std::size_t serial_length) noexcept override;
    void setPluginNamespace(const char* plugin_namespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection field_collection_{};
    std::string namespace_;
};

}  // namespace edge
