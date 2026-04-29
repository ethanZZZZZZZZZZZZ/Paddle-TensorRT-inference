#include "yolo_decode_plugin.h"

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

#include "cuda/decode_kernel.h"
#include "cuda/nms_kernel.h"

namespace edge {
namespace {

constexpr char kPluginType[] = "YoloDecodeNMS_TRT";
constexpr char kPluginVersion[] = "3";
constexpr int kNbInputs = 2;
constexpr int kNbOutputs = 2;
constexpr int kCandidateValues = 7;
constexpr int kMetaValues = 7;
constexpr int kSerializationVersion = 3;

template <typename T> void WriteToBuffer(char *&buffer, const T &value) {
  std::memcpy(buffer, &value, sizeof(T));
  buffer += sizeof(T);
}

template <typename T>
bool ReadFromBuffer(const char *&buffer, const char *end, T *value) {
  if (buffer + sizeof(T) > end) {
    return false;
  }
  std::memcpy(value, buffer, sizeof(T));
  buffer += sizeof(T);
  return true;
}

int SanitizePositive(int value, int fallback) {
  return value > 0 ? value : fallback;
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

char *AlignPtr(char *ptr, std::size_t alignment) {
  const auto address = reinterpret_cast<std::uintptr_t>(ptr);
  const auto aligned = AlignUp(address, alignment);
  return reinterpret_cast<char *>(aligned);
}

std::size_t WorkspaceBytes(int batch, int top_k) {
  const auto batch_size = static_cast<std::size_t>(std::max(1, batch));
  const auto top_k_size = static_cast<std::size_t>(std::max(1, top_k));
  std::size_t offset = 0;
  offset = AlignUp(offset, alignof(float));
  offset += batch_size * top_k_size * kCandidateValues * sizeof(float);
  offset = AlignUp(offset, alignof(int));
  offset += batch_size * sizeof(int);
  return offset + alignof(float) + alignof(int);
}

} // namespace

YoloDecodePlugin::YoloDecodePlugin(float score_threshold, float nms_threshold,
                                   int top_k, int input_width, int input_height,
                                   float int8_input_scale)
    : score_threshold_(score_threshold), nms_threshold_(nms_threshold),
      top_k_(SanitizePositive(top_k, 100)),
      input_width_(SanitizePositive(input_width, 640)),
      input_height_(SanitizePositive(input_height, 640)),
      int8_input_scale_(int8_input_scale > 0.0F ? int8_input_scale : 1.0F) {}

YoloDecodePlugin::YoloDecodePlugin(const void *data, std::size_t length) {
  if (data == nullptr) {
    return;
  }

  const char *cursor = static_cast<const char *>(data);
  const char *end = cursor + length;
  int version = 0;
  int top_k = 0;
  int input_width = 0;
  int input_height = 0;
  float score_threshold = 0.0F;
  float nms_threshold = 0.0F;
  float int8_input_scale = 1.0F;

  if (ReadFromBuffer(cursor, end, &version) &&
      version == kSerializationVersion &&
      ReadFromBuffer(cursor, end, &score_threshold) &&
      ReadFromBuffer(cursor, end, &nms_threshold) &&
      ReadFromBuffer(cursor, end, &top_k) &&
      ReadFromBuffer(cursor, end, &input_width) &&
      ReadFromBuffer(cursor, end, &input_height) &&
      ReadFromBuffer(cursor, end, &int8_input_scale)) {
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    top_k_ = SanitizePositive(top_k, 100);
    input_width_ = SanitizePositive(input_width, 640);
    input_height_ = SanitizePositive(input_height, 640);
    int8_input_scale_ = int8_input_scale > 0.0F ? int8_input_scale : 1.0F;
  }
}

const char *YoloDecodePlugin::getPluginType() const noexcept {
  return kPluginType;
}

const char *YoloDecodePlugin::getPluginVersion() const noexcept {
  return kPluginVersion;
}

int YoloDecodePlugin::getNbOutputs() const noexcept { return kNbOutputs; }

nvinfer1::DimsExprs YoloDecodePlugin::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) noexcept {
  nvinfer1::DimsExprs output{};
  if (inputs == nullptr || nb_inputs != kNbInputs) {
    output.nbDims = 0;
    return output;
  }

  if (output_index == 0) {
    output.nbDims = 3;
    output.d[0] = inputs[0].d[0];
    output.d[1] = expr_builder.constant(top_k_);
    output.d[2] = expr_builder.constant(kCandidateValues);
    return output;
  }

  if (output_index == 1) {
    output.nbDims = 1;
    output.d[0] = inputs[0].d[0];
    return output;
  }

  output.nbDims = 0;
  return output;
}

bool YoloDecodePlugin::supportsFormatCombination(
    int position, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) noexcept {
  if (in_out == nullptr || nb_inputs != kNbInputs || nb_outputs != kNbOutputs ||
      position < 0 || position >= nb_inputs + nb_outputs) {
    return false;
  }

  const auto &desc = in_out[position];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }

  switch (position) {
  case 0:
    return desc.type == nvinfer1::DataType::kFLOAT ||
           desc.type == nvinfer1::DataType::kHALF ||
           desc.type == nvinfer1::DataType::kINT8;
  case 1:
  case 2:
    return desc.type == nvinfer1::DataType::kFLOAT;
  case 3:
    return desc.type == nvinfer1::DataType::kINT32;
  default:
    return false;
  }
}

void YoloDecodePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int nb_outputs) noexcept {
  (void)in;
  (void)nb_inputs;
  (void)out;
  (void)nb_outputs;
}

std::size_t YoloDecodePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nb_inputs,
    const nvinfer1::PluginTensorDesc *outputs, int nb_outputs) const noexcept {
  (void)inputs;
  (void)nb_inputs;
  (void)outputs;
  (void)nb_outputs;
  if (inputs == nullptr || nb_inputs != kNbInputs ||
      inputs[0].dims.nbDims != 3) {
    return 0;
  }
  return WorkspaceBytes(inputs[0].dims.d[0], top_k_);
}

int YoloDecodePlugin::enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                              const nvinfer1::PluginTensorDesc *output_desc,
                              const void *const *inputs, void *const *outputs,
                              void *workspace, cudaStream_t stream) noexcept {
  (void)output_desc;

  if (input_desc == nullptr || inputs == nullptr || outputs == nullptr ||
      inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr ||
      outputs[1] == nullptr || workspace == nullptr) {
    return 1;
  }

  const auto &model_dims = input_desc[0].dims;
  const auto &meta_dims = input_desc[1].dims;
  if (model_dims.nbDims != 3 || meta_dims.nbDims != 2 || model_dims.d[0] <= 0 ||
      model_dims.d[1] <= 0 || model_dims.d[2] <= 0 ||
      meta_dims.d[1] != kMetaValues) {
    return 1;
  }

  const int batch = model_dims.d[0];
  if (meta_dims.d[0] != batch) {
    return 1;
  }
  const bool bcn_layout =
      model_dims.d[1] <= 256 && model_dims.d[2] > model_dims.d[1];
  const int channels = bcn_layout ? model_dims.d[1] : model_dims.d[2];
  const int boxes = bcn_layout ? model_dims.d[2] : model_dims.d[1];
  const int values_per_box = model_dims.d[2];
  if ((!bcn_layout && values_per_box < 6) || (bcn_layout && channels < 5)) {
    return 1;
  }

  const auto *preprocess_metas = static_cast<const float *>(inputs[1]);
  auto *detections = static_cast<float *>(outputs[0]);
  auto *detection_counts = static_cast<int *>(outputs[1]);

  char *workspace_cursor = static_cast<char *>(workspace);
  workspace_cursor = AlignPtr(workspace_cursor, alignof(float));
  auto *pre_nms_candidates = reinterpret_cast<float *>(workspace_cursor);
  workspace_cursor +=
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(top_k_) *
      static_cast<std::size_t>(kCandidateValues) * sizeof(float);
  workspace_cursor = AlignPtr(workspace_cursor, alignof(int));
  auto *pre_nms_counts = reinterpret_cast<int *>(workspace_cursor);

  const std::size_t candidates_bytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(top_k_) *
      static_cast<std::size_t>(kCandidateValues) * sizeof(float);
  const std::size_t counts_bytes =
      static_cast<std::size_t>(batch) * sizeof(int);

  cudaError_t status =
      cudaMemsetAsync(pre_nms_candidates, 0, candidates_bytes, stream);
  if (status != cudaSuccess) {
    return 1;
  }
  status = cudaMemsetAsync(pre_nms_counts, 0, counts_bytes, stream);
  if (status != cudaSuccess) {
    return 1;
  }
  status = cudaMemsetAsync(detections, 0, candidates_bytes, stream);
  if (status != cudaSuccess) {
    return 1;
  }
  status = cudaMemsetAsync(detection_counts, 0, counts_bytes, stream);
  if (status != cudaSuccess) {
    return 1;
  }

  if (input_desc[0].type == nvinfer1::DataType::kFLOAT) {
    const auto *model_output = static_cast<const float *>(inputs[0]);
    status =
        bcn_layout
            ? LaunchDecodePreNMSBcnFloatMetaKernel(
                  model_output, batch, channels, boxes, preprocess_metas,
                  score_threshold_, top_k_, input_width_, input_height_,
                  pre_nms_candidates, pre_nms_counts, stream)
            : LaunchDecodePreNMSFloatMetaKernel(
                  model_output, batch, boxes, values_per_box, preprocess_metas,
                  score_threshold_, top_k_, input_width_, input_height_,
                  pre_nms_candidates, pre_nms_counts, stream);
  } else if (input_desc[0].type == nvinfer1::DataType::kHALF) {
    status =
        bcn_layout
            ? LaunchDecodePreNMSBcnHalfMetaKernel(
                  inputs[0], batch, channels, boxes, preprocess_metas,
                  score_threshold_, top_k_, input_width_, input_height_,
                  pre_nms_candidates, pre_nms_counts, stream)
            : LaunchDecodePreNMSHalfMetaKernel(
                  inputs[0], batch, boxes, values_per_box, preprocess_metas,
                  score_threshold_, top_k_, input_width_, input_height_,
                  pre_nms_candidates, pre_nms_counts, stream);
  } else if (input_desc[0].type == nvinfer1::DataType::kINT8) {
    status =
        bcn_layout
            ? LaunchDecodePreNMSBcnInt8MetaKernel(
                  inputs[0], int8_input_scale_, batch, channels, boxes,
                  preprocess_metas, score_threshold_, top_k_, input_width_,
                  input_height_, pre_nms_candidates, pre_nms_counts, stream)
            : LaunchDecodePreNMSInt8MetaKernel(
                  inputs[0], int8_input_scale_, batch, boxes, values_per_box,
                  preprocess_metas, score_threshold_, top_k_, input_width_,
                  input_height_, pre_nms_candidates, pre_nms_counts, stream);
  } else {
    return 1;
  }
  if (status != cudaSuccess) {
    return 1;
  }
  status = LaunchNMSFloatCandidatesKernel(pre_nms_candidates, pre_nms_counts,
                                          batch, top_k_, nms_threshold_,
                                          detections, detection_counts, stream);
  return status == cudaSuccess ? 0 : 1;
}

nvinfer1::DataType
YoloDecodePlugin::getOutputDataType(int index,
                                    const nvinfer1::DataType *input_types,
                                    int nb_inputs) const noexcept {
  (void)input_types;
  (void)nb_inputs;
  return index == 1 ? nvinfer1::DataType::kINT32 : nvinfer1::DataType::kFLOAT;
}

int YoloDecodePlugin::initialize() noexcept { return 0; }

void YoloDecodePlugin::terminate() noexcept {}

std::size_t YoloDecodePlugin::getSerializationSize() const noexcept {
  return sizeof(int) + sizeof(float) + sizeof(float) + sizeof(int) +
         sizeof(int) + sizeof(int) + sizeof(float);
}

void YoloDecodePlugin::serialize(void *buffer) const noexcept {
  auto *cursor = static_cast<char *>(buffer);
  WriteToBuffer(cursor, kSerializationVersion);
  WriteToBuffer(cursor, score_threshold_);
  WriteToBuffer(cursor, nms_threshold_);
  WriteToBuffer(cursor, top_k_);
  WriteToBuffer(cursor, input_width_);
  WriteToBuffer(cursor, input_height_);
  WriteToBuffer(cursor, int8_input_scale_);
}

nvinfer1::IPluginV2DynamicExt *YoloDecodePlugin::clone() const noexcept {
  auto *plugin = new (std::nothrow)
      YoloDecodePlugin(score_threshold_, nms_threshold_, top_k_, input_width_,
                       input_height_, int8_input_scale_);
  if (plugin != nullptr) {
    plugin->setPluginNamespace(namespace_.c_str());
  }
  return plugin;
}

void YoloDecodePlugin::destroy() noexcept { delete this; }

void YoloDecodePlugin::setPluginNamespace(
    const char *plugin_namespace) noexcept {
  namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

const char *YoloDecodePlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 8
bool YoloDecodePlugin::isOutputBroadcastAcrossBatch(
    int output_index, const bool *input_is_broadcasted,
    int nb_inputs) const noexcept {
  (void)output_index;
  (void)input_is_broadcasted;
  (void)nb_inputs;
  return false;
}

bool YoloDecodePlugin::canBroadcastInputAcrossBatch(
    int input_index) const noexcept {
  (void)input_index;
  return false;
}
#endif

#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
void YoloDecodePlugin::attachToContext(
    cudnnContext *cudnn_context, cublasContext *cublas_context,
    nvinfer1::IGpuAllocator *gpu_allocator) noexcept {
  (void)cudnn_context;
  (void)cublas_context;
  (void)gpu_allocator;
}

void YoloDecodePlugin::detachFromContext() noexcept {}
#endif

YoloDecodePluginCreator::YoloDecodePluginCreator() {
  fields_.emplace_back(nvinfer1::PluginField{
      "score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
  fields_.emplace_back(nvinfer1::PluginField{
      "nms_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
  fields_.emplace_back(nvinfer1::PluginField{
      "top_k", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
  fields_.emplace_back(nvinfer1::PluginField{
      "input_width", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
  fields_.emplace_back(nvinfer1::PluginField{
      "input_height", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
  fields_.emplace_back(nvinfer1::PluginField{
      "int8_input_scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});

  field_collection_.nbFields = static_cast<int>(fields_.size());
  field_collection_.fields = fields_.data();
}

const char *YoloDecodePluginCreator::getPluginName() const noexcept {
  return kPluginType;
}

const char *YoloDecodePluginCreator::getPluginVersion() const noexcept {
  return kPluginVersion;
}

const nvinfer1::PluginFieldCollection *
YoloDecodePluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV2 *YoloDecodePluginCreator::createPlugin(
    const char *name,
    const nvinfer1::PluginFieldCollection *field_collection) noexcept {
  (void)name;

  float score_threshold = 0.25F;
  float nms_threshold = 0.45F;
  int top_k = 100;
  int input_width = 640;
  int input_height = 640;
  float int8_input_scale = 1.0F;

  if (field_collection != nullptr && field_collection->fields != nullptr) {
    for (int i = 0; i < field_collection->nbFields; ++i) {
      const auto &field = field_collection->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length <= 0) {
        continue;
      }

      const std::string field_name(field.name);
      if (field_name == "score_threshold" &&
          field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        score_threshold = *static_cast<const float *>(field.data);
      } else if (field_name == "nms_threshold" &&
                 field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        nms_threshold = *static_cast<const float *>(field.data);
      } else if (field_name == "top_k" &&
                 field.type == nvinfer1::PluginFieldType::kINT32) {
        top_k = *static_cast<const int *>(field.data);
      } else if (field_name == "input_width" &&
                 field.type == nvinfer1::PluginFieldType::kINT32) {
        input_width = *static_cast<const int *>(field.data);
      } else if (field_name == "input_height" &&
                 field.type == nvinfer1::PluginFieldType::kINT32) {
        input_height = *static_cast<const int *>(field.data);
      } else if (field_name == "int8_input_scale" &&
                 field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        int8_input_scale = *static_cast<const float *>(field.data);
      }
    }
  }

  auto *plugin = new (std::nothrow)
      YoloDecodePlugin(score_threshold, nms_threshold, top_k, input_width,
                       input_height, int8_input_scale);
  if (plugin != nullptr) {
    plugin->setPluginNamespace(namespace_.c_str());
  }
  return plugin;
}

nvinfer1::IPluginV2 *
YoloDecodePluginCreator::deserializePlugin(const char *name,
                                           const void *serial_data,
                                           std::size_t serial_length) noexcept {
  (void)name;
  auto *plugin =
      new (std::nothrow) YoloDecodePlugin(serial_data, serial_length);
  if (plugin != nullptr) {
    plugin->setPluginNamespace(namespace_.c_str());
  }
  return plugin;
}

void YoloDecodePluginCreator::setPluginNamespace(
    const char *plugin_namespace) noexcept {
  namespace_ = plugin_namespace != nullptr ? plugin_namespace : "";
}

const char *YoloDecodePluginCreator::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

} // namespace edge

using EdgeYoloDecodePluginCreator = edge::YoloDecodePluginCreator;
REGISTER_TENSORRT_PLUGIN(EdgeYoloDecodePluginCreator);
