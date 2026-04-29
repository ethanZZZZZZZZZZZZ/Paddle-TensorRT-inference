#include "inference/paddle_infer_engine.h"

#ifdef EDGE_ENABLE_PADDLE

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "common/logging.h"

namespace edge {
namespace {

std::vector<int> ToPaddleShape(const std::vector<int64_t> &shape) {
  std::vector<int> out;
  out.reserve(shape.size());
  for (const int64_t dim : shape) {
    out.push_back(static_cast<int>(dim));
  }
  return out;
}

size_t NumElements(const std::vector<int> &shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                         [](size_t acc, int dim) {
                           return dim > 0 ? acc * static_cast<size_t>(dim) : 0;
                         });
}

std::vector<int64_t> ToTensorShape(const std::vector<int> &shape) {
  std::vector<int64_t> out;
  out.reserve(shape.size());
  for (const int dim : shape) {
    out.push_back(static_cast<int64_t>(dim));
  }
  return out;
}

paddle_infer::PrecisionType ToPaddleTrtPrecision(const std::string &precision) {
  if (precision == "fp16") {
    return paddle_infer::PrecisionType::kHalf;
  }
  if (precision == "int8") {
    return paddle_infer::PrecisionType::kInt8;
  }
  return paddle_infer::PrecisionType::kFloat32;
}

std::string Trim(std::string value) {
  const auto begin =
      std::find_if_not(value.begin(), value.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; });
  const auto end =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
      }).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

bool ParseShapeList(const std::string &text, std::vector<int> &shape) {
  shape.clear();
  std::string normalized = text;
  for (char &ch : normalized) {
    if (ch == '[' || ch == ']' || ch == 'x' || ch == 'X' || ch == ';') {
      ch = ',';
    }
  }

  std::stringstream ss(normalized);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = Trim(token);
    if (token.empty()) {
      continue;
    }
    try {
      const int dim = std::stoi(token);
      if (dim <= 0) {
        EDGE_LOG_ERROR("TensorRT dynamic shape dimension must be positive, got "
                       << dim);
        return false;
      }
      shape.push_back(dim);
    } catch (const std::exception &ex) {
      EDGE_LOG_ERROR("Failed to parse TensorRT dynamic shape '"
                     << text << "': " << ex.what());
      return false;
    }
  }

  if (shape.empty()) {
    EDGE_LOG_ERROR("TensorRT dynamic shape must not be empty");
    return false;
  }
  return true;
}

bool EnsureDirectory(const std::string &path, const std::string &label) {
  if (path.empty()) {
    return true;
  }

  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec) {
    EDGE_LOG_ERROR("Failed to create " << label << ": " << path
                                       << ", error=" << ec.message());
    return false;
  }
  return true;
}

} // namespace

bool PaddleInferEngine::Init(const AppConfig &config) {
  EDGE_LOG_INFO("PaddleInferEngine init started");
  EDGE_LOG_INFO("Paddle Inference enabled: true");

  paddle_infer::Config predictor_config;
  if (!ConfigureModel(config.paddle, predictor_config)) {
    return false;
  }

  if (config.paddle.use_gpu) {
    predictor_config.EnableUseGpu(config.paddle.gpu_mem_mb,
                                  config.paddle.gpu_device_id);
    EDGE_LOG_INFO("PaddleInferEngine GPU enabled, gpu_mem_mb="
                  << config.paddle.gpu_mem_mb
                  << ", gpu_device_id=" << config.paddle.gpu_device_id);
  } else {
    predictor_config.DisableGpu();
    EDGE_LOG_INFO("PaddleInferEngine GPU disabled; using CPU inference");
  }

  predictor_config.SwitchIrOptim(config.paddle.enable_ir_optim);
  if (config.paddle.enable_memory_optim) {
    predictor_config.EnableMemoryOptim();
  }

  if (!ConfigureTensorRt(config.trt, predictor_config)) {
    return false;
  }

  EDGE_LOG_INFO("PaddleInferEngine options: ir_optim="
                << (config.paddle.enable_ir_optim ? "true" : "false")
                << ", memory_optim="
                << (config.paddle.enable_memory_optim ? "true" : "false"));

  predictor_ = paddle_infer::CreatePredictor(predictor_config);
  if (!predictor_) {
    EDGE_LOG_ERROR("Failed to create Paddle Inference predictor");
    return false;
  }

  input_names_ = predictor_->GetInputNames();
  output_names_ = predictor_->GetOutputNames();
  if (input_names_.empty()) {
    EDGE_LOG_ERROR("Paddle predictor has no input tensors");
    return false;
  }
  if (output_names_.empty()) {
    EDGE_LOG_ERROR("Paddle predictor has no output tensors");
    return false;
  }

  EDGE_LOG_INFO("PaddleInferEngine initialized");
  for (const auto &name : input_names_) {
    EDGE_LOG_INFO("  paddle_input=" << name);
  }
  for (const auto &name : output_names_) {
    EDGE_LOG_INFO("  paddle_output=" << name);
  }
  return true;
}

bool PaddleInferEngine::Infer(const TensorBuffer &input, TensorBuffer &output) {
  if (!RunPredictor(input)) {
    return false;
  }
  return CopyOutput(output);
}

bool PaddleInferEngine::Infer(const TensorBuffer &input, InferOutput &output) {
  output = InferOutput{};
  if (!RunPredictor(input)) {
    return false;
  }
  return ExportOutput(output);
}

bool PaddleInferEngine::Infer(const DeviceTensorView &input,
                              InferOutput &output) {
  output = InferOutput{};
  if (!RunPredictor(input)) {
    return false;
  }
  return ExportOutput(output);
}

bool PaddleInferEngine::RunPredictor(const TensorBuffer &input) {
  if (!predictor_) {
    EDGE_LOG_ERROR("PaddleInferEngine must be initialized before Infer()");
    return false;
  }

  try {
    if (!CopyInput(input)) {
      return false;
    }
    if (!predictor_->Run()) {
      EDGE_LOG_ERROR(
          "Paddle predictor Run() failed. input_shape="
          << ShapeToString(input.shape)
          << ". If the model was exported with fixed batch=1, "
          << "set infer.batch_size: 1 and infer.enable_dynamic_batch: false.");
      return false;
    }
    return true;
  } catch (const std::exception &ex) {
    EDGE_LOG_ERROR(
        "PaddleInferEngine exception during inference. input_shape="
        << ShapeToString(input.shape) << ", message=" << ex.what()
        << ". If the traceback mentions ReshapeOp capacity mismatch, "
        << "the model likely has a fixed batch dimension; use batch_size=1 "
        << "or re-export the model with dynamic batch support.");
    return false;
  }
}

bool PaddleInferEngine::RunPredictor(const DeviceTensorView &input) {
  if (!predictor_) {
    EDGE_LOG_ERROR("PaddleInferEngine must be initialized before Infer()");
    return false;
  }

  try {
    if (!BindDeviceInput(input)) {
      return false;
    }
    if (!predictor_->Run()) {
      EDGE_LOG_ERROR(
          "Paddle predictor Run() failed for GPU input. input_shape="
          << ShapeToString(input.shape)
          << ". If the model was exported with fixed batch=1, "
          << "set infer.batch_size: 1 and infer.enable_dynamic_batch: false.");
      return false;
    }
    return true;
  } catch (const std::exception &ex) {
    EDGE_LOG_ERROR(
        "PaddleInferEngine exception during GPU-input inference. input_shape="
        << ShapeToString(input.shape) << ", message=" << ex.what()
        << ". If this fails inside Paddle GPU input binding, disable "
        << "cuda.enable_full_gpu_pipeline and fall back to CopyFromCpu().");
    return false;
  }
}

std::string PaddleInferEngine::Name() const { return "PaddleInferEngine"; }

bool PaddleInferEngine::ConfigureModel(
    const PaddleConfig &paddle_config,
    paddle_infer::Config &predictor_config) const {
  const bool has_model_dir = !paddle_config.model_dir.empty();
  const bool has_model_pair =
      !paddle_config.model_file.empty() && !paddle_config.params_file.empty();

  if (has_model_dir && has_model_pair) {
    EDGE_LOG_ERROR("Set either paddle.model_dir or paddle.model_file + "
                   "paddle.params_file, not both");
    return false;
  }
  if (has_model_dir) {
    predictor_config.SetModel(paddle_config.model_dir);
    EDGE_LOG_INFO("PaddleInferEngine model_dir=" << paddle_config.model_dir);
    return true;
  }
  if (has_model_pair) {
    predictor_config.SetModel(paddle_config.model_file,
                              paddle_config.params_file);
    EDGE_LOG_INFO("PaddleInferEngine model_file=" << paddle_config.model_file
                                                  << ", params_file="
                                                  << paddle_config.params_file);
    return true;
  }

  EDGE_LOG_ERROR("PaddleInferEngine requires paddle.model_dir or "
                 "paddle.model_file + paddle.params_file");
  return false;
}

bool PaddleInferEngine::ConfigureTensorRt(
    const TrtConfig &trt_config, paddle_infer::Config &predictor_config) const {
  EDGE_LOG_INFO("TensorRT enabled: " << (trt_config.enable ? "true" : "false"));
  EDGE_LOG_INFO("TensorRT precision="
                << trt_config.precision
                << ", max_batch_size=" << trt_config.max_batch_size
                << ", min_subgraph_size=" << trt_config.min_subgraph_size
                << ", workspace_size=" << trt_config.workspace_size
                << ", use_static=" << (trt_config.use_static ? "true" : "false")
                << ", use_calib_mode="
                << (trt_config.use_calib_mode ? "true" : "false")
                << ", cache_dir=" << trt_config.cache_dir);
  EDGE_LOG_INFO("TensorRT INT8 calibration: int8_calib_images="
                << trt_config.int8_calib_images
                << ", calib_batch_size=" << trt_config.calib_batch_size
                << ", calib_num_batches=" << trt_config.calib_num_batches
                << ", calib_cache_dir=" << trt_config.calib_cache_dir);
  EDGE_LOG_INFO("TensorRT dynamic_shape="
                << (trt_config.enable_dynamic_shape ? "true" : "false")
                << ", input_name=" << trt_config.dynamic_shape_input_name
                << ", min_input_shape=" << trt_config.min_input_shape
                << ", opt_input_shape=" << trt_config.opt_input_shape
                << ", max_input_shape=" << trt_config.max_input_shape
                << ", disable_plugin_fp16="
                << (trt_config.disable_plugin_fp16 ? "true" : "false"));
  if (!trt_config.enable) {
    EDGE_LOG_INFO(
        "TensorRT disabled; using native Paddle Inference execution path");
    return true;
  }

  EDGE_LOG_INFO("[TRT_ANALYSIS] begin Paddle-TRT configuration snapshot");
  EDGE_LOG_INFO("[TRT_ANALYSIS] precision="
                << trt_config.precision
                << ", max_batch_size=" << trt_config.max_batch_size
                << ", min_subgraph_size=" << trt_config.min_subgraph_size
                << ", workspace_size=" << trt_config.workspace_size
                << ", use_static=" << (trt_config.use_static ? "true" : "false")
                << ", use_calib_mode="
                << (trt_config.use_calib_mode ? "true" : "false"));
  EDGE_LOG_INFO("[TRT_ANALYSIS] dynamic_shape="
                << (trt_config.enable_dynamic_shape ? "true" : "false")
                << ", input_name=" << trt_config.dynamic_shape_input_name
                << ", min_input_shape=" << trt_config.min_input_shape
                << ", opt_input_shape=" << trt_config.opt_input_shape
                << ", max_input_shape=" << trt_config.max_input_shape);
  EDGE_LOG_INFO("[TRT_ANALYSIS] cache_dir=" << trt_config.cache_dir
                                            << ", calib_cache_dir="
                                            << trt_config.calib_cache_dir);
  EDGE_LOG_INFO("[TRT_ANALYSIS] capture stdout/stderr and parse with "
                "scripts/parse_paddle_trt_log.py");

  if (trt_config.precision == "int8" && trt_config.use_calib_mode) {
    if (trt_config.calib_cache_dir.empty()) {
      EDGE_LOG_ERROR("TensorRT INT8 calibration requires trt.calib_cache_dir");
      return false;
    }
    if (!EnsureDirectory(trt_config.calib_cache_dir,
                         "TensorRT INT8 calibration cache dir")) {
      return false;
    }
    EDGE_LOG_INFO(
        "TensorRT INT8 calibration mode enabled. Paddle will build calibration "
        "data "
        "while running representative batches. Calibration image list="
        << trt_config.int8_calib_images);
  } else if (!trt_config.cache_dir.empty()) {
    if (!EnsureDirectory(trt_config.cache_dir,
                         "TensorRT optimized cache dir")) {
      return false;
    }
  }

  const auto precision = ToPaddleTrtPrecision(trt_config.precision);
  predictor_config.EnableTensorRtEngine(
      trt_config.workspace_size, trt_config.max_batch_size,
      trt_config.min_subgraph_size, precision, trt_config.use_static,
      trt_config.use_calib_mode);

  if (trt_config.enable_dynamic_shape) {
    std::vector<int> min_shape;
    std::vector<int> opt_shape;
    std::vector<int> max_shape;
    if (!ParseShapeList(trt_config.min_input_shape, min_shape) ||
        !ParseShapeList(trt_config.opt_input_shape, opt_shape) ||
        !ParseShapeList(trt_config.max_input_shape, max_shape)) {
      return false;
    }
    if (min_shape.size() != opt_shape.size() ||
        opt_shape.size() != max_shape.size()) {
      EDGE_LOG_ERROR("TensorRT dynamic shape rank mismatch: min="
                     << min_shape.size() << ", opt=" << opt_shape.size()
                     << ", max=" << max_shape.size());
      return false;
    }
    std::map<std::string, std::vector<int>> min_input_shape{
        {trt_config.dynamic_shape_input_name, min_shape}};
    std::map<std::string, std::vector<int>> opt_input_shape{
        {trt_config.dynamic_shape_input_name, opt_shape}};
    std::map<std::string, std::vector<int>> max_input_shape{
        {trt_config.dynamic_shape_input_name, max_shape}};
    predictor_config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                            opt_input_shape,
                                            trt_config.disable_plugin_fp16);
    EDGE_LOG_INFO("TensorRT dynamic shape info configured for input="
                  << trt_config.dynamic_shape_input_name);
  }

  const std::string effective_cache_dir =
      (trt_config.precision == "int8" && trt_config.use_calib_mode &&
       !trt_config.calib_cache_dir.empty())
          ? trt_config.calib_cache_dir
          : trt_config.cache_dir;
  if (!effective_cache_dir.empty()) {
    predictor_config.SetOptimCacheDir(effective_cache_dir);
    EDGE_LOG_INFO(
        "TensorRT optimized/calibration cache dir=" << effective_cache_dir);
  }

  EDGE_LOG_INFO(
      "TensorRT subgraph acceleration configured through Paddle Inference");
  return true;
}

bool PaddleInferEngine::CopyInput(const TensorBuffer &input) {
  if (input_names_.empty()) {
    EDGE_LOG_ERROR("PaddleInferEngine has no cached input tensor names");
    return false;
  }
  if (input.shape.empty() || input.NumElements() == 0 ||
      input.Data() == nullptr) {
    EDGE_LOG_ERROR("PaddleInferEngine received empty input tensor");
    return false;
  }

  const auto input_shape = ToPaddleShape(input.shape);
  const auto *input_data = input.Data();
  auto input_handle = predictor_->GetInputHandle(input_names_.front());
  input_handle->Reshape(input_shape);
  input_handle->CopyFromCpu(input_data);

  EDGE_LOG_INFO("PaddleInferEngine copied input name="
                << input_names_.front()
                << ", shape=" << ShapeToString(input.shape));
  if (input_names_.size() > 1) {
    EDGE_LOG_WARN("PaddleInferEngine currently binds only the first input "
                  "tensor; extra inputs are ignored");
  }
  return true;
}

bool PaddleInferEngine::BindDeviceInput(const DeviceTensorView &input) {
#ifdef EDGE_ENABLE_PADDLE_GPU_INPUT_SHARE
  if (input_names_.empty()) {
    EDGE_LOG_ERROR("PaddleInferEngine has no cached input tensor names");
    return false;
  }
  if (!input.IsGpuFloat()) {
    EDGE_LOG_ERROR("PaddleInferEngine GPU input path requires a GPU FP32 "
                   "tensor view, got shape="
                   << ShapeToString(input.shape)
                   << ", num_elements=" << input.num_elements);
    return false;
  }

  const auto input_shape = ToPaddleShape(input.shape);
  auto input_handle = predictor_->GetInputHandle(input_names_.front());
  auto *input_data =
      const_cast<float *>(static_cast<const float *>(input.data));
  input_handle->ShareExternalData<float>(input_data, input_shape,
                                         paddle_infer::PlaceType::kGPU);

  EDGE_LOG_INFO("PaddleInferEngine shared GPU input name="
                << input_names_.front() << ", shape="
                << ShapeToString(input.shape) << ", producer=" << input.producer
                << ". The caller must keep the GPU buffer alive until "
                   "predictor->Run() completes.");
  if (input_names_.size() > 1) {
    EDGE_LOG_WARN("PaddleInferEngine currently binds only the first input "
                  "tensor; extra inputs are ignored");
  }
  return true;
#else
  (void)input;
  EDGE_LOG_ERROR(
      "PaddleInferEngine GPU input ShareExternalData path was disabled at "
      "build time. "
      << "Reconfigure with -DENABLE_PADDLE_GPU_INPUT_SHARE=ON or disable "
      << "cuda.enable_full_gpu_pipeline.");
  return false;
#endif
}

bool PaddleInferEngine::CopyOutput(TensorBuffer &output) {
  if (output_names_.empty()) {
    EDGE_LOG_ERROR("PaddleInferEngine has no cached output tensor names");
    return false;
  }

  auto output_handle = predictor_->GetOutputHandle(output_names_.front());
  const std::vector<int> output_shape = output_handle->shape();
  const size_t output_elements = NumElements(output_shape);
  output.shape = ToTensorShape(output_shape);
  output.ClearExternalHostData();
  output.host_data.resize(output_elements);
  if (output_elements > 0) {
    output_handle->CopyToCpu(output.host_data.data());
  }

  EDGE_LOG_INFO("PaddleInferEngine copied output name="
                << output_names_.front()
                << ", shape=" << ShapeToString(output.shape));
  if (output_names_.size() > 1) {
    EDGE_LOG_WARN("PaddleInferEngine currently reads only the first output "
                  "tensor; extra outputs are ignored");
  }
  return true;
}

bool PaddleInferEngine::ExportOutput(InferOutput &output) {
  if (output_names_.empty()) {
    EDGE_LOG_ERROR("PaddleInferEngine has no cached output tensor names");
    return false;
  }

  output.has_host_tensor = false;
  output.has_device_tensor = false;
  output.host_tensor.host_data.clear();
  output.host_tensor.ClearExternalHostData();
  output.host_tensor.shape.clear();
  output.device_tensor = DeviceTensorView{};

  auto output_handle = predictor_->GetOutputHandle(output_names_.front());
  const std::vector<int> output_shape = output_handle->shape();
  const size_t output_elements = NumElements(output_shape);
  const auto tensor_shape = ToTensorShape(output_shape);

  paddle_infer::PlaceType place = paddle_infer::PlaceType::kUNK;
  int raw_size = 0;
  const void *data = nullptr;
  TensorDataType dtype = TensorDataType::kUnknown;
  size_t element_bytes = 0;
  const auto output_type = output_handle->type();
  if (output_type == paddle_infer::DataType::FLOAT32) {
    dtype = TensorDataType::kFloat32;
    element_bytes = sizeof(float);
  } else if (output_type == paddle_infer::DataType::FLOAT16) {
    dtype = TensorDataType::kFloat16;
    element_bytes = sizeof(uint16_t);
  } else if (output_type == paddle_infer::DataType::INT8) {
    dtype = TensorDataType::kInt8;
    element_bytes = sizeof(int8_t);
  } else {
    EDGE_LOG_ERROR(
        "PaddleInferEngine output dtype is not supported by the current "
        "pipeline. "
        << "Supported device-output dtypes: FLOAT32, FLOAT16, INT8.");
    return false;
  }

  // Paddle Inference binaries only export Tensor::data<T>() for selected T.
  // The full-GPU path only needs the raw GPU address here; dtype and element
  // size are tracked separately by DeviceTensorView for TensorRT plugin input.
  data = output_handle->data<float>(&place, &raw_size);

  if (data != nullptr && place == paddle_infer::PlaceType::kGPU &&
      output_elements > 0) {
    output.device_tensor.data = data;
    output.device_tensor.shape = tensor_shape;
    output.device_tensor.num_elements = output_elements;
    output.device_tensor.element_bytes = element_bytes;
    output.device_tensor.place = TensorMemoryPlace::kGPU;
    output.device_tensor.dtype = dtype;
    output.device_tensor.producer = output_names_.front();
    output.has_device_tensor = true;
    EDGE_LOG_INFO("PaddleInferEngine exported GPU output view name="
                  << output_names_.front()
                  << ", shape=" << ShapeToString(tensor_shape)
                  << ", element_bytes=" << element_bytes
                  << ", raw_size_bytes=" << raw_size
                  << ". The pointer is valid until the next predictor run.");
  } else if (output_type == paddle_infer::DataType::FLOAT32) {
    output.host_tensor.shape = tensor_shape;
    output.host_tensor.ClearExternalHostData();
    output.host_tensor.host_data.resize(output_elements);
    if (output_elements > 0) {
      output_handle->CopyToCpu(output.host_tensor.host_data.data());
    }
    output.has_host_tensor = true;
    EDGE_LOG_WARN("PaddleInferEngine output is not a GPU tensor view; copied "
                  "output to host. "
                  << "place_is_gpu="
                  << (place == paddle_infer::PlaceType::kGPU ? "true" : "false")
                  << ", shape=" << ShapeToString(tensor_shape));
  } else {
    EDGE_LOG_ERROR(
        "PaddleInferEngine received non-FP32 output outside GPU memory. "
        << "Host fallback is only implemented for FLOAT32 output tensors.");
    return false;
  }

  if (output_names_.size() > 1) {
    EDGE_LOG_WARN("PaddleInferEngine currently reads only the first output "
                  "tensor; extra outputs are ignored");
  }
  return output.has_device_tensor || output.has_host_tensor;
}

} // namespace edge

#endif // EDGE_ENABLE_PADDLE
