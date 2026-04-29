# Full GPU Pipeline

This stage connects the existing CUDA and Paddle/TensorRT pieces into one
device-resident path:

```text
OpenCV frame
  -> CUDA preprocess kernel
  -> Paddle GPU input binding
  -> Paddle Inference / Paddle-TRT
  -> Paddle GPU output view
  -> TensorRT plugin decode + NMS
  -> CPU detection vector for logs/results
```

The key switch is:

```yaml
cuda:
  enable_full_gpu_pipeline: true
```

Required runtime configuration:

```yaml
preprocess:
  type: gpu

infer:
  backend: paddle_trt

paddle:
  use_gpu: true

postprocess:
  mode: trt_yolo
  decode_backend: trt_plugin
  nms_backend: trt_plugin
```

Required CMake options:

```bash
export PADDLE_INFERENCE_DIR=/path/to/paddle_inference
export TENSORRT_ROOT=/path/to/TensorRT_or_system_prefix
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCV=ON \
  -DENABLE_CUDA=ON \
  -DENABLE_PADDLE=ON \
  -DENABLE_TENSORRT_PLUGIN=ON \
  -DENABLE_PADDLE_GPU_INPUT_SHARE=ON \
  -DPADDLE_INFERENCE_DIR="${PADDLE_INFERENCE_DIR}" \
  -DPADDLE_INFERENCE_THIRD_PARTY_DIR="${PADDLE_INFERENCE_DIR}/third_party" \
  -DTENSORRT_ROOT="${TENSORRT_ROOT}"
cmake --build build -j
```

Example run:

```bash
mkdir -p logs benchmarks outputs trt_cache
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle_trt_fp16_full_gpu_plugin.yaml 2>&1 | tee logs/full_gpu_fp16.log
```

Expected local log markers:

```text
full_gpu_pipeline=true
skip preprocess D2H copy
PaddleInferEngine shared GPU input
device_input=true
device_output=true
zero_copy_model_output=true
```

## Paddle Input Binding

`PaddleInferEngine` exposes an `Infer(const DeviceTensorView&, InferOutput&)`
path. It binds the CUDA preprocess output with Paddle Inference's
`Tensor::ShareExternalData<float>(..., PlaceType::kGPU)` API. The caller keeps
the `GpuTensorBuffer` alive until `predictor->Run()` completes.

When `PredictorPool` is enabled, the main thread stores the `GpuTensorBuffer`
inside `PreparedBatch` until the worker result has been postprocessed and
released.

## Plugin DType Support

The TensorRT postprocess plugin now accepts:

```text
input[0] model_output: FP32 / FP16 / INT8
input[1] preprocess_meta: FP32
output[0] detections: FP32
output[1] detection_count: INT32
```

FP16 and INT8 input values are converted to FP32 inside the CUDA decode kernel
before score filtering and NMS. `postprocess.plugin_int8_input_scale` is used
for INT8 input dequantization and to set the standalone plugin engine input
dynamic range when a direct INT8 TensorRT producer feeds the plugin.

## Current Boundaries

- This does not insert the postprocess plugin into Paddle Inference's internal
  Paddle-TRT subgraph.
- Final detections are still copied D2H because the current pipeline writes CPU
  logs and result files.
- The full-GPU path depends on local Paddle headers supporting
  `ShareExternalData` GPU binding.
- Compile, runtime, Nsight, and benchmark validation are pending user local
  execution.
