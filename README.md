# Edge Infer

A C++17 edge AI inference optimization project focused on multi-stream video
inference infrastructure, Paddle Inference deployment, Paddle-TRT acceleration,
CUDA preprocessing, TensorRT plugin postprocessing, and benchmark/profiling
tooling.

The project is intentionally infrastructure-oriented. It is not a model
training repo and it is not tied to a specific detection algorithm beyond the
provided YOLO-style postprocess plugin examples.

## Highlights

- Config-driven C++17 inference pipeline.
- Synthetic, video-file, image-list, and RTSP-style OpenCV input paths.
- Multi-stream frame scheduling and dynamic batch aggregation.
- CPU preprocessing baseline: resize, letterbox, BGR to RGB, normalize,
  HWC to CHW, NCHW batch tensor.
- CUDA fused preprocessing kernel with optional pinned-memory support.
- Mock inference backend for dependency-light validation.
- Paddle Inference backend with Paddle native and Paddle-TRT execution modes.
- Paddle-TRT FP32, FP16, and INT8 calibration configuration.
- Paddle-TRT dynamic shape profile, engine cache, subgraph log parsing, and
  fallback report generation.
- Predictor pool for batch-level concurrent inference.
- CUDA decode/pre-NMS kernel and CUDA NMS path.
- TensorRT `IPluginV2DynamicExt` YOLO decode + NMS plugin.
- Full-GPU experimental path:
  GPU preprocess -> Paddle GPU input binding -> Paddle/Paddle-TRT GPU output
  -> TensorRT plugin postprocess.
- Per-stage benchmark CSV, average/percentile meters, and optional NVTX ranges.

## Architecture

```text
VideoSource
  -> BatchScheduler
  -> CPUPreprocessor or GpuPreprocessor
  -> MockInferEngine or PaddleInferEngine
       -> optional PredictorPool
       -> optional Paddle-TRT subgraph acceleration
  -> CPU / CUDA / TensorRT-plugin Postprocessor
  -> Profiler + CSV + logs
```

Full-GPU path:

```text
OpenCV frame
  -> CUDA fused preprocess
  -> Paddle Tensor::ShareExternalData GPU input
  -> Paddle Inference / Paddle-TRT
  -> Paddle GPU output view
  -> TensorRT decode + NMS plugin
  -> CPU detection vector for logs/results
```

## Repository Layout

```text
include/      Public C++ interfaces and data structures
src/          Pipeline, config, inference, preprocess, postprocess, profiling
cuda/         CUDA preprocess, decode, and NMS kernels
plugins/      TensorRT YOLO decode + NMS plugin
configs/      Mock, Paddle, Paddle-TRT, INT8, full-GPU examples
scripts/      Build, run, benchmark, profiling, and log parsing helpers
tests/        Lightweight unit-test targets
benchmarks/   Local CSV outputs and generated analysis reports
docs/         Stage-by-stage design notes
models/       Placeholder for local Paddle model files
data/         Placeholder for local videos/images/calibration data
```

## Build Options

All heavy dependencies are optional and isolated by CMake options.

| Option | Default | Purpose |
|---|---:|---|
| `BUILD_TESTS` | `ON` | Build lightweight unit-test targets |
| `ENABLE_OPENCV` | `ON` | Enable OpenCV video/image input |
| `ENABLE_CUDA` | `OFF` | Enable CUDA preprocess/decode/NMS modules |
| `ENABLE_PADDLE` | `OFF` | Enable Paddle Inference backend |
| `ENABLE_TENSORRT_PLUGIN` | `OFF` | Build and use TensorRT decode + NMS plugin |
| `ENABLE_PADDLE_GPU_INPUT_SHARE` | `ON` | Enable Paddle GPU input zero-copy path |
| `ENABLE_NVTX` | `OFF` | Enable NVTX ranges for Nsight Systems |

## Quick Start: Mock Pipeline

The mock pipeline does not require Paddle, TensorRT, or CUDA.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCV=OFF \
  -DENABLE_PADDLE=OFF \
  -DENABLE_CUDA=OFF \
  -DENABLE_TENSORRT_PLUGIN=OFF

cmake --build build -j

./build/edge_infer --config configs/mock.yaml
```

Expected local behavior:

- Load YAML config.
- Create synthetic mock input.
- Run CPU preprocess.
- Run `MockInferEngine`.
- Decode mock YOLO-style output.
- Print stage logs and optional benchmark CSV.

## Build: Paddle-TRT Full GPU Path

Example for a Linux server with Paddle Inference, CUDA, OpenCV, and TensorRT
installed:

```bash
export PADDLE_INFERENCE_DIR=/workspace/3rd/paddle_inference
export PADDLE_INFERENCE_THIRD_PARTY_DIR="${PADDLE_INFERENCE_DIR}/third_party"

# For Debian/Ubuntu TensorRT packages, these paths are often valid:
export TENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu
export TENSORRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so
export TENSORRT_PLUGIN_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCV=ON \
  -DENABLE_CUDA=ON \
  -DENABLE_PADDLE=ON \
  -DENABLE_TENSORRT_PLUGIN=ON \
  -DENABLE_PADDLE_GPU_INPUT_SHARE=ON \
  -DENABLE_NVTX=OFF \
  -DPADDLE_INFERENCE_DIR="${PADDLE_INFERENCE_DIR}" \
  -DPADDLE_INFERENCE_THIRD_PARTY_DIR="${PADDLE_INFERENCE_THIRD_PARTY_DIR}" \
  -DTENSORRT_INCLUDE_DIR="${TENSORRT_INCLUDE_DIR}" \
  -DTENSORRT_LIBRARY="${TENSORRT_LIBRARY}" \
  -DTENSORRT_PLUGIN_LIBRARY="${TENSORRT_PLUGIN_LIBRARY}"

cmake --build build -j
```

For unstable terminals or memory-limited containers, use the logged helper:

```bash
bash scripts/build_full_gpu_logged.sh Release 2
```

It writes:

```text
logs/cmake_full_gpu_<timestamp>.log
logs/build_full_gpu_<timestamp>.log
logs/build_full_gpu_<timestamp>_summary.log
```

## Runtime Environment

Paddle Inference packages usually ship runtime dependencies such as oneDNN,
MKLML, glog, gflags, protobuf, and OpenMP libraries. Before launching the
binary, prepare `LD_LIBRARY_PATH`:

```bash
export PADDLE_INFERENCE_DIR=/workspace/3rd/paddle_inference
source scripts/paddle_runtime_env.sh

ldd ./build/edge_infer | grep -E "not found|paddle|dnnl|iomp|nvinfer"
```

If the container cannot see the GPU, Paddle will fail with `GPU count is: 0`.
Verify the container was started with GPU access:

```bash
nvidia-smi
ls -l /dev/nvidia*
```

## Run Examples

Mock:

```bash
./build/edge_infer --config configs/mock.yaml
```

Native Paddle:

```bash
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle.yaml
```

Paddle-TRT FP16 with CPU preprocess and TensorRT plugin postprocess:

```bash
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle_trt_fp16_plugin_postprocess.yaml \
  2>&1 | tee logs/paddle_trt_fp16_plugin.log
```

Full-GPU Paddle-TRT FP16 path:

```bash
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle_trt_fp16_full_gpu_plugin.yaml \
  2>&1 | tee logs/paddle_trt_fp16_full_gpu_plugin.log
```

Useful full-GPU log markers:

```text
full_gpu_pipeline=true
skip preprocess D2H copy
PaddleInferEngine shared GPU input
device_input=true
device_output=true
zero_copy_model_output=true
```

## Configuration

The project uses a lightweight YAML-style config reader. Important sections:

```yaml
input:
  source_type: video_file
  path: "rtsp://host:8554/stream1"
  num_streams: 1
  num_frames: 0

model:
  input_width: 640
  input_height: 640
  num_classes: 6

preprocess:
  type: gpu

infer:
  backend: paddle_trt
  precision: fp16
  batch_size: 1
  enable_dynamic_batch: false
  dynamic_batch_timeout_ms: 10
  predictor_pool_size: 1

cuda:
  stream_pool_size: 2
  enable_pinned_memory: true
  enable_full_gpu_pipeline: true

paddle:
  model_file: "../models/inference.pdmodel"
  params_file: "../models/inference.pdiparams"
  use_gpu: true
  gpu_device_id: 0

trt:
  enable: true
  precision: fp16
  min_subgraph_size: 5
  enable_dynamic_shape: true
  dynamic_shape_input_name: "x2paddle_images"
  min_input_shape: "1,3,640,640"
  opt_input_shape: "1,3,640,640"
  max_input_shape: "1,3,640,640"
  use_static: true
  cache_dir: "../trt_cache/fp16_full_gpu_plugin"

postprocess:
  mode: trt_yolo
  decode_backend: trt_plugin
  nms_backend: trt_plugin

benchmark:
  warmup_iters: 20
  benchmark_iters: 200
  output_csv: "../benchmarks/paddle_trt_fp16_full_gpu_plugin.csv"
```

Paths in config files are resolved relative to the config file location.

## Benchmarking

Each benchmark row is recorded after warmup. CSV columns include:

```text
iter,num_streams,batch_size,actual_batch_size,predictor_pool_size,
predictor_worker_id,preprocess_backend,batch_wait_ms,batch_latency_ms,
inference_queue_wait_ms,video_decode_ms,preprocess_ms,cpu_preprocess_ms,
gpu_preprocess_ms,d2h_copy_ms,inference_ms,decode_backend,nms_backend,
decode_ms,cpu_decode_ms,gpu_decode_pre_nms_ms,nms_ms,gpu_nms_ms,
trt_plugin_ms,postprocess_ms,e2e_ms,fps
```

The repository may contain local CSV files under `benchmarks/`. These are
environment-dependent local runs, not portable reference results. Re-run on the
target machine before using the numbers in a report.

Example local comparison currently present in `benchmarks/`:

| Scenario | Preprocess P50 | Inference P50 | Postprocess P50 | Batch latency P50 | E2E P50 |
|---|---:|---:|---:|---:|---:|
| Paddle-TRT FP16 core, CPU preprocess | 2.006 ms | 1.554 ms | 0.210 ms | 3.792 ms | 4.824 ms |
| Full-GPU FP16 + plugin | 0.766 ms | 0.817 ms | 0.204 ms | 1.870 ms | 2.630 ms |

Interpretation of this local run:

- GPU fused preprocessing reduced preprocessing latency.
- Paddle GPU input binding removed the CPU-to-GPU input copy path and reduced
  measured inference-stage latency.
- TensorRT plugin postprocess stayed around 0.2 ms in this scene; the test video
  had few targets, so NMS was not the bottleneck.
- RTSP tests at 15 FPS can make `video_decode_ms` approach 66.7 ms once the
  pipeline catches up with the stream. For compute comparison, focus on
  `preprocess_ms`, `inference_ms`, `postprocess_ms`, and `batch_latency_ms`.

## Paddle-TRT Subgraph Analysis

Paddle Inference can print TensorRT subgraph conversion logs. Capture and parse
them with:

```bash
bash scripts/run_paddle_trt_analysis.sh \
  configs/paddle_trt_fp16.yaml \
  logs/paddle_trt_fp16_analysis.log \
  benchmarks/trt_subgraph_report_fp16.json \
  benchmarks/trt_subgraph_report_fp16.md
```

The parser reports:

- TRT subgraph count.
- Total nodes reported in TRT subgraphs.
- Engine build events.
- Fallback/unsupported op candidates with log evidence when present.
- TensorRT inserted-copy evidence such as unsupported striding copies.
- Error candidate lines.

The script does not invent a coverage percentage when Paddle logs do not expose
the total original graph op count.

## INT8 Calibration

Prepare representative image list:

```bash
python3 scripts/prepare_int8_calibration_list.py data/calib_images \
  --output data/int8_calib_images.txt \
  --limit 512 \
  --relative-to data
```

Run calibration config:

```bash
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle_trt_int8_calib.yaml
```

Then run INT8 inference config that points to the generated cache:

```bash
./build/edge_infer --config configs/paddle_trt_int8.yaml
```

After FP32/FP16/INT8 runs with `output.save_result: true`, compare detection
drift:

```bash
python3 scripts/compare_detection_outputs.py \
  --fp32 outputs/paddle_trt_fp32.txt \
  --fp16 outputs/paddle_trt_fp16.txt \
  --int8 outputs/paddle_trt_int8.txt \
  --summary benchmarks/int8_accuracy_regression.md
```

## Can This Run Any Paddle Model?

Not arbitrary models out of the box.

The current framework is best suited for image inference models with:

- One primary image input tensor.
- NCHW layout.
- FP32 input after preprocessing.
- Configurable static or dynamic batch dimension.
- A first output tensor that can be copied to host or exposed as a GPU output
  view.

What already works:

- Native Paddle Inference can load `model_dir` or `model_file + params_file`.
- `postprocess.mode: raw` can run an unknown Paddle model and log output shape
  without decoding detections.
- YOLO-style full postprocess is available for the expected detection-head
  output shape used by the plugin path.

What requires adapter work:

- Multiple model inputs.
- Non-image inputs, variable-resolution inputs, or non-NCHW layouts.
- Models that require custom preprocessing beyond resize/letterbox/normalize.
- Multiple output tensors or output formats different from the current YOLO
  decode path.
- Segmentation, OCR, classification, keypoint, ReID, or tracking-specific
  postprocess logic.
- Models with fixed batch-dependent reshape ops when dynamic batch is enabled.

Recommended extension point:

```text
ModelAdapter
  -> declares input names, shapes, dtypes, and layout
  -> owns preprocessing metadata rules
  -> maps Paddle outputs to task-specific postprocess
```

Adding this adapter layer would make the framework practical for a wider set of
Paddle models without changing the pipeline core.

## Roadmap

High-priority infrastructure improvements:

- Add a formal `ModelAdapter` registry for arbitrary Paddle model input/output
  contracts.
- Support multiple Paddle inputs and multiple outputs.
- Add direct GPU output adapters for FP16/INT8 Paddle tensors with stronger
  runtime validation.
- Add a full asynchronous multi-stage pipeline:
  decode queue -> preprocess queue -> inference queue -> postprocess queue.
- Add NVDEC or hardware decode path to reduce CPU/OpenCV decode bottlenecks.
- Expand predictor pool experiments to multi-stream and larger batch workloads.
- Add CUDA stream ownership per batch and better overlap between H2D, kernels,
  inference, and D2H.
- Add reusable GPU memory pools for preprocess, inference input, model output,
  candidates, and detection buffers.
- Add direct TensorRT engine demo for the plugin, independent of Paddle-TRT.
- Investigate Paddle-TRT custom plugin/converter integration so postprocess can
  be inserted into the internal TRT subgraph when supported by the Paddle
  version.
- Add structured benchmark summary generation from CSV to Markdown/JSON.
- Add Dockerfile and CI jobs for mock-only build, CUDA build, and documentation
  checks.
- Add accuracy regression fixtures for FP32 vs FP16 vs INT8.

Research/performance experiments:

- Stress-test postprocess with dense target scenes and low score thresholds.
- Compare CPU decode + CPU NMS, CUDA decode + CPU NMS, CUDA decode + CUDA NMS,
  and TensorRT plugin decode + NMS.
- Profile TensorRT engine build/cache behavior with `trt.use_static`.
- Analyze Paddle-TRT fallback and inserted-copy events per model family.
- Try CUDA Graph capture for stable batch-size inference.
- Explore batch-size and timeout policies for 4/8/12 stream RTSP workloads.

## Limitations

- The TensorRT plugin is currently used after Paddle/Paddle-TRT output; it is
  not automatically inserted into Paddle Inference's internal TRT subgraph.
- RTSP end-to-end FPS is bounded by stream frame rate. At 15 FPS, frame arrival
  interval is about 66.7 ms, so `video_decode_ms` can dominate once buffered
  frames are consumed.
- The lightweight config parser supports simple YAML-style sections and scalar
  values, not the full YAML spec.
- Local benchmark numbers depend on GPU, driver, CUDA, TensorRT, Paddle
  Inference build, video source, model export, and config.
- Full compile/runtime/benchmark validation must be performed in the user's
  local Linux environment.

## License

Add a license file before publishing the repository publicly.
