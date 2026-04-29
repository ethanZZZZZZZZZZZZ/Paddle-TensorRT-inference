# edge_infer

Dependency-free bootstrap skeleton for an edge multi-stream AI inference
optimization project.

This stage uses OpenCV only for video-file input. CUDA, Paddle Inference, and
TensorRT are still optional. The mock inference pipeline remains dependency
isolated. Paddle Inference can be enabled explicitly with `ENABLE_PADDLE=ON`.

The current config reader supports a small YAML subset with one-level sections
and scalar `key: value` entries. `yaml-cpp` is intentionally not required in
this stage.

OpenCV is used for `input.source_type: video_file` and
`input.source_type: image_list`. Configure with `-DENABLE_OPENCV=OFF` to keep
the synthetic mock source available without OpenCV.

CUDA GPU preprocessing is optional. Configure with `-DENABLE_CUDA=ON` only
when CUDA Toolkit and OpenCV are available.

## Current Pipeline

```text
SyntheticVideoSource or OpenCV VideoSource
  -> BatchScheduler
  -> CPUPreprocessor or GpuPreprocessor
  -> MockInferEngine or PaddleInferEngine, optionally through PredictorPool
  -> CPU/GPU/TensorRT-plugin Postprocessor
  -> logs
```

## Build

Manual CMake flow from the project root:

```bash
mkdir -p build
cd build
cmake .. -DENABLE_OPENCV=ON
make -j
```

Helper script:

```bash
bash scripts/build.sh Release ON
```

Build synthetic-only mock mode without OpenCV:

```bash
bash scripts/build.sh Release OFF
```

Build with NVTX ranges enabled for Nsight Systems:

```bash
bash scripts/build.sh Release OFF ON
```

Build with Paddle Inference enabled:

```bash
export PADDLE_INFERENCE_DIR=/path/to/paddle_inference
bash scripts/build.sh Release OFF OFF ON
```

Equivalent manual CMake flow:

```bash
cmake -S . -B build \
  -DENABLE_OPENCV=OFF \
  -DENABLE_NVTX=OFF \
  -DENABLE_PADDLE=ON \
  -DPADDLE_INFERENCE_DIR=/path/to/paddle_inference
cmake --build build -j
```

Build with CUDA GPU preprocessing enabled:

```bash
bash scripts/build.sh Release ON OFF OFF ON
```

Equivalent manual CMake flow:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCV=ON \
  -DENABLE_CUDA=ON \
  -DENABLE_PADDLE=OFF \
  -DENABLE_NVTX=OFF
cmake --build build -j
```

## Run

From the build directory:

```bash
./edge_infer --config ../configs/mock.yaml
```

Or from the project root after building:

```bash
bash scripts/run_mock.sh
```

## Paddle Inference Backend

`ENABLE_PADDLE` is disabled by default. With it disabled, the project does not
include Paddle headers, does not link Paddle libraries, and the mock backend
continues to work.

When enabled, CMake searches for:

```text
paddle_inference_api.h
libpaddle_inference
```

Supported path options:

```bash
-DPADDLE_INFERENCE_DIR=/path/to/paddle_inference
-DPADDLE_INFERENCE_INCLUDE_DIR=/path/to/paddle/include
-DPADDLE_INFERENCE_LIB_DIR=/path/to/paddle/lib
-DPADDLE_INFERENCE_LIBRARY=/path/to/libpaddle_inference.so
-DPADDLE_INFERENCE_THIRD_PARTY_DIR=/path/to/paddle_inference/third_party
-DPADDLE_INFERENCE_EXTRA_LIB_DIRS="/extra/lib/dir1;/extra/lib/dir2"
-DPADDLE_INFERENCE_EXTRA_LIBS="/extra/lib/libfoo.so;/extra/lib/libbar.so"
```

If `PADDLE_INFERENCE_DIR` does not match the expected package layout, CMake
falls back to recursively locating `paddle_inference_api.h` and
`libpaddle_inference.so` under that root. You can inspect the exact local paths:

```bash
find "${PADDLE_INFERENCE_DIR}" \( -name 'paddle_inference_api.h' -o -name 'libpaddle_inference.so' \)
```

Then pass explicit paths if needed:

```bash
cmake -S . -B build \
  -DENABLE_PADDLE=ON \
  -DPADDLE_INFERENCE_INCLUDE_DIR=/path/to/include/that/contains/paddle_inference_api.h \
  -DPADDLE_INFERENCE_LIBRARY=/path/to/libpaddle_inference.so
```

When `ENABLE_PADDLE=ON`, CMake links `libpaddle_inference.so`, scans common
Paddle runtime libraries under `paddle/lib`, scans `third_party` or
`third_party/install` for libraries such as `libdnnl.so*`, `libiomp5.so*`, and
adds those directories as linker `rpath-link` entries. This keeps the mock
backend independent while making the Paddle backend work with typical Linux
Paddle Inference release packages.

Paddle config example:

```yaml
input:
  num_streams: 1

infer:
  backend: paddle
  batch_size: 1
  enable_dynamic_batch: false

paddle:
  model_file: "../models/inference.pdmodel"
  params_file: "../models/inference.pdiparams"
  model_dir: ""
  use_gpu: false
  gpu_mem_mb: 512
  gpu_device_id: 0
  enable_ir_optim: true
  enable_memory_optim: true

postprocess:
  mode: raw
```

Use either `model_dir` or `model_file + params_file`, not both.

Start real Paddle model validation with `batch_size: 1`. Many exported Paddle
models contain fixed batch dimensions inside reshape operations; feeding batch
4 into such a model can produce errors like `ReshapeOp ... capacity != in_size`.
Only raise `infer.batch_size`, `input.num_streams`, or enable dynamic batch
after confirming the exported model supports dynamic batch.

For real Paddle models whose output tensor format is not integrated yet, keep
`postprocess.mode: raw`. The pipeline will run inference, print the output
tensor shape, skip detection decode, and write timing CSV. Use
`postprocess.mode: mock_yolo` only for the mock engine output format
`[batch, boxes, 6]`.

Run after local Paddle build:

```bash
bash scripts/run_paddle.sh configs/paddle.yaml
```

If a direct launch exits with code 1, capture the program logs:

```bash
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle.yaml 2>&1 | tee paddle_run.log
```

Linux linker troubleshooting:

```bash
find "${PADDLE_INFERENCE_DIR}" \( -name 'libdnnl.so*' -o -name 'libiomp5.so*' -o -name 'libgomp.so*' \)
```

If these libraries are outside the standard Paddle directory layout, pass their
root or concrete library directories explicitly:

```bash
cmake -S . -B build \
  -DENABLE_PADDLE=ON \
  -DPADDLE_INFERENCE_DIR="${PADDLE_INFERENCE_DIR}" \
  -DPADDLE_INFERENCE_THIRD_PARTY_DIR="${PADDLE_INFERENCE_DIR}/third_party" \
  -DPADDLE_INFERENCE_EXTRA_LIB_DIRS="/path/to/dnnl/lib;/path/to/iomp/lib"
```

At runtime, if the loader reports missing Paddle shared libraries, export
`LD_LIBRARY_PATH` with the same Paddle runtime directories before launching the
binary.

The helper below discovers shared-library directories under
`PADDLE_INFERENCE_DIR` and prepends them to `LD_LIBRARY_PATH`:

```bash
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle.yaml
```

Use `--config`, not `--configs`, when launching `edge_infer`.

## Paddle + TensorRT Subgraph

This project uses Paddle Inference's internal TensorRT subgraph acceleration.
It does not create or manage a raw TensorRT engine directly.

Baseline Paddle config:

```yaml
infer:
  backend: paddle

trt:
  enable: false
```

TensorRT FP16 config:

```yaml
input:
  num_streams: 1

infer:
  backend: paddle_trt
  precision: fp16
  batch_size: 1
  enable_dynamic_batch: false

paddle:
  use_gpu: true

trt:
  enable: true
  workspace_size: 1073741824
  max_batch_size: 1
  min_subgraph_size: 3
  precision: fp16
  use_static: false
  use_calib_mode: false
  cache_dir: "../trt_cache"
  enable_dynamic_shape: false
  dynamic_shape_input_name: ""
  min_input_shape: "1,3,640,640"
  opt_input_shape: "1,3,640,640"
  max_input_shape: "1,3,640,640"
  disable_plugin_fp16: true
```

If TensorRT engine build exits with a message like
`network.cpp::addConcatenation ... inputs[j] != nullptr`, first run native
Paddle and copy the logged `paddle_input=` name into
`trt.dynamic_shape_input_name`, then set `trt.enable_dynamic_shape: true`.
Clear `trt.cache_dir` after changing TRT settings.

TensorRT parameters:

- `trt.enable`: whether Paddle should enable TensorRT subgraph acceleration.
- `trt.workspace_size`: TensorRT workspace bytes reserved during engine build.
- `trt.max_batch_size`: max batch used by the TensorRT subgraph engine.
- `trt.min_subgraph_size`: minimum Paddle subgraph size before TensorRT conversion.
- `trt.precision`: `fp32`, `fp16`, or `int8`.
- `trt.use_static`: whether to reuse serialized TensorRT engine/cache artifacts.
- `trt.use_calib_mode`: calibration mode for INT8.
- `trt.cache_dir`: optimized engine/cache directory used by Paddle Inference.
- `trt.int8_calib_images`: representative calibration image list.
- `trt.calib_batch_size`: intended INT8 calibration batch size.
- `trt.calib_num_batches`: intended number of INT8 calibration batches.
- `trt.calib_cache_dir`: directory used for INT8 calibration cache artifacts.

Build with Paddle support:

```bash
export PADDLE_INFERENCE_DIR=/path/to/paddle_inference
bash scripts/build.sh Release OFF OFF ON
```

Run native Paddle baseline:

```bash
bash scripts/run_paddle.sh configs/paddle.yaml
```

Run Paddle + TensorRT FP16:

```bash
bash scripts/run_paddle_trt.sh configs/paddle_trt_fp16.yaml
```

Run Paddle + TensorRT with subgraph analysis:

```bash
bash scripts/run_paddle_trt_analysis.sh \
  configs/paddle_trt_fp16.yaml \
  logs/paddle_trt_fp16_analysis.log \
  benchmarks/trt_subgraph_report_fp16.json \
  benchmarks/trt_subgraph_report_fp16.md
```

The analysis script captures Paddle stdout/stderr, parses TRT subgraph lines,
engine build events, verified fallback/unsupported op names, fallback evidence
lines, TensorRT inserted-copy events caused by unsupported striding, ignored
fallback keyword noise, and error candidate lines. It does not invent a full
coverage percentage when Paddle logs do not expose total graph op count. See
`docs/stage_paddle_trt_subgraph_analysis.md`.

Only compare baseline and TensorRT numbers after both runs are executed locally
on the same machine, model, input, batch, and benchmark settings.

## Paddle + TensorRT INT8 Calibration

Prepare a representative calibration image list:

```bash
python3 scripts/prepare_int8_calibration_list.py data/calib_images \
  --output data/int8_calib_images.txt \
  --limit 512 \
  --relative-to data
```

Build with OpenCV and Paddle enabled because the INT8 example uses
`input.source_type: image_list`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCV=ON \
  -DENABLE_PADDLE=ON \
  -DENABLE_NVTX=OFF \
  -DPADDLE_INFERENCE_DIR=/path/to/paddle_inference
cmake --build build -j
```

Run INT8 calibration mode:

```bash
bash scripts/run_paddle_trt_int8_calib.sh configs/paddle_trt_int8_calib.yaml
```

Run INT8 inference from the generated cache:

```bash
bash scripts/run_paddle_trt.sh configs/paddle_trt_int8.yaml
```

For this non-calibration INT8 run, `trt.cache_dir` must already exist and point
to the cache generated during calibration.

Relevant config:

```yaml
input:
  source_type: image_list
  path: "../data/int8_calib_images.txt"

infer:
  backend: paddle_trt
  precision: int8
  batch_size: 1

trt:
  enable: true
  precision: int8
  use_calib_mode: true
  int8_calib_images: "../data/int8_calib_images.txt"
  calib_batch_size: 1
  calib_num_batches: 512
  calib_cache_dir: "../trt_cache/int8_calib"
```

With `input.source_type: image_list` and `input.num_frames: 0`, the loader uses
`trt.calib_batch_size * trt.calib_num_batches` as the effective frame count.

After local FP32/FP16/INT8 runs with `output.save_result: true`, compare
detection drift:

```bash
python3 scripts/compare_detection_outputs.py \
  --fp32 outputs/paddle_trt_fp32.txt \
  --fp16 outputs/paddle_trt_fp16.txt \
  --int8 outputs/paddle_trt_int8.txt \
  --summary benchmarks/int8_accuracy_regression.md
```

Clear `trt.calib_cache_dir` after changing model files, preprocessing, input
shape, calibration images, or TensorRT precision settings.

## Run Multi-Stream Video

Prepare a local mp4:

```bash
mkdir -p data
cp /path/to/demo.mp4 data/demo.mp4
```

Build with OpenCV enabled:

```bash
bash scripts/build.sh Release ON
```

Run one video file as four simulated streams:

```bash
bash scripts/run_multi_video.sh
```

Equivalent manual command from the build directory:

```bash
./edge_infer --config ../configs/multi_video.yaml
```

## Dynamic Batch

Dynamic batching is controlled under `infer`:

```yaml
infer:
  batch_size: 4
  enable_dynamic_batch: true
  dynamic_batch_timeout_ms: 10
  predictor_pool_size: 1
```

Behavior:

- Frames are read round-robin from active streams.
- A batch is emitted when `actual_batch_size == infer.batch_size`.
- A partial batch is emitted when pending frames exceed `dynamic_batch_timeout_ms`.
- Remaining frames are flushed when sources end.

To simulate more streams in mock mode, edit `input.num_streams` to `8` or `12`
and keep `input.source_type: synthetic`.

## Predictor Pool

Batch-level concurrent inference is controlled by:

```yaml
infer:
  predictor_pool_size: 2
```

When `predictor_pool_size > 1`, the pipeline creates one independent
`InferEngine` per worker. For Paddle/Paddle-TRT this means one independent
Paddle Predictor per worker. The main thread still performs video decode,
dynamic batch, preprocess, postprocess, result writing, and profiler updates;
worker threads only run inference on preprocessed batch tensors.

Mock smoke config:

```bash
./build/edge_infer --config configs/mock_predictor_pool.yaml
```

Paddle-TRT example config:

```bash
./build/edge_infer --config configs/paddle_trt_fp16_predictor_pool.yaml
```

Useful log fields:

```text
predictor_pool_size
predictor_worker_id
inference_queue_wait_ms
```

For the Paddle output zero-copy plugin path, the worker blocks after publishing
the inference result until the main thread completes postprocess and releases
the result. This keeps Paddle-owned output GPU pointers valid until the plugin
finishes consuming them.

## GPU Preprocessing

GPU preprocessing is selected with:

```yaml
preprocess:
  type: gpu
```

The CUDA path fuses:

```text
BGR cv::Mat
  -> bilinear resize
  -> letterbox padding
  -> BGR to RGB
  -> normalize to float
  -> HWC to CHW
  -> GPU NCHW tensor
```

The current InferEngine interface still consumes host tensors, so the Pipeline
copies the GPU output back to host after `GpuPreprocessor`. This D2H copy is
reported as `d2h_copy_ms` and can be removed in a later GPU inference path.

Benchmark CPU vs GPU preprocessing locally:

```bash
bash scripts/build.sh Release ON OFF OFF ON
bash scripts/benchmark_preprocess.sh
```

Manual commands:

```bash
./build/edge_infer --config configs/mock_cpu_preprocess_benchmark.yaml
./build/edge_infer --config configs/mock_gpu_preprocess.yaml
python3 scripts/compare_preprocess_benchmark.py \
  benchmarks/cpu_preprocess.csv benchmarks/gpu_preprocess.csv \
  --summary benchmarks/preprocess_summary.md
```

## GPU Postprocess

GPU decode/pre-NMS candidate selection is selected with:

```yaml
postprocess:
  mode: mock_yolo
  decode_backend: gpu
  nms_backend: cpu
```

This stage moves score filtering, bbox coordinate inverse transform, and
candidate selection to CUDA. CPU NMS remains in use.

Full GPU postprocess is selected with:

```yaml
postprocess:
  mode: mock_yolo
  decode_backend: gpu
  nms_backend: gpu
```

This path runs decode/pre-NMS and class-aware greedy NMS on CUDA, then copies
only final detections back to host.

Benchmark CPU decode + CPU NMS, GPU decode + CPU NMS, and GPU decode + GPU NMS
locally:

```bash
bash scripts/build.sh Release ON OFF OFF ON
bash scripts/benchmark_decode.sh
```

Manual commands:

```bash
./build/edge_infer --config configs/mock_cpu_decode_benchmark.yaml
./build/edge_infer --config configs/mock_gpu_decode_pre_nms.yaml
./build/edge_infer --config configs/mock_gpu_decode_gpu_nms.yaml
python3 scripts/compare_decode_benchmark.py \
  benchmarks/cpu_decode.csv \
  benchmarks/gpu_decode_pre_nms.csv \
  benchmarks/gpu_decode_gpu_nms.csv \
  --summary benchmarks/postprocess_summary.md
```

## TensorRT Decode Plugin

The TensorRT plugin is optional and disabled by default. It wraps the existing
CUDA decode + score filter + class-aware NMS postprocess path as an
`IPluginV2DynamicExt` shared library target. The same plugin can also be used
inside the main pipeline as a postprocess engine after Paddle/Paddle-TRT
inference.

Build the pipeline and plugin target when TensorRT and CUDA development files
are available:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PADDLE=ON \
  -DENABLE_TENSORRT_PLUGIN=ON \
  -DPADDLE_INFERENCE_DIR=/path/to/paddle_inference \
  -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j
```

Or pass explicit paths:

```bash
cmake -S . -B build \
  -DENABLE_TENSORRT_PLUGIN=ON \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include \
  -DTENSORRT_LIBRARY=/path/to/TensorRT/lib/libnvinfer.so \
  -DTENSORRT_PLUGIN_LIBRARY=/path/to/TensorRT/lib/libnvinfer_plugin.so
```

Enable Paddle-TRT body inference followed by TensorRT plugin postprocess:

```yaml
infer:
  backend: paddle_trt

postprocess:
  mode: trt_yolo
  decode_backend: trt_plugin
  nms_backend: trt_plugin
```

Example config:

```bash
./build/edge_infer --config configs/paddle_trt_fp16_plugin_postprocess.yaml
```

For a dependency-light plugin smoke path without Paddle, use the mock backend:

```bash
./build/edge_infer --config configs/mock_trt_plugin_postprocess.yaml
```

This is a two-engine path: Paddle Inference internally runs the model's
Paddle-TRT subgraph, then the project runs a small TensorRT plugin engine for
decode + NMS. When Paddle exposes the output tensor as a GPU FP32 view through
`Tensor::data<float>(&place, &size)`, the pipeline feeds that pointer directly
to the plugin engine and skips the Paddle output `CopyToCpu()` plus plugin input
H2D copy. Logs show this as `zero_copy_model_output=true`. If Paddle returns a
CPU output or an unsupported dtype, the pipeline falls back to the host path and
logs `zero_copy_model_output=false`. See
`docs/stage_paddle_output_zero_copy.md`, `docs/stage_tensorrt_plugin.md`, and
`plugins/yolo_decode_plugin/README.md` for the plugin tensor formats and current
limits.

## Benchmark CSV

The mock pipeline records per-iteration timing metrics after warmup and writes:

```text
benchmarks/results.csv
```

Configured by:

```yaml
benchmark:
  warmup_iters: 0
  benchmark_iters: 3
  output_csv: "../benchmarks/results.csv"
```

CSV columns:

```text
iter,num_streams,batch_size,actual_batch_size,predictor_pool_size,predictor_worker_id,preprocess_backend,batch_wait_ms,batch_latency_ms,inference_queue_wait_ms,video_decode_ms,preprocess_ms,cpu_preprocess_ms,gpu_preprocess_ms,d2h_copy_ms,inference_ms,decode_backend,nms_backend,decode_ms,cpu_decode_ms,gpu_decode_pre_nms_ms,nms_ms,gpu_nms_ms,trt_plugin_ms,postprocess_ms,e2e_ms,fps
```

These numbers are generated only when the user runs the binary locally. Do not
copy example or placeholder values into reports.

Helper script:

```bash
bash scripts/benchmark.sh configs/mock.yaml
```

## Nsight Systems Profiling

NVTX is optional and disabled by default. When enabled, the pipeline emits
NVTX ranges for:

```text
pipeline_e2e
video_decode
preprocess
inference
postprocess
decode
nms
```

Build with NVTX:

```bash
bash scripts/build.sh Release OFF ON
```

Run Nsight Systems:

```bash
bash scripts/profile_nsys.sh configs/mock.yaml profiles/edge_infer_profile
```

If NVTX headers or libraries are unavailable, build with `-DENABLE_NVTX=OFF`.
The mock pipeline remains runnable without NVTX.

Expected behavior after local execution:

- The app prints the final effective config.
- The pipeline initializes each stage.
- The synthetic source emits mock frames, or the OpenCV source opens one mp4 per simulated stream.
- The batch scheduler prints `batch_id`, actual batch size, trigger reason, and per-item stream/frame mapping.
- The CPU preprocessor creates an NCHW float tensor and logs preprocessing latency.
- `MockInferEngine` produces deterministic fake detections.
- The postprocessor decodes mock model output, applies score filtering and CPU/GPU NMS depending on config, then prints final boxes.

## Current Scope

Implemented:

- C++17 CMake project skeleton.
- Lightweight config loader for simple YAML-style key/value files.
- Config-driven input, model, infer, profile, and output options.
- `VideoSource` interface with synthetic and OpenCV video-file implementations.
- OpenCV image-list source for INT8 calibration image manifests.
- Round-robin multi-stream frame reading for `input.num_streams`.
- Single-threaded `BatchScheduler` for dynamic batch aggregation.
- `CPUPreprocessor` baseline with resize, letterbox, BGR to RGB, normalize, HWC to CHW, and batch output.
- Optional `GpuPreprocessor` with fused CUDA resize, letterbox, normalize, and HWC to CHW.
- `InferEngine` interface and `MockInferEngine`.
- Optional `PaddleInferEngine` behind `ENABLE_PADDLE`.
- Optional batch-level `PredictorPool` with one independent infer engine per worker.
- Optional Paddle Inference + TensorRT subgraph config through `trt.*`.
- Paddle-TRT log capture and subgraph/fallback report generation through `trt_analysis.*`.
- Paddle TensorRT INT8 calibration config, cache directory handling, calibration list script, and detection-output regression script.
- `CPUPostprocessor` with mock output decode, score filtering, letterbox inverse transform, and CPU NMS.
- Optional `GpuDecodePreNMS` with CUDA score filtering, coordinate inverse transform, candidate selection, and optional CUDA NMS.
- Optional TensorRT `IPluginV2DynamicExt` YOLO decode + NMS plugin and post-Paddle/Paddle-TRT plugin postprocess engine behind `ENABLE_TENSORRT_PLUGIN`.
- Timer, average/percentile meters, periodic console profiling, benchmark CSV output, and optional NVTX ranges.
- `Pipeline` orchestration.
- Mock engine, CPU preprocess, coordinate transform, CPU NMS, GPU decode, and GPU NMS unit test targets.

Not implemented yet:

- Threaded multi-stream input queues.
- Full multi-stage async pipeline with overlapping decode/preprocess/inference/postprocess.
- Full direct hand-written TensorRT model engine integration.
- Paddle-TRT custom op / converter path for automatically inserting the plugin into Paddle Inference's internal TensorRT subgraph.
- Fully GPU-resident inference input path without D2H copy.
- Benchmark JSON generation.

All compile and runtime validation is pending user local execution.
