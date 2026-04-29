# TensorRT Plugin Postprocess

This stage adds an optional TensorRT plugin target that wraps the CUDA
postprocess path:

```text
decode -> score filter -> candidate selection -> class-aware NMS
```

Build isolation:

```bash
-DENABLE_TENSORRT_PLUGIN=OFF
```

keeps the main project independent of TensorRT. The plugin target and optional
pipeline postprocess engine are compiled only when explicitly enabled:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PADDLE=ON \
  -DENABLE_TENSORRT_PLUGIN=ON \
  -DPADDLE_INFERENCE_DIR=/path/to/paddle_inference \
  -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j
```

If TensorRT is installed in a non-standard layout, pass explicit paths:

```bash
cmake -S . -B build \
  -DENABLE_TENSORRT_PLUGIN=ON \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include \
  -DTENSORRT_LIBRARY=/path/to/TensorRT/lib/libnvinfer.so \
  -DTENSORRT_PLUGIN_LIBRARY=/path/to/TensorRT/lib/libnvinfer_plugin.so
```

The generated library target is:

```text
yolo_decode_plugin
```

The exact shared-library filename is platform dependent, for example:

```text
build/libyolo_decode_plugin.so
```

## Pipeline Integration

The plugin can be run after Paddle/Paddle-TRT inference as a separate
TensorRT postprocess engine:

```yaml
infer:
  backend: paddle_trt

postprocess:
  mode: trt_yolo
  decode_backend: trt_plugin
  nms_backend: trt_plugin
```

Example:

```bash
./build/edge_infer --config configs/paddle_trt_fp16_plugin_postprocess.yaml
```

Smoke-test the plugin postprocess engine without Paddle:

```bash
./build/edge_infer --config configs/mock_trt_plugin_postprocess.yaml
```

This is not the same as inserting the plugin into Paddle Inference's internal
Paddle-TRT subgraph. The current path is:

```text
Paddle Inference + Paddle-TRT model body
  -> Paddle output tensor GPU pointer when available
  -> TensorRT plugin postprocess engine
  -> detections
```

When Paddle exposes the output tensor with `Tensor::data<float>(&place, &size)`
and `place == kGPU`, the pipeline passes that GPU pointer directly into the
postprocess TensorRT engine. This removes the Paddle output `CopyToCpu()` and
the plugin input H2D copy. If the output tensor is not on GPU, the pipeline
falls back to the old host `TensorBuffer` path and logs
`zero_copy_model_output=false`.

This is still not "plugin inside Paddle-TRT subgraph". Metadata is still copied
H2D, final detections are copied D2H, and the postprocess engine is a separate
TensorRT engine launched after `predictor->Run()`.

## Plugin Type

```text
type:    YoloDecodeNMS_TRT
version: 2
```

The creator accepts these fields:

```text
score_threshold  FP32 scalar
nms_threshold    FP32 scalar
top_k            INT32 scalar
input_width      INT32 scalar
input_height     INT32 scalar
```

## Inputs

`input[0]` is the model output tensor. Two FP32 linear layouts are supported:

```text
shape:  [B, boxes, values_per_box]
layout: x1, y1, x2, y2, score, class_id, ...
```

and:

```text
shape:  [B, channels, boxes]
layout: cx, cy, w, h, class_score_0, class_score_1, ...
example: [B, 10, 8400]
```

The `[B, channels, boxes]` path assumes `xywh + class scores`; confirm this
against the exported model before using it as a production decoder.

`input[1]` is preprocess metadata:

```text
dtype:  FP32
format: linear
shape:  [B, 7]
layout: original_width, original_height, input_width, input_height, scale, pad_x, pad_y
```

## Outputs

`output[0]` contains final detections:

```text
dtype:  FP32
format: linear
shape:  [B, top_k, 7]
layout: x1, y1, x2, y2, score, class_id, batch_index
```

`output[1]` contains detection counts:

```text
dtype:  INT32
format: linear
shape:  [B]
```

Each count is capped to `top_k`.

## Enqueue Path

`YoloDecodePlugin::enqueue` clears workspace/output buffers and calls:

```cpp
LaunchDecodePreNMSFloatMetaKernel(...)
// or LaunchDecodePreNMSBcnFloatMetaKernel(...) for [B, C, N]
LaunchNMSFloatCandidatesKernel(...)
```

These launchers live in `cuda/decode_kernel.cu` and `cuda/nms_kernel.cu`.

## Zero-Copy Boundary

The zero-copy boundary currently covers only:

```text
Paddle output GPU buffer -> TensorRT postprocess plugin input
```

It does not cover:

```text
CPU/GPU preprocess output -> Paddle input
TensorRT plugin detections -> CPU result vector
```

The first gap requires Paddle input `ShareExternalData` or `mutable_data` based
binding. The second gap is usually acceptable because detection output is small,
but it can be kept on GPU if a downstream GPU consumer is added later.

## Current Limits

- This stage integrates the plugin after Paddle Inference, not inside Paddle
  Inference's internal TRT subgraph.
- The Paddle output zero-copy path depends on Paddle returning a GPU FP32 output
  tensor view. Otherwise the host fallback path is used.
- GPU NMS is a simple greedy per-batch CUDA baseline, not an optimized bitmask
  NMS.
- FP16/INT8 plugin execution is not implemented.
- Paddle-TRT custom op / converter integration remains a separate future
  stage.
- Compile and runtime validation must be performed locally on a machine with
  matching CUDA and TensorRT development files.
