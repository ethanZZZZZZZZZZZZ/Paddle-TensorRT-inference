# YOLO Decode + NMS TensorRT Plugin

This directory contains a TensorRT `IPluginV2DynamicExt` skeleton for wrapping
the CUDA postprocess path:

```text
decode -> score filter -> candidate selection -> class-aware NMS
```

The plugin is isolated behind:

```bash
-DENABLE_TENSORRT_PLUGIN=ON
```

In the pipeline it can run after Paddle/Paddle-TRT inference as a separate
TensorRT postprocess engine. When Paddle exposes the model output as a GPU FP32
view, that pointer is passed directly to the plugin engine; otherwise the
pipeline uses the host fallback path.

## Tensor Formats

Inputs:

```text
input[0] model_output     FP32, linear
input[1] preprocess_meta FP32, linear, shape [B, 7]
```

`model_output` supports two layouts:

```text
[B, boxes, values_per_box]
  values_per_box layout: x1, y1, x2, y2, score, class_id, ...

[B, channels, boxes]
  channels layout: cx, cy, w, h, class_score_0, class_score_1, ...
  Example: [B, 10, 8400]
```

`preprocess_meta` layout:

```text
original_width, original_height, input_width, input_height, scale, pad_x, pad_y
```

Outputs:

```text
output[0] detections       FP32, linear, shape [B, top_k, 7]
output[1] detection_count  INT32, linear, shape [B]
```

Detection layout:

```text
x1, y1, x2, y2, score, class_id, batch_index
```

`detection_count[b]` is capped to `top_k`.

## Plugin Fields

```text
score_threshold float
nms_threshold   float
top_k           int
input_width     int
input_height    int
```

## Supported Types

- `model_output`: FP32 only
- `preprocess_meta`: FP32 only
- `detections`: FP32 only
- `detection_count`: INT32 only
- Tensor format: linear only

## Current Limits

- This is a post-Paddle TensorRT plugin engine, not a plugin inserted into
  Paddle Inference's internal TensorRT subgraph.
- Paddle output zero-copy requires a GPU FP32 Paddle output tensor view.
- NMS uses a simple greedy per-batch CUDA baseline; it is correct-oriented, not
  a highly optimized bitmask NMS.
- FP16/INT8 plugin execution is not implemented yet.
- The `[B, channels, boxes]` decoder assumes `cx, cy, w, h + class scores`.
  Confirm this against the exported model before using it as a production
  decoder.
- Paddle custom op / Paddle-TRT converter work remains a future stage if the
  plugin must be fused into Paddle's internal TRT subgraph.
