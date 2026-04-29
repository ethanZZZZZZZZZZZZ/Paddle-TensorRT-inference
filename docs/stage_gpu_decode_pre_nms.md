# GPU Decode + NMS

This stage adds optional CUDA postprocess paths. It is controlled by:

```yaml
postprocess:
  mode: mock_yolo
  decode_backend: gpu
  nms_backend: cpu  # or gpu
```

Build isolation:

```bash
-DENABLE_CUDA=OFF
```

keeps the project on the CPU postprocess path. GPU postprocess is compiled only
with:

```bash
-DENABLE_CUDA=ON
```

## Backend Modes

CPU baseline:

```text
CPU decode -> CPU NMS
```

GPU decode baseline:

```text
model output [batch, boxes, 6]
  -> H2D copy
  -> CUDA decode/pre-NMS:
       score filter
       bbox coordinate inverse letterbox transform
       candidate selection up to postprocess.top_k per batch item
  -> D2H candidate copy
  -> CPU NMS
```

Full GPU postprocess baseline:

```text
model output [batch, boxes, 6]
  -> H2D copy
  -> CUDA decode/pre-NMS
  -> CUDA class-aware greedy NMS
  -> D2H final detection copy
```

The candidate/detection structure contains:

```text
x1, y1, x2, y2, score, class_id, batch_index
```

`batch_index` is used on host to recover `stream_id` and `frame_id`.

## Benchmark Locally

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

CSV additions:

```text
decode_backend
nms_backend
cpu_decode_ms
gpu_decode_pre_nms_ms
nms_ms
gpu_nms_ms
```

Generated files after local benchmark execution:

```text
benchmarks/cpu_decode.csv
benchmarks/gpu_decode_pre_nms.csv
benchmarks/gpu_decode_gpu_nms.csv
benchmarks/postprocess_summary.md
```

## Limitations

- GPU NMS is a simple greedy per-batch CUDA baseline, not a highly optimized
  bitmask NMS implementation.
- The current pipeline path still performs H2D model-output copy because the
  generic `InferEngine` interface returns host tensors.
- The pipeline GPU postprocess path targets the mock `[batch, boxes, 6]`
  output layout. The TensorRT plugin also includes a `[B, channels, boxes]`
  decoder for layouts such as `[B, 10, 8400]`, but that path still needs local
  validation against the exported model's real output semantics.

All compile, correctness, and benchmark validation is pending user local
execution.
