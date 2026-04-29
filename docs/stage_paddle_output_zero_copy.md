# Paddle Output To TensorRT Plugin Zero-Copy

This stage optimizes the boundary between Paddle/Paddle-TRT inference and the
separate TensorRT postprocess plugin engine.

## Implemented Path

```text
CPU or GPU preprocess
  -> Paddle Inference / Paddle-TRT predictor
  -> Paddle output GPU pointer
  -> TensorRT decode + NMS plugin engine
  -> CPU detection vector
```

`PaddleInferEngine` uses Paddle Inference's C++ Tensor API:

```cpp
auto* out_data = output_tensor->data<float>(&place, &size);
```

If `place == paddle_infer::PlaceType::kGPU`, the pointer is wrapped in a local
`DeviceTensorView` and passed directly to `TrtPostprocessEngine::RunDevice`.
The pointer is owned by Paddle and is valid only until the next predictor run.
When `infer.predictor_pool_size > 1`, each worker owns its own Paddle Predictor
and waits for the main thread to release the result after plugin postprocess.
This prevents the worker's next `predictor->Run()` from overwriting the output
buffer while the TensorRT postprocess plugin is still consuming it.

## What This Removes

- Paddle output `CopyToCpu()` for the postprocess-plugin path.
- TensorRT plugin input H2D copy of the model output.

## What Still Copies

- Preprocess metadata `[B, 7]` is copied H2D before plugin enqueue.
- Final detections `[B, top_k, 7]` and counts `[B]` are copied D2H because the
  current pipeline writes CPU-side result logs and CSV metadata.
- Paddle input binding still uses `CopyFromCpu()`.

## Runtime Logs

The pipeline logs whether the optimized path is used:

```text
device_output=true
zero_copy_model_output=true
```

If Paddle returns a CPU tensor view or the output dtype is unsupported, the
pipeline falls back to the host path and logs:

```text
zero_copy_model_output=false
```

## Technical Boundary

This is not a Paddle custom op and it does not insert the plugin into Paddle's
internal TensorRT subgraph. It is a post-inference TensorRT engine that consumes
Paddle's GPU output pointer.

## Validation Status

No compile, runtime, or benchmark result is recorded here. Validation is pending
user local execution on the Linux CUDA/Paddle/TensorRT server.
