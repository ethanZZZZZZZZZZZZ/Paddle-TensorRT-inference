# Predictor Pool

This stage adds batch-level concurrent inference with a pool of independent
predictors.

## Goal

The previous pipeline owned one `InferEngine`, so batches were processed in a
strict sequence:

```text
preprocess batch N -> inference batch N -> postprocess batch N
```

The predictor pool changes only the inference stage:

```text
main thread:
  decode -> dynamic batch -> preprocess -> submit inference task
  collect completed inference result -> postprocess -> profiler/result output

worker thread i:
  owns InferEngine i
  runs InferEngine::Infer()
```

For Paddle/Paddle-TRT, each worker owns an independent Paddle Predictor. This
avoids sharing one predictor across threads.

## Configuration

```yaml
infer:
  predictor_pool_size: 2
```

`predictor_pool_size: 1` keeps the original single-predictor execution path.

## Lifecycle

1. `Pipeline::Init()` creates `PredictorPool` when `predictor_pool_size > 1`.
2. `PredictorPool::Init()` creates and initializes one infer engine per worker.
3. Each worker thread waits for a preprocessed `TensorBuffer`.
4. The main thread stores batch metadata and submits the input tensor.
5. The worker runs inference and publishes a `PredictorPool::Result`.
6. The main thread postprocesses the result, writes logs/results, updates CSV
   metrics, and releases the result.

## Paddle Output Zero-Copy Interaction

If the postprocess backend is `trt_plugin`, workers request `InferOutput` from
`PaddleInferEngine`.

When Paddle exposes a GPU FP32 output view, the result carries a
`DeviceTensorView`. The worker then waits until the main thread releases the
result after postprocess. This is necessary because the device pointer is owned
by the worker's Paddle Predictor and can be overwritten by the worker's next
`predictor->Run()`.

## Profiling Fields

CSV and console logs include:

```text
predictor_pool_size
predictor_worker_id
inference_queue_wait_ms
```

`inference_ms` measures only `InferEngine::Infer()` inside the worker.
`inference_queue_wait_ms` measures time between submit and worker start.
`e2e_ms` still uses per-batch wall-clock latency from preprocessing start to
postprocess completion plus decode/batch wait accounting.

## Current Limits

- This is batch-level inference concurrency, not a full multi-stage async
  runtime.
- Video decode, dynamic batching, preprocessing, postprocessing, result writing,
  and profiler aggregation remain on the main thread.
- CUDA multi-stream overlap is not guaranteed. Existing CUDA stages may still
  synchronize internally.
- Predictor pool size should be chosen based on GPU memory. Each Paddle worker
  owns its own predictor and TRT engine/cache context.
- Compile and runtime validation are pending user local execution.
