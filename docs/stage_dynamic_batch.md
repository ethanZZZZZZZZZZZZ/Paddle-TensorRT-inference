# Dynamic Batch Scheduler

This stage changes the mock pipeline from single-frame processing to
batch aggregation.

Pipeline shape:

```text
VideoSource round-robin read
  -> BatchScheduler
  -> CPUPreprocessor RunBatch
  -> MockInferEngine batch inference
  -> CPUPostprocessor batch decode/NMS
```

Configuration:

```yaml
infer:
  batch_size: 4
  enable_dynamic_batch: true
  dynamic_batch_timeout_ms: 10
  predictor_pool_size: 1
```

Behavior:

- `infer.batch_size` is the max batch size.
- `infer.enable_dynamic_batch: false` keeps single-frame batches.
- When dynamic batching is enabled, a batch is emitted when the pending frame
  count reaches `batch_size`.
- If pending frames wait for at least `dynamic_batch_timeout_ms`, a partial
  batch is emitted.
- When all sources are exhausted, the remaining partial batch is drained.

Per-batch metadata is retained through:

- `FrameBatch.frames[i].meta.stream_id`
- `FrameBatch.frames[i].meta.frame_id`
- `PreprocessMeta` returned by `CPUPreprocessor::RunBatch`

The postprocessor receives frame metadata and preprocess metadata in batch
order, so detections are mapped back to the correct source frame.

CSV additions:

```text
actual_batch_size
batch_wait_ms
batch_latency_ms
```

`batch_latency_ms` is defined as scheduler wait time plus batch processing time
for preprocess, inference, and postprocess. It is a timing contract for local
benchmarking, not a measured result in this repository.

The scheduler itself remains single-threaded. If `infer.predictor_pool_size > 1`,
the emitted batches can be submitted to the predictor pool for concurrent
inference after preprocessing.
