# Profiling Infrastructure

This stage adds a reproducible timing contract for the mock pipeline.

Timed stages:

```text
batch_wait
batch_latency
inference_queue_wait
video_decode
preprocess
cpu_preprocess
gpu_preprocess
d2h_copy
inference
decode
cpu_decode
gpu_decode_pre_nms
nms
postprocess
e2e
```

`Profiler` stores per-iteration metrics, prints aggregate statistics, and
writes CSV rows after warmup iterations.

CSV schema:

```text
iter,num_streams,batch_size,actual_batch_size,predictor_pool_size,predictor_worker_id,preprocess_backend,batch_wait_ms,batch_latency_ms,inference_queue_wait_ms,video_decode_ms,preprocess_ms,cpu_preprocess_ms,gpu_preprocess_ms,d2h_copy_ms,inference_ms,decode_backend,nms_backend,decode_ms,cpu_decode_ms,gpu_decode_pre_nms_ms,nms_ms,gpu_nms_ms,trt_plugin_ms,postprocess_ms,e2e_ms,fps
```

Configuration:

```yaml
benchmark:
  warmup_iters: 0
  benchmark_iters: 3
  output_csv: "../benchmarks/results.csv"
```

The current implementation records CPU wall-clock times with `std::chrono`.
NVTX ranges are added in `docs/stage_nvtx_profiling.md`. CUDA event timing is
intentionally left for a later GPU-specific stage.
