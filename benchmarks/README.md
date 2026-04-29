# Benchmarks

Benchmark CSV outputs are written here by local runs.

Expected default file:

```text
benchmarks/results.csv
```

The CSV is produced by:

```bash
./edge_infer --config ../configs/mock.yaml
```

Current schema:

```text
iter,num_streams,batch_size,actual_batch_size,predictor_pool_size,predictor_worker_id,preprocess_backend,batch_wait_ms,batch_latency_ms,inference_queue_wait_ms,video_decode_ms,preprocess_ms,cpu_preprocess_ms,gpu_preprocess_ms,d2h_copy_ms,inference_ms,decode_backend,nms_backend,decode_ms,cpu_decode_ms,gpu_decode_pre_nms_ms,nms_ms,gpu_nms_ms,trt_plugin_ms,postprocess_ms,e2e_ms,fps
```

Do not fill latency, throughput, GPU utilization, or memory fields manually
until they are produced by local benchmark runs.
