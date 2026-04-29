# Postprocess Baseline

Current mock model output format:

```text
shape = [batch, num_boxes, 6]
values = [x1, y1, x2, y2, score, class_id]
```

Coordinates are in the model input letterbox coordinate system. The CPU
postprocessor maps them back to original frame coordinates with `PreprocessMeta`,
then applies score filtering and class-aware CPU NMS scoped by
`stream_id/frame_id`.

This stage still uses `MockInferEngine`; no Paddle, TensorRT, or CUDA decode is
implemented yet. Latency logs are local runtime observations and must be
validated by the user.
