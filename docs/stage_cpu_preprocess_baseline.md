# CPU Preprocess Baseline

The CPU preprocessor implements the YOLO input baseline:

```text
cv::Mat BGR
  -> resize with keep-ratio
  -> letterbox padding
  -> BGR to RGB
  -> normalize to float [0, 1]
  -> HWC to CHW
  -> NCHW tensor
```

Output shape:

```text
[batch, 3, input_height, input_width]
```

`PreprocessMeta` records:

```text
original_width
original_height
input_width
input_height
scale
pad_x
pad_y
```

The current pipeline still processes one frame at a time, but
`CPUPreprocessor::RunBatch` supports multiple frames and preserves per-sample
metadata for future dynamic batch work.

No performance conclusions should be made from this stage. The latency log is
only a local runtime observation until benchmark CSV/JSON support is added.
