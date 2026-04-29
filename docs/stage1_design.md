# Stage 1 Design

Stage 1 bootstraps a dependency-free C++17 project.

Current pipeline:

```text
SyntheticVideoSource
  -> CPUPreprocessor
  -> MockInferEngine
  -> CPUPostprocessor
  -> log detections
```

The source generates deterministic RGB frames in memory. The preprocessor
resizes by nearest-neighbor sampling and normalizes pixels to float values in
`[0, 1]`. The mock inference engine returns stable synthetic detections so the
pipeline can be exercised before Paddle Inference, TensorRT, CUDA, or OpenCV
are introduced.

No performance values are recorded in this stage. All runtime validation is
pending user local execution.
