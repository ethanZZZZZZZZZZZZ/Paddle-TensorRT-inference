# GPU Preprocessing

This stage adds an optional CUDA preprocessing path behind:

```bash
-DENABLE_CUDA=ON
```

Default builds keep `ENABLE_CUDA=OFF`, so the mock/Paddle CPU preprocessing
pipeline remains dependency-free.

Current GPU path:

```text
cv::Mat BGR host input
  -> cudaMemcpy2DAsync H2D
  -> fused CUDA kernel:
       bilinear resize
       letterbox padding
       BGR to RGB
       normalize to float
       HWC to CHW
  -> GPU NCHW tensor
  -> D2H compatibility copy for current InferEngine
```

`GpuPreprocessor` uses the same `PreprocessMeta` fields as `CPUPreprocessor`:

```text
original_width
original_height
input_width
input_height
scale
pad_x
pad_y
```

YAML selection:

```yaml
preprocess:
  type: gpu
```

Build locally:

```bash
bash scripts/build.sh Release ON OFF OFF ON
```

Run the focused preprocess benchmark:

```bash
bash scripts/benchmark_preprocess.sh
```

The helper runs both configs and writes:

```text
benchmarks/cpu_preprocess.csv
benchmarks/gpu_preprocess.csv
benchmarks/preprocess_summary.md
```

Manual comparison:

```bash
./build/edge_infer --config configs/mock_cpu_preprocess_benchmark.yaml
./build/edge_infer --config configs/mock_gpu_preprocess.yaml
python3 scripts/compare_preprocess_benchmark.py \
  benchmarks/cpu_preprocess.csv benchmarks/gpu_preprocess.csv \
  --summary benchmarks/preprocess_summary.md
```

CSV additions:

```text
preprocess_backend
cpu_preprocess_ms
gpu_preprocess_ms
d2h_copy_ms
```

The current implementation prioritizes correctness and integration. It still
allocates temporary input buffers per frame and copies GPU output back to host
for existing inference backends. Later stages can remove the D2H copy and reuse
device buffers when Paddle/TensorRT input binding supports GPU-resident tensors.

All compile, correctness, and benchmark validation is pending user local
execution.
