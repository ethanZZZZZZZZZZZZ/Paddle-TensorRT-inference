# Paddle Inference + TensorRT Config

This stage enables TensorRT through Paddle Inference's subgraph acceleration
API. It is not a direct TensorRT engine implementation.

Backend selection:

```yaml
infer:
  backend: paddle      # native Paddle baseline
```

```yaml
infer:
  backend: paddle_trt  # Paddle Inference with TensorRT subgraphs
```

`ENABLE_PADDLE=OFF` remains dependency-free and does not compile the Paddle or
Paddle+TensorRT backend implementation.

TensorRT config:

```yaml
trt:
  enable: true
  workspace_size: 1073741824
  max_batch_size: 1
  min_subgraph_size: 3
  precision: fp16
  use_static: false
  use_calib_mode: false
  cache_dir: "../trt_cache"
  int8_calib_images: ""
  calib_batch_size: 1
  calib_num_batches: 1
  calib_cache_dir: ""
  enable_dynamic_shape: false
  dynamic_shape_input_name: ""
  min_input_shape: "1,3,640,640"
  opt_input_shape: "1,3,640,640"
  max_input_shape: "1,3,640,640"
  disable_plugin_fp16: true
```

Parameter meaning:

- `trt.enable`: enables Paddle's TensorRT subgraph conversion.
- `trt.workspace_size`: TensorRT workspace size in bytes.
- `trt.max_batch_size`: maximum TensorRT batch size.
- `trt.min_subgraph_size`: minimum number of Paddle ops required to convert a
  subgraph to TensorRT.
- `trt.precision`: TensorRT precision mode, one of `fp32`, `fp16`, `int8`.
- `trt.use_static`: reuse serialized engine/cache artifacts.
- `trt.use_calib_mode`: INT8 calibration mode.
- `trt.cache_dir`: Paddle optimized cache directory.
- `trt.int8_calib_images`: representative calibration image manifest.
- `trt.calib_batch_size`: intended INT8 calibration batch size.
- `trt.calib_num_batches`: intended number of INT8 calibration batches.
- `trt.calib_cache_dir`: Paddle TensorRT cache directory used during INT8
  calibration.
- `trt.enable_dynamic_shape`: pass TensorRT dynamic shape info through Paddle.
- `trt.dynamic_shape_input_name`: Paddle input tensor name, copied from native
  Paddle logs such as `paddle_input=image`.
- `trt.min_input_shape`, `trt.opt_input_shape`, `trt.max_input_shape`: input
  shapes for TensorRT engine build, written as comma-separated integers.
- `trt.disable_plugin_fp16`: disable FP16 inside TRT plugin layers while keeping
  TensorRT engine precision set to FP16. This is useful for isolating plugin
  compatibility problems.

FP16 example:

```yaml
paddle:
  use_gpu: true

trt:
  enable: true
  precision: fp16
  use_calib_mode: false
  enable_dynamic_shape: true
  dynamic_shape_input_name: "image"
  min_input_shape: "1,3,640,640"
  opt_input_shape: "1,3,640,640"
  max_input_shape: "1,3,640,640"
  disable_plugin_fp16: true
```

Replace `image` with the actual `paddle_input=` name printed by a native Paddle
run. If the model has multiple required inputs, this project still only binds
the first input and needs a model-specific input binding stage before Paddle-TRT
can be validated reliably.

INT8 calibration example:

```yaml
input:
  source_type: image_list
  path: "../data/int8_calib_images.txt"

infer:
  backend: paddle_trt
  precision: int8
  batch_size: 1

trt:
  enable: true
  precision: int8
  use_calib_mode: true
  int8_calib_images: "../data/int8_calib_images.txt"
  calib_batch_size: 1
  calib_num_batches: 512
  calib_cache_dir: "../trt_cache/int8_calib"
```

Prepare the image manifest before running:

```bash
python3 scripts/prepare_int8_calibration_list.py data/calib_images \
  --output data/int8_calib_images.txt \
  --limit 512 \
  --relative-to data
```

Run INT8 calibration locally:

```bash
bash scripts/run_paddle_trt_int8_calib.sh configs/paddle_trt_int8_calib.yaml
```

Run INT8 inference from the generated cache:

```bash
bash scripts/run_paddle_trt.sh configs/paddle_trt_int8.yaml
```

TensorRT builder failure triage:

1. Run `configs/paddle.yaml` first and confirm native Paddle inference works.
2. Use `infer.batch_size: 1`, `input.num_streams: 1`, and `trt.max_batch_size: 1`.
3. Enable dynamic shape using the real Paddle input name.
4. Remove old TensorRT cache files under `trt.cache_dir`.
5. If FP16 still fails, try `trt.precision: fp32` to separate TensorRT
   conversion issues from FP16/plugin issues.
6. If the log still fails in `tensorrt_subgraph_pass`, set
   `trt.min_subgraph_size` above the detected node count, for example `200`
   when the log says `detect a sub-graph with 168 nodes`, to confirm that the
   crash is inside that converted TRT subgraph. This disables that subgraph and
   is a diagnostic fallback, not a performance result.

Baseline comparison:

```yaml
infer:
  backend: paddle

trt:
  enable: false
```

Run commands for local validation:

```bash
bash scripts/build.sh Release OFF OFF ON
bash scripts/run_paddle.sh configs/paddle.yaml
bash scripts/run_paddle_trt.sh configs/paddle_trt_fp16.yaml
```

Run Paddle-TRT with subgraph analysis:

```bash
bash scripts/run_paddle_trt_analysis.sh \
  configs/paddle_trt_fp16.yaml \
  logs/paddle_trt_fp16_analysis.log \
  benchmarks/trt_subgraph_report_fp16.json \
  benchmarks/trt_subgraph_report_fp16.md
```

The logs should show whether Paddle Inference and TensorRT are enabled, plus
TensorRT precision, max batch size, min subgraph size, workspace size,
`use_static`, and `use_calib_mode`.

The generated subgraph report is log-based. It records TRT subgraph count,
subgraph node counts, engine build events, verified fallback/unsupported op
names, fallback evidence lines, TensorRT inserted-copy events caused by
unsupported striding, ignored fallback keyword noise, and error candidate lines
when Paddle logs expose them.

No benchmark values are included in this repository. All build, runtime, and
performance validation is pending user local execution.
