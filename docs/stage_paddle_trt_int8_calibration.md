# Paddle TensorRT INT8 Calibration

This stage adds configuration and tooling for Paddle Inference's TensorRT INT8
calibration path. It still uses Paddle Inference internal TensorRT subgraph
acceleration, not a hand-written TensorRT engine.

## Config Keys

Required for INT8 calibration:

```yaml
trt:
  enable: true
  precision: int8
  use_calib_mode: true
  int8_calib_images: "../data/int8_calib_images.txt"
  calib_batch_size: 1
  calib_num_batches: 512
  calib_cache_dir: "../trt_cache/int8_calib"
```

Meaning:

- `trt.int8_calib_images`: image manifest used for representative
  calibration data.
- `trt.calib_batch_size`: intended calibration batch size. Keep it aligned
  with `infer.batch_size` unless the exported model is confirmed to support
  dynamic batch.
- `trt.calib_num_batches`: intended number of representative calibration
  batches.
- `trt.calib_cache_dir`: directory passed to Paddle's TensorRT optimized cache
  path during INT8 calibration.

`PaddleInferEngine` calls:

```cpp
EnableTensorRtEngine(..., PrecisionType::kInt8, use_static, use_calib_mode)
SetOptimCacheDir(trt.calib_cache_dir)
```

when `trt.precision: int8` and `trt.use_calib_mode: true`.

When `input.source_type: image_list` and `input.num_frames: 0`, the config
loader derives `input.num_frames = trt.calib_batch_size *
trt.calib_num_batches` so the calibration run consumes the intended number of
samples.

## Prepare Calibration Images

Create a representative image list:

```bash
python3 scripts/prepare_int8_calibration_list.py data/calib_images \
  --output data/int8_calib_images.txt \
  --limit 512 \
  --relative-to data
```

The script accepts either a directory or an existing text list. It only scans
paths; it does not decode images.

## Run Calibration

Build with OpenCV and Paddle enabled. OpenCV is required for
`input.source_type: image_list`.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCV=ON \
  -DENABLE_PADDLE=ON \
  -DENABLE_NVTX=OFF \
  -DPADDLE_INFERENCE_DIR=/path/to/paddle_inference
cmake --build build -j
```

Run:

```bash
bash scripts/run_paddle_trt_int8_calib.sh configs/paddle_trt_int8_calib.yaml
```

Then run INT8 inference using the generated cache:

```bash
bash scripts/run_paddle_trt.sh configs/paddle_trt_int8.yaml
```

For INT8 inference with `trt.use_calib_mode: false`, `trt.cache_dir` must
already exist and point to the cache generated during calibration.

Expected local side effects after a successful run:

```text
trt_cache/int8_calib/
benchmarks/paddle_trt_int8_calib.csv
```

The exact calibration cache filenames are determined by Paddle Inference and
TensorRT.

## Accuracy Regression Interface

After running FP32, FP16, and INT8 configs with `output.save_result: true`, run:

```bash
python3 scripts/compare_detection_outputs.py \
  --fp32 outputs/paddle_trt_fp32.txt \
  --fp16 outputs/paddle_trt_fp16.txt \
  --int8 outputs/paddle_trt_int8.txt \
  --summary benchmarks/int8_accuracy_regression.md
```

The script compares detection count, matched boxes, IoU drift, and score drift.
It is a regression check, not a replacement for dataset-level mAP evaluation.

## Notes

- Use real representative images. Synthetic data is not suitable for INT8
  calibration.
- Start with `infer.batch_size: 1` and `trt.max_batch_size: 1` unless the model
  is confirmed to support dynamic batch.
- Remove stale files under `trt.calib_cache_dir` after changing model,
  precision, input shape, preprocessing, or calibration data.
- Keep `postprocess.mode: raw` until the real Paddle output decoder is wired.
- All compile, calibration, and accuracy-regression validation is pending user
  local execution.
