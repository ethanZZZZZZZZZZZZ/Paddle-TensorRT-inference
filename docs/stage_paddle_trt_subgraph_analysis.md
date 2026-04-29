# Paddle-TRT Subgraph Coverage Analysis

This stage adds log-based Paddle Inference + TensorRT subgraph analysis. It
does not change inference behavior and does not require hand-written TensorRT
engine integration.

## What It Captures

The analysis flow captures `edge_infer` stdout/stderr and parses Paddle logs
for:

```text
TRT subgraph count
TRT subgraph node counts
TensorRT engine build events
verified fallback / unsupported op names when the log exposes them
fallback evidence lines that need raw-log confirmation
ignored fallback keyword noise, such as Paddle startup flag lists
TensorRT inserted-copy events caused by unsupported striding
error candidate lines
```

It also records the Paddle-TRT config snapshot printed by
`PaddleInferEngine`.

## Config

Add this section to Paddle-TRT configs:

```yaml
trt_analysis:
  enable: true
  log_path: "../logs/paddle_trt_fp16_analysis.log"
  report_json: "../benchmarks/trt_subgraph_report_fp16.json"
  report_md: "../benchmarks/trt_subgraph_report_fp16.md"
```

`trt_analysis.enable: true` requires `trt.enable: true`.

## Run

Build with Paddle enabled:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENCV=OFF \
  -DENABLE_PADDLE=ON \
  -DENABLE_NVTX=OFF \
  -DPADDLE_INFERENCE_DIR=/path/to/paddle_inference
cmake --build build -j
```

Run and parse:

```bash
bash scripts/run_paddle_trt_analysis.sh \
  configs/paddle_trt_fp16.yaml \
  logs/paddle_trt_fp16_analysis.log \
  benchmarks/trt_subgraph_report_fp16.json \
  benchmarks/trt_subgraph_report_fp16.md
```

The script sets default log verbosity:

```bash
GLOG_v=1
FLAGS_v=1
```

You can override them when local Paddle builds require different verbosity:

```bash
GLOG_v=2 FLAGS_v=2 bash scripts/run_paddle_trt_analysis.sh configs/paddle_trt_fp16.yaml
```

## Parse Existing Logs

If a run log already exists:

```bash
python3 scripts/parse_paddle_trt_log.py logs/paddle_trt_fp16_analysis.log \
  --json benchmarks/trt_subgraph_report_fp16.json \
  --markdown benchmarks/trt_subgraph_report_fp16.md \
  --config configs/paddle_trt_fp16.yaml
```

## Output

Generated after local execution:

```text
logs/paddle_trt_fp16_analysis.log
benchmarks/trt_subgraph_report_fp16.json
benchmarks/trt_subgraph_report_fp16.md
```

The Markdown report contains:

```text
Summary
Parsed config snapshot
TRT subgraphs
Fallback / unsupported candidates
Error candidates
Notes
```

## Limitations

- This is log-based analysis. Paddle Inference does not always print total
  graph op count, so the parser does not invent a full coverage percentage.
- Verified fallback op names are reported only when Paddle logs include
  parseable op names.
- Lines that only contain generic flag names such as `enable_fusion_fallback`
  are ignored and are not treated as real model fallback.
- Fallback evidence lines without op names are weaker signals and should be
  checked against the raw log before being used in a technical report.
- Lines like `Generating copy ... because input does not support striding` are
  TensorRT inserted-copy events. They are useful performance evidence around
  concat/layout handling, but they are not Paddle fallback ops.
- If fallback details are missing, increase Paddle/GLOG verbosity and keep the
  raw log for manual inspection.
- All build, run, and report validation is pending user local execution.
