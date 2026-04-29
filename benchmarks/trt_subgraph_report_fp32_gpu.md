# Paddle-TRT Subgraph Coverage Report

All values are parsed from local Paddle Inference logs.

## Inputs

- Log: `/workspace/project/logs/paddle_trt_fp32_gpu_analysis.log`
- Config: `/workspace/project/configs/paddle.yaml`

## Summary

- TRT subgraph count: `1`
- TRT subgraph total nodes: `338`
- Engine build event count: `8`
- Verified fallback op count: `0`
- Fallback evidence lines: `0`
- Ignored fallback keyword lines: `1`
- TensorRT copy / striding events: `0`
- Error candidate lines: `0`

Coverage ratio is not computed because Paddle logs do not always expose total graph op count.

## Parsed Config Snapshot

| Key | Value |
|---|---|
| `cache_dir` | `/workspace/project/configs/../trt_cache/fp32` |
| `dynamic_shape` | `true` |
| `dynamic_shape_input_name` | `x2paddle_images` |
| `max_batch_size` | `1` |
| `max_input_shape` | `1,3,640,640` |
| `min_input_shape` | `1,3,640,640` |
| `min_subgraph_size` | `5` |
| `opt_input_shape` | `1,3,640,640` |
| `precision` | `fp32` |
| `use_calib_mode` | `false` |
| `use_static` | `false` |
| `workspace_size` | `1073741824` |

## TRT Subgraphs

| Index | Log Line | Nodes | Text |
|---:|---:|---:|---|
| 1 | 149 | 338 | `I0428 15:06:07.529218 747062 tensorrt_subgraph_pass.cc:320] ---  detect a sub-graph with 338 nodes` |

## Verified Fallback / Unsupported Ops

No verified fallback or unsupported op names were extracted from the log.

## Ignored Fallback Keyword Lines

The previous report counted Paddle startup flags as fallback candidates because the flag list contained names such as `enable_fusion_fallback`. These are parser noise, not real model fallback.

## Notes

- This report is a log parser output, not proof that all unsupported ops were found.
- Verified fallback ops require parseable op names in Paddle logs.
- Fallback evidence lines are weaker signals and should be checked against the raw log.
- Increase Paddle/GLOG verbosity if fallback details are missing.
- Keep the raw log for manual inspection when TensorRT build fails.
