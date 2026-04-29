# Paddle Inference Backend

This stage adds an optional Paddle Inference backend without changing the mock
backend contract.

Build isolation:

```bash
-DENABLE_PADDLE=OFF
```

is the default and does not require Paddle headers or libraries.

When enabled:

```bash
-DENABLE_PADDLE=ON
```

CMake searches for `paddle_inference_api.h` and `libpaddle_inference`.

Supported CMake path hints:

```bash
-DPADDLE_INFERENCE_DIR=/path/to/paddle_inference
-DPADDLE_INFERENCE_INCLUDE_DIR=/path/to/paddle/include
-DPADDLE_INFERENCE_LIB_DIR=/path/to/paddle/lib
-DPADDLE_INFERENCE_LIBRARY=/path/to/libpaddle_inference.so
-DPADDLE_INFERENCE_THIRD_PARTY_DIR=/path/to/paddle_inference/third_party
-DPADDLE_INFERENCE_EXTRA_LIB_DIRS="/extra/lib/dir1;/extra/lib/dir2"
-DPADDLE_INFERENCE_EXTRA_LIBS="/extra/lib/libfoo.so;/extra/lib/libbar.so"
```

If the root path does not match the expected release-package layout, CMake
recursively searches that root for `paddle_inference_api.h` and
`libpaddle_inference.so`. To inspect the exact package layout locally:

```bash
find "${PADDLE_INFERENCE_DIR}" \( -name 'paddle_inference_api.h' -o -name 'libpaddle_inference.so' \)
```

Linux Paddle Inference packages often ship required shared libraries such as
`libdnnl.so*`, `libiomp5.so*`, `libgomp.so*`, protobuf, glog, and gflags under
`third_party/install/*/lib`. The CMake integration scans those directories and
adds them to the link line and linker `rpath-link`. If a package has a custom
layout, pass `PADDLE_INFERENCE_THIRD_PARTY_DIR`,
`PADDLE_INFERENCE_EXTRA_LIB_DIRS`, or `PADDLE_INFERENCE_EXTRA_LIBS`.

Useful local inspection command:

```bash
find "${PADDLE_INFERENCE_DIR}" \( -name 'libdnnl.so*' -o -name 'libiomp5.so*' -o -name 'libgomp.so*' \)
```

If the binary links successfully but fails at launch with a loader error such
as `libdnnl.so.3: cannot open shared object file`, prepare runtime library
paths before launching:

```bash
source scripts/paddle_runtime_env.sh
./build/edge_infer --config configs/paddle.yaml
```

The executable argument is `--config`; `--configs` is not a supported flag.

Runtime config:

```yaml
input:
  num_streams: 1

infer:
  backend: paddle
  batch_size: 1
  enable_dynamic_batch: false

paddle:
  model_file: "../models/inference.pdmodel"
  params_file: "../models/inference.pdiparams"
  model_dir: ""
  use_gpu: false
  gpu_mem_mb: 512
  gpu_device_id: 0
  enable_ir_optim: true
  enable_memory_optim: true

postprocess:
  mode: raw
```

Use exactly one model source:

- `paddle.model_dir`
- `paddle.model_file + paddle.params_file`

Start with `infer.batch_size: 1` for real Paddle models. A traceback from
`ReshapeOp` that reports a capacity mismatch, for example input shape
`[4, 256, 20, 20]` but reshape target `[1, 2, 128, 400]`, means the exported
model contains fixed batch=1 assumptions. Keep batch=1 or re-export the model
with dynamic batch support before testing multi-stream dynamic batching.

`PaddleInferEngine` currently binds the first Paddle input tensor and uses the
first Paddle output tensor. For normal CPU postprocess it copies the output into
the existing host `TensorBuffer`. For `postprocess.decode_backend: trt_plugin`,
it first tries to expose the Paddle output as a GPU FP32 tensor view and feeds
that pointer directly to the postprocess TensorRT engine. Multi-input and
multi-output model adaptation is intentionally left for a later model-specific
integration stage.

`postprocess.mode: raw` is intended for real Paddle models before a
model-specific decoder is implemented. It logs the Paddle output shape and
skips detection decode, so the inference path and profiling CSV can be
validated without assuming the mock `[batch, boxes, 6]` tensor layout.

TensorRT subgraph options are documented in `docs/stage_paddle_trt_config.md`.

No Paddle build or runtime result is included in this repository. All validation
is pending user local execution.
