# NVTX Profiling

NVTX integration is optional and controlled by:

```bash
-DENABLE_NVTX=ON
```

Default builds keep `ENABLE_NVTX=OFF`, so NVTX is not a required dependency.

When enabled, `PROFILE_RANGE("stage_name")` maps to an RAII wrapper around:

```text
nvtxRangePushA
nvtxRangePop
```

When disabled, `PROFILE_RANGE` compiles to a no-op.

Current NVTX ranges:

```text
pipeline_e2e
video_decode
preprocess
cpu_preprocess
gpu_preprocess
d2h_copy
inference
postprocess
decode
nms
```

Build and profile locally:

```bash
bash scripts/build.sh Release OFF ON
bash scripts/profile_nsys.sh configs/mock.yaml profiles/edge_infer_profile
```

The generated `.nsys-rep` file is produced by the user's local Nsight Systems
installation. No profiling result is included in this repository.
