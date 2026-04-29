#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run CPU decode, GPU decode/pre-NMS + CPU NMS, and GPU decode + GPU NMS
#   benchmark configs and print a CSV summary. Requires a build configured
#   with -DENABLE_CUDA=ON -DENABLE_OPENCV=ON.
#
# Usage:
#   bash scripts/benchmark_decode.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT_DIR}/build/edge_infer"
CPU_CONFIG="${ROOT_DIR}/configs/mock_cpu_decode_benchmark.yaml"
GPU_CONFIG="${ROOT_DIR}/configs/mock_gpu_decode_pre_nms.yaml"
GPU_NMS_CONFIG="${ROOT_DIR}/configs/mock_gpu_decode_gpu_nms.yaml"

if [[ ! -x "${APP}" ]]; then
    echo "[benchmark_decode.sh] missing executable: ${APP}" >&2
    echo "[benchmark_decode.sh] build first with: bash scripts/build.sh Release ON OFF OFF ON" >&2
    exit 1
fi

echo "[benchmark_decode.sh] run CPU decode baseline"
"${APP}" --config "${CPU_CONFIG}"

echo "[benchmark_decode.sh] run GPU decode/pre-NMS baseline"
"${APP}" --config "${GPU_CONFIG}"

echo "[benchmark_decode.sh] run GPU decode + GPU NMS baseline"
"${APP}" --config "${GPU_NMS_CONFIG}"

python3 "${ROOT_DIR}/scripts/compare_decode_benchmark.py" \
    "${ROOT_DIR}/benchmarks/cpu_decode.csv" \
    "${ROOT_DIR}/benchmarks/gpu_decode_pre_nms.csv" \
    "${ROOT_DIR}/benchmarks/gpu_decode_gpu_nms.csv" \
    --summary "${ROOT_DIR}/benchmarks/postprocess_summary.md"
