#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run CPU and GPU preprocessing benchmark configs and print a CSV summary.
#   Requires a build configured with -DENABLE_CUDA=ON -DENABLE_OPENCV=ON.
#
# Usage:
#   bash scripts/benchmark_preprocess.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT_DIR}/build/edge_infer"
CPU_CONFIG="${ROOT_DIR}/configs/mock_cpu_preprocess_benchmark.yaml"
GPU_CONFIG="${ROOT_DIR}/configs/mock_gpu_preprocess.yaml"

if [[ ! -x "${APP}" ]]; then
    echo "[benchmark_preprocess.sh] missing executable: ${APP}" >&2
    echo "[benchmark_preprocess.sh] build first with: bash scripts/build.sh Release ON OFF OFF ON" >&2
    exit 1
fi

echo "[benchmark_preprocess.sh] run CPU preprocess baseline"
"${APP}" --config "${CPU_CONFIG}"

echo "[benchmark_preprocess.sh] run GPU preprocess baseline"
"${APP}" --config "${GPU_CONFIG}"

python3 "${ROOT_DIR}/scripts/compare_preprocess_benchmark.py" \
    "${ROOT_DIR}/benchmarks/cpu_preprocess.csv" \
    "${ROOT_DIR}/benchmarks/gpu_preprocess.csv" \
    --summary "${ROOT_DIR}/benchmarks/preprocess_summary.md"
