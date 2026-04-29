#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run Paddle Inference + TensorRT INT8 calibration mode after building with
#   ENABLE_PADDLE=ON and a Paddle package that includes TensorRT support.
#
# Usage:
#   bash scripts/run_paddle_trt_int8_calib.sh [configs/paddle_trt_int8_calib.yaml]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="${1:-configs/paddle_trt_int8_calib.yaml}"
APP="${ROOT_DIR}/build/edge_infer"
CONFIG="${ROOT_DIR}/${CONFIG_REL}"

if [[ ! -x "${APP}" ]]; then
    echo "[run_paddle_trt_int8_calib.sh] missing executable: ${APP}" >&2
    echo "[run_paddle_trt_int8_calib.sh] build first with: bash scripts/build.sh Release ON OFF ON" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[run_paddle_trt_int8_calib.sh] missing config: ${CONFIG}" >&2
    exit 1
fi

source "${ROOT_DIR}/scripts/paddle_runtime_env.sh"

echo "[run_paddle_trt_int8_calib.sh] run command:"
echo "\"${APP}\" --config \"${CONFIG}\""
"${APP}" --config "${CONFIG}"
