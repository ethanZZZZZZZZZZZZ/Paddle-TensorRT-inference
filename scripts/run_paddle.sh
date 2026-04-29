#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run the Paddle Inference backend after building with ENABLE_PADDLE=ON.
#
# Usage:
#   bash scripts/run_paddle.sh [configs/paddle.yaml]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="${1:-configs/paddle.yaml}"
APP="${ROOT_DIR}/build/edge_infer"
CONFIG="${ROOT_DIR}/${CONFIG_REL}"

if [[ ! -x "${APP}" ]]; then
    echo "[run_paddle.sh] missing executable: ${APP}" >&2
    echo "[run_paddle.sh] build first with: bash scripts/build.sh Release OFF OFF ON" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[run_paddle.sh] missing config: ${CONFIG}" >&2
    exit 1
fi

source "${ROOT_DIR}/scripts/paddle_runtime_env.sh"

echo "[run_paddle.sh] run command:"
echo "\"${APP}\" --config \"${CONFIG}\""
"${APP}" --config "${CONFIG}"
