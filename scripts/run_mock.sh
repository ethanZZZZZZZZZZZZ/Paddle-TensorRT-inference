#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run the mock dynamic-batch pipeline after the project has been built.
#
# Usage:
#   bash scripts/run_mock.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT_DIR}/build/edge_infer"
CONFIG="${ROOT_DIR}/configs/mock.yaml"

if [[ ! -x "${APP}" ]]; then
    echo "[run_mock.sh] missing executable: ${APP}" >&2
    echo "[run_mock.sh] build first with: bash scripts/build.sh Release" >&2
    exit 1
fi

echo "[run_mock.sh] run command:"
echo "\"${APP}\" --config \"${CONFIG}\""
"${APP}" --config "${CONFIG}"
