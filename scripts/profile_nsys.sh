#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run Nsight Systems profiling for the mock pipeline.
#
# Usage:
#   bash scripts/profile_nsys.sh [configs/mock.yaml] [profiles/edge_infer_profile]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="${1:-configs/mock.yaml}"
OUTPUT_REL="${2:-profiles/edge_infer_profile}"
CONFIG="${ROOT_DIR}/${CONFIG_REL}"
OUTPUT="${ROOT_DIR}/${OUTPUT_REL}"
APP="${ROOT_DIR}/build/edge_infer"

if [[ ! -x "${APP}" ]]; then
    echo "[profile_nsys.sh] missing executable: ${APP}" >&2
    echo "[profile_nsys.sh] build first with: bash scripts/build.sh Release OFF ON" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[profile_nsys.sh] missing config: ${CONFIG}" >&2
    exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "[profile_nsys.sh] nsys not found in PATH. Install Nsight Systems or add nsys to PATH." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "[profile_nsys.sh] run command:"
echo "nsys profile -o \"${OUTPUT}\" --force-overwrite=true --trace=cuda,nvtx,osrt \"${APP}\" --config \"${CONFIG}\""
nsys profile \
    -o "${OUTPUT}" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    "${APP}" --config "${CONFIG}"
