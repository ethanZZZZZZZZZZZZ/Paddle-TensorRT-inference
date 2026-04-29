#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run the OpenCV video-file source with one mp4 duplicated across multiple
#   simulated camera streams.
#
# Usage:
#   bash scripts/run_multi_video.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${ROOT_DIR}/build/edge_infer"
CONFIG="${ROOT_DIR}/configs/multi_video.yaml"
VIDEO="${ROOT_DIR}/data/demo.mp4"

if [[ ! -x "${APP}" ]]; then
    echo "[run_multi_video.sh] missing executable: ${APP}" >&2
    echo "[run_multi_video.sh] build first with: bash scripts/build.sh Release ON" >&2
    exit 1
fi

if [[ ! -f "${VIDEO}" ]]; then
    echo "[run_multi_video.sh] missing test video: ${VIDEO}" >&2
    echo "[run_multi_video.sh] prepare it with: mkdir -p data && cp /path/to/demo.mp4 data/demo.mp4" >&2
    exit 1
fi

echo "[run_multi_video.sh] run command:"
echo "\"${APP}\" --config \"${CONFIG}\""
"${APP}" --config "${CONFIG}"
