#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run the mock pipeline and write benchmark CSV according to the config.
#
# Usage:
#   bash scripts/benchmark.sh [configs/mock.yaml]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="${1:-configs/mock.yaml}"
CONFIG="${ROOT_DIR}/${CONFIG_REL}"
APP="${ROOT_DIR}/build/edge_infer"

if [[ ! -x "${APP}" ]]; then
    echo "[benchmark.sh] missing executable: ${APP}" >&2
    echo "[benchmark.sh] build first with: bash scripts/build.sh Release OFF" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[benchmark.sh] missing config: ${CONFIG}" >&2
    exit 1
fi

echo "[benchmark.sh] run command:"
echo "\"${APP}\" --config \"${CONFIG}\""
"${APP}" --config "${CONFIG}"
