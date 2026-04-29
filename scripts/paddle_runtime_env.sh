#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Prepare runtime linker paths for Paddle Inference shared libraries.
#
# Usage:
#   source scripts/paddle_runtime_env.sh
#
# Inputs:
#   Optional PADDLE_INFERENCE_DIR. If it is not set, this script tries to read
#   PADDLE_INFERENCE_DIR from build/CMakeCache.txt.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PADDLE_ROOT="${PADDLE_INFERENCE_DIR:-}"

if [[ -z "${PADDLE_ROOT}" && -f "${ROOT_DIR}/build/CMakeCache.txt" ]]; then
    PADDLE_ROOT="$(grep '^PADDLE_INFERENCE_DIR:PATH=' "${ROOT_DIR}/build/CMakeCache.txt" | cut -d= -f2- || true)"
fi

if [[ -z "${PADDLE_ROOT}" || ! -d "${PADDLE_ROOT}" ]]; then
    echo "[paddle_runtime_env.sh] PADDLE_INFERENCE_DIR is not set or does not exist; skip LD_LIBRARY_PATH update" >&2
    return 0 2>/dev/null || exit 0
fi

mapfile -t PADDLE_LIB_DIRS < <(
    find "${PADDLE_ROOT}" -type f -name '*.so*' -printf '%h\n' | sort -u
)

if [[ "${#PADDLE_LIB_DIRS[@]}" -eq 0 ]]; then
    echo "[paddle_runtime_env.sh] no shared libraries found under: ${PADDLE_ROOT}" >&2
    return 0 2>/dev/null || exit 0
fi

PADDLE_LD_LIBRARY_PATH="$(IFS=:; echo "${PADDLE_LIB_DIRS[*]}")"
export LD_LIBRARY_PATH="${PADDLE_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

echo "[paddle_runtime_env.sh] Paddle runtime lib dirs:"
printf '  %s\n' "${PADDLE_LIB_DIRS[@]}"
