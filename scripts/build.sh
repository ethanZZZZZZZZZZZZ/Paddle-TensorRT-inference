#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Configure and build the C++17 project.
#
# Usage:
#   bash scripts/build.sh [Debug|Release] [ENABLE_OPENCV:ON|OFF] [ENABLE_NVTX:ON|OFF] [ENABLE_PADDLE:ON|OFF] [ENABLE_CUDA:ON|OFF]

BUILD_TYPE="${1:-Release}"
ENABLE_OPENCV="${2:-ON}"
ENABLE_NVTX="${3:-OFF}"
ENABLE_PADDLE="${4:-OFF}"
ENABLE_CUDA="${5:-OFF}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "[build.sh] build type: ${BUILD_TYPE}"
echo "[build.sh] ENABLE_OPENCV: ${ENABLE_OPENCV}"
echo "[build.sh] ENABLE_NVTX: ${ENABLE_NVTX}"
echo "[build.sh] ENABLE_PADDLE: ${ENABLE_PADDLE}"
echo "[build.sh] ENABLE_CUDA: ${ENABLE_CUDA}"
echo "[build.sh] configure command:"
echo "cmake -S \"${ROOT_DIR}\" -B \"${BUILD_DIR}\" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_OPENCV=${ENABLE_OPENCV} -DENABLE_NVTX=${ENABLE_NVTX} -DENABLE_PADDLE=${ENABLE_PADDLE} -DENABLE_CUDA=${ENABLE_CUDA}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DENABLE_OPENCV="${ENABLE_OPENCV}" -DENABLE_NVTX="${ENABLE_NVTX}" -DENABLE_PADDLE="${ENABLE_PADDLE}" -DENABLE_CUDA="${ENABLE_CUDA}"

echo "[build.sh] build command:"
echo "cmake --build \"${BUILD_DIR}\" -j"
cmake --build "${BUILD_DIR}" -j
