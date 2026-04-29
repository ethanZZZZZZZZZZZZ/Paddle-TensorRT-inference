#!/usr/bin/env bash
set -uo pipefail

# Purpose:
#   Configure and build the full GPU Paddle-TRT + TensorRT plugin target while
#   preserving CMake/build logs for debugging compiler crashes, linker errors,
#   or terminals that close immediately after failure.
#
# Usage:
#   bash scripts/build_full_gpu_logged.sh [Release|Debug] [jobs]
#
# Required environment:
#   PADDLE_INFERENCE_DIR
#
# Optional environment:
#   PADDLE_INFERENCE_THIRD_PARTY_DIR
#   TENSORRT_ROOT
#   TENSORRT_INCLUDE_DIR
#   TENSORRT_LIBRARY
#   TENSORRT_PLUGIN_LIBRARY
#   BUILD_DIR
#   LOG_DIR
#   ENABLE_PADDLE_GPU_INPUT_SHARE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${1:-Release}"
JOBS="${2:-2}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CMAKE_LOG="${LOG_DIR}/cmake_full_gpu_${TIMESTAMP}.log"
BUILD_LOG="${LOG_DIR}/build_full_gpu_${TIMESTAMP}.log"
SUMMARY_LOG="${LOG_DIR}/build_full_gpu_${TIMESTAMP}_summary.log"

mkdir -p "${LOG_DIR}" "${ROOT_DIR}/benchmarks" "${ROOT_DIR}/outputs" "${ROOT_DIR}/trt_cache"

log() {
    echo "$@" | tee -a "${SUMMARY_LOG}"
}

append_command() {
    local label="$1"
    shift
    log "[build_full_gpu_logged.sh] ${label}:"
    printf '  %q' "$@" | tee -a "${SUMMARY_LOG}"
    echo | tee -a "${SUMMARY_LOG}"
}

tail_log() {
    local path="$1"
    local lines="${2:-120}"
    if [[ -f "${path}" ]]; then
        log "[build_full_gpu_logged.sh] last ${lines} lines from ${path}:"
        tail -n "${lines}" "${path}" | tee -a "${SUMMARY_LOG}"
    fi
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        log "[build_full_gpu_logged.sh][ERROR] missing environment variable: ${name}"
        log "[build_full_gpu_logged.sh] example: export ${name}=/path/to/value"
        exit 2
    fi
}

log "[build_full_gpu_logged.sh] root=${ROOT_DIR}"
log "[build_full_gpu_logged.sh] build_dir=${BUILD_DIR}"
log "[build_full_gpu_logged.sh] build_type=${BUILD_TYPE}"
log "[build_full_gpu_logged.sh] jobs=${JOBS}"
log "[build_full_gpu_logged.sh] cmake_log=${CMAKE_LOG}"
log "[build_full_gpu_logged.sh] build_log=${BUILD_LOG}"

require_env PADDLE_INFERENCE_DIR

PADDLE_INFERENCE_THIRD_PARTY_DIR="${PADDLE_INFERENCE_THIRD_PARTY_DIR:-${PADDLE_INFERENCE_DIR}/third_party}"
ENABLE_PADDLE_GPU_INPUT_SHARE="${ENABLE_PADDLE_GPU_INPUT_SHARE:-ON}"

log "[build_full_gpu_logged.sh] PADDLE_INFERENCE_DIR=${PADDLE_INFERENCE_DIR}"
log "[build_full_gpu_logged.sh] PADDLE_INFERENCE_THIRD_PARTY_DIR=${PADDLE_INFERENCE_THIRD_PARTY_DIR}"
log "[build_full_gpu_logged.sh] TENSORRT_ROOT=${TENSORRT_ROOT:-}"
log "[build_full_gpu_logged.sh] TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR:-}"
log "[build_full_gpu_logged.sh] TENSORRT_LIBRARY=${TENSORRT_LIBRARY:-}"
log "[build_full_gpu_logged.sh] TENSORRT_PLUGIN_LIBRARY=${TENSORRT_PLUGIN_LIBRARY:-}"
log "[build_full_gpu_logged.sh] ENABLE_PADDLE_GPU_INPUT_SHARE=${ENABLE_PADDLE_GPU_INPUT_SHARE}"

cmake_args=(
    cmake
    -S "${ROOT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DENABLE_OPENCV=ON
    -DENABLE_CUDA=ON
    -DENABLE_PADDLE=ON
    -DENABLE_TENSORRT_PLUGIN=ON
    -DENABLE_PADDLE_GPU_INPUT_SHARE="${ENABLE_PADDLE_GPU_INPUT_SHARE}"
    -DENABLE_NVTX=OFF
    -DPADDLE_INFERENCE_DIR="${PADDLE_INFERENCE_DIR}"
    -DPADDLE_INFERENCE_THIRD_PARTY_DIR="${PADDLE_INFERENCE_THIRD_PARTY_DIR}"
)

if [[ -n "${TENSORRT_ROOT:-}" ]]; then
    cmake_args+=(-DTENSORRT_ROOT="${TENSORRT_ROOT}")
fi
if [[ -n "${TENSORRT_INCLUDE_DIR:-}" ]]; then
    cmake_args+=(-DTENSORRT_INCLUDE_DIR="${TENSORRT_INCLUDE_DIR}")
fi
if [[ -n "${TENSORRT_LIBRARY:-}" ]]; then
    cmake_args+=(-DTENSORRT_LIBRARY="${TENSORRT_LIBRARY}")
fi
if [[ -n "${TENSORRT_PLUGIN_LIBRARY:-}" ]]; then
    cmake_args+=(-DTENSORRT_PLUGIN_LIBRARY="${TENSORRT_PLUGIN_LIBRARY}")
fi

append_command "configure command" "${cmake_args[@]}"
set +e
"${cmake_args[@]}" 2>&1 | tee "${CMAKE_LOG}"
cmake_status=${PIPESTATUS[0]}
set -u
log "[build_full_gpu_logged.sh] cmake exit code=${cmake_status}"
if [[ ${cmake_status} -ne 0 ]]; then
    tail_log "${CMAKE_LOG}" 120
    log "[build_full_gpu_logged.sh][ERROR] CMake configure failed. Full log: ${CMAKE_LOG}"
    exit "${cmake_status}"
fi

build_args=(
    cmake
    --build "${BUILD_DIR}"
    --parallel "${JOBS}"
    --verbose
)

append_command "build command" "${build_args[@]}"
set +e
"${build_args[@]}" 2>&1 | tee "${BUILD_LOG}"
build_status=${PIPESTATUS[0]}
set -u
log "[build_full_gpu_logged.sh] build exit code=${build_status}"
if [[ ${build_status} -ne 0 ]]; then
    tail_log "${BUILD_LOG}" 160
    log "[build_full_gpu_logged.sh][ERROR] Build failed. Full log: ${BUILD_LOG}"
    exit "${build_status}"
fi

log "[build_full_gpu_logged.sh] build finished. Logs:"
log "  ${CMAKE_LOG}"
log "  ${BUILD_LOG}"
log "  ${SUMMARY_LOG}"
