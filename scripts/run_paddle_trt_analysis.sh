#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run Paddle Inference + TensorRT, capture stdout/stderr, and parse Paddle-TRT
#   subgraph coverage signals into JSON and Markdown reports.
#
# Usage:
#   bash scripts/run_paddle_trt_analysis.sh [config] [log] [report_json] [report_md]
#
# Example:
#   bash scripts/run_paddle_trt_analysis.sh \
#     configs/paddle_trt_fp16.yaml \
#     logs/paddle_trt_fp16_analysis.log \
#     benchmarks/trt_subgraph_report_fp16.json \
#     benchmarks/trt_subgraph_report_fp16.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_REL="${1:-configs/paddle_trt_fp16.yaml}"
LOG_REL="${2:-logs/paddle_trt_analysis.log}"
REPORT_JSON_REL="${3:-benchmarks/trt_subgraph_report.json}"
REPORT_MD_REL="${4:-benchmarks/trt_subgraph_report.md}"

APP="${ROOT_DIR}/build/edge_infer"
CONFIG="${ROOT_DIR}/${CONFIG_REL}"
LOG_PATH="${ROOT_DIR}/${LOG_REL}"
REPORT_JSON="${ROOT_DIR}/${REPORT_JSON_REL}"
REPORT_MD="${ROOT_DIR}/${REPORT_MD_REL}"

if [[ ! -x "${APP}" ]]; then
    echo "[run_paddle_trt_analysis.sh] missing executable: ${APP}" >&2
    echo "[run_paddle_trt_analysis.sh] build first with: bash scripts/build.sh Release OFF OFF ON" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[run_paddle_trt_analysis.sh] missing config: ${CONFIG}" >&2
    exit 1
fi

mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${REPORT_JSON}")" "$(dirname "${REPORT_MD}")"
source "${ROOT_DIR}/scripts/paddle_runtime_env.sh"

# Paddle and glog verbosity are version-dependent. These defaults are kept
# conservative and can be overridden by the caller.
export GLOG_v="${GLOG_v:-1}"
export FLAGS_v="${FLAGS_v:-1}"

echo "[run_paddle_trt_analysis.sh] run command:"
echo "\"${APP}\" --config \"${CONFIG}\" 2>&1 | tee \"${LOG_PATH}\""

set +e
"${APP}" --config "${CONFIG}" 2>&1 | tee "${LOG_PATH}"
APP_STATUS=${PIPESTATUS[0]}
set -e

python3 "${ROOT_DIR}/scripts/parse_paddle_trt_log.py" "${LOG_PATH}" \
    --json "${REPORT_JSON}" \
    --markdown "${REPORT_MD}" \
    --config "${CONFIG}"

if [[ ${APP_STATUS} -ne 0 ]]; then
    echo "[run_paddle_trt_analysis.sh] edge_infer exited with status ${APP_STATUS}; reports were still generated from captured logs" >&2
fi

exit "${APP_STATUS}"
