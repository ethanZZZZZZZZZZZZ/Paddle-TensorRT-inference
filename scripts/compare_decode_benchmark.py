#!/usr/bin/env python3
"""Compare CPU decode, GPU decode/pre-NMS, and full GPU postprocess CSV files.

Usage:
  python3 scripts/compare_decode_benchmark.py \
      benchmarks/cpu_decode.csv \
      benchmarks/gpu_decode_pre_nms.csv \
      benchmarks/gpu_decode_gpu_nms.csv \
      benchmarks/mock_trt_plugin_postprocess.csv \
      --summary benchmarks/postprocess_summary.md
"""

import argparse
import csv
import statistics
from pathlib import Path


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(rows, key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return statistics.mean(values) if values else 0.0


def build_summary(label: str, rows):
    return {
        "label": label,
        "samples": len(rows),
        "avg_actual_batch_size": mean(rows, "actual_batch_size"),
        "avg_decode_ms": mean(rows, "decode_ms"),
        "avg_cpu_decode_ms": mean(rows, "cpu_decode_ms"),
        "avg_gpu_decode_pre_nms_ms": mean(rows, "gpu_decode_pre_nms_ms"),
        "avg_gpu_nms_ms": mean(rows, "gpu_nms_ms"),
        "avg_trt_plugin_ms": mean(rows, "trt_plugin_ms"),
        "avg_nms_ms": mean(rows, "nms_ms"),
        "avg_postprocess_ms": mean(rows, "postprocess_ms"),
        "avg_e2e_ms": mean(rows, "e2e_ms"),
        "avg_fps": mean(rows, "fps"),
    }


def print_summary(summary):
    label = summary["label"]
    print(f"[{label}] samples={summary['samples']}")
    print(f"[{label}] avg_actual_batch_size={summary['avg_actual_batch_size']:.6f}")
    print(f"[{label}] avg_decode_ms={summary['avg_decode_ms']:.6f}")
    print(f"[{label}] avg_cpu_decode_ms={summary['avg_cpu_decode_ms']:.6f}")
    print(f"[{label}] avg_gpu_decode_pre_nms_ms={summary['avg_gpu_decode_pre_nms_ms']:.6f}")
    print(f"[{label}] avg_nms_ms={summary['avg_nms_ms']:.6f}")
    print(f"[{label}] avg_gpu_nms_ms={summary['avg_gpu_nms_ms']:.6f}")
    print(f"[{label}] avg_trt_plugin_ms={summary['avg_trt_plugin_ms']:.6f}")
    print(f"[{label}] avg_postprocess_ms={summary['avg_postprocess_ms']:.6f}")
    print(f"[{label}] avg_e2e_ms={summary['avg_e2e_ms']:.6f}")
    print(f"[{label}] avg_fps={summary['avg_fps']:.6f}")


def write_markdown(path: Path, summaries):
    cpu_summary = summaries[0]
    gpu_decode_summary = summaries[1] if len(summaries) > 1 else None
    gpu_full_summary = summaries[2] if len(summaries) > 2 else None
    cpu_decode = cpu_summary["avg_cpu_decode_ms"] or cpu_summary["avg_decode_ms"]
    gpu_decode = (
        (gpu_decode_summary["avg_gpu_decode_pre_nms_ms"] or gpu_decode_summary["avg_decode_ms"])
        if gpu_decode_summary is not None else 0.0
    )
    speedup = cpu_decode / gpu_decode if cpu_decode > 0.0 and gpu_decode > 0.0 else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Postprocess Backend Benchmark\n\n")
        f.write("All values are generated only when the user runs the benchmark locally.\n\n")
        f.write("| Backend | Samples | Actual Batch | Decode ms | CPU Decode ms | GPU Decode Pre-NMS ms | NMS ms | GPU NMS ms | TRT Plugin ms | Postprocess ms | E2E ms | FPS |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for summary in summaries:
            f.write(
                f"| {summary['label']} "
                f"| {summary['samples']} "
                f"| {summary['avg_actual_batch_size']:.6f} "
                f"| {summary['avg_decode_ms']:.6f} "
                f"| {summary['avg_cpu_decode_ms']:.6f} "
                f"| {summary['avg_gpu_decode_pre_nms_ms']:.6f} "
                f"| {summary['avg_nms_ms']:.6f} "
                f"| {summary['avg_gpu_nms_ms']:.6f} "
                f"| {summary['avg_trt_plugin_ms']:.6f} "
                f"| {summary['avg_postprocess_ms']:.6f} "
                f"| {summary['avg_e2e_ms']:.6f} "
                f"| {summary['avg_fps']:.6f} |\n"
            )
        f.write("\n")
        f.write(f"- CPU decode baseline ms: {cpu_decode:.6f}\n")
        if gpu_decode_summary is not None:
            f.write(f"- GPU decode pre-NMS ms: {gpu_decode:.6f}\n")
            f.write(f"- CPU/GPU decode pre-NMS speedup: {speedup:.6f}\n")
        if gpu_full_summary is not None:
            full_gpu_post = gpu_full_summary["avg_postprocess_ms"]
            cpu_post = cpu_summary["avg_postprocess_ms"]
            post_speedup = cpu_post / full_gpu_post if cpu_post > 0.0 and full_gpu_post > 0.0 else 0.0
            f.write(f"- Full GPU postprocess ms: {full_gpu_post:.6f}\n")
            f.write(f"- CPU/full-GPU postprocess speedup: {post_speedup:.6f}\n")
        f.write("\n")
        f.write("Notes:\n\n")
        f.write("- GPU decode includes current host-to-device model output copy and device-to-host candidate copy.\n")
        f.write("- Full GPU postprocess runs decode/pre-NMS and NMS on CUDA, then copies final detections to host.\n")
        f.write("- Do not manually fill numbers; use locally generated CSV files only.\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CPU decode, GPU decode/pre-NMS, and full GPU postprocess CSV files.")
    parser.add_argument("cpu_csv", type=Path)
    parser.add_argument("gpu_csv", type=Path)
    parser.add_argument("gpu_nms_csv", type=Path, nargs="?")
    parser.add_argument("trt_plugin_csv", type=Path, nargs="?")
    parser.add_argument("--summary", type=Path, default=None, help="Optional Markdown summary output path.")
    args = parser.parse_args()

    if not args.cpu_csv.is_file():
        print(f"missing CPU CSV: {args.cpu_csv}")
        return 1
    if not args.gpu_csv.is_file():
        print(f"missing GPU CSV: {args.gpu_csv}")
        return 1
    if args.gpu_nms_csv is not None and not args.gpu_nms_csv.is_file():
        print(f"missing GPU NMS CSV: {args.gpu_nms_csv}")
        return 1
    if args.trt_plugin_csv is not None and not args.trt_plugin_csv.is_file():
        print(f"missing TRT plugin CSV: {args.trt_plugin_csv}")
        return 1

    cpu_summary = build_summary("cpu_decode", load_rows(args.cpu_csv))
    gpu_summary = build_summary("gpu_decode_pre_nms", load_rows(args.gpu_csv))
    summaries = [cpu_summary, gpu_summary]
    if args.gpu_nms_csv is not None:
        summaries.append(build_summary("gpu_decode_gpu_nms", load_rows(args.gpu_nms_csv)))
    if args.trt_plugin_csv is not None:
        summaries.append(build_summary("trt_plugin_postprocess", load_rows(args.trt_plugin_csv)))
    for summary in summaries:
        print_summary(summary)

    cpu_decode = cpu_summary["avg_cpu_decode_ms"] or cpu_summary["avg_decode_ms"]
    gpu_decode = gpu_summary["avg_gpu_decode_pre_nms_ms"] or gpu_summary["avg_decode_ms"]
    if cpu_decode > 0.0 and gpu_decode > 0.0:
        print(f"[compare] cpu_decode_ms / gpu_decode_pre_nms_ms = {cpu_decode / gpu_decode:.6f}")
    if args.summary is not None:
        write_markdown(args.summary, summaries)
        print(f"[compare] wrote summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
