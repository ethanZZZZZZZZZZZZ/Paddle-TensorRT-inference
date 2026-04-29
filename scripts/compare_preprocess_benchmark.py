#!/usr/bin/env python3
"""Compare CPU and GPU preprocessing benchmark CSV files.

Usage:
  python3 scripts/compare_preprocess_benchmark.py \
      benchmarks/cpu_preprocess.csv benchmarks/gpu_preprocess.csv

  python3 scripts/compare_preprocess_benchmark.py \
      benchmarks/cpu_preprocess.csv benchmarks/gpu_preprocess.csv \
      --summary benchmarks/preprocess_summary.md
"""

import argparse
import csv
import statistics
import sys
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
        "avg_preprocess_ms": mean(rows, "preprocess_ms"),
        "avg_cpu_preprocess_ms": mean(rows, "cpu_preprocess_ms"),
        "avg_gpu_preprocess_ms": mean(rows, "gpu_preprocess_ms"),
        "avg_d2h_copy_ms": mean(rows, "d2h_copy_ms"),
        "avg_e2e_ms": mean(rows, "e2e_ms"),
        "avg_fps": mean(rows, "fps"),
    }


def print_summary(summary):
    label = summary["label"]
    print(f"[{label}] samples={summary['samples']}")
    print(f"[{label}] avg_actual_batch_size={summary['avg_actual_batch_size']:.6f}")
    print(f"[{label}] avg_preprocess_ms={summary['avg_preprocess_ms']:.6f}")
    print(f"[{label}] avg_cpu_preprocess_ms={summary['avg_cpu_preprocess_ms']:.6f}")
    print(f"[{label}] avg_gpu_preprocess_ms={summary['avg_gpu_preprocess_ms']:.6f}")
    print(f"[{label}] avg_d2h_copy_ms={summary['avg_d2h_copy_ms']:.6f}")
    print(f"[{label}] avg_e2e_ms={summary['avg_e2e_ms']:.6f}")
    print(f"[{label}] avg_fps={summary['avg_fps']:.6f}")


def write_markdown(path: Path, cpu_summary, gpu_summary):
    cpu_pre = cpu_summary["avg_cpu_preprocess_ms"] or cpu_summary["avg_preprocess_ms"]
    gpu_pre = gpu_summary["avg_gpu_preprocess_ms"]
    speedup = cpu_pre / gpu_pre if cpu_pre > 0.0 and gpu_pre > 0.0 else 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# CPU vs GPU Preprocess Benchmark\n\n")
        f.write("All values are generated only when the user runs the benchmark locally.\n\n")
        f.write("| Backend | Samples | Actual Batch | Preprocess ms | CPU Preprocess ms | GPU Preprocess ms | D2H Copy ms | E2E ms | FPS |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for summary in (cpu_summary, gpu_summary):
            f.write(
                f"| {summary['label']} "
                f"| {summary['samples']} "
                f"| {summary['avg_actual_batch_size']:.6f} "
                f"| {summary['avg_preprocess_ms']:.6f} "
                f"| {summary['avg_cpu_preprocess_ms']:.6f} "
                f"| {summary['avg_gpu_preprocess_ms']:.6f} "
                f"| {summary['avg_d2h_copy_ms']:.6f} "
                f"| {summary['avg_e2e_ms']:.6f} "
                f"| {summary['avg_fps']:.6f} |\n"
            )
        f.write("\n")
        f.write(f"- CPU preprocess baseline ms: {cpu_pre:.6f}\n")
        f.write(f"- GPU kernel preprocess ms: {gpu_pre:.6f}\n")
        f.write(f"- CPU/GPU preprocess speedup: {speedup:.6f}\n")
        f.write("\n")
        f.write("Notes:\n\n")
        f.write("- `gpu_preprocess_ms` excludes the current compatibility D2H copy.\n")
        f.write("- `preprocess_ms` for GPU mode includes `gpu_preprocess_ms + d2h_copy_ms`.\n")
        f.write("- Do not treat this file as a verified result until it is generated locally.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CPU and GPU preprocessing benchmark CSV files.")
    parser.add_argument("cpu_csv", type=Path)
    parser.add_argument("gpu_csv", type=Path)
    parser.add_argument("--summary", type=Path, default=None, help="Optional Markdown summary output path.")
    args = parser.parse_args()

    cpu_path = args.cpu_csv
    gpu_path = args.gpu_csv
    if not cpu_path.is_file():
        print(f"missing CPU CSV: {cpu_path}", file=sys.stderr)
        return 1
    if not gpu_path.is_file():
        print(f"missing GPU CSV: {gpu_path}", file=sys.stderr)
        return 1

    cpu_rows = load_rows(cpu_path)
    gpu_rows = load_rows(gpu_path)
    cpu_summary = build_summary("cpu", cpu_rows)
    gpu_summary = build_summary("gpu", gpu_rows)
    print_summary(cpu_summary)
    print_summary(gpu_summary)

    cpu_pre = cpu_summary["avg_cpu_preprocess_ms"] or cpu_summary["avg_preprocess_ms"]
    gpu_pre = gpu_summary["avg_gpu_preprocess_ms"]
    if cpu_pre > 0.0 and gpu_pre > 0.0:
        print(f"[compare] cpu_preprocess_ms / gpu_preprocess_ms = {cpu_pre / gpu_pre:.6f}")
    if args.summary is not None:
        write_markdown(args.summary, cpu_summary, gpu_summary)
        print(f"[compare] wrote summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
