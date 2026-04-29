#!/usr/bin/env python3
"""Compare detection result files across FP32/FP16/INT8 runs.

Input files use the pipeline result format:
  stream_id,frame_id,class_id,score,x1,y1,x2,y2

Usage:
  python3 scripts/compare_detection_outputs.py \
      --fp32 outputs/paddle_trt_fp32.txt \
      --fp16 outputs/paddle_trt_fp16.txt \
      --int8 outputs/paddle_trt_int8.txt \
      --summary benchmarks/int8_accuracy_regression.md
"""

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class Detection:
    stream_id: int
    frame_id: int
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def parse_detection_file(path: Path) -> List[Detection]:
    detections: List[Detection] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row or row[0].strip().startswith("#"):
                continue
            if line_no == 1 and row[0].strip().lower() in {"stream_id", "stream"}:
                continue
            if len(row) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 columns, got {len(row)}")
            detections.append(
                Detection(
                    stream_id=int(row[0]),
                    frame_id=int(row[1]),
                    class_id=int(row[2]),
                    score=float(row[3]),
                    x1=float(row[4]),
                    y1=float(row[5]),
                    x2=float(row[6]),
                    y2=float(row[7]),
                )
            )
    return detections


def area(det: Detection) -> float:
    return max(0.0, det.x2 - det.x1) * max(0.0, det.y2 - det.y1)


def iou(lhs: Detection, rhs: Detection) -> float:
    ix1 = max(lhs.x1, rhs.x1)
    iy1 = max(lhs.y1, rhs.y1)
    ix2 = min(lhs.x2, rhs.x2)
    iy2 = min(lhs.y2, rhs.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(lhs) + area(rhs) - inter
    return inter / union if union > 0.0 else 0.0


def group_by_frame_class(detections: Iterable[Detection]) -> Dict[Tuple[int, int, int], List[Detection]]:
    grouped: Dict[Tuple[int, int, int], List[Detection]] = {}
    for det in detections:
        key = (det.stream_id, det.frame_id, det.class_id)
        grouped.setdefault(key, []).append(det)
    for values in grouped.values():
        values.sort(key=lambda det: det.score, reverse=True)
    return grouped


def compare_variant(name: str, baseline: List[Detection], variant: List[Detection], iou_threshold: float):
    base_groups = group_by_frame_class(baseline)
    variant_groups = group_by_frame_class(variant)

    matched_iou = []
    matched_score_diff = []
    missing = 0
    extra = 0
    low_iou = 0

    all_keys = set(base_groups) | set(variant_groups)
    for key in all_keys:
        base_items = base_groups.get(key, [])
        variant_items = variant_groups.get(key, [])
        used_variant = set()
        for base_det in base_items:
            best_idx = -1
            best_iou = -1.0
            for idx, variant_det in enumerate(variant_items):
                if idx in used_variant:
                    continue
                current_iou = iou(base_det, variant_det)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_idx = idx
            if best_idx >= 0:
                used_variant.add(best_idx)
                matched_iou.append(best_iou)
                matched_score_diff.append(abs(base_det.score - variant_items[best_idx].score))
                if best_iou < iou_threshold:
                    low_iou += 1
            else:
                missing += 1
        extra += max(0, len(variant_items) - len(used_variant))

    return {
        "name": name,
        "baseline_count": len(baseline),
        "variant_count": len(variant),
        "matched": len(matched_iou),
        "missing": missing,
        "extra": extra,
        "low_iou": low_iou,
        "mean_iou": statistics.mean(matched_iou) if matched_iou else 0.0,
        "min_iou": min(matched_iou) if matched_iou else 0.0,
        "mean_score_abs_diff": statistics.mean(matched_score_diff) if matched_score_diff else 0.0,
        "max_score_abs_diff": max(matched_score_diff) if matched_score_diff else 0.0,
    }


def print_summary(summary):
    print(
        f"[{summary['name']}] baseline={summary['baseline_count']} "
        f"variant={summary['variant_count']} matched={summary['matched']} "
        f"missing={summary['missing']} extra={summary['extra']} "
        f"low_iou={summary['low_iou']} mean_iou={summary['mean_iou']:.6f} "
        f"mean_score_abs_diff={summary['mean_score_abs_diff']:.6f}"
    )


def write_markdown(path: Path, summaries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Precision Regression Summary\n\n")
        f.write("All values are generated only when the user runs local inference.\n\n")
        f.write("| Compare | Baseline Dets | Variant Dets | Matched | Missing | Extra | Low IoU | Mean IoU | Min IoU | Mean Score Diff | Max Score Diff |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for summary in summaries:
            f.write(
                f"| {summary['name']} "
                f"| {summary['baseline_count']} "
                f"| {summary['variant_count']} "
                f"| {summary['matched']} "
                f"| {summary['missing']} "
                f"| {summary['extra']} "
                f"| {summary['low_iou']} "
                f"| {summary['mean_iou']:.6f} "
                f"| {summary['min_iou']:.6f} "
                f"| {summary['mean_score_abs_diff']:.6f} "
                f"| {summary['max_score_abs_diff']:.6f} |\n"
            )
        f.write("\n")
        f.write("Review manually before accepting INT8. This script does not define model accuracy.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare detection outputs across precisions.")
    parser.add_argument("--fp32", type=Path, required=True, help="FP32 baseline result file.")
    parser.add_argument("--fp16", type=Path, default=None, help="Optional FP16 result file.")
    parser.add_argument("--int8", type=Path, default=None, help="Optional INT8 result file.")
    parser.add_argument("--iou-threshold", type=float, default=0.95)
    parser.add_argument("--summary", type=Path, default=None, help="Optional Markdown summary path.")
    args = parser.parse_args()

    if not args.fp32.is_file():
        raise FileNotFoundError(f"missing FP32 result file: {args.fp32}")
    if args.fp16 is None and args.int8 is None:
        raise ValueError("provide at least one of --fp16 or --int8")

    baseline = parse_detection_file(args.fp32)
    summaries = []
    if args.fp16 is not None:
        if not args.fp16.is_file():
            raise FileNotFoundError(f"missing FP16 result file: {args.fp16}")
        summaries.append(compare_variant("fp16_vs_fp32", baseline, parse_detection_file(args.fp16), args.iou_threshold))
    if args.int8 is not None:
        if not args.int8.is_file():
            raise FileNotFoundError(f"missing INT8 result file: {args.int8}")
        summaries.append(compare_variant("int8_vs_fp32", baseline, parse_detection_file(args.int8), args.iou_threshold))

    for summary in summaries:
        print_summary(summary)
    if args.summary is not None:
        write_markdown(args.summary, summaries)
        print(f"[compare_detection_outputs] wrote summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
