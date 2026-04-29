#!/usr/bin/env python3
"""Parse Paddle Inference + TensorRT logs into a subgraph coverage report.

The parser is intentionally conservative: it reports TRT subgraph counts and
node counts only when Paddle logs expose them. Fallback reporting is split into
verified op names and weaker evidence lines so generic Paddle startup flags
such as enable_fusion_fallback are not reported as real model fallback.
TensorRT inserted-copy lines caused by unsupported striding are reported
separately because they are performance evidence, not Paddle fallback.

Usage:
  python3 scripts/parse_paddle_trt_log.py logs/paddle_trt_analysis.log \
      --json benchmarks/trt_subgraph_report.json \
      --markdown benchmarks/trt_subgraph_report.md \
      --config configs/paddle_trt_fp16.yaml
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


SUBGRAPH_PATTERNS = [
    re.compile(r"detect\s+a\s+sub-?graph\s+with\s+(\d+)\s+nodes", re.IGNORECASE),
    re.compile(r"sub-?graph\s+with\s+(\d+)\s+nodes", re.IGNORECASE),
]
ENGINE_BUILD_PATTERNS = [
    re.compile(r"prepare\s+trt\s+engine", re.IGNORECASE),
    re.compile(r"tensorrt.*engine", re.IGNORECASE),
]
COPY_EVENT_PATTERN = re.compile(
    r"Generating copy for\s+(.+?)\s+to\s+(.+?)\s+because input does not support striding",
    re.IGNORECASE,
)
FALLBACK_PATTERNS = [
    re.compile(r"\bfallback\b", re.IGNORECASE),
    re.compile(r"\bnot\s+support(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bunsupported\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+convert\b", re.IGNORECASE),
    re.compile(r"\bfailed?\s+to\s+convert\b", re.IGNORECASE),
    re.compile(r"\bskip(?:ped)?\s+(?:op|node)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+converted\b", re.IGNORECASE),
    re.compile(r"\bno\s+converter\b", re.IGNORECASE),
    re.compile(r"\bconverter\s+is\s+not\s+implemented\b", re.IGNORECASE),
]
ERROR_KEYWORDS = (
    "[error]",
    " error ",
    "error code",
    "invalidargumenterror",
    "fatal",
    "segmentation fault",
    "core dumped",
)
GLOG_ERROR_PATTERN = re.compile(r"^E\d{4}\s")
TRT_RELATED_KEYWORDS = (
    "trt",
    "tensorrt",
    "subgraph",
    "converter",
    "engine",
    "unsupported",
    "fallback",
)
OP_PATTERNS = [
    re.compile(r"\bop(?:erator)?(?:\s+type)?\s*[:=]\s*([A-Za-z0-9_./-]+)", re.IGNORECASE),
    re.compile(r"\btype\s*[:=]\s*([A-Za-z0-9_./-]+)", re.IGNORECASE),
    re.compile(r"\bno\s+converter\s+for\s+op\s+([A-Za-z0-9_./-]+)", re.IGNORECASE),
    re.compile(r"\bconverter\s+for\s+(?:op\s+)?([A-Za-z0-9_./-]+)", re.IGNORECASE),
    re.compile(r"\bunsupported\s+op\s+([A-Za-z0-9_./-]+)", re.IGNORECASE),
    re.compile(r"\bskip(?:ped)?\s+op\s+([A-Za-z0-9_./-]+)", re.IGNORECASE),
    re.compile(r"\bfallback\s+op\s+([A-Za-z0-9_./-]+)", re.IGNORECASE),
]
CONFIG_PATTERNS = {
    "precision": re.compile(r"TensorRT precision=([A-Za-z0-9_]+)"),
    "max_batch_size": re.compile(r"max_batch_size=([0-9]+)"),
    "min_subgraph_size": re.compile(r"min_subgraph_size=([0-9]+)"),
    "workspace_size": re.compile(r"workspace_size=([0-9]+)"),
    "use_static": re.compile(r"use_static=(true|false)", re.IGNORECASE),
    "use_calib_mode": re.compile(r"use_calib_mode=(true|false)", re.IGNORECASE),
    "cache_dir": re.compile(r"cache_dir=([^,\s]+)"),
    "dynamic_shape": re.compile(r"TensorRT dynamic_shape=(true|false)", re.IGNORECASE),
    "dynamic_shape_input_name": re.compile(r"input_name=([^,\s]*)"),
    "min_input_shape": re.compile(r"min_input_shape=(.*?)(?:,\s*opt_input_shape=|$)"),
    "opt_input_shape": re.compile(r"opt_input_shape=(.*?)(?:,\s*max_input_shape=|$)"),
    "max_input_shape": re.compile(r"max_input_shape=(.*?)(?:,\s*disable_plugin_fp16=|$)"),
}
NOISE_PATTERNS = [
    re.compile(r"\bInit commandline\b", re.IGNORECASE),
    re.compile(r"\bBefore Parse:\b", re.IGNORECASE),
    re.compile(r"\bAfter Parse:\b", re.IGNORECASE),
    re.compile(r"--tryfromenv=", re.IGNORECASE),
    re.compile(r"\bFLAGS_[A-Za-z0-9_]+\b"),
]
TRT_CONTEXT_PATTERNS = [
    re.compile(r"tensorrt", re.IGNORECASE),
    re.compile(r"\btrt\b", re.IGNORECASE),
    re.compile(r"op_converter", re.IGNORECASE),
    re.compile(r"subgraph", re.IGNORECASE),
    re.compile(r"converter", re.IGNORECASE),
]


def classify_line(line: str) -> Dict[str, bool]:
    lowered = f" {line.lower()} "
    return {
        "trt_related": any(keyword in lowered for keyword in TRT_RELATED_KEYWORDS),
        "fallback_keyword": has_fallback_keyword(line),
        "error_candidate": any(keyword in lowered for keyword in ERROR_KEYWORDS) or
        GLOG_ERROR_PATTERN.match(line) is not None,
    }


def has_fallback_keyword(line: str) -> bool:
    return any(pattern.search(line) is not None for pattern in FALLBACK_PATTERNS)


def is_noise_line(line: str) -> bool:
    return any(pattern.search(line) is not None for pattern in NOISE_PATTERNS)


def parse_copy_event(line: str) -> Optional[Dict[str, str]]:
    match = COPY_EVENT_PATTERN.search(line)
    if not match:
        return None
    return {
        "source": match.group(1).strip(),
        "target": match.group(2).strip(),
    }


def has_trt_context(line: str) -> bool:
    return any(pattern.search(line) is not None for pattern in TRT_CONTEXT_PATTERNS)


def is_fallback_evidence_line(line: str) -> bool:
    if not has_fallback_keyword(line):
        return False
    if parse_copy_event(line) is not None:
        return False
    if is_noise_line(line):
        return False
    return has_trt_context(line) or extract_op_name(line) is not None


def extract_op_name(line: str) -> Optional[str]:
    for pattern in OP_PATTERNS:
        match = pattern.search(line)
        if match:
            value = match.group(1).strip().strip(",.;")
            if value:
                return value
    return None


def update_config_snapshot(config_snapshot: Dict[str, str], line: str) -> None:
    for key, pattern in CONFIG_PATTERNS.items():
        match = pattern.search(line)
        if match:
            config_snapshot[key] = match.group(1).strip()


def parse_log(path: Path, config_path: Optional[Path]) -> Dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    subgraphs: List[Dict] = []
    engine_build_events: List[Dict] = []
    fallback_evidence_lines: List[Dict] = []
    ignored_fallback_keyword_lines: List[Dict] = []
    copy_events: List[Dict] = []
    error_lines: List[Dict] = []
    trt_related_lines: List[Dict] = []
    op_counter: Counter = Counter()
    config_snapshot: Dict[str, str] = {}

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        update_config_snapshot(config_snapshot, stripped)

        for pattern in SUBGRAPH_PATTERNS:
            match = pattern.search(stripped)
            if match:
                subgraphs.append({
                    "line": index,
                    "nodes": int(match.group(1)),
                    "text": stripped,
                })
                break

        if any(pattern.search(stripped) for pattern in ENGINE_BUILD_PATTERNS):
            engine_build_events.append({"line": index, "text": stripped})

        flags = classify_line(stripped)
        if flags["trt_related"]:
            trt_related_lines.append({"line": index, "text": stripped})

        copy_event = parse_copy_event(stripped)
        if copy_event is not None:
            copy_event["line"] = index
            copy_event["text"] = stripped
            copy_events.append(copy_event)

        if flags["fallback_keyword"] and not is_fallback_evidence_line(stripped) and copy_event is None:
            ignored_fallback_keyword_lines.append({"line": index, "text": stripped})

        if is_fallback_evidence_line(stripped):
            op_name = extract_op_name(stripped)
            item = {"line": index, "text": stripped}
            if op_name:
                item["op"] = op_name
                op_counter[op_name] += 1
            fallback_evidence_lines.append(item)

        if flags["error_candidate"]:
            error_lines.append({"line": index, "text": stripped})

    total_trt_nodes = sum(item["nodes"] for item in subgraphs)
    report = {
        "log_path": str(path),
        "config_path": str(config_path) if config_path is not None else "",
        "line_count": len(lines),
        "config_snapshot": config_snapshot,
        "trt_subgraph_count": len(subgraphs),
        "trt_subgraph_total_nodes": total_trt_nodes,
        "trt_subgraphs": subgraphs,
        "engine_build_event_count": len(engine_build_events),
        "engine_build_events": engine_build_events[:50],
        "fallback_candidate_count": len(fallback_evidence_lines),
        "fallback_evidence_line_count": len(fallback_evidence_lines),
        "fallback_candidates": fallback_evidence_lines[:200],
        "fallback_evidence_lines": fallback_evidence_lines[:200],
        "ignored_fallback_keyword_line_count": len(ignored_fallback_keyword_lines),
        "ignored_fallback_keyword_lines": ignored_fallback_keyword_lines[:100],
        "fallback_op_counts": dict(op_counter.most_common()),
        "verified_fallback_op_count": sum(op_counter.values()),
        "trt_copy_event_count": len(copy_events),
        "trt_copy_events": copy_events[:200],
        "error_candidate_count": len(error_lines),
        "error_candidates": error_lines[:200],
        "trt_related_line_count": len(trt_related_lines),
        "trt_related_lines_sample": trt_related_lines[:200],
        "coverage_note": (
            "This report counts Paddle log lines that explicitly mention TRT subgraphs. "
            "It does not compute a full graph coverage ratio unless Paddle logs expose total graph op counts."
        ),
    }
    return report


def write_json(path: Path, report: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subgraphs = report["trt_subgraphs"]
    fallback_ops = report["fallback_op_counts"]
    config = report["config_snapshot"]

    lines: List[str] = []
    lines.append("# Paddle-TRT Subgraph Coverage Report")
    lines.append("")
    lines.append("All values are parsed from local Paddle Inference logs.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Log: `{report['log_path']}`")
    if report.get("config_path"):
        lines.append(f"- Config: `{report['config_path']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- TRT subgraph count: `{report['trt_subgraph_count']}`")
    lines.append(f"- TRT subgraph total nodes: `{report['trt_subgraph_total_nodes']}`")
    lines.append(f"- Engine build event count: `{report['engine_build_event_count']}`")
    lines.append(f"- Verified fallback op count: `{report['verified_fallback_op_count']}`")
    lines.append(f"- Fallback evidence lines: `{report['fallback_evidence_line_count']}`")
    lines.append(f"- Ignored fallback keyword lines: `{report['ignored_fallback_keyword_line_count']}`")
    lines.append(f"- TensorRT copy / striding events: `{report['trt_copy_event_count']}`")
    lines.append(f"- Error candidate lines: `{report['error_candidate_count']}`")
    lines.append("")
    lines.append("Coverage ratio is not computed because Paddle logs do not always expose total graph op count.")
    lines.append("")

    if config:
        lines.append("## Parsed Config Snapshot")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|---|---|")
        for key in sorted(config):
            lines.append(f"| `{key}` | `{config[key]}` |")
        lines.append("")

    lines.append("## TRT Subgraphs")
    lines.append("")
    if subgraphs:
        lines.append("| Index | Log Line | Nodes | Text |")
        lines.append("|---:|---:|---:|---|")
        for idx, item in enumerate(subgraphs, start=1):
            text = item["text"].replace("|", "\\|")
            lines.append(f"| {idx} | {item['line']} | {item['nodes']} | `{text}` |")
    else:
        lines.append("No TRT subgraph lines were found. Check Paddle log verbosity and whether `trt.enable` is true.")
    lines.append("")

    lines.append("## Verified Fallback / Unsupported Ops")
    lines.append("")
    if fallback_ops:
        lines.append("| Op | Count |")
        lines.append("|---|---:|")
        for op_name, count in fallback_ops.items():
            lines.append(f"| `{op_name}` | {count} |")
    else:
        lines.append("No verified fallback or unsupported op names were extracted from the log.")
    lines.append("")

    fallback_evidence = report["fallback_evidence_lines"]
    if fallback_evidence:
        lines.append("### Fallback Evidence Lines")
        lines.append("")
        lines.append("| Log Line | Op | Text |")
        lines.append("|---:|---|---|")
        for item in fallback_evidence[:50]:
            text = item["text"].replace("|", "\\|")
            op_name = item.get("op", "")
            lines.append(f"| {item['line']} | `{op_name}` | `{text}` |")
        lines.append("")

    ignored_lines = report["ignored_fallback_keyword_lines"]
    if ignored_lines:
        lines.append("### Ignored Fallback Keyword Lines")
        lines.append("")
        lines.append("These lines contain words such as fallback but are treated as parser noise, not real model fallback.")
        lines.append("")
        lines.append("| Log Line | Text |")
        lines.append("|---:|---|")
        for item in ignored_lines[:20]:
            text = item["text"].replace("|", "\\|")
            lines.append(f"| {item['line']} | `{text}` |")
        lines.append("")

    copy_events = report["trt_copy_events"]
    if copy_events:
        lines.append("## TensorRT Copy / Striding Events")
        lines.append("")
        lines.append("These are TensorRT inserted copies caused by unsupported striding. They are not Paddle fallback ops, but they can add overhead before concat layers.")
        lines.append("")
        lines.append("| Log Line | Source Tensor | Target Tensor |")
        lines.append("|---:|---|---|")
        for item in copy_events[:50]:
            source = item["source"].replace("|", "\\|")
            target = item["target"].replace("|", "\\|")
            lines.append(f"| {item['line']} | `{source}` | `{target}` |")
        lines.append("")

    if report["error_candidates"]:
        lines.append("## Error Candidates")
        lines.append("")
        lines.append("| Log Line | Text |")
        lines.append("|---:|---|")
        for item in report["error_candidates"][:50]:
            text = item["text"].replace("|", "\\|")
            lines.append(f"| {item['line']} | `{text}` |")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is a log parser output, not proof that all unsupported ops were found.")
    lines.append("- Verified fallback ops require parseable op names in Paddle logs.")
    lines.append("- Fallback evidence lines are weaker signals and should be checked against the raw log.")
    lines.append("- TensorRT copy / striding events indicate inserted copies inside TRT, not Paddle fallback.")
    lines.append("- Increase Paddle/GLOG verbosity if fallback details are missing.")
    lines.append("- Keep the raw log for manual inspection when TensorRT build fails.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Paddle-TRT logs into coverage reports.")
    parser.add_argument("log", type=Path, help="Captured Paddle-TRT stdout/stderr log.")
    parser.add_argument("--json", type=Path, default=Path("benchmarks/trt_subgraph_report.json"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmarks/trt_subgraph_report.md"))
    parser.add_argument("--config", type=Path, default=None, help="Optional config path used for the run.")
    args = parser.parse_args()

    if not args.log.is_file():
        raise FileNotFoundError(f"missing log file: {args.log}")

    report = parse_log(args.log, args.config)
    write_json(args.json, report)
    write_markdown(args.markdown, report)
    print(f"[parse_paddle_trt_log] trt_subgraph_count={report['trt_subgraph_count']}")
    print(f"[parse_paddle_trt_log] trt_subgraph_total_nodes={report['trt_subgraph_total_nodes']}")
    print(f"[parse_paddle_trt_log] verified_fallback_op_count={report['verified_fallback_op_count']}")
    print(f"[parse_paddle_trt_log] fallback_evidence_line_count={report['fallback_evidence_line_count']}")
    print(f"[parse_paddle_trt_log] ignored_fallback_keyword_line_count={report['ignored_fallback_keyword_line_count']}")
    print(f"[parse_paddle_trt_log] trt_copy_event_count={report['trt_copy_event_count']}")
    print(f"[parse_paddle_trt_log] wrote json: {args.json}")
    print(f"[parse_paddle_trt_log] wrote markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
