#!/usr/bin/env python3
"""Prepare an INT8 calibration image list.

Usage:
  python3 scripts/prepare_int8_calibration_list.py data/calib_images \
      --output data/int8_calib_images.txt \
      --limit 512 \
      --relative-to data

The script accepts either an image directory or an existing text list. It does
not decode images; it only discovers and normalizes file paths for later
calibration runs.
"""

import argparse
import random
from pathlib import Path
from typing import Optional


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_extensions(text: str):
    exts = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        exts.append(item if item.startswith(".") else f".{item}")
    return tuple(exts) if exts else DEFAULT_EXTENSIONS


def discover_images(source: Path, extensions):
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

    if source.is_file():
        base_dir = source.parent
        paths = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = base_dir / path
            if path.suffix.lower() in extensions:
                paths.append(path)
        return sorted(paths)

    raise FileNotFoundError(f"calibration source does not exist: {source}")


def format_path(path: Path, relative_to: Optional[Path]):
    resolved = path.resolve()
    if relative_to is None:
        return str(resolved)
    try:
        return str(resolved.relative_to(relative_to.resolve()))
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare INT8 calibration image list.")
    parser.add_argument("source", type=Path, help="Image directory or existing image-list file.")
    parser.add_argument("--output", type=Path, default=Path("data/int8_calib_images.txt"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum images to keep; 0 keeps all.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=2026, help="Shuffle seed.")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=Path("."),
        help="Write paths relative to this directory when possible; use empty string for absolute paths.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    extensions = parse_extensions(args.extensions)
    images = discover_images(args.source, extensions)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(images)
    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        raise RuntimeError(f"no calibration images found under: {args.source}")

    relative_to = None if str(args.relative_to) == "" else args.relative_to
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for image in images:
            f.write(format_path(image, relative_to) + "\n")

    print(f"[prepare_int8_calibration_list] wrote {len(images)} images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
