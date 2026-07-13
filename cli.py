"""Command-line interface for pp_ocrv4_mobile_engine.

Examples
--------
::

    python -m pp_ocrv4_mobile_engine --image photo.jpg
    python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind mobile --out results.json
    python -m pp_ocrv4_mobile_engine --image-dir ./imgs --jsonl --kind server
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import DEFAULT_JPG_QUALITY, DEFAULT_MAX_LONG, PaddleMobileEngine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pp-ocrv4-mobile-engine",
        description="PP-OCRv4 (mobile/server) OCR engine — single CLI.",
    )
    ap.add_argument("--image", help="single image to OCR")
    ap.add_argument("--image-dir", help="directory of images to OCR (jpg/jpeg/png/webp/bmp)")
    ap.add_argument(
        "--kind",
        choices=PaddleMobileEngine.VALID_KINDS,
        default="mobile",
        help="which PP-OCRv4 weight set to load (default: mobile)",
    )
    ap.add_argument(
        "--max-long",
        type=int,
        default=DEFAULT_MAX_LONG,
        help=f"long-edge clip before OCR (default {DEFAULT_MAX_LONG}; only img ≥ this is downscaled)",
    )
    ap.add_argument(
        "--jpg-quality",
        type=int,
        default=DEFAULT_JPG_QUALITY,
        help=f"JPEG quality of preprocessed tile (default {DEFAULT_JPG_QUALITY}; do NOT exceed 95)",
    )
    ap.add_argument(
        "--use-gpu",
        action="store_true",
        help="attempt GPU inference (PaddlePaddle's own CUDA path; not benchmarked here)",
    )
    ap.add_argument(
        "--out",
        help="write JSON results here (default: stdout pretty-printed)",
    )
    ap.add_argument(
        "--jsonl",
        action="store_true",
        help="write JSON Lines (one result per line) instead of a single object",
    )
    args = ap.parse_args(argv)

    if not args.image and not args.image_dir:
        ap.error("either --image or --image-dir is required")

    paths: list[Path] = []
    if args.image:
        paths.append(Path(args.image))
    if args.image_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            paths.extend(sorted(Path(args.image_dir).glob(ext)))
        if not paths:
            print(f"no images found under {args.image_dir}", file=sys.stderr)
            return 1

    engine = PaddleMobileEngine(
        engine_kind=args.kind,
        use_gpu=args.use_gpu,
        max_long=args.max_long,
        jpg_quality=args.jpg_quality,
    )
    results = engine.recognize_batch(paths)

    payload = [r.to_dict() for r in results]

    if args.jsonl:
        out = sys.stdout
        if args.out:
            out = open(args.out, "w", encoding="utf-8")
        try:
            for row in payload:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            if args.out:
                out.close()
    elif args.out:
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
