"""Reproduce the F1 sweet-spot benchmark on RCTW first-10.

This is the script that produced the numbers in the README:

    | engine_kind | F1_s  | s/img |
    |-------------|-------|-------|
    | mobile      | 0.449 | 2.42  |
    | server      | 0.553 | 27.30 |

Usage
-----
::

    python examples/benchmark_rctw.py
    python examples/benchmark_rctw.py --kind server --out server_results.json

The script assumes the RCTW images are at
``img/rctw/rctw_default_01..10.jpg`` and the GT file at
``img/ground_truth_rctw.json`` (matching ``app5/`` layout).  Adjust
``--image-dir`` / ``--gt`` for any other layout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pp_ocrv4_mobile_engine import PaddleMobileEngine

# Lazy import — the scorer is project-local to app5.  When this example is
# vendored into a project that does not ship it, the run still works and
# only the optional F1-printing section is skipped.
try:
    from score_rctw_universal import score_results  # type: ignore
except Exception:
    score_results = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="img/rctw")
    ap.add_argument("--gt", default="img/ground_truth_rctw.json")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--kind", choices=PaddleMobileEngine.VALID_KINDS, default="mobile")
    ap.add_argument("--max-long", type=int, default=4096)
    ap.add_argument("--jpg-quality", type=int, default=90)
    ap.add_argument("--out", default="pp_mobile_results.json")
    args = ap.parse_args()

    img_dir = Path(args.image_dir)
    paths = sorted(img_dir.glob("rctw_default_*.jpg"))[: args.limit]
    if not paths:
        print(f"no images found under {img_dir}", file=sys.stderr)
        return 1

    engine = PaddleMobileEngine(
        engine_kind=args.kind,
        max_long=args.max_long,
        jpg_quality=args.jpg_quality,
    )

    t0 = time.time()
    results = engine.recognize_batch(paths)
    total = time.time() - t0

    # Save in the shape consumed by score_rctw_universal.py.
    out = {Path(r.image_path).name: r.lines for r in results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}  ({len(results)} images, {total:.1f}s total)", file=sys.stderr)

    avg = sum(r.elapsed_s for r in results) / max(len(results), 1)
    print(
        f"avg latency: {avg:.2f}s/img  (kind={args.kind}, "
        f"max_long={args.max_long}, jpg_quality={args.jpg_quality})",
        file=sys.stderr,
    )

    if score_results is not None and Path(args.gt).exists():
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        # score_results returns either a string or dict depending on version;
        # accept both.
        scored = score_results(out, gt)
        if isinstance(scored, dict):
            print(json.dumps(scored, ensure_ascii=False, indent=2))
        else:
            print(scored)
    else:
        print(
            "(scorer not present in this project; skipping optional F1 re-score)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
