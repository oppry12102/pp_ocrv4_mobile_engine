"""Minimal GPU V10 demo — run after the model files exist on disk.

Mirrors ``examples/basic_usage.py`` but uses ``engine_kind='gpu_v10'``
(the ONNX + MIGraphX path ported from agentocr).  Falls back to CPU
automatically when MIGraphX is not available — the only loss is speed.

Prereqs:
    pip install onnxruntime pyclipper shapely opencv-python numpy
    # the three ONNX files at /home/oppry/work/tools/agentocr/workspace/models/onnx/
"""
from __future__ import annotations

from pp_ocrv4_mobile_engine import PaddleMobileEngine

IMG = "/home/oppry/RCTW-171/train_images/image_0.jpg"  # any sign / scene photo


def main() -> int:
    # Same API as the Paddle path — just flip the constructor flag.
    engine = PaddleMobileEngine(engine_kind="gpu_v10", use_gpu=True)
    print(repr(engine))
    result = engine.recognize(IMG)
    print(f"image:  {result.image_path}")
    print(f"lines:  {len(result.lines)}")
    print(f"time:   {result.elapsed_s:.2f}s")
    for line in result.lines:
        print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())