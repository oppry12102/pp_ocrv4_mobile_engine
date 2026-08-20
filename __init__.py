"""pp_ocrv4_mobile_engine — production OCR wrapper around PP-OCRv4.

Three engines share one ``recognize()`` API:

* ``engine_kind='mobile'``  — PaddleOCR mobile checkpoint on CPU.
* ``engine_kind='server'``  — PaddleOCR server checkpoint on CPU.
* ``engine_kind='gpu_v10'`` — ONNX Runtime + MIGraphX path (AMD ROCm)
  ported from ``agentocr/workspace/gpu_v10_ocr.py``.  Falls back to CPU
  if MIGraphX is unavailable.

Quickstart
----------
::

    from pp_ocrv4_mobile_engine import PaddleMobileEngine

    engine = PaddleMobileEngine()                       # mobile on CPU
    print(engine.recognize("sign.jpg").lines)

    # Same API, ~10× faster, +0.25 F1 absolute on RCTW-171:
    engine_gpu = PaddleMobileEngine(engine_kind="gpu_v10", use_gpu=True)
    print(engine_gpu.recognize("sign.jpg").lines)

    # Direct access to the GPU V10 impl (same interface):
    from pp_ocrv4_mobile_engine import GPUV10Engine
    engine = GPUV10Engine(use_gpu=True)
"""
from __future__ import annotations

from .engine import (
    DEFAULT_JPG_QUALITY,
    DEFAULT_LANG,
    DEFAULT_MAX_LONG,
    OCRLine,
    OCRResult,
    PaddleMobileEngine,
    RecognizeError,
    preprocess_image,
)

# Re-export the GPU V10 engine at package level.  The import is wrapped
# so a system without onnxruntime / pyclipper / shapely can still use
# the Paddle path; only selecting engine_kind='gpu_v10' forces those
# imports to be available.
try:
    from .gpu_v10 import GPUV10Engine, GPUV10Error  # noqa: F401
except ImportError:  # pragma: no cover - optional dep
    GPUV10Engine = None  # type: ignore[assignment]
    GPUV10Error = None  # type: ignore[assignment]


__all__ = [
    "DEFAULT_JPG_QUALITY",
    "DEFAULT_LANG",
    "DEFAULT_MAX_LONG",
    "OCRLine",
    "OCRResult",
    "PaddleMobileEngine",
    "RecognizeError",
    "preprocess_image",
    "GPUV10Engine",
    "GPUV10Error",
]

__version__ = "0.2.0"