"""pp_ocrv4_mobile_engine — production OCR wrapper around PP-OCRv4 (mobile/server).

High-level API: :class:`PaddleMobileEngine` — see :mod:`.engine` for the
implementation.

Quickstart
----------
::

    from pp_ocrv4_mobile_engine import PaddleMobileEngine

    engine = PaddleMobileEngine()                       # mobile on CPU
    result = engine.recognize("sign.jpg")
    print(result.lines, result.elapsed_s)

    # Same API but heavier/accurate model
    engine_srv = PaddleMobileEngine(engine_kind="server")
    print(engine_srv.recognize("sign.jpg").lines)
"""
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

__all__ = [
    "DEFAULT_JPG_QUALITY",
    "DEFAULT_LANG",
    "DEFAULT_MAX_LONG",
    "OCRLine",
    "OCRResult",
    "PaddleMobileEngine",
    "RecognizeError",
    "preprocess_image",
]

__version__ = "0.1.0"
