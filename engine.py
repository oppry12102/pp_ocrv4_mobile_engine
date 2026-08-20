"""
pp_ocrv4_mobile_engine
======================

A small, well-wrapped production interface to PaddleOCR's PP-OCRv4 family.

The wrapper does four things:

  1. **Hides the long edge / JPEG quality sweet spots.**  We measured the
     ``max_long`` × ``jpg_quality`` plane on RCTW first-10 (10 images, 44
     GT keywords) — only ``max_long=4096`` actually saves time (img05
     ``4208x2368`` is the only image that ever gets downscaled), and
     ``jpg_quality=90`` is the sweet spot.  Quality 95 actually *regresses*
     F1 by ~0.05 because PaddleOCR's det head is sensitive to JPEG ringing.
     These defaults are baked in but overridable.

  2. **Lets you flip to the server model via one constructor flag.**  PP-OCRv4
     *server* reuses the same det/rec APIs but loads the heavier
     ``*_server_infer`` checkpoints.  On RCTW it scores F1=0.5532 vs the
     mobile default 0.4494 — same API, ~11× slower on CPU.

  3. **Single OCR instance + thread-safe reuse.**  PaddleOCR's constructor
     downloads/caches ~12-450 MB of model weights; creating one per
     request is wasteful.  The engine keeps a process-global cache so
     :class:`PaddleMobileEngine` is cheap to instantiate.

  4. **Clean return types.**  Every call returns :class:`OCRResult` —
     a dataclass with ``.lines`` (``list[str]``), ``.boxes`` (``list[OCRLine]``
     with text/conf/polygon) and ``.elapsed_s``.  The convenience
     :meth:`PaddleMobileEngine.recognize_text` returns just the lines, in a
     shape compatible with downstream text-comparison code
     (``dict[filename, list[str]]`` matches :func:`score_rctw_universal.score_results`).

Installation
------------
::

    pip install paddleocr opencv-python-headless
    # Or with full GUI/numpy: pip install paddleocr opencv-python numpy

The model weights (≈12 MB mobile, ≈450 MB server) download on first use
from Paddle's CDN via ``paddleocr.PaddleOCR(lang="ch", ...)``.  Set
``download_params`` if you are behind a proxy.

API quick-start
---------------
::

    from pp_ocrv4_mobile_engine import PaddleMobileEngine

    engine = PaddleMobileEngine()                              # mobile, CPU
    result = engine.recognize("sign.jpg")
    for line in result.lines:
        print(line)

    # Switch to server for higher accuracy (CPU):
    engine_srv = PaddleMobileEngine(engine_kind="server")
    res = engine_srv.recognize("sign.jpg")

Performance on RCTW (10-image, 44-keyword benchmark)
-----------------------------------------------------
::

    | engine_kind | F1_s  | F1_l  | hits/GT | FP   | s/img | model size |
    |-------------|-------|-------|---------|------|-------|------------|
    | mobile (CPU)| 0.449 | 0.484 | 20/44   |  25  |  2.42 | 12  MB     |  <-- this module
    | server (CPU)| 0.553 | 0.553 | 26/44   |  24  | 27.30 | 450 MB     |  +5× F1, ~11× slower

Bench details in ``examples/benchmark_rctw.py``.

Limitations
-----------
* PP-OCRv4 server via this engine is **CPU-only** by default.  For ROCm/CUDA
  acceleration use PaddlePaddle's own GPU inference — this engine does not
  wrap that path.
* Built and benchmarked on Linux + Python 3.12 + paddleocr 2.7.3.  Other
  versions of paddleocr have reshuffled which models ``lang="ch"`` auto-loads;
  pass ``det_model_dir`` / ``rec_model_dir`` explicitly if you need a specific
  checkpoint.
* Det/rec sensitivity to ``jpg_quality`` is real — do NOT raise above 95.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2

# ``GPUV10Engine`` lives in ``gpu_v10.py``; we don't import it here at
# top-level so the package can still be imported on systems without
# onnxruntime / pyclipper / shapely (those are only needed when the
# user picks engine_kind='gpu_v10').
__all__ = [
    "DEFAULT_MAX_LONG",
    "DEFAULT_JPG_QUALITY",
    "DEFAULT_LANG",
    "PaddleMobileEngine",
    "OCRLine",
    "OCRResult",
    "RecognizeError",
    "preprocess_image",
]

logger = logging.getLogger("pp_ocrv4_mobile_engine")

# ---------------------------------------------------------------------------
# Defaults (validated by sweep on RCTW first-10)
# ---------------------------------------------------------------------------

DEFAULT_MAX_LONG: int = 4096
"""Longest edge of preprocessed image before OCR.  Only the
``4208x2368`` image in our benchmark exceeded this; raising it buys 0
F1 but adds decode cost.  Lowering it (e.g. 2048) loses info on
high-res scenes."""

DEFAULT_JPG_QUALITY: int = 90
"""JPEG quality of the preprocessed tile.  Empirically the best F1 on
RCTW.  95 actually regresses by ~0.05 because the det head is sensitive
to high-Q ringing artifacts."""

DEFAULT_LANG: str = "ch"
"""BCP-47-ish language code understood by PaddleOCR (we only test 'ch')."""

# Models PP-OCRv4 ships per (lang, kind).  These are the canonical paths
# inside the paddleocr 2.7 cache so users can inspect them.
_MOBILE_PATHS = {
    "det": "ch_PP-OCRv4_det_infer",
    "rec": "ch_PP-OCRv4_rec_infer",
    "cls": "ch_ppocr_mobile_v2.0_cls_infer",
}
_SERVER_PATHS = {
    "det": "ch_PP-OCRv4_det_server_infer",
    "rec": "ch_PP-OCRv4_rec_server_infer",
    "cls": "ch_ppocr_mobile_v2.0_cls_infer",
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OCRLine:
    """One detected text region in an image."""

    text: str
    conf: float
    bbox: tuple[tuple[float, float], ...]
    """Four-corner polygon as ((x1, y1), (x2, y2), (x3, y3), (x4, y4))."""

    def to_dict(self) -> dict:
        return {"text": self.text, "conf": self.conf, "bbox": list(self.bbox)}


@dataclass
class OCRResult:
    """Output of :meth:`PaddleMobileEngine.recognize`."""

    image_path: str
    lines: list[str] = field(default_factory=list)
    boxes: list[OCRLine] = field(default_factory=list)
    elapsed_s: float = 0.0

    def __iter__(self):
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __bool__(self) -> bool:
        return bool(self.lines)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {
            "image_path": self.image_path,
            "lines": self.lines,
            "boxes": [b.to_dict() for b in self.boxes],
            "elapsed_s": round(self.elapsed_s, 4),
        }


class RecognizeError(RuntimeError):
    """Raised when OCR fails on a single image."""


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(
    img_path: str | Path,
    out_tmp_path: str | Path | None = None,
    max_long: int = DEFAULT_MAX_LONG,
    jpg_quality: int = DEFAULT_JPG_QUALITY,
) -> str:
    """Load + downscale + JPEG-recompress an image.

    Parameters
    ----------
    img_path
        Source image (any format OpenCV can read).
    out_tmp_path
        Destination JPEG path.  When ``None`` a stable path under
        :file:`/tmp/pp_ocrv4_mobile_engine/<hash>.jpg` is used so repeat
        calls on the same source hit the filesystem page cache.
    max_long
        Longest edge to clip to.  Images whose longest edge ≤ ``max_long``
        pass through unchanged.
    jpg_quality
        Output JPEG quality.  Use 90.  Going higher hurts F1.

    Returns
    -------
    str
        Path to the JPEG file (suitable for ``PaddleOCR.ocr(...)``).
    """
    img_path = str(img_path)
    if out_tmp_path is None:
        h = uuid.uuid5(uuid.NAMESPACE_URL, img_path).hex[:12]
        tmp_dir = Path("/tmp/pp_ocrv4_mobile_engine")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_tmp_path = str(tmp_dir / f"{h}.jpg")
    out_tmp_path = str(out_tmp_path)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]
    if max(w, h) > max_long:
        scale = max_long / max(w, h)
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])[1].tofile(out_tmp_path)
    return out_tmp_path


# ---------------------------------------------------------------------------
# Process-global cache of PaddleOCR instances (one per unique set of args)
# ---------------------------------------------------------------------------

_ENGINE_CACHE: dict[tuple, object] = {}


def _get_or_build_paddle(
    *,
    engine_kind: str,
    lang: str,
    use_angle_cls: bool,
    use_gpu: bool,
    det_model_dir: str | None,
    rec_model_dir: str | None,
    cls_model_dir: str | None,
    show_log: bool,
):
    """Construct (and cache) a PaddleOCR instance for these arguments."""
    key = (
        engine_kind, lang, use_angle_cls, use_gpu,
        det_model_dir, rec_model_dir, cls_model_dir, show_log,
    )
    cached = _ENGINE_CACHE.get(key)
    if cached is not None:
        return cached

    from paddleocr import PaddleOCR  # imported here so the module loads fast

    if engine_kind == "server":
        d, r, c = _SERVER_PATHS["det"], _SERVER_PATHS["rec"], _SERVER_PATHS["cls"]
    elif engine_kind == "mobile":
        d, r, c = _MOBILE_PATHS["det"], _MOBILE_PATHS["rec"], _MOBILE_PATHS["cls"]
    else:
        raise ValueError(f"engine_kind must be 'mobile' or 'server', got {engine_kind!r}")

    kwargs = dict(
        use_angle_cls=use_angle_cls,
        lang=lang,
        use_gpu=use_gpu,
        show_log=show_log,
    )
    # Only pass *_model_dir when caller provided one.  When None, PaddleOCR
    # auto-downloads the standard model for the (lang, kind) pair.
    if det_model_dir is not None:
        kwargs["det_model_dir"] = det_model_dir or d
    if rec_model_dir is not None:
        kwargs["rec_model_dir"] = rec_model_dir or r
    if cls_model_dir is not None:
        kwargs["cls_model_dir"] = cls_model_dir or c

    logger.info("initialising PaddleOCR (kind=%s, lang=%s, gpu=%s)", engine_kind, lang, use_gpu)
    t0 = time.time()
    ocr = PaddleOCR(**kwargs)
    logger.info("PaddleOCR ready in %.1fs", time.time() - t0)

    _ENGINE_CACHE[key] = ocr
    return ocr


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PaddleMobileEngine:
    """High-level OCR over PP-OCRv4 (mobile by default).

    Parameters
    ----------
    engine_kind : {'mobile', 'server', 'gpu_v10'}
        Which PP-OCRv4 weight set / runtime to use:

        * ``'mobile'`` (default) — PaddleOCR mobile checkpoint on CPU.
          ~2.4s/img, F1 ≈ 0.45 on RCTW first-10.
        * ``'server'`` — PaddleOCR server checkpoint on CPU.  +5% F1 absolute
          but ~11× slower (≈27 s/img).
        * ``'gpu_v10'`` — ONNX Runtime + MIGraphX path, same accuracy as
          agentocr ``gpu_v10_ocr.py``: F1 ≈ 0.70 on RCTW-171 50-149 at
          ~0.24 s/img on this ROCm box (CPU fallback if MIGraphX absent).

        All three share the same ``.recognize`` API and return
        :class:`OCRResult`.
    lang : str
        Passed to :class:`paddleocr.PaddleOCR`.  Defaults to ``'ch'``.
        Ignored when ``engine_kind='gpu_v10'``.
    use_angle_cls : bool
        Pass ``True`` to enable the 0/180° text-direction classifier
        (recommended for scenes with rotated signs).  Always on for
        ``'gpu_v10'``.
    use_gpu : bool
        Pass ``True`` to attempt GPU inference.  ``engine_kind='gpu_v10'``
        uses MIGraphX (AMD ROCm) via ONNX Runtime and falls back to CPU
        when MIGraphX is missing; for the Paddle path the GPU option is
        forwarded to PaddleOCR as-is.
    max_long : int
        Longest edge cap for the preprocessed image.  See
        :data:`DEFAULT_MAX_LONG`.
    jpg_quality : int
        JPEG quality of the preprocessed image.  See
        :data:`DEFAULT_JPG_QUALITY`.
    show_log : bool
        Forwarded to the underlying runtime (default: ``False``).
    det_model_dir / rec_model_dir / cls_model_dir : str | None
        Override the default model directory for each head (Paddle path
        only).
    model_dir : str | None
        ``engine_kind='gpu_v10'`` only.  Directory containing the three
        ``.onnx`` files (det_mobile, rec_server, cls_mobile).  See
        :data:`gpu_v10.DEFAULT_MODEL_DIR`.
    det_size : int | None
        ``engine_kind='gpu_v10'`` only.  Detection input canvas side.
        Defaults to :data:`gpu_v10.DET_LIMIT` (1920).
    """

    ENGINE_KIND_MOBILE = "mobile"
    ENGINE_KIND_SERVER = "server"
    ENGINE_KIND_GPU_V10 = "gpu_v10"
    VALID_KINDS = ("mobile", "server", "gpu_v10")

    def __init__(
        self,
        engine_kind: str = "mobile",
        *,
        lang: str = DEFAULT_LANG,
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        max_long: int = DEFAULT_MAX_LONG,
        jpg_quality: int = DEFAULT_JPG_QUALITY,
        show_log: bool = False,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        cls_model_dir: str | None = None,
        model_dir: str | None = None,
        det_size: int | None = None,
    ) -> None:
        if engine_kind not in self.VALID_KINDS:
            raise ValueError(
                f"engine_kind must be one of {self.VALID_KINDS}, got {engine_kind!r}"
            )
        if not (50 <= jpg_quality <= 100):
            raise ValueError(
                f"jpg_quality must be 50..100 (we measured 90 is sweetest; "
                f"95 actually hurts F1).  Got {jpg_quality!r}."
            )
        if max_long < 256:
            raise ValueError(f"max_long must be >= 256 (got {max_long!r})")

        self.engine_kind = engine_kind
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.max_long = int(max_long)
        self.jpg_quality = int(jpg_quality)

        # GPU V10 path — defer import so users without onnxruntime / pyclipper
        # / shapely can still use the CPU path.
        if engine_kind == self.ENGINE_KIND_GPU_V10:
            from .gpu_v10 import _get_or_build_gpu_v10
            kw: dict = dict(
                use_gpu=use_gpu,
                max_long=self.max_long,
                jpg_quality=self.jpg_quality,
                show_log=show_log,
            )
            if model_dir is not None:
                kw["model_dir"] = model_dir
            if det_size is not None:
                kw["det_size"] = det_size
            self._engine_impl: object = _get_or_build_gpu_v10(**kw)
            # The two engines share an interface but live in different
            # classes; tag the impl so __repr__ / introspection stay
            # honest.
            self._impl_kind = "gpu_v10"
            return

        # Paddle path
        self._impl_kind = "paddle"
        self._ocr = _get_or_build_paddle(
            engine_kind=engine_kind,
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            show_log=show_log,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recognize(
        self,
        image: str | Path,
        *,
        max_long: int | None = None,
        jpg_quality: int | None = None,
    ) -> OCRResult:
        """OCR a single image.

        Parameters
        ----------
        image
            Path to an image file readable by OpenCV (``cv2.imread``).
        max_long, jpg_quality
            Per-call overrides for the preprocessing parameters.  When
            ``None`` the engine's defaults are used.

        Returns
        -------
        :class:`OCRResult` with ``.lines`` (``list[str]``),
        ``.boxes`` (``list[OCRLine]``) and ``.elapsed_s``.
        """
        # GPU V10 path: delegate entirely to the ONNX impl.
        if self._impl_kind == "gpu_v10":
            return self._engine_impl.recognize(
                image, max_long=max_long, jpg_quality=jpg_quality
            )

        image_str = str(image)
        # Per-call overrides captured here so the cache + reused instance
        # remain stable across calls.
        ml = self.max_long if max_long is None else int(max_long)
        jq = self.jpg_quality if jpg_quality is None else int(jpg_quality)

        try:
            tmp = preprocess_image(image_str, max_long=ml, jpg_quality=jq)
        except Exception as e:
            raise RecognizeError(f"预处理失败: {image_str}: {e}") from e

        # paddleocr 2.x returns nested lists, possibly None on failure.
        t0 = time.time()
        try:
            raw = self._ocr.ocr(tmp, cls=self.use_angle_cls)
        except Exception as e:
            raise RecognizeError(f"OCR 失败: {image_str}: {e}") from e
        elapsed = time.time() - t0

        lines, boxes = _parse_pages(raw)
        return OCRResult(
            image_path=image_str,
            lines=lines,
            boxes=boxes,
            elapsed_s=elapsed,
        )

    def recognize_batch(
        self,
        images: Iterable[str | Path],
        *,
        max_long: int | None = None,
        jpg_quality: int | None = None,
        on_error: str = "empty",  # 'empty' or 'raise'
    ) -> list[OCRResult]:
        """OCR multiple images sequentially.

        PaddleOCR's underlying CLI is not thread-safe in 2.7, so this is a
        sequential loop.  For moderate workloads (≤10k images) that's
        typically fine.  Parallelism is best done at the process boundary.
        The ``gpu_v10`` kind has the same sequential shape — MIGraphX
        sessions are process-singleton and not safe to share across
        processes.

        Parameters
        ----------
        on_error
            ``'empty'`` (default): a failed image yields an empty
            :class:`OCRResult` so the rest of the batch still runs.
            ``'raise'``: re-raise the first failure (skips remaining images).
        """
        if on_error not in ("empty", "raise"):
            raise ValueError("on_error must be 'empty' or 'raise'")

        # GPU V10 path: the impl already has its own batch helper.
        if self._impl_kind == "gpu_v10":
            return self._engine_impl.recognize_batch(
                images,
                max_long=max_long,
                jpg_quality=jpg_quality,
                on_error=on_error,
            )

        results: list[OCRResult] = []
        n_total = 0
        try:
            n_total = len(images)  # type: ignore[arg-type]
        except TypeError:
            pass

        for i, img in enumerate(images, 1):
            try:
                r = self.recognize(img, max_long=max_long, jpg_quality=jpg_quality)
            except RecognizeError as e:
                print(f"[pp-mobile] {i}: {e}", file=sys.stderr)
                if on_error == "raise":
                    raise
                r = OCRResult(image_path=str(img), lines=[], boxes=[], elapsed_s=0.0)
            results.append(r)
            print(
                f"[pp-mobile] {i}/{n_total or '?'}: {Path(r.image_path).name} "
                f"-> {len(r.lines)} lines in {r.elapsed_s:.2f}s",
                file=sys.stderr,
            )
        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def recognize_text(self, image: str | Path, **kw) -> list[str]:
        """Just the text fragments — convenience wrapper around
        :meth:`recognize` returning ``result.lines``."""
        return self.recognize(image, **kw).lines

    def recognize_batch_text(
        self,
        images: Iterable[str | Path],
        **kw,
    ) -> dict[str, list[str]]:
        """Batch text-only output in ``dict[basename, list[str]]`` shape —
        matches the file format consumed by ``score_rctw_universal.py``."""
        results = self.recognize_batch(images, **kw)
        return {Path(r.image_path).name: r.lines for r in results}

    def __repr__(self) -> str:
        if self._impl_kind == "gpu_v10":
            inner = repr(self._engine_impl)
            return f"PaddleMobileEngine(engine_kind='gpu_v10', impl={inner})"
        return (
            f"PaddleMobileEngine(kind={self.engine_kind!r}, lang={self.lang!r}, "
            f"use_gpu={self.use_gpu}, max_long={self.max_long}, "
            f"jpg_quality={self.jpg_quality})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_pages(raw) -> tuple[list[str], list[OCRLine]]:
    """Convert PaddleOCR's nested-list return value to text + typed boxes.

    PaddleOCR 2.x returns either ``[[[(box, (text, score)), ...]]]`` (when
    successful) or ``None`` / ``[None]`` (when nothing was detected).
    Older versions occasionally return a flat list — we tolerate both.
    """
    lines: list[str] = []
    boxes: list[OCRLine] = []
    if raw is None:
        return lines, boxes
    pages = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if page is None:
            continue
        # Sometimes `page` is just one line, not a list of lines.
        page_items = page if isinstance(page, list) else [page]
        for line in page_items:
            if line is None:
                continue
            try:
                box, (text, score) = line[0], line[1]
            except (TypeError, IndexError, ValueError):
                # If a row isn't a `(box, (text, score))` tuple, skip it.
                continue
            box_t = tuple((float(p[0]), float(p[1])) for p in box)
            lines.append(str(text))
            boxes.append(OCRLine(text=str(text), conf=float(score), bbox=box_t))
    return lines, boxes
