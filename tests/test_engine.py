"""Unit tests for pp_ocrv4_mobile_engine.

Run with::

    cd /home/oppry/work/pp_ocrv4_mobile_engine
    python -m pytest tests/ -v

or directly::

    python tests/test_engine.py

The PaddleOCR-backed integration test (TestEngineIntegration) is
skipped by default because it requires the model weights.  Set
``PP_OCR_ENGINE_TEST_IMAGE=path/to/image.jpg`` to enable it.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Make the package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from pp_ocrv4_mobile_engine import (  # noqa: E402
    DEFAULT_JPG_QUALITY,
    DEFAULT_MAX_LONG,
    OCRResult,
    PaddleMobileEngine,
    preprocess_image,
)
from pp_ocrv4_mobile_engine.engine import _parse_pages  # noqa: E402


# ---------------------------------------------------------------------------
# preprocess_image — pure-Python, no OCR model needed
# ---------------------------------------------------------------------------

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("/tmp/pp_ocrv4_mobile_engine_test")
        self.tmp.mkdir(parents=True, exist_ok=True)

    def test_small_image_passes_through(self):
        # Make a 800x600 RGB image, save as PNG, preprocess.
        src = self.tmp / "small.png"
        cv2.imwrite(str(src), np.zeros((600, 800, 3), dtype=np.uint8))
        out = preprocess_image(src, max_long=4096, jpg_quality=90)
        self.assertTrue(os.path.exists(out))
        img = cv2.imread(out)
        # Since longest edge (800) < 4096, no resize.
        self.assertEqual(img.shape, (600, 800, 3))

    def test_long_image_clipped(self):
        src = self.tmp / "tall.png"
        cv2.imwrite(str(src), np.zeros((2400, 3000, 3), dtype=np.uint8))
        out = preprocess_image(src, max_long=2048, jpg_quality=90)
        img = cv2.imread(out)
        # 3000 long edge -> 2048; scale = 2048/3000 ≈ 0.6827
        self.assertEqual(img.shape[1], 2048)  # width
        # height = 2400 * 0.6827 ≈ 1638
        self.assertAlmostEqual(img.shape[0], 2400 * 2048 / 3000, delta=2)

    def test_jpg_quality_in_output_file_size(self):
        # Output JPEG should differ between q=60 and q=95.
        src = self.tmp / "mid.png"
        cv2.imwrite(str(src), (np.random.rand(500, 500, 3) * 255).astype(np.uint8))
        low = Path(preprocess_image(src, out_tmp_path=self.tmp / "low.jpg",
                                    jpg_quality=60))
        high = Path(preprocess_image(src, out_tmp_path=self.tmp / "high.jpg",
                                     jpg_quality=95))
        self.assertGreater(high.stat().st_size, low.stat().st_size)

    def test_defaults_sensible(self):
        self.assertEqual(DEFAULT_MAX_LONG, 4096)
        self.assertEqual(DEFAULT_JPG_QUALITY, 90)


# ---------------------------------------------------------------------------
# _parse_pages — defensive parsing of PaddleOCR's nested output
# ---------------------------------------------------------------------------

class TestParsePages(unittest.TestCase):
    def test_none_input(self):
        self.assertEqual(_parse_pages(None), ([], []))

    def test_typical_nested_list(self):
        # Real PaddleOCR 2.x shape: [[[(box, (text, score)), ...]]]
        raw = [
            [
                [[[10, 20], [100, 20], [100, 50], [10, 50]],
                 ("hello", 0.97)],
                [[[10, 60], [200, 60], [200, 90], [10, 90]],
                 ("world", 0.88)],
            ]
        ]
        lines, boxes = _parse_pages(raw)
        self.assertEqual(lines, ["hello", "world"])
        self.assertEqual([round(b.conf, 2) for b in boxes], [0.97, 0.88])
        self.assertEqual(boxes[0].bbox, ((10.0, 20.0), (100.0, 20.0),
                                        (100.0, 50.0), (10.0, 50.0)))

    def test_handles_no_detection(self):
        # PaddleOCR returns [None] on images where det finds nothing.
        raw = [None]
        self.assertEqual(_parse_pages(raw), ([], []))

    def test_handles_skipped_lines(self):
        # Some lines might be None, malformed, or have wrong shape.
        raw = [
            [None,
             [[[0, 0], [10, 0], [10, 10], [0, 10]], ("kept", 0.5)],
             "this is not a valid row",  # will be skipped
             [[[0, 0], [5, 0], [5, 5], [0, 5]], "missing-score-row"],  # skipped
            ]
        ]
        lines, boxes = _parse_pages(raw)
        self.assertEqual(lines, ["kept"])
        self.assertEqual(len(boxes), 1)


# ---------------------------------------------------------------------------
# Engine validation — no PaddleOCR init required
# ---------------------------------------------------------------------------

class TestEngineValidation(unittest.TestCase):
    def test_invalid_kind(self):
        with self.assertRaises(ValueError):
            PaddleMobileEngine(engine_kind="flagship_v9")

    def test_invalid_jpg_quality(self):
        with self.assertRaises(ValueError):
            PaddleMobileEngine(jpg_quality=120)
        with self.assertRaises(ValueError):
            PaddleMobileEngine(jpg_quality=30)

    def test_max_long_too_small(self):
        with self.assertRaises(ValueError):
            PaddleMobileEngine(max_long=64)


# ---------------------------------------------------------------------------
# Integration — only when an actual image is available
# ---------------------------------------------------------------------------

class TestEngineIntegration(unittest.TestCase):
    """Runs the real engine on a provided image; skipped otherwise.

    Set ``PP_OCR_ENGINE_TEST_IMAGE=/path/to/image.jpg`` to enable.
    Uses mobile model only (smaller download) and only checks structural
    fields (no F1 assertion here).
    """

    IMAGE_ENV = "PP_OCR_ENGINE_TEST_IMAGE"

    @classmethod
    def setUpClass(cls):
        cls.image = os.environ.get(cls.IMAGE_ENV)
        if cls.image is None:
            raise unittest.SkipTest(f"{cls.IMAGE_ENV} not set")

    def test_mobile_returns_ocr_result(self):
        engine = PaddleMobileEngine(engine_kind="mobile", use_gpu=False)
        result = engine.recognize(self.image)
        self.assertIsInstance(result, OCRResult)
        self.assertIsInstance(result.lines, list)
        self.assertIsInstance(result.boxes, list)
        self.assertGreaterEqual(len(result.lines), 0)
        self.assertGreater(result.elapsed_s, 0.0)
        for box in result.boxes:
            self.assertEqual(len(box.bbox), 4)


if __name__ == "__main__":
    unittest.main()
