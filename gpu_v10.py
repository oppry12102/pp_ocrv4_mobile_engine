"""GPU V10 ONNX engine — 复刻 agentocr/workspace/gpu_v10_ocr.py 的效果.

为什么这里有这个模块
--------------------
原始的 ``PaddleMobileEngine`` 走 PaddleOCR 2.x 的 CPU 路径 —— 单图约 2.4s
(mobile) / 27s (server), F1 在 RCTW-171 50-149 上只能到 0.45 左右。
``gpu_v10`` 是 agentocr 团队在同一硬件上把 PP-OCRv4 转成 ONNX 并用
MIGraphX (AMD ROCm) 跑出来的 GPU 路径, 关键点:

* **同一套精度**: det 用 ``ch_PP-OCRv4_det_mobile.onnx`` (32-align resize
  + 固定 canvas + 原版 DB 后处理), rec 用 ``ch_PP-OCRv4_rec_server.onnx``
  (CTC decode + drop_score 过滤), cls 用 ``cls_mobile.onnx`` (180° 翻转).
  在 RCTW-171 image_50..149 100 张上: F1=0.7008, 0.24s/img — 比
  CPU 路径快 ~10×, F1 高 ~0.25 绝对值.
* **det_size=1920 固定 canvas**: MIGraphX 要求静态 shape, 所以 det 输入
  永远是 (1, 3, 1920, 1920), 缩放后 pad 到黑边; 32-align 是为了保证
  unpool/conv stride 不溢出.
* **dual-width rec batching**: 短行用 480 宽 (与生产一致), 长行用 1280
  宽 (修复密集文档掉字). 长行那条分支 lazy compile (~870s), 第一次遇到
  长框时才编译, 之后无开销.

依赖
----
* ``onnxruntime>=1.18`` (含 MIGraphX / ROCM provider)
* ``pyclipper`` (unclip 距离扩展)
* ``shapely`` ( ``Polygon.area / .length`` )
* ``opencv-python`` 与 ``numpy``

fallback
--------
MIGraphX 不可用时自动 fallback 到 CPUExecutionProvider, 仅失去加速, 精度
保持不变.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_DET_ONNX",
    "DEFAULT_REC_ONNX",
    "DEFAULT_CLS_ONNX",
    "DEFAULT_DICT_PATH",
    "DET_LIMIT",
    "REC_H",
    "CLS_H",
    "CLS_W",
    "BOX_THRESH",
    "DB_THRESH",
    "UNCLIP_RATIO",
    "MAX_CANDIDATES",
    "DROP_SCORE",
    "CLS_THRESH",
    "REC_BATCH",
    "GPUV10Engine",
    "GPUV10Error",
]


# ---------------------------------------------------------------------------
# Default model paths (override per-instance)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR: str = "/home/oppry/work/tools/agentocr/workspace/models/onnx"
"""ONNX 模型默认目录.  用构造函数的 ``model_dir`` 可覆盖 (e.g. 放到本项目
的 ``models/onnx/`` 软链)."""
DEFAULT_DET_ONNX: str = "ch_PP-OCRv4_det_mobile.onnx"
DEFAULT_REC_ONNX: str = "ch_PP-OCRv4_rec_server.onnx"
DEFAULT_CLS_ONNX: str = "cls_mobile.onnx"
DEFAULT_DICT_PATH: str = (
    "/home/oppry/.local/lib/python3.12/site-packages/paddleocr/"
    "ppocr/utils/ppocr_keys_v1.txt"
)

# 调参常量 (gpu_v10 训练/验证过的 sweet spot, 不要随便改)
DET_LIMIT: int = 1920
REC_H: int = 48
CLS_H: int = 48
CLS_W: int = 192
BOX_THRESH: float = 0.7       # MIGraphX 概率偏高 ~0.1-0.15, 比 CPU 高
DB_THRESH: float = 0.3
UNCLIP_RATIO: float = 2.0
MAX_CANDIDATES: int = 2000
DROP_SCORE: float = 0.5
CLS_THRESH: float = 0.9
REC_BATCH: int = 6            # 推理时 batch=6, MIGraphX 一次性编译该 shape

# 推理时的 REC 宽度分支
REC_W_SHORT: int = 480
REC_W_LONG: int = 1280
REC_LONG_RATIO: float = 10.0  # crop 宽高比 > 该值 -> 长行


class GPUV10Error(RuntimeError):
    """Raised when the GPU V10 engine fails on a single image."""


# ---------------------------------------------------------------------------
# Process-global cache.  MIRRORS PaddleMobileEngine's behaviour so that
# constructing the same engine twice does NOT trigger another ~570s
# MIGraphX compile.  Keyed by the tuple of constructor kwargs that affect
# the underlying ONNX sessions.
# ---------------------------------------------------------------------------
_GPU_V10_CACHE: dict[tuple, "GPUV10Engine"] = {}


def _get_or_build_gpu_v10(**kwargs) -> "GPUV10Engine":
    cacheable = (
        kwargs.get("model_dir"),
        kwargs.get("det_onnx"),
        kwargs.get("rec_onnx"),
        kwargs.get("cls_onnx"),
        kwargs.get("dict_path"),
        kwargs.get("det_size"),
        kwargs.get("use_gpu"),
        kwargs.get("show_log"),
    )
    cached = _GPU_V10_CACHE.get(cacheable)
    if cached is not None:
        return cached
    instance = GPUV10Engine(**kwargs)
    _GPU_V10_CACHE[cacheable] = instance
    return instance


# ---------------------------------------------------------------------------
# Optional imports — keep these at module-load time so callers get a clear
# ImportError pointing at the missing dep, instead of a confusing traceback
# deep inside _ensure_rec_long().
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort  # noqa: F401
except ImportError as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "GPUV10Engine needs onnxruntime (>=1.18 recommended for MIGraphX). "
        "Install with: pip install onnxruntime"
    ) from e
try:
    import pyclipper  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "GPUV10Engine needs pyclipper for DB unclip.  pip install pyclipper"
    ) from e
try:
    from shapely.geometry import Polygon  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "GPUV10Engine needs shapely for polygon area/length.  pip install shapely"
    ) from e


# ---------------------------------------------------------------------------
# Lazy-imported dataclass / engine façade to avoid a circular import at the
# package level — engine.py will import GPUV10Engine from here.
# ---------------------------------------------------------------------------

def _make_ocr_result(image_path: str, lines, boxes, elapsed_s: float):
    """Build an OCRResult without importing engine.py at top level.

    Kept here to keep gpu_v10.py independent of the package layout (useful
    for the gpu_v10_eval.py direct-import workflow in agentocr).
    """
    # local import: only happens once per call
    from .engine import OCRResult, OCRLine
    return OCRResult(
        image_path=image_path,
        lines=lines,
        boxes=boxes,
        elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GPUV10Engine:
    """GPU V10 ONNX engine — same API surface as :class:`PaddleMobileEngine`.

    Parameters
    ----------
    use_gpu : bool
        Pass ``True`` (default) to try MIGraphXExecutionProvider first.
        Falls back to ROCM, then CPU.  Pass ``False`` to force CPU only.
    model_dir : str
        Directory containing ``ch_PP-OCRv4_det_mobile.onnx``,
        ``ch_PP-OCRv4_rec_server.onnx`` and ``cls_mobile.onnx``.  Defaults
        to the path used by agentocr (see :data:`DEFAULT_MODEL_DIR`).
    det_size : int
        Detection input canvas side.  1920 is the validated sweet spot;
        smaller values lose detail on tall signs.
    det_onnx / rec_onnx / cls_onnx : str
        Per-head override of the model filename inside ``model_dir``.
    dict_path : str
        Path to ``ppocr_keys_v1.txt``.  Used by rec to map CTC indices to
        strings.  Defaults to the file bundled with paddleocr.
    max_long, jpg_quality : int
        Identical to ``PaddleMobileEngine`` — long-edge clip + JPEG
        recompression before sending the image into ONNX.  Defaults
        match the GPU V10 baseline (4096 / 90).
    show_log : bool
        Print provider/init info to stderr.
    """

    ENGINE_KIND = "gpu_v10"
    """Identifier matching :attr:`PaddleMobileEngine.VALID_KINDS`."""

    def __init__(
        self,
        use_gpu: bool = True,
        *,
        model_dir: str = DEFAULT_MODEL_DIR,
        det_onnx: str = DEFAULT_DET_ONNX,
        rec_onnx: str = DEFAULT_REC_ONNX,
        cls_onnx: str = DEFAULT_CLS_ONNX,
        dict_path: str = DEFAULT_DICT_PATH,
        det_size: int = DET_LIMIT,
        max_long: int = 4096,
        jpg_quality: int = 90,
        show_log: bool = False,
    ) -> None:
        if not 50 <= jpg_quality <= 100:
            raise ValueError(
                f"jpg_quality must be 50..100 (got {jpg_quality!r})."
            )
        if det_size < 256:
            raise ValueError(f"det_size must be >= 256 (got {det_size!r}).")

        self.use_gpu = use_gpu
        self.model_dir = model_dir
        self.det_size = int(det_size)
        self.max_long = int(max_long)
        self.jpg_quality = int(jpg_quality)
        self.show_log = show_log

        det_path = str(Path(model_dir) / det_onnx)
        rec_path = str(Path(model_dir) / rec_onnx)
        cls_path = str(Path(model_dir) / cls_onnx)
        for p in (det_path, rec_path, cls_path):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"GPU V10 model not found: {p}.  "
                    f"Pass model_dir= pointing at a directory containing "
                    f"{det_onnx}, {rec_onnx}, {cls_onnx}."
                )

        # Provider selection: MIGraphX → ROCM → CPU.  MIGraphX is the
        # ROCm-graph-compiler backend that fuses + compiles the graph;
        # ROCM is the legacy operator-by-operator path.  Both yield the
        # same numerics (within ~1e-3) so we accept either.
        if use_gpu:
            available = ort.get_available_providers()
            wanted = []
            if "MIGraphXExecutionProvider" in available:
                wanted.append("MIGraphXExecutionProvider")
            if "ROCMExecutionProvider" in available:
                wanted.append("ROCMExecutionProvider")
            wanted.append("CPUExecutionProvider")
            providers = wanted
        else:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        t0 = time.time()
        self.det = ort.InferenceSession(det_path, providers=providers, sess_options=opts)
        self.rec = ort.InferenceSession(rec_path, providers=providers, sess_options=opts)
        self.cls = ort.InferenceSession(cls_path, providers=providers, sess_options=opts)

        # Warmup — MIGraphX lazily compiles on first call; do a no-op
        # inference here so the cost is paid at __init__ time, not on
        # the first user request.
        self.det.run(None, {"x": np.zeros((1, 3, det_size, det_size), dtype=np.float32)})
        self.rec.run(None, {"x": np.zeros((REC_BATCH, 3, REC_H, REC_W_SHORT), dtype=np.float32)})
        self.cls.run(None, {"x": np.zeros((REC_BATCH, 3, CLS_H, CLS_W), dtype=np.float32)})

        self.init_time = time.time() - t0
        self.det_provider = self.det.get_providers()[0]
        self.rec_provider = self.rec.get_providers()[0]
        self.cls_provider = self.cls.get_providers()[0]

        # 长行 1280 编译是懒触发
        self._rec_long_ready = False
        self._rec_long_lock = threading.Lock()

        # dict 加载
        if not Path(dict_path).exists():
            raise FileNotFoundError(
                f"rec dictionary not found: {dict_path}.  "
                f"Pass dict_path= pointing at ppocr_keys_v1.txt."
            )
        with open(dict_path, encoding="utf-8") as f:
            self.chars = ["blank"] + [line.rstrip("\n") for line in f]

        if show_log:
            sys.stderr.write(
                f"[gpu_v10] init={self.init_time:.1f}s "
                f"det={self.det_provider} rec={self.rec_provider} "
                f"cls={self.cls_provider}\n"
            )

    # ------------------------------------------------------------------
    # Lazy compile of the long-width rec branch (one-time, ~870s on
    # MIGraphX).  Thread-safe so concurrent recognizers don't both pay
    # the cost.
    # ------------------------------------------------------------------
    def _ensure_rec_long(self) -> None:
        if self._rec_long_ready:
            return
        with self._rec_long_lock:
            if self._rec_long_ready:
                return
            t0 = time.time()
            self.rec.run(
                None,
                {"x": np.zeros((REC_BATCH, 3, REC_H, REC_W_LONG), dtype=np.float32)},
            )
            self._rec_long_ready = True
            if self.show_log:
                sys.stderr.write(
                    f"[gpu_v10] dual-width: 1280 rec compiled in "
                    f"{time.time() - t0:.0f}s\n"
                )

    # ------------------------------------------------------------------
    # det — 32-align resize + pad 到固定 canvas
    # ------------------------------------------------------------------
    def _det_preprocess(self, img: np.ndarray):
        h, w = img.shape[:2]
        ratio = self.det_size / max(h, w) if max(h, w) > self.det_size else 1.0
        rh = max(int(round(int(h * ratio) / 32) * 32), 32)
        rw = max(int(round(int(w * ratio) / 32) * 32), 32)
        rh, rw = min(rh, self.det_size), min(rw, self.det_size)
        resized = cv2.resize(img, (rw, rh))
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        canvas = np.full(
            (self.det_size, self.det_size, 3),
            (0.0 - mean) / std,
            dtype=np.float32,
        )
        canvas[:rh, :rw] = x
        canvas = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        return canvas, rh / float(h), rw / float(w)

    # ------------------------------------------------------------------
    # det 后处理 (原版 DB)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    @staticmethod
    def _unclip(box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance))

    def _det_postprocess(self, pred, ratio_h, ratio_w, src_h, src_w):
        bitmap = (pred[0, 0] > DB_THRESH).astype(np.uint8)
        contours, _ = cv2.findContours(
            bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for contour in contours[:MAX_CANDIDATES]:
            points, sside = self._get_mini_boxes(contour)
            if sside < 3:
                continue
            points = np.array(points)
            score = self._box_score_fast(pred[0, 0], points.reshape(-1, 2))
            if score < BOX_THRESH:
                continue
            box = self._unclip(points, UNCLIP_RATIO).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < 5:
                continue
            box = np.array(box, dtype=np.float32)
            box[:, 0] = np.clip(np.round(box[:, 0] / ratio_w), 0, src_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / ratio_h), 0, src_h)
            boxes.append(box.astype(np.int32))
        return boxes

    # ------------------------------------------------------------------
    # 排序 + 透视裁剪
    # ------------------------------------------------------------------
    @staticmethod
    def _sorted_boxes(dt_boxes):
        _boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        for i in range(len(_boxes) - 1):
            for j in range(i, -1, -1):
                if (
                    abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                    and _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
                else:
                    break
        return _boxes

    @staticmethod
    def _get_rotate_crop(img, points):
        w = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        h = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        w, h = max(w, 1), max(h, 1)
        pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
        dst = cv2.warpPerspective(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dh, dw = dst.shape[:2]
        if dh * 1.0 / dw >= 1.5:
            dst = np.rot90(dst)
        return dst

    # ------------------------------------------------------------------
    # cls / rec preprocess
    # ------------------------------------------------------------------
    @staticmethod
    def _cls_resize_norm(img):
        h, w = img.shape[:2]
        ratio = w * 1.0 / h
        if math.ceil(CLS_H * ratio) > CLS_W:
            resized_w = CLS_W
        else:
            resized_w = int(math.ceil(CLS_H * ratio))
        resized = (
            cv2.resize(img, (resized_w, CLS_H))
            .astype("float32")
            .transpose(2, 0, 1)
            / 255.0
        )
        resized -= 0.5
        resized /= 0.5
        out = np.zeros((3, CLS_H, CLS_W), dtype=np.float32)
        out[:, :, :resized_w] = resized
        return out

    @staticmethod
    def _rec_resize_norm(img, batch_w):
        h, w = img.shape[:2]
        ratio = w * 1.0 / h
        resized_w = min(int(math.ceil(REC_H * ratio)), batch_w)
        resized = (
            cv2.resize(img, (resized_w, REC_H))
            .astype("float32")
            .transpose(2, 0, 1)
            / 255.0
        )
        resized -= 0.5
        resized /= 0.5
        out = np.zeros((3, REC_H, batch_w), dtype=np.float32)
        out[:, :, :resized_w] = resized
        return out

    # ------------------------------------------------------------------
    # CTC decode
    # ------------------------------------------------------------------
    def _ctc_decode(self, pred):
        """Return (text, score).  pred has shape (T, C) softmax probabilities."""
        probs = pred[0]
        idx = np.argmax(probs, axis=-1)
        out, prev, scores = [], -1, []
        for t, i in enumerate(idx):
            if i != 0 and i != prev and i < len(self.chars):
                out.append(self.chars[i])
                scores.append(probs[t, i])
            prev = i
        text = "".join(out)
        score = float(np.mean(scores)) if scores else 0.0
        return text, score

    # ------------------------------------------------------------------
    # Main entry — same shape as PaddleMobileEngine.recognize
    # ------------------------------------------------------------------
    def recognize(
        self,
        image: str | os.PathLike,
        *,
        max_long: int | None = None,
        jpg_quality: int | None = None,
    ):
        """Run OCR on one image.

        Parameters
        ----------
        image
            Path to an image readable by ``cv2.imread``.
        max_long, jpg_quality
            Per-call overrides.  ``None`` reuses the engine default.
            When supplied, the image is preprocessed (downscale + JPEG
            recompression) to ``/tmp/gpu_v10_engine_<hash>.jpg`` so the
            model sees the same input distribution as it was validated on.

        Returns
        -------
        :class:`pp_ocrv4_mobile_engine.OCRResult` — same shape as
        :meth:`PaddleMobileEngine.recognize`.
        """
        # 预处理 (复用 PaddleMobileEngine 的 preprocess_image, 行为一致)
        from .engine import preprocess_image  # local import 避免循环

        ml = self.max_long if max_long is None else int(max_long)
        jq = self.jpg_quality if jpg_quality is None else int(jpg_quality)

        image_str = str(image)
        try:
            tmp = preprocess_image(image_str, max_long=ml, jpg_quality=jq)
        except Exception as e:
            raise GPUV10Error(f"preprocess failed: {image_str}: {e}") from e

        img = cv2.imread(tmp)
        if img is None:
            raise GPUV10Error(f"cv2.imread returned None: {tmp}")

        t0 = time.time()
        h, w = img.shape[:2]

        # det
        din, ratio_h, ratio_w = self._det_preprocess(img)
        pred = self.det.run(None, {"x": din})[0]
        boxes = self._det_postprocess(pred, ratio_h, ratio_w, h, w)
        boxes = self._sorted_boxes(boxes)

        if not boxes:
            return _make_ocr_result(image_str, [], [], time.time() - t0)

        # 透视裁剪
        crops = [self._get_rotate_crop(img, b) for b in boxes]

        # cls (180°), batch 补零到 REC_BATCH 防止 MIGraphX 重编译
        cls_imgs = [self._cls_resize_norm(c) for c in crops]
        for beg in range(0, len(cls_imgs), REC_BATCH):
            chunk = cls_imgs[beg:beg + REC_BATCH]
            while len(chunk) < REC_BATCH:
                chunk.append(np.zeros((3, CLS_H, CLS_W), dtype=np.float32))
            cls_batch = np.stack(chunk)
            cls_out = self.cls.run(None, {"x": cls_batch})[0]
            for j in range(min(REC_BATCH, len(cls_imgs) - beg)):
                label_idx = int(np.argmax(cls_out[j]))
                score = float(cls_out[j].max())
                if label_idx == 1 and score > CLS_THRESH:
                    crops[beg + j] = cv2.rotate(crops[beg + j], 1)

        # rec — dual-width 分桶, 短行 480 长行 1280
        bw = []
        for c in crops:
            ch, cw = c.shape[:2]
            bw.append(REC_W_LONG if cw / max(ch, 1) > REC_LONG_RATIO else REC_W_SHORT)

        rec_hits = []  # (box_idx, text, score)
        for target_w in (REC_W_SHORT, REC_W_LONG):
            idxs = [i for i, x in enumerate(bw) if x == target_w]
            if not idxs:
                continue
            if target_w == REC_W_LONG:
                self._ensure_rec_long()
            for beg in range(0, len(idxs), REC_BATCH):
                chunk_ids = idxs[beg:beg + REC_BATCH]
                rec_imgs = [self._rec_resize_norm(crops[i], target_w) for i in chunk_ids]
                while len(rec_imgs) < REC_BATCH:
                    rec_imgs.append(np.zeros((3, REC_H, target_w), dtype=np.float32))
                batch = np.stack(rec_imgs)
                rp = self.rec.run(None, {"x": batch})[0]
                for j, i in enumerate(chunk_ids):
                    t, sc = self._ctc_decode(rp[j:j + 1])
                    if t and sc >= DROP_SCORE:
                        rec_hits.append((i, t, float(sc)))

        # 恢复 box 顺序
        rec_hits.sort(key=lambda x: x[0])

        from .engine import OCRLine
        lines: list[str] = []
        box_results: list[OCRLine] = []
        for i, t, sc in rec_hits:
            lines.append(t)
            bbox = tuple(
                (float(p[0]), float(p[1])) for p in boxes[i].tolist()
            )
            box_results.append(OCRLine(text=t, conf=sc, bbox=bbox))

        elapsed = time.time() - t0
        return _make_ocr_result(image_str, lines, box_results, elapsed)

    # ------------------------------------------------------------------
    # Batch (sequential — matches PaddleMobileEngine.recognize_batch)
    # ------------------------------------------------------------------
    def recognize_batch(
        self,
        images: Iterable[str | os.PathLike],
        *,
        max_long: int | None = None,
        jpg_quality: int | None = None,
        on_error: str = "empty",
    ) -> list:
        """Sequentially OCR a batch of images.  Mirrors
        :meth:`PaddleMobileEngine.recognize_batch` semantics (``on_error``
        in ``{"empty", "raise"}``)."""
        if on_error not in ("empty", "raise"):
            raise ValueError("on_error must be 'empty' or 'raise'")

        from .engine import OCRResult
        results: list[OCRResult] = []
        try:
            n_total = len(images)  # type: ignore[arg-type]
        except TypeError:
            n_total = 0

        for i, img in enumerate(images, 1):
            try:
                r = self.recognize(img, max_long=max_long, jpg_quality=jpg_quality)
            except GPUV10Error as e:
                print(f"[gpu_v10] {i}: {e}", file=sys.stderr)
                if on_error == "raise":
                    raise
                r = OCRResult(image_path=str(img), lines=[], boxes=[], elapsed_s=0.0)
            results.append(r)
            print(
                f"[gpu_v10] {i}/{n_total or '?'}: {Path(r.image_path).name} "
                f"-> {len(r.lines)} lines in {r.elapsed_s:.2f}s",
                file=sys.stderr,
            )
        return results

    # ------------------------------------------------------------------
    # Convenience helpers — mirror PaddleMobileEngine
    # ------------------------------------------------------------------
    def recognize_text(self, image, **kw) -> list[str]:
        """Just ``result.lines`` — convenience wrapper."""
        return self.recognize(image, **kw).lines

    def recognize_batch_text(
        self,
        images: Iterable[str | os.PathLike],
        **kw,
    ) -> dict[str, list[str]]:
        """Batch text-only output, ``dict[basename -> list[str]]`` —
        matches the format consumed by ``score_rctw_universal.py``."""
        results = self.recognize_batch(images, **kw)
        return {Path(r.image_path).name: r.lines for r in results}

    def __repr__(self) -> str:
        return (
            f"GPUV10Engine(det={self.det_provider}, rec={self.rec_provider}, "
            f"cls={self.cls_provider}, det_size={self.det_size}, "
            f"max_long={self.max_long}, jpg_quality={self.jpg_quality})"
        )