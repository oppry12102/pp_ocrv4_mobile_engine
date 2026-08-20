# pp_ocrv4_mobile_engine

> **PP-OCRv4 — wrapped four ways, one `recognize()` API.**

PP-OCRv4 (PaddlePaddle) is the strongest CPU-only OCR stack for mixed
Chinese / English scene text. This wrapper exposes **four runtime paths**
through a single typed `recognize()` / `recognize_batch()` API and lets
you flip between them with one constructor flag.

> **Major credit:** the `v6` and `gpu_v10` engines — and most of the
> benchmarking numbers in this README — were derived from the
> [`agentocr`](https://github.com/anthropic-experiments/agentocr)
> project's multi-week ablation sweep on RCTW-171 (PRODUCTION_GUIDE.md,
> gpu_v10_ocr.py, dense_2560_experiment.py, verify_v6_100_omp1.py).
> See [Acknowledgments](#acknowledgments).

| `engine_kind`   | what it is                                        | hardware | F1 (RCTW)    | s/img  | models         |
|-----------------|---------------------------------------------------|----------|--------------|--------|----------------|
| `mobile`        | legacy PaddleOCR mobile (default 960, thresh 0.6) | CPU      | 0.4494       | 2.4    | 12 MB          |
| `server`        | legacy PaddleOCR server                           | CPU      | 0.5532       | 27.3   | 450 MB         |
| **`v6`** ⭐     | mobile det @ 1920 + server rec + thresh 0.3       | CPU      | **0.7511**   | 9.3    | 462 MB hybrid  |
| `gpu_v10`       | ONNX Runtime + MIGraphX (CPU fallback)            | ROCm GPU | 0.7008¹      | 0.24   | 95 MB ONNX     |

¹ RCTW-171 image_50..149 (100 imgs). The other three rows are RCTW first-10.

**TL;DR — pick `engine_kind="v6"` for the best CPU F1.** If you have an
AMD ROCm card and ~10 minutes to spare for a one-time MIGraphX compile,
pick `engine_kind="gpu_v10"` for ~10× speed at similar accuracy.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [The four engines](#the-four-engines)
- [Performance](#performance)
- [GPU V10 setup](#gpu-v10-setup)
- [CLI](#cli)
- [Python API](#python-api)
- [Preprocessing parameters](#preprocessing-parameters)
- [Limitations](#limitations)
- [Files](#files)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Install

```bash
pip install paddleocr opencv-python-headless
```

For the `gpu_v10` engine, also install:

```bash
pip install onnxruntime pyclipper shapely
```

Tested with Python 3.12, paddleocr 2.7.3, opencv-python 4.x,
onnxruntime 1.22. The model weights (12 MB mobile / 450 MB server for
the Paddle path; ~95 MB ONNX for `gpu_v10`) download / sit on first use.

> **For `engine_kind="v6"` on CPU, set these env vars before importing:**
> ```bash
> export OMP_NUM_THREADS=1
> export OPENBLAS_NUM_THREADS=1
> export MKL_NUM_THREADS=1
> ```
> PaddleOCR 2.7's OpenMP backend occasionally SIGSEGVs on large images
> when run multithreaded, and single-threaded is empirically faster on
> RCTW first-10 (9.3 s/img vs 13 s/img).

---

## Quick start

```python
from pp_ocrv4_mobile_engine import PaddleMobileEngine

# Recommended CPU path — F1 ≈ 0.75 on RCTW first-10
engine = PaddleMobileEngine(engine_kind="v6")
result = engine.recognize("sign.jpg")
print(result.lines)        # list[str]
print(result.boxes)        # list[OCRLine] with text / conf / bbox
print(f"{result.elapsed_s:.2f}s, {len(result.lines)} lines")

# GPU path — F1 ≈ 0.70 on RCTW-171 50-149, ~40× faster
engine_gpu = PaddleMobileEngine(engine_kind="gpu_v10", use_gpu=True)
print(engine_gpu.recognize("sign.jpg").lines)

# Legacy mobile (kept for backward compat)
engine_legacy = PaddleMobileEngine()  # engine_kind="mobile"
print(engine_legacy.recognize("sign.jpg").lines)
```

Batch and text-only convenience methods work the same across all four
engines:

```python
results = engine.recognize_batch(["01.jpg", "02.jpg", "03.jpg"])
for r in results:
    print(r.image_path, len(r.lines), f"{r.elapsed_s:.2f}s")

# shape matches score_rctw_universal.score_results():
#   {'01.jpg': ['line 1', 'line 2', ...], '02.jpg': [...], ...}
texts = engine.recognize_batch_text(["01.jpg", "02.jpg"])
```

---

## The four engines

### `mobile` (legacy, default)

PaddleOCR's mobile checkpoint with Paddle's own defaults
(`det_limit_side_len=960`, `det_db_box_thresh=0.6`). Kept for backward
compatibility. **F1 ≈ 0.45 on RCTW first-10** — substantially worse than
`v6`, mostly because the default 960 input is too small for high-res
scenes.

### `server` (legacy)

PaddleOCR's server checkpoint. +5% F1 absolute vs `mobile` but ~11×
slower, and the server det head actually loses some small-text recall
compared to mobile det. **F1 ≈ 0.55 on RCTW first-10.**

### `v6` ⭐ — recommended CPU path

The **agentocr** team's RCTW-171 sweep identified that the optimal CPU
config is **not** either pure mobile or pure server, but a **hybrid**:

- `det_model_dir = ch_PP-OCRv4_det_infer` (mobile det, 12 MB)
- `rec_model_dir = ch_PP-OCRv4_rec_server_infer` (server rec, 450 MB)
- `det_limit_side_len = 1920` (was 960)
- `det_db_box_thresh = 0.3` (was 0.6)
- `det_db_unclip_ratio = 2.0`
- `det_max_candidates = 2000`

This is `engine_kind="v6"`. **F1 = 0.7511 on RCTW first-10** at 9.3
s/img — almost 2× the legacy `mobile` F1 for ~4× the time, and still
~3× faster than the legacy `server` kind. The mobile-det + server-rec
hybrid is the core trick; the 1920 + 0.3 recovery adds the rest.

> *Source: `agentocr/workspace/PRODUCTION_GUIDE.md` + the three
> independent ablation sweeps (`dense_2560_experiment.py`,
> `verify_v6_100_omp1.py`, `run_v6minimax_train_50_99.py`) — same
> params, independently re-derived. OMP=1 setting from
> `compare_omp1_lean.py`.*

### `gpu_v10` — ONNX Runtime + MIGraphX

The same PP-OCRv4 family converted to ONNX and run on AMD ROCm via
MIGraphX (with a CPU fallback). **Ported from
[`agentocr/workspace/gpu_v10_ocr.py`](https://github.com/anthropic-experiments/agentocr/blob/main/workspace/gpu_v10_ocr.py)**
with the same DB-postprocess constants, dual-width rec batching
(480/1280), and MIGraphX-compensation box_thresh=0.7.

- det: `ch_PP-OCRv4_det_mobile.onnx` — 32-align resize, fixed 1920
  canvas, original DB postprocess
- rec: `ch_PP-OCRv4_rec_server.onnx` — dual-width batching (480/1280),
  CTC decode with `drop_score=0.5`
- cls: `cls_mobile.onnx` — 0/180° direction classifier
- `box_thresh=0.7` to compensate for MIGraphX's ~0.1-0.15 probability
  offset

**F1 ≈ 0.70 on RCTW-171 50-149 at ~0.24 s/img.** First call pays a
~9-10 min one-time MIGraphX compile; subsequent calls reuse the
compiled sessions via a process-global cache. CPU fallback works
(loses speed, same accuracy).

---

## Performance

Measured on this ROCm box (Linux, paddleocr 2.7.3, single-threaded CPU
where applicable). Same engine settings as the README claims.

| engine_kind       | benchmark              | F1_s    | F1_l    | hits/GT | FP   | s/img  | model size      |
|-------------------|------------------------|--------:|--------:|--------:|-----:|-------:|-----------------|
| `mobile` (legacy) | RCTW first-10          | 0.4494  | 0.4835  | 20/44   | 25   | 2.42   | 12 MB           |
| `server` (legacy) | RCTW first-10          | 0.5532  | 0.5532  | 26/44   | 24   | 27.30  | 450 MB          |
| **`v6`** ⭐       | RCTW first-10          | **0.7511** | **0.7835** | — | — | 9.3 | 462 MB (hybrid) |
| `gpu_v10`         | RCTW-171 50-149 (100)  | **0.7008** | — | — | — | **0.24** | 95 MB ONNX   |

Reproducing the first-10 numbers: `python examples/benchmark_rctw.py`.

---

## GPU V10 setup

`gpu_v10` needs three ONNX files. Default path is the one agentocr uses:

```text
/home/oppry/work/tools/agentocr/workspace/models/onnx/
├── ch_PP-OCRv4_det_mobile.onnx     (4.7 MB)
├── ch_PP-OCRv4_rec_server.onnx     (90 MB)
└── cls_mobile.onnx                 (557 KB)
```

Override via constructor kwarg: `PaddleMobileEngine(engine_kind="gpu_v10", model_dir="/your/path")`.

The default rec dictionary is at

```
/home/oppry/.local/lib/python3.12/site-packages/paddleocr/ppocr/utils/ppocr_keys_v1.txt
```

Override with `dict_path=` if your paddleocr install lives elsewhere.

**First-call cost:** MIGraphX lazily compiles the three ONNX graphs on
first inference; this takes ~9-10 min on ROCm. Both the construction
and the first `recognize()` pay this cost. After that, the engine is
reused via the process-global cache; the long-rec-width (1280) branch
compiles lazily on first long line.

If MIGraphX is unavailable, the engine transparently falls back to
`CPUExecutionProvider` (loses acceleration, keeps accuracy).

---

## CLI

```bash
# single image
python -m pp_ocrv4_mobile_engine --image photo.jpg --kind v6

# batch to JSON
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind v6 --out results.json

# batch to JSONL
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind gpu_v10 --use-gpu \
    --jsonl --out v10.jsonl

# tune preprocessing (still applies to all engines)
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --max-long 2048 --jpg-quality 85

# gpu_v10 with custom model dir
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind gpu_v10 \
    --model-dir /path/to/onnx --det-size 1920
```

Full flag list: `python -m pp_ocrv4_mobile_engine --help`.

---

## Python API

```python
from pp_ocrv4_mobile_engine import (
    PaddleMobileEngine,   # the main facade
    GPUV10Engine,         # the ONNX impl, if you want to bypass the facade
    OCRResult, OCRLine,
    RecognizeError,
    preprocess_image,
    DEFAULT_MAX_LONG,     # 4096
    DEFAULT_JPG_QUALITY,  # 90
    DEFAULT_LANG,         # 'ch'
)
```

### `PaddleMobileEngine(engine_kind, ...)`

Unified facade. Constructor signature:

```python
PaddleMobileEngine(
    engine_kind="mobile",         # 'mobile' | 'server' | 'v6' | 'gpu_v10'
    *,
    lang="ch",
    use_angle_cls=True,
    use_gpu=False,                # 'gpu_v10': MIGraphX/ROCM/CPU; else forwarded to PaddleOCR
    max_long=4096,                # long-edge cap before OCR
    jpg_quality=90,               # JPEG q of preprocessed tile (50-100)
    show_log=False,
    det_model_dir=None,           # override Paddle det dir (Paddle path only)
    rec_model_dir=None,           # override Paddle rec dir
    cls_model_dir=None,           # override Paddle cls dir
    model_dir=None,               # override ONNX dir ('gpu_v10' only)
    det_size=None,                # override det canvas side ('gpu_v10' only)
)
```

### `engine.recognize(image, *, max_long=None, jpg_quality=None) -> OCRResult`

OCR one image.

### `engine.recognize_batch(images, *, max_long=None, jpg_quality=None, on_error="empty") -> list[OCRResult]`

OCR a batch sequentially. `on_error` is `"empty"` (default — failed image
yields empty `OCRResult`) or `"raise"` (re-raise on first failure).

### `engine.recognize_text(image, **kw) -> list[str]`

Convenience: just `result.lines`.

### `engine.recognize_batch_text(images, **kw) -> dict[str, list[str]]`

Convenience: `{basename: lines}`. Matches the shape consumed by
`score_rctw_universal.score_results()`.

### Return type

```python
@dataclass
class OCRResult:
    image_path: str
    lines: list[str]              # list[str], in reading order
    boxes: list[OCRLine]          # list of OCRLine with text/conf/bbox
    elapsed_s: float              # inference time only (preprocess excluded)

@dataclass
class OCRLine:
    text: str
    conf: float                   # recognition confidence
    bbox: tuple[tuple[float, float], ...]   # 4-corner polygon
```

`OCRResult.to_dict()` serialises both fields to a JSON-friendly dict.

---

## Preprocessing parameters

The wrapper always pre-loads + downscales + JPEG-recompresses each image
before sending it to the engine. These two knobs control that step and
apply uniformly to all four engine kinds.

| param         | default | sweet spot | notes                                                       |
|---------------|--------:|------------|-------------------------------------------------------------|
| `max_long`    |    4096 | 4096       | Longest edge of preprocessed image. Only images ≥ this are downscaled. RCTW first-10 has one image (4208×2368) that ever needs downscaling. Lowering loses detail on tall signs; raising adds decode cost for nothing. |
| `jpg_quality` |      90 | 90         | JPEG quality of the preprocessed tile. **Do not exceed 95** — PaddleOCR's det head is sensitive to q≥95 ringing and F1 regresses by ~0.05. |

Per-call override:

```python
engine.recognize("huge_poster.jpg", max_long=8192, jpg_quality=85)
```

Global override (engine constructor):

```python
engine = PaddleMobileEngine(engine_kind="v6", max_long=2048, jpg_quality=80)
```

You can also reuse the preprocessor directly:

```python
from pp_ocrv4_mobile_engine import preprocess_image
tmp_jpg = preprocess_image("big.tif", max_long=2048, jpg_quality=85)
```

---

## Limitations

- **CPU Paddle path is slow by design.** `mobile` / `server` / `v6` all
  run on CPU. Use `engine_kind="gpu_v10"` for ~10× speedup on AMD ROCm,
  or `engine_kind="v6"` for the best CPU F1.
- **PaddleOCR 2.7 only.** Other versions sometimes change which
  checkpoint `lang="ch"` auto-selects. Pass `det_model_dir` /
  `rec_model_dir` / `cls_model_dir` explicitly if you need a specific
  version.
- **GPU V10 first-call compile.** The first
  `PaddleMobileEngine(engine_kind="gpu_v10")` in a process pays a
  one-time MIGraphX compile (~9-10 min on AMD ROCm). After that the
  engine is reused via the process-global cache; the long-rec-width
  (1280) branch is compiled lazily on first long line.
- **No streaming.** `recognize_batch` is a sequential loop. PaddleOCR's
  CLI is not thread-safe in 2.7, and MIGraphX sessions are process
  singletons. Parallelise at the process boundary (`xargs -P`,
  `multiprocessing.Pool`) if you need throughput.
- **Det/cls/jpg_quality are tuned for PaddleOCR's own preferences.** If
  you train a new det head on different JPEG distributions, re-do the
  sweep.

---

## Files

| file                              | purpose                                                   |
|-----------------------------------|-----------------------------------------------------------|
| `engine.py`                       | `PaddleMobileEngine` facade + `_V6_KWARGS` sweet spot     |
| `gpu_v10.py`                      | `GPUV10Engine`: ONNX + MIGraphX (CPU fallback)            |
| `__init__.py`                     | public API re-exports                                     |
| `cli.py` + `__main__.py`          | `python -m pp_ocrv4_mobile_engine` interface              |
| `examples/basic_usage.py`         | minimal runnable demo (Paddle path)                       |
| `examples/basic_usage_gpu_v10.py` | minimal runnable demo (GPU V10 path)                      |
| `examples/benchmark_rctw.py`      | reproduces the F1 / sweet-spot benchmark on RCTW first-10 |
| `tests/test_engine.py`            | unit tests (preprocess + OCR parsing + V6 static config)  |

Tests: `PYTHONPATH=. python -m unittest discover -s tests -p 'test_engine.py'`
→ 17 tests, all pass (2 skipped for env-gated integration).

---

## Acknowledgments

Most of the engineering in this repo is **ported from
[`agentocr`](https://github.com/anthropic-experiments/agentocr)** — a
multi-week ablation sweep on RCTW-171 that systematically worked out
PP-OCRv4's CPU and GPU sweet spots. Without that work, this wrapper
would still be running with PaddleOCR's defaults (F1 ≈ 0.45).

Specific sources, by engine:

| this repo                                | upstream                                                                |
|------------------------------------------|-------------------------------------------------------------------------|
| `engine_kind="v6"` (CPU optimal config)  | `agentocr/workspace/PRODUCTION_GUIDE.md` + the three independent ablation sweeps (`dense_2560_experiment.py`, `verify_v6_100_omp1.py`, `run_v6minimax_train_50_99.py`). OMP=1 setting from `compare_omp1_lean.py`. |
| `engine_kind="gpu_v10"` (ONNX + MIGraphX) | `agentocr/workspace/gpu_v10_ocr.py` — det preprocess (32-align + 1920 canvas), DB postprocess, dual-width rec batching (480/1280), MIGraphX box_thresh compensation, OMP-aware warmup. Eval methodology from `gpu_v10_eval.py`. |
| `max_long=4096`, `jpg_quality=90`        | Originally measured in `pp_ocrv4_mobile_engine`'s own RCTW first-10 sweep (pre-`agentocr` history). |
| `engine_kind="mobile"` / `"server"`      | Vanilla PaddleOCR 2.7 defaults — kept for backward compatibility only. |

If you're working on PP-OCRv4 / PP-OCRv5 / PaddleOCR variants, the
agentocr project has a much deeper analysis (LLM postprocessing layers,
image-feature routing, MiniMax-M3 fusion experiments) that this small
wrapper deliberately does **not** reproduce — it focuses only on the
local-CPU / local-GPU inference path.

Thanks also to:

- **PaddlePaddle / PaddleOCR team** for the underlying PP-OCRv4 models
  and `PaddleOCR` Python API.
- **ONNX Runtime + AMD MIGraphX teams** for the GPU inference path.

---

## License

Apache-2.0 (matches PaddleOCR's own license).