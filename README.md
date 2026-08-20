# pp-ocrv4-mobile-engine

**A small, well-wrapped production OCR interface around PP-OCRv4
(mobile / server / v6 via PaddleOCR, or `gpu_v10` via ONNX Runtime + MIGraphX).**

PP-OCRv4 (PaddlePaddle) is the strongest **CPU** OCR stack for mixed
Chinese/English scene text. The `v6` engine (mobile det @ 1920 + server
rec + box_thresh=0.3, ported from agentocr's PRODUCTION_GUIDE) reaches
**F1 ≈ 0.75 on RCTW first-10 on CPU alone** — almost 2× the legacy
`mobile` default.  The same checkpoints converted to ONNX and run on AMD
ROCm via MIGraphX (`gpu_v10`) hit **~10× the speed** at comparable
accuracy. This wrapper exposes all four paths through one `recognize()`
/ `recognize_batch()` API with typed returns, and lets you flip between
them with a single constructor flag.

## Why this wrapper exists

When we first wrapped PaddleOCR with five lines of inline code we
discovered two things that hurt F1 and five things that bit us in
production:

- **`max_long=4096` is the sweet spot, not higher.** RCTW first-10 has
  one image (`rctw_default_05.jpg`) longer than 4096 px on its long edge.
  All other images are passed through at native resolution. So raising
  the cap above 4096 changes nothing; lowering it loses detail on tall
  signs.
- **JPEG quality ≥ 95 actively regresses F1 by ~0.05.** PaddleOCR's det
  head was trained on images whose JPEG compression artifacts live in
  the q=70-90 range. Pushing to q=95 introduces subtle ringing that
  the NMS picks up as extra boxes.

Beyond F1:

- `PaddleOCR(...)` takes ~5-30 s to build (it downloads & caches
  det/rec/cls weights on first call). Wrapping it in a process-global
  cache makes `PaddleMobileEngine()` essentially free on subsequent
  constructions.
- The raw return shape (`[page][line] = (box, (text, score))` lists of
  lists with a couple of corner cases) is awkward to consume. The
  wrapper exposes `OCRResult` with `.lines: list[str]` and `.boxes:
  list[OCRLine]`.
- Switching to the **server** model (F1 0.553 on RCTW) requires pointing
  `PaddleOCR` at the heavier `ch_PP-OCRv4_{det,rec}_server_infer`
  directories. The wrapper does that based on `engine_kind="server"`.

## Install

```bash
pip install paddleocr opencv-python-headless
# `opencv-python` is fine if you also need to draw boxes etc.
```

Tested versions:

- Python 3.12
- paddleocr 2.7.3 (lang="ch" gives you PP-OCRv4 mobile by default)
- opencv-python 4.x

The model weights (12 MB mobile / 450 MB server) download on first use
to `~/.paddleocr/whl/`.

## Quick start

```python
from pp_ocrv4_mobile_engine import PaddleMobileEngine

# Legacy mobile — 12 MB, ~2.4 s/img on CPU, F1 ≈ 0.45 on RCTW first-10
engine = PaddleMobileEngine()
result = engine.recognize("sign.jpg")
print(result.lines)         # list[str]
print(result.boxes)         # list[OCRLine] with text/conf/bbox
print(f"{result.elapsed_s:.2f}s, {len(result.lines)} lines")

# Legacy server — 450 MB, ~27 s/img on CPU, F1 ≈ 0.55 on RCTW first-10
engine_srv = PaddleMobileEngine(engine_kind="server")
print(engine_srv.recognize("sign.jpg").lines)

# V6 (recommended CPU) — mobile det @ 1920 + server rec, ~9 s/img,
# F1 ≈ 0.75 on RCTW first-10.  Almost 2x the legacy mobile F1 for ~4x
# the time.  Add the suggested env vars below for best determinism.
engine_v6 = PaddleMobileEngine(engine_kind="v6", use_gpu=False)
print(engine_v6.recognize("sign.jpg").lines)

# GPU V10 — ONNX Runtime + MIGraphX, F1 ≈ 0.70 on RCTW-171 50-149
# at ~0.24 s/img.  Requires onnxruntime + pyclipper + shapely
# (`pip install onnxruntime pyclipper shapely`) and the three
# `*.onnx` files on disk (see "GPU V10 setup" below).
engine_v10 = PaddleMobileEngine(engine_kind="gpu_v10", use_gpu=True)
print(engine_v10.recognize("sign.jpg").lines)
```

Batch API returns the same shape across all four engines:

```python
results = engine.recognize_batch(
    ["img/01.jpg", "img/02.jpg", "img/03.jpg"],
)
for r in results:
    print(r.image_path, len(r.lines), f"{r.elapsed_s:.2f}s")
```

For text-only consumers, there is a convenience `recognize_batch_text()`
returning `dict[basename -> list[str]]`:

```python
texts = engine.recognize_batch_text(["img/01.jpg", "img/02.jpg"])
# {'01.jpg': [...], '02.jpg': [...]}
```

This dict shape matches what `score_rctw_universal.score_results()`
reads, so you can pipe engine output straight into the scorer.

### Recommended env for `engine_kind="v6"`

PaddleOCR 2.7's OpenMP backend occasionally SIGSEGVs on large images
when run with multiple threads.  Single-threaded is also empirically
faster on RCTW first-10 (agentocr measured 9.3 vs 13 s/img):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## CLI

```bash
python -m pp_ocrv4_mobile_engine --image photo.jpg --kind mobile
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind mobile --out results.json
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind server --jsonl --out results.jsonl
OMP_NUM_THREADS=1 python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind v6 --out v6.json
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind gpu_v10 --use-gpu --jsonl --out v10.jsonl
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --max-long 2048 --jpg-quality 85
```

(See `cli.py`.)

## Preprocessing parameters

| param        | default | sweep-tested sweet spot | what it does                                     |
|--------------|--------:|-------------------------|--------------------------------------------------|
| `max_long`   |    4096 | 4096                    | Longest edge of preprocessed image (only img ≥ this gets downscaled). Lowering loses detail on tall signs; raising adds decode cost for nothing. |
| `jpg_quality`|      90 | 90                      | JPEG quality of preprocessed tile. **Do not exceed 95.** |

Override per call:

```python
engine.recognize("huge_poster.jpg", max_long=8192, jpg_quality=85)
```

or globally:

```python
engine = PaddleMobileEngine(engine_kind="mobile", max_long=2048, jpg_quality=80)
```

## Performance

| engine_kind | benchmark | F1_s  | F1_l  | hits/GT | FP   | s/img | model size |
|-------------|-----------|------:|------:|--------:|-----:|------:|-----------:|
| **mobile** (legacy) | RCTW first-10 | 0.4494 | 0.4835 | 20/44 | 25 | 2.42 | 12 MB |
| server (legacy)     | RCTW first-10 | 0.5532 | 0.5532 | 26/44 | 24 | 27.30 | 450 MB |
| **v6** ⭐ (CPU)     | RCTW first-10 | **0.7511** | 0.7835 | — | — | 9.3 | 462 MB (hybrid) |
| **gpu_v10**         | RCTW-171 50-149 (100 imgs) | 0.7008 | — | — | — | 0.24 | 95 MB ONNX |

`v6` (recommended CPU): mobile det + server rec + 1920 input +
`det_db_box_thresh=0.3` + `det_db_unclip_ratio=2.0` + 2000 candidates.
Same PP-OCRv4 family, very different parametrisation.  +0.30 F1 absolute
over the legacy mobile default for ~4× the cost — still ~3× faster than
the legacy server kind.  Set `OMP_NUM_THREADS=1` for full benefit.

`gpu_v10` (ONNX + MIGraphX): the same model family converted to ONNX and
run on AMD ROCm — det_mobile + rec_server + cls head, dual-width rec
batching (480/1280), 32-align resize + fixed 1920-canvas pad for det,
original DB postprocessing with `box_thresh=0.7` to compensate for
MIGraphX's ~0.1-0.15 probability offset. **Compared to `mobile` (CPU):
~10× faster, +0.25 F1 absolute on RCTW-171.** CPU fallback works (loses
speed, same accuracy).

### GPU V10 setup

```bash
pip install onnxruntime pyclipper shapely opencv-python numpy

# Three ONNX files.  Default path is the one agentocr uses:
#   /home/oppry/work/tools/agentocr/workspace/models/onnx/
#   ├─ ch_PP-OCRv4_det_mobile.onnx     (4.7 MB)
#   ├─ ch_PP-OCRv4_rec_server.onnx     (90 MB)
#   └─ cls_mobile.onnx                 (557 KB)
# Override via constructor kwarg: model_dir=/path/to/your/onnx/
```

The default rec dictionary is at
`/home/oppry/.local/lib/python3.12/site-packages/paddleocr/ppocr/utils/ppocr_keys_v1.txt`
— override via `dict_path=` if you have a different paddleocr install.

The first `PaddleMobileEngine(engine_kind="gpu_v10")` pays a one-time
MIGraphX compile cost (~9-10 min).  Subsequent constructions in the
same process reuse the compiled sessions — no re-compile.

## Limitations

- **CPU Paddle path is slow by design.** `mobile` / `server` / `v6`
  all run on CPU.  Use `engine_kind="gpu_v10"` for ~10× speedup on AMD
  ROCm, or `engine_kind="v6"` for the best CPU F1.
- **PaddleOCR 2.7 only.** Other versions sometimes change which
  checkpoint `lang="ch"` auto-selects. Pass
  `det_model_dir`/`rec_model_dir`/`cls_model_dir` explicitly if you
  need a specific version.
- **GPU V10 first-call compile.** The first `PaddleMobileEngine(engine_kind="gpu_v10")`
  in a process pays a one-time MIGraphX compile (~9-10 min on AMD ROCm).
  After that the engine is reused via the process-global cache; the
  long-rec-width (1280) branch is compiled lazily on first long line.
- **No streaming.** `recognize_batch` is a sequential loop (PaddleOCR's
  CLI is not thread-safe in 2.7, and MIGraphX sessions are process
  singletons). Parallelise at the process boundary (`xargs -P`,
  `multiprocessing.Pool`) if you need throughput.
- **Det/cls/jpg_quality are tuned for PaddleOCR's own preferences.** If
  you train a new det head on different JPEG distributions, re-do the
  sweep.

## Files

| file                       | purpose                                                |
|----------------------------|--------------------------------------------------------|
| `engine.py`                | PaddleMobileEngine facade (single file, ~610 lines)    |
| `gpu_v10.py`               | GPU V10 ONNX engine (det_mobile + rec_server + cls)    |
| `__init__.py`              | public API re-exports                                  |
| `cli.py` + `__main__.py`   | `python -m pp_ocrv4_mobile_engine` interface           |
| `examples/basic_usage.py`  | minimal runnable demo (Paddle path)                    |
| `examples/basic_usage_gpu_v10.py` | minimal runnable demo (GPU V10 path)             |
| `examples/benchmark_rctw.py` | reproduces the F1/sweet-spot benchmark on RCTW first-10 |
| `tests/test_engine.py`     | unit tests (preprocess + OCR parsing, no GPU needed)   |

## License

Apache-2.0 (matches PaddleOCR's own license).
