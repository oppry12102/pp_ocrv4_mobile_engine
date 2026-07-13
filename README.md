# pp-ocrv4-mobile-engine

**A small, well-wrapped production OCR interface around PaddleOCR's PP-OCRv4 (mobile or server).**

PP-OCRv4 (PaddlePaddle) is the strongest **CPU-only** OCR stack for mixed
Chinese/English scene text. This wrapper bakes in the preprocessing
sweet spots we measured on RCTW first-10, gives you a `recognize()` /
`recognize_batch()` API with typed returns, and lets you flip between
the 12 MB *mobile* model and the 450 MB *server* model with a single
constructor flag.

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

# Mobile — 12 MB, ~2.4 s/img on CPU, F1 ≈ 0.45 on RCTW
engine = PaddleMobileEngine()
result = engine.recognize("sign.jpg")
print(result.lines)         # list[str]
print(result.boxes)         # list[OCRLine] with text/conf/bbox
print(f"{result.elapsed_s:.2f}s, {len(result.lines)} lines")

# Server — 450 MB, ~27 s/img on CPU, F1 ≈ 0.55 on RCTW
engine_srv = PaddleMobileEngine(engine_kind="server")
print(engine_srv.recognize("sign.jpg").lines)
```

Batch API returns the same shape:

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

## CLI

```bash
python -m pp_ocrv4_mobile_engine --image photo.jpg --kind mobile
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind mobile --out results.json
python -m pp_ocrv4_mobile_engine --image-dir ./imgs --kind server --jsonl --out results.jsonl
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

## Performance on RCTW (10 imgs, 44 GT keywords)

| engine_kind | F1_s  | F1_l  | hits/GT | FP   | s/img | model size |
|-------------|------:|------:|--------:|-----:|------:|-----------:|
| **mobile (CPU, this module)** | 0.4494 | 0.4835 | 20/44 | 25 | 2.42 | 12 MB |
| server (CPU) | **0.5532** | 0.5532 | 26/44 | 24 | 27.30 | 450 MB |

Difference: **+0.10 F1 absolute** for an **11× slower, 38× heavier**
model. Both are CPU-only here — PaddlePaddle has its own CUDA path but
this wrapper does not bundle it.

## Limitations

- **No GPU acceleration.** PaddlePaddle's CUDA/CNCL stack is shipped
  separately; this wrapper sticks to the stable CPU path. If you have a
  CUDA card, follow Paddle's own instructions for `paddleocr.PaddleOCR`
  with `use_gpu=True` — and consider whether this wrapper earns its
  keep at all (it adds almost nothing over raw PaddleOCR on GPU).
- **PaddleOCR 2.7 only.** Other versions sometimes change which
  checkpoint `lang="ch"` auto-selects. Pass
  `det_model_dir`/`rec_model_dir`/`cls_model_dir` explicitly if you
  need a specific version.
- **No streaming.** `recognize_batch` is a sequential loop (PaddleOCR's
  CLI is not thread-safe in 2.7). Parallelise at the process boundary
  (`xargs -P`, `multiprocessing.Pool`) if you need throughput.
- **Det/cls/jpg_quality are tuned for PaddleOCR's own preferences.** If
  you train a new det head on different JPEG distributions, re-do the
  sweep.

## Files

| file                       | purpose                                                |
|----------------------------|--------------------------------------------------------|
| `engine.py`                | the wrapper (single file, ~340 lines)                  |
| `__init__.py`              | public API re-exports                                  |
| `cli.py` + `__main__.py`   | `python -m pp_ocrv4_mobile_engine` interface           |
| `examples/basic_usage.py`  | minimal runnable demo                                  |
| `examples/benchmark_rctw.py` | reproduces the F1/sweet-spot benchmark on RCTW first-10 |
| `tests/test_engine.py`     | unit tests (preprocess + OCR parsing, no GPU needed)   |

## License

Apache-2.0 (matches PaddleOCR's own license).
