"""Minimal example: OCR a single image and print detected lines + boxes."""
from pp_ocrv4_mobile_engine import PaddleMobileEngine

# Lazy: PaddleOCR downloads the model on first use (~30s for mobile, ~1m for server).
engine = PaddleMobileEngine()  # default = mobile on CPU

result = engine.recognize("sign.jpg")
print(f"image:    {result.image_path}")
print(f"elapsed:  {result.elapsed_s:.2f}s")
print(f"lines:    {len(result.lines)}")
print("---")
for line in result.lines:
    print(line)
print("---")
print("boxes:")
for box in result.boxes:
    print(f"  text={box.text!r}, conf={box.conf:.3f}, bbox={box.bbox}")
