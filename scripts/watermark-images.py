#!/usr/bin/env python3
"""
Add a visible copyright watermark to all images in static/img/.
Usage: python3 scripts/watermark-images.py
Requires: pip install Pillow
"""
import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required. Run: pip3 install Pillow")
    exit(1)

# Your name for the watermark
COPYRIGHT_TEXT = "© Varchaleswari Ganugapati"
IMG_DIR = Path(__file__).resolve().parent.parent / "static" / "img"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# Skip tiny icons / UI assets if you add any later
SKIP_NAMES = {"favicon.ico", "favicon.png"}


def add_watermark(image_path: Path) -> bool:
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"  Skip {image_path.name}: {e}")
        return False

    w, h = img.size
    # Scale font with image size (roughly 2–3% of min dimension)
    font_size = max(12, min(w, h) // 40)
    padding = max(8, min(w, h) // 80)

    # Try a nice font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Get text bbox for positioning (bottom-right)
    bbox = draw.textbbox((0, 0), COPYRIGHT_TEXT, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = w - tw - padding
    y = h - th - padding

    # Semi-transparent dark background strip for readability
    draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 2], fill=(0, 0, 0, 140))
    draw.text((x, y), COPYRIGHT_TEXT, fill=(255, 255, 255, 230), font=font)

    out = Image.alpha_composite(img, overlay)
    # Save back as original format (no alpha for JPEG)
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        out = out.convert("RGB")
    out.save(image_path, quality=92, optimize=True)
    return True


def main():
    if not IMG_DIR.is_dir():
        print(f"Image directory not found: {IMG_DIR}")
        return
    count = 0
    for path in sorted(IMG_DIR.iterdir()):
        if path.name in SKIP_NAMES or path.suffix not in EXTENSIONS:
            continue
        print(f"Watermarking: {path.name}")
        if add_watermark(path):
            count += 1
    print(f"Done. Watermarked {count} image(s).")


if __name__ == "__main__":
    main()
