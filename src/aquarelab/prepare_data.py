"""
Resize a folder of images once, before training, so data loading is fast.

Run from the project root:

    uv run python -m aquarelab.prepare_data datasets/coco/train2017 datasets/coco/train2017_320

Images are resized so the shorter side is 320 px. Smaller images are kept as
they are. Grayscale images stay grayscale, so the training loader can still
skip them. Files already in the output folder are skipped, so the script can be
stopped and restarted.
"""

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from aquarelab.constants import IMAGE_EXTENSIONS


def resize_one(task: tuple[Path, Path, int, int]) -> str:
    """Resize one image. Returns 'ok', 'skipped' or 'failed'."""
    src, dst, short_side, quality = task
    if dst.exists():
        return "skipped"
    try:
        with Image.open(src) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            width, height = img.size
            scale = short_side / min(width, height)
            if scale < 1:
                new_size = (round(width * scale), round(height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, quality=quality)
        return "ok"
    except (UnidentifiedImageError, OSError):
        return "failed"


def main():
    parser = argparse.ArgumentParser(
        description="Resize a folder of images for training."
    )
    parser.add_argument("src", type=Path, help="Folder with the original images.")
    parser.add_argument("dst", type=Path, help="Folder for the resized images.")
    parser.add_argument("--short-side", type=int, default=320)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    sources = sorted(
        p for p in args.src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    tasks = [
        (
            p,
            (args.dst / p.relative_to(args.src)).with_suffix(".jpg"),
            args.short_side,
            args.quality,
        )
        for p in sources
    ]
    print(f"Found {len(tasks)} images. Resizing with {args.workers} workers...")

    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(resize_one, tasks, chunksize=64), start=1):
            counts[result] += 1
            if i % 5000 == 0 or i == len(tasks):
                print(
                    f"  {i}/{len(tasks)} | ok {counts['ok']} | skipped {counts['skipped']} "
                    f"| failed {counts['failed']}"
                )

    print(f"Done. Resized images are in {args.dst}")


if __name__ == "__main__":
    main()
