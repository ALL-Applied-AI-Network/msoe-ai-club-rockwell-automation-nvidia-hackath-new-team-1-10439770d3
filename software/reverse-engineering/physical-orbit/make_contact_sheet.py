"""Create labeled contact sheets for reconstruction input/output QA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=320)
    args = parser.parse_args()
    paths = sorted([*args.images.glob("*.png"), *args.images.glob("*.jpg")])
    if not paths:
        raise RuntimeError(f"No images in {args.images}")
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = args.thumb_width / image.shape[1]
        resized = cv2.resize(image, (args.thumb_width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (resized.shape[1], 26), (12, 18, 28), -1)
        cv2.putText(resized, path.stem[:40], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        thumbs.append(resized)
    cell_h = max(item.shape[0] for item in thumbs)
    rows = math.ceil(len(thumbs) / args.columns)
    canvas = np.full((rows * cell_h, args.columns * args.thumb_width, 3), 245, np.uint8)
    for index, item in enumerate(thumbs):
        row, column = divmod(index, args.columns)
        canvas[row * cell_h:row * cell_h + item.shape[0], column * args.thumb_width:(column + 1) * args.thumb_width] = item
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    print(f"Wrote {args.out} ({len(thumbs)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
