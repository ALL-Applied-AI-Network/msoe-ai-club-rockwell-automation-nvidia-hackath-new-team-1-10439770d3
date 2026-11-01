"""Encode saved Isaac orbit RGB observations into a compact MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--fps", type=float, default=18.0)
args = parser.parse_args()

root = Path(args.dataset) / "observations"
paths = sorted(root.glob("*_rgb.png"))
if not paths:
    raise SystemExit("no RGB frames found")
first = cv2.imread(str(paths[0]))
height, width = first.shape[:2]
writer = cv2.VideoWriter(str(Path(args.out)), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
for path in paths:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(path)
    writer.write(frame)
writer.release()
print(f"wrote {len(paths)} frames to {args.out}")
