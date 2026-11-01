"""Prepare fair image-only reconstruction inputs from the physical wrist orbit.

This module never reads the measured CAD.  It decodes the real RGB wrist video,
finds stable high-quality frames, derives color/silhouette masks from the visible
blue housing, and writes raw, masked, alpha, and crop variants plus a manifest.

The selected frame is the sharpest low-motion frame in each temporal bin.  This
matches the physical orbit controller, which pauses at discrete viewpoints,
without assuming access to simulator object poses or depth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


# Executed 18 unique IK-limited orbit endpoints in the clean 85 s clip.  The
# duplicate first-angle loop-closure occurs at 54.9 s and is stored separately.
ORBIT_ENDPOINT_TIMES_S = [
    12.4,
    13.5,
    16.0,
    18.6,
    21.1,
    24.1,
    25.4,
    29.2,
    31.3,
    33.9,
    36.3,
    38.9,
    41.0,
    42.4,
    43.9,
    47.0,
    50.0,
    52.1,
]
ORBIT_LOOP_CLOSURE_TIME_S = 54.9


@dataclass
class FrameMetric:
    frame: int
    time_s: float
    sharpness: float
    motion: float
    mask_area_fraction: float
    bbox: list[int]
    border_touch: bool
    score: float = 0.0


def largest_component(binary: np.ndarray, min_area: int = 250) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary)
    candidates = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)]
    area, label = max(candidates)
    if area < min_area:
        return np.zeros_like(binary)
    return np.where(labels == label, 255, 0).astype(np.uint8)


def object_mask(frame: np.ndarray) -> np.ndarray:
    """Segment the blue housing; fill its external silhouette.

    The green gear is contained within that silhouette, so filling the blue
    housing contour correctly retains the complete visible object including the
    gear holes.  This deliberately avoids any learned model or CAD prior.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (88, 45, 22), (145, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue = largest_component(blue)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(blue)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def mask_bbox(mask: np.ndarray) -> tuple[list[int], bool]:
    points = cv2.findNonZero(mask)
    if points is None:
        return [0, 0, 0, 0], True
    x, y, w, h = cv2.boundingRect(points)
    height, width = mask.shape
    touch = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
    return [int(x), int(y), int(w), int(h)], bool(touch)


def normalized(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [5, 95])
    if hi <= lo + 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def analyze_video(video: Path, start_s: float, end_s: float) -> tuple[list[np.ndarray], list[FrameMetric], float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    start_frame = max(0, int(round(start_s * fps)))
    end_frame = int(round(end_s * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    frame_ids: list[int] = []
    while True:
        frame_id = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id >= end_frame:
            break
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        frame_ids.append(frame_id)
    capture.release()
    if len(frames) < 3:
        raise RuntimeError("The selected interval has too few frames")

    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    metrics: list[FrameMetric] = []
    for index, (frame_id, frame, gray) in enumerate(zip(frame_ids, frames, grays)):
        mask = object_mask(frame)
        bbox, border = mask_bbox(mask)
        area_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap[mask > 0].var()) if np.count_nonzero(mask) > 50 else 0.0
        prev_gray = grays[max(0, index - 1)]
        next_gray = grays[min(len(grays) - 1, index + 1)]
        motion = 0.5 * (
            float(cv2.absdiff(prev_gray, gray).mean())
            + float(cv2.absdiff(next_gray, gray).mean())
        )
        metrics.append(
            FrameMetric(
                frame=frame_id,
                time_s=frame_id / fps,
                sharpness=sharpness,
                motion=motion,
                mask_area_fraction=area_fraction,
                bbox=bbox,
                border_touch=border,
            )
        )

    sharp = normalized(np.log1p(np.array([m.sharpness for m in metrics])))
    stable = 1.0 - normalized(np.array([m.motion for m in metrics]))
    areas = np.array([m.mask_area_fraction for m in metrics])
    area_good = np.clip(areas / 0.035, 0.0, 1.0)
    for index, metric in enumerate(metrics):
        metric.score = float(0.46 * sharp[index] + 0.44 * stable[index] + 0.10 * area_good[index])
        if metric.border_touch or metric.mask_area_fraction < 0.005:
            metric.score -= 1.0
    return frames, metrics, fps


def choose_binned(metrics: list[FrameMetric], count: int) -> list[int]:
    edges = np.linspace(0, len(metrics), count + 1).astype(int)
    chosen: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        candidates = range(left, max(left + 1, right))
        chosen.append(max(candidates, key=lambda idx: metrics[idx].score))
    return sorted(set(chosen))


def choose_endpoint_frames(metrics: list[FrameMetric], times_s: list[float], search_radius_s: float = 0.25) -> list[int]:
    """Choose a sharp stable frame at each measured orbit endpoint."""

    chosen: list[int] = []
    for target in times_s:
        candidates = [
            index for index, metric in enumerate(metrics)
            if abs(metric.time_s - target) <= search_radius_s
        ]
        if not candidates:
            candidates = [min(range(len(metrics)), key=lambda idx: abs(metrics[idx].time_s - target))]
        chosen.append(max(candidates, key=lambda idx: metrics[idx].score))
    return chosen


def square_crop(frame: np.ndarray, mask: np.ndarray, bbox: list[int], size: int = 512) -> tuple[np.ndarray, np.ndarray, dict]:
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    side = int(math.ceil(max(w, h) * 1.45))
    side = max(side, 32)
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - width), max(0, y1 - height)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    crop = frame[y0c:y1c, x0c:x1c]
    crop_mask = mask[y0c:y1c, x0c:x1c]
    crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    crop_mask = cv2.copyMakeBorder(crop_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
    crop_mask = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    transform = {
        "source_crop_xyxy": [x0, y0, x1, y1],
        "padding_ltrb": [pad_left, pad_top, pad_right, pad_bottom],
        "output_size": [size, size],
        "source_to_crop_scale": size / float(side),
    }
    return crop, crop_mask, transform


def write_set(root: Path, name: str, frames: list[np.ndarray], metrics: list[FrameMetric], chosen: list[int]) -> list[dict]:
    set_root = root / name
    for folder in ("raw", "masked", "alpha", "masks", "crops", "crop_masks"):
        (set_root / folder).mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for output_index, local_index in enumerate(chosen):
        frame = frames[local_index]
        metric = metrics[local_index]
        mask = object_mask(frame)
        stem = f"{output_index:03d}_f{metric.frame:06d}_t{metric.time_s:08.3f}"
        cv2.imwrite(str(set_root / "raw" / f"{stem}.png"), frame)
        white = np.full_like(frame, 255)
        masked = np.where(mask[..., None] > 0, frame, white)
        cv2.imwrite(str(set_root / "masked" / f"{stem}.png"), masked)
        cv2.imwrite(str(set_root / "masks" / f"{stem}.png"), mask)
        alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        alpha[:, :, 3] = mask
        cv2.imwrite(str(set_root / "alpha" / f"{stem}.png"), alpha)
        crop, crop_mask, transform = square_crop(frame, mask, metric.bbox)
        crop_white = np.full_like(crop, 255)
        crop_masked = np.where(crop_mask[..., None] > 0, crop, crop_white)
        cv2.imwrite(str(set_root / "crops" / f"{stem}.png"), crop_masked)
        cv2.imwrite(str(set_root / "crop_masks" / f"{stem}.png"), crop_mask)
        record = asdict(metric)
        record.update({"output_index": output_index, "stem": stem, "crop_transform": transform})
        manifest.append(record)
    with (set_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=12.1)
    parser.add_argument("--end-s", type=float, default=55.2)
    parser.add_argument("--counts", type=int, nargs="+", default=[18, 36, 60])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames, metrics, fps = analyze_video(args.video, args.start_s, args.end_s)
    all_rows: list[dict] = []
    summary = {
        "source_video": str(args.video.resolve()),
        "fps": fps,
        "interval_s": [args.start_s, args.end_s],
        "decoded_frames": len(frames),
        "selection": "sharpest low-motion frame in each equal temporal bin",
        "depth_used": False,
        "cad_used": False,
        "sets": {},
    }
    endpoint_indices = choose_endpoint_frames(metrics, ORBIT_ENDPOINT_TIMES_S)
    endpoint_records = write_set(args.out, "orbit_endpoints_18", frames, metrics, endpoint_indices)
    closure_index = choose_endpoint_frames(metrics, [ORBIT_LOOP_CLOSURE_TIME_S])[0]
    write_set(args.out, "orbit_loop_closure_1", frames, metrics, [closure_index])
    summary["orbit_endpoint_times_s"] = ORBIT_ENDPOINT_TIMES_S
    summary["loop_closure_time_s"] = ORBIT_LOOP_CLOSURE_TIME_S
    summary["sets"]["orbit_endpoints_18"] = {
        "count": len(endpoint_records),
        "mean_sharpness": float(np.mean([r["sharpness"] for r in endpoint_records])),
        "mean_motion": float(np.mean([r["motion"] for r in endpoint_records])),
        "mean_mask_area_fraction": float(np.mean([r["mask_area_fraction"] for r in endpoint_records])),
        "first_time_s": endpoint_records[0]["time_s"],
        "last_time_s": endpoint_records[-1]["time_s"],
    }
    all_rows.extend({"set": "orbit_endpoints_18", **record} for record in endpoint_records)
    for count in args.counts:
        chosen = choose_binned(metrics, count)
        records = write_set(args.out, f"keyframes_{count}", frames, metrics, chosen)
        summary["sets"][str(count)] = {
            "count": len(records),
            "mean_sharpness": float(np.mean([r["sharpness"] for r in records])),
            "mean_motion": float(np.mean([r["motion"] for r in records])),
            "mean_mask_area_fraction": float(np.mean([r["mask_area_fraction"] for r in records])),
            "first_time_s": records[0]["time_s"],
            "last_time_s": records[-1]["time_s"],
        }
        all_rows.extend({"set": str(count), **record} for record in records)
    with (args.out / "preparation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    fieldnames = ["set", "output_index", "frame", "time_s", "sharpness", "motion", "mask_area_fraction", "bbox", "border_touch", "score", "stem", "crop_transform"]
    with (args.out / "selected_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
