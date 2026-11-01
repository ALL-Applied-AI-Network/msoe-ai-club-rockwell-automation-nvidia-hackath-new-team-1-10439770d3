"""Reproducible hand-tracking benchmark for the six demonstration videos.

The benchmark intentionally uses only packages and model weights already present
on this workstation.  It compares detector-only, temporally-smoothed, motion-
gated, KCF, and CSRT OpenCV baselines on two representative clips.  If requested,
it also evaluates the locally cached SAM2.1 Hiera Small video model with an
automatically generated first-frame box prompt.

The reported proxy-overlap metrics use the classical skin mask as a consistency
signal, not as ground-truth accuracy.  Runtime, detection uptime, dropouts, box
count, and temporal motion statistics are measured directly.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "input"
ANNOTATED_DIR = HERE / "annotated"
QA_DIR = HERE / "qa"

SAMPLES = (
    ("overhead_multihand", "PXL_20261020_165311341.mp4", 0.0, 12.0),
    ("closeup_singlehand", "PXL_20261020_165414567.mp4", 0.0, 12.0),
)

COLORS = {
    "skin_ycrcb_hsv": (31, 177, 255),
    "skin_temporal": (80, 215, 106),
    "motion_gated_skin": (255, 157, 65),
    "kcf_reinit": (237, 99, 174),
    "csrt_reinit": (132, 104, 255),
    "sam2_prompted": (64, 220, 190),
}


@dataclass
class Result:
    boxes: list[tuple[int, int, int, int]]
    mask: np.ndarray | None
    elapsed_ms: float


def package_inventory() -> dict[str, object]:
    modules = [
        "cv2",
        "mediapipe",
        "ultralytics",
        "torch",
        "torchvision",
        "onnxruntime",
        "transformers",
        "supervision",
    ]
    available = {name: importlib.util.find_spec(name) is not None for name in modules}
    versions: dict[str, str] = {}
    for name, present in available.items():
        if not present:
            continue
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "present"))
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[name] = f"present, import error: {exc}"

    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--sam2.1-hiera-small"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "available": available,
        "versions": versions,
        "opencv_trackers": {
            "KCF": hasattr(cv2, "TrackerKCF_create"),
            "CSRT": hasattr(cv2, "TrackerCSRT_create"),
            "MOSSE": hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"),
        },
        "sam2_hiera_small_cache_present": cache.exists(),
    }


def video_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(SOURCE_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        rows.append(
            {
                "file": path.name,
                "width": width,
                "height": height,
                "fps": fps,
                "frames": frames,
                "duration_s": frames / fps if fps else None,
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def read_sample(path: Path, start_s: float, duration_s: float, target_fps: float = 12.0, width: int = 960) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    start_frame = int(round(start_s * source_fps))
    end_frame = int(round((start_s + duration_s) * source_fps))
    stride = max(1, int(round(source_fps / target_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    index = start_frame
    while index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (index - start_frame) % stride == 0:
            height = int(round(frame.shape[0] * width / frame.shape[1]))
            frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
        index += 1
    cap.release()
    return frames


def skin_mask(frame: np.ndarray) -> np.ndarray:
    """Conservative skin proxy tuned for these fixed, white-table recordings."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycc = cv2.inRange(ycrcb, np.array([20, 125, 70], np.uint8), np.array([255, 185, 150], np.uint8))
    hsv_mask = cv2.inRange(hsv, np.array([0, 18, 45], np.uint8), np.array([28, 205, 255], np.uint8))
    mask = cv2.bitwise_and(ycc, hsv_mask)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    # The red/orange cubes overlap the broad skin chroma range.  Hands and
    # forearms are much larger connected regions in this fixed overhead view;
    # remove small isolated regions before using color as a hand proposal.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape
    clean = np.zeros_like(mask)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_edge = x <= 2 or y <= 2 or x + component_width >= width - 2 or y + component_height >= height - 2
        # In these fixed-camera demonstrations, every real hand is attached to
        # a forearm entering through a frame boundary.  A fully interior region
        # must therefore be very large before it is accepted; this rejects a
        # cluster of red cubes in the bin that otherwise resembles skin chroma.
        minimum = width * height * (0.0022 if touches_edge else 0.025)
        if area >= minimum:
            clean[labels == label] = 255
    return clean


def boxes_from_mask(mask: np.ndarray, max_boxes: int = 2) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    min_area = max(900.0, w * h * 0.0024)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
        if len(boxes) >= max_boxes:
            break
    return boxes


class SkinMethod:
    name = "skin_ycrcb_hsv"

    def process(self, frame: np.ndarray, frame_idx: int) -> Result:
        start = time.perf_counter()
        mask = skin_mask(frame)
        boxes = boxes_from_mask(mask)
        return Result(boxes, mask, (time.perf_counter() - start) * 1000)


class TemporalSkinMethod:
    name = "skin_temporal"

    def __init__(self) -> None:
        self.tracks: list[dict[str, object]] = []

    @staticmethod
    def _center(box: np.ndarray) -> np.ndarray:
        return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], dtype=np.float32)

    def process(self, frame: np.ndarray, frame_idx: int) -> Result:
        start = time.perf_counter()
        mask = skin_mask(frame)
        detections = [np.array(b, dtype=np.float32) for b in boxes_from_mask(mask)]
        if not self.tracks:
            self.tracks = [{"box": box, "misses": 0} for box in detections]
        else:
            candidates: list[tuple[float, int, int]] = []
            for ti, track in enumerate(self.tracks):
                for di, detection in enumerate(detections):
                    distance = float(np.linalg.norm(self._center(track["box"]) - self._center(detection)))
                    candidates.append((distance, ti, di))
            assigned_tracks: set[int] = set()
            assigned_detections: set[int] = set()
            for distance, ti, di in sorted(candidates):
                if distance > 260 or ti in assigned_tracks or di in assigned_detections:
                    continue
                previous = self.tracks[ti]["box"]
                self.tracks[ti]["box"] = previous * 0.42 + detections[di] * 0.58
                self.tracks[ti]["misses"] = 0
                assigned_tracks.add(ti)
                assigned_detections.add(di)
            for ti, track in enumerate(self.tracks):
                if ti not in assigned_tracks:
                    track["misses"] = int(track["misses"]) + 1
            for di, detection in enumerate(detections):
                if di not in assigned_detections and len(self.tracks) < 2:
                    self.tracks.append({"box": detection, "misses": 0})
            self.tracks = [track for track in self.tracks if int(track["misses"]) <= 4]
        boxes = [tuple(int(round(v)) for v in track["box"]) for track in self.tracks]
        return Result(boxes, mask, (time.perf_counter() - start) * 1000)


class MotionGatedSkinMethod:
    name = "motion_gated_skin"

    def __init__(self) -> None:
        self.background = cv2.createBackgroundSubtractorMOG2(history=70, varThreshold=18, detectShadows=False)

    def process(self, frame: np.ndarray, frame_idx: int) -> Result:
        start = time.perf_counter()
        skin = skin_mask(frame)
        motion = self.background.apply(frame)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        motion = cv2.dilate(motion, np.ones((25, 25), np.uint8), iterations=2)
        mask = cv2.bitwise_and(skin, motion)
        boxes = boxes_from_mask(mask)
        return Result(boxes, mask, (time.perf_counter() - start) * 1000)


class OpenCVTrackerMethod:
    def __init__(self, kind: str, reinit_interval: int = 36) -> None:
        self.kind = kind.upper()
        self.name = f"{kind.lower()}_reinit"
        self.reinit_interval = reinit_interval
        self.trackers: list[object] = []

    def _new_tracker(self):
        if self.kind == "KCF":
            return cv2.TrackerKCF_create()
        if self.kind == "CSRT":
            return cv2.TrackerCSRT_create()
        raise ValueError(self.kind)

    def _initialize(self, frame: np.ndarray) -> None:
        self.trackers = []
        for box in boxes_from_mask(skin_mask(frame)):
            tracker = self._new_tracker()
            tracker.init(frame, tuple(int(v) for v in box))
            self.trackers.append(tracker)

    def process(self, frame: np.ndarray, frame_idx: int) -> Result:
        start = time.perf_counter()
        if not self.trackers or frame_idx % self.reinit_interval == 0:
            self._initialize(frame)
        boxes: list[tuple[int, int, int, int]] = []
        survivors: list[object] = []
        for tracker in self.trackers:
            ok, box = tracker.update(frame)
            if not ok:
                continue
            x, y, width, height = (int(round(v)) for v in box)
            if width > 0 and height > 0:
                boxes.append((x, y, width, height))
                survivors.append(tracker)
        self.trackers = survivors
        return Result(boxes, None, (time.perf_counter() - start) * 1000)


class Metrics:
    def __init__(self, name: str) -> None:
        self.name = name
        self.frame_times: list[float] = []
        self.valid: list[bool] = []
        self.counts: list[int] = []
        self.area_fractions: list[float] = []
        self.proxy_recall: list[float] = []
        self.proxy_precision: list[float] = []
        self.centers: list[tuple[float, float] | None] = []

    @staticmethod
    def prediction_mask(result: Result, shape: tuple[int, int]) -> np.ndarray:
        if result.mask is not None:
            return result.mask > 0
        prediction = np.zeros(shape, dtype=np.uint8)
        for x, y, width, height in result.boxes:
            cv2.rectangle(prediction, (x, y), (x + width, y + height), 1, -1)
        return prediction > 0

    def add(self, result: Result, proxy: np.ndarray) -> None:
        prediction = self.prediction_mask(result, proxy.shape)
        proxy_bool = proxy > 0
        intersection = np.logical_and(prediction, proxy_bool).sum()
        self.frame_times.append(result.elapsed_ms)
        self.valid.append(bool(result.boxes))
        self.counts.append(len(result.boxes))
        self.area_fractions.append(float(prediction.mean()))
        self.proxy_recall.append(float(intersection / max(1, proxy_bool.sum())))
        self.proxy_precision.append(float(intersection / max(1, prediction.sum())))
        if result.boxes:
            x, y, width, height = max(result.boxes, key=lambda b: b[2] * b[3])
            self.centers.append((x + width / 2, y + height / 2))
        else:
            self.centers.append(None)

    def summary(self, width: int, height: int) -> dict[str, float | int | str]:
        valid_centers = [center for center in self.centers if center is not None]
        steps: list[float] = []
        diagonal = math.hypot(width, height)
        previous = None
        for center in self.centers:
            if center is not None and previous is not None:
                steps.append(math.dist(center, previous) / diagonal)
            previous = center
        dropouts = sum(1 for prev, current in zip(self.valid, self.valid[1:]) if prev and not current)
        elapsed = sum(self.frame_times)
        return {
            "method": self.name,
            "frames": len(self.valid),
            "processing_ms_total": elapsed,
            "processing_fps": len(self.valid) * 1000 / max(1e-9, elapsed),
            "mean_ms_per_frame": float(np.mean(self.frame_times)),
            "p95_ms_per_frame": float(np.percentile(self.frame_times, 95)),
            "detection_uptime": float(np.mean(self.valid)),
            "dropout_transitions": dropouts,
            "mean_boxes": float(np.mean(self.counts)),
            "multi_box_rate": float(np.mean(np.asarray(self.counts) >= 2)),
            "median_predicted_area_fraction": float(np.median(self.area_fractions)),
            "skin_proxy_recall": float(np.mean(self.proxy_recall)),
            "skin_proxy_precision": float(np.mean(self.proxy_precision)),
            "median_primary_step_normalized": float(np.median(steps)) if steps else 0.0,
            "p95_primary_step_normalized": float(np.percentile(steps, 95)) if steps else 0.0,
            "valid_center_samples": len(valid_centers),
        }


def annotate(frame: np.ndarray, result: Result, name: str, color: tuple[int, int, int], frame_idx: int) -> np.ndarray:
    canvas = frame.copy()
    if result.mask is not None:
        tint = np.zeros_like(canvas)
        tint[:] = color
        alpha = (result.mask.astype(np.float32) / 255.0 * 0.28)[:, :, None]
        canvas = np.uint8(canvas * (1 - alpha) + tint * alpha)
        contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
    for index, (x, y, width, height) in enumerate(result.boxes, start=1):
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"H{index}", (x + 5, max(24, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (12, 17, 22), -1)
    cv2.putText(canvas, name.upper().replace("_", " "), (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 247, 249), 2, cv2.LINE_AA)
    detail = f"frame {frame_idx:03d} | hands {len(result.boxes)} | {result.elapsed_ms:5.1f} ms"
    cv2.putText(canvas, detail, (canvas.shape[1] - 390, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 188, 198), 1, cv2.LINE_AA)
    return canvas


def source_panel(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (12, 17, 22), -1)
    cv2.putText(canvas, "SOURCE", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 247, 249), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_idx:03d}", (canvas.shape[1] - 150, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 188, 198), 1, cv2.LINE_AA)
    return canvas


def grid_frame(panels: list[np.ndarray]) -> np.ndarray:
    resized = [cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA) for panel in panels]
    while len(resized) < 6:
        resized.append(np.zeros_like(resized[0]))
    return np.vstack((np.hstack(resized[:3]), np.hstack(resized[3:6])))


def run_opencv_sample(sample_name: str, frames: list[np.ndarray], fps: float = 12.0) -> tuple[list[dict[str, object]], Path]:
    methods = [
        SkinMethod(),
        TemporalSkinMethod(),
        MotionGatedSkinMethod(),
        OpenCVTrackerMethod("KCF"),
        OpenCVTrackerMethod("CSRT"),
    ]
    metrics = {method.name: Metrics(method.name) for method in methods}
    output = ANNOTATED_DIR / f"{sample_name}_opencv_comparison.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 720))
    for frame_idx, frame in enumerate(frames):
        proxy = skin_mask(frame)
        panels = [source_panel(frame, frame_idx)]
        for method in methods:
            result = method.process(frame, frame_idx)
            metrics[method.name].add(result, proxy)
            panels.append(annotate(frame, result, method.name, COLORS[method.name], frame_idx))
        writer.write(grid_frame(panels))
    writer.release()
    rows = [collector.summary(frames[0].shape[1], frames[0].shape[0]) for collector in metrics.values()]
    for row in rows:
        row["sample"] = sample_name
    return rows, output


def _mask_array(post_processed) -> np.ndarray:
    array = post_processed.detach().cpu().numpy()
    while array.ndim > 3 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None]
    return array > 0


def run_sam2(frames: list[np.ndarray], fps: float = 6.0) -> tuple[dict[str, object], Path | None]:
    """Run cached SAM2 on a deliberately short clip because this host is CPU-only."""
    try:
        import torch
        from PIL import Image
        from transformers import Sam2VideoModel, Sam2VideoProcessor
    except Exception as exc:
        return {"method": "sam2_prompted", "status": "unavailable", "error": str(exc)}, None

    model_id = "facebook/sam2.1-hiera-small"
    sampled = frames[::2][:24]
    if not sampled:
        return {"method": "sam2_prompted", "status": "no frames"}, None
    # Seed on the frame with the largest visible skin region instead of blindly
    # using frame zero, where the hand is only a clipped sliver at the top edge.
    prompt_frame_idx = int(np.argmax([np.count_nonzero(skin_mask(frame)) for frame in sampled]))
    boxes = boxes_from_mask(skin_mask(sampled[prompt_frame_idx]))
    if not boxes:
        return {"method": "sam2_prompted", "status": "no automatic first-frame prompt"}, None
    xyxy = [[x, y, x + width, y + height] for x, y, width, height in boxes]
    rgb_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in sampled]

    load_start = time.perf_counter()
    processor = Sam2VideoProcessor.from_pretrained(model_id, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(model_id, local_files_only=True).eval()
    load_seconds = time.perf_counter() - load_start
    torch.set_num_threads(max(1, min(12, (os.cpu_count() or 4) - 1)))

    infer_start = time.perf_counter()
    session = processor.init_video_session(video=rgb_frames, inference_device="cpu", dtype=torch.float32)
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=prompt_frame_idx,
        obj_ids=list(range(1, len(xyxy) + 1)),
        input_boxes=[[box for box in xyxy]],
    )
    model(inference_session=session, frame_idx=prompt_frame_idx)
    segment_masks: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for reverse in (True, False):
            for output in model.propagate_in_video_iterator(session, start_frame_idx=prompt_frame_idx, reverse=reverse):
                masks = processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True,
                )[0]
                segment_masks[int(output.frame_idx)] = _mask_array(masks)
    inference_seconds = time.perf_counter() - infer_start

    output_path = ANNOTATED_DIR / "closeup_singlehand_sam2_prompted.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
    collector = Metrics("sam2_prompted")
    per_frame_ms = inference_seconds * 1000 / max(1, len(sampled))
    for frame_idx, frame in enumerate(sampled):
        object_masks = segment_masks.get(frame_idx)
        union = np.any(object_masks, axis=0).astype(np.uint8) * 255 if object_masks is not None else np.zeros(frame.shape[:2], np.uint8)
        result = Result(boxes_from_mask(union), union, per_frame_ms)
        collector.add(result, skin_mask(frame))
        left = cv2.resize(source_panel(frame, frame_idx), (640, 360), interpolation=cv2.INTER_AREA)
        right = cv2.resize(annotate(frame, result, "sam2_prompted", COLORS["sam2_prompted"], frame_idx), (640, 360), interpolation=cv2.INTER_AREA)
        writer.write(np.hstack((left, right)))
    writer.release()
    summary = collector.summary(sampled[0].shape[1], sampled[0].shape[0])
    summary.update(
        {
            "sample": "closeup_singlehand_short",
            "status": "ok",
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "prompt": "automatic maximum-visible-area keyframe box",
            "prompt_frame_idx": prompt_frame_idx,
            "prompt_objects": len(xyxy),
            "note": "CPU-only short-run benchmark; model load excluded from processing_fps",
        }
    )
    return summary, output_path


def transcode_h264(path: Path) -> Path:
    destination = path.with_name(path.stem + "_h264.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return destination


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    global SOURCE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the six source demonstration videos",
    )
    parser.add_argument("--with-sam2", action="store_true", help="Run the cached SAM2.1 model on a short CPU sample")
    parser.add_argument("--sam2-only", action="store_true", help="Preserve existing OpenCV results and run only SAM2")
    args = parser.parse_args()
    SOURCE_DIR = args.input_dir.resolve()
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    outputs: list[Path] = []
    closeup_frames: list[np.ndarray] | None = None
    if args.sam2_only:
        results_path = HERE / "benchmark_results.json"
        if results_path.exists():
            all_rows = [row for row in json.loads(results_path.read_text(encoding="utf-8")) if row.get("method") != "sam2_prompted"]
        _, filename, start_s, duration_s = next(sample for sample in SAMPLES if sample[0] == "closeup_singlehand")
        closeup_frames = read_sample(SOURCE_DIR / filename, start_s, duration_s)
        outputs.extend(sorted(ANNOTATED_DIR.glob("*_opencv_comparison_h264.mp4")))
    else:
        for sample_name, filename, start_s, duration_s in SAMPLES:
            frames = read_sample(SOURCE_DIR / filename, start_s, duration_s)
            if sample_name == "closeup_singlehand":
                closeup_frames = frames
            rows, output = run_opencv_sample(sample_name, frames)
            all_rows.extend(rows)
            outputs.append(transcode_h264(output))

    sam2_summary: dict[str, object] | None = None
    if (args.with_sam2 or args.sam2_only) and closeup_frames:
        sam2_summary, sam2_output = run_sam2(closeup_frames)
        all_rows.append(sam2_summary)
        if sam2_output is not None:
            outputs.append(transcode_h264(sam2_output))

    inventory = {
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_directory": str(SOURCE_DIR),
        "packages": package_inventory(),
        "videos": video_inventory(),
        "sample_outputs": [str(path) for path in outputs],
        "metric_warning": "Skin-proxy overlap is diagnostic agreement, not labeled accuracy.",
    }
    (HERE / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (HERE / "benchmark_results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    write_csv(all_rows, HERE / "benchmark_results.csv")
    print(json.dumps({"rows": all_rows, "outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
