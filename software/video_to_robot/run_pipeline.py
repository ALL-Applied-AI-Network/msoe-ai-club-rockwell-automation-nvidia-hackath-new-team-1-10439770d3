"""Hybrid hand/object/action annotation for the gearbox demonstration clips.

The code intentionally records uncertainty and never promotes pixel coordinates to robot
coordinates.  It is a practical evidence layer that can later accept SAM/FoundationPose
backends when calibrated RGB-D and CAD are available.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


COLORS = {
    "ink": (244, 244, 240),
    "muted": (176, 184, 190),
    "panel": (22, 25, 29),
    "hand": (232, 190, 62),
    "hand_fill": (196, 142, 32),
    "cube": (76, 88, 238),
    "bin": (84, 214, 126),
    "active": (74, 201, 244),
}


@dataclass
class Detection:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: float
    inside_bin: bool = False


class CentroidTracker:
    """Small deterministic tracker suitable for the few cubes in this task."""

    def __init__(self, max_distance: float = 95.0, max_missing: int = 18):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    def update(self, boxes: list[tuple[int, int, int, int]], areas: list[float]) -> list[Detection]:
        centers = [((x + w / 2), (y + h / 2)) for x, y, w, h in boxes]
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self.tracks.items():
            for i, c in enumerate(centers):
                pairs.append((math.dist(tr["center"], c), tid, i))
        for distance, tid, i in sorted(pairs):
            if distance > self.max_distance or tid in assigned_tracks or i in assigned_dets:
                continue
            self.tracks[tid].update(center=centers[i], bbox=boxes[i], area=areas[i], missing=0)
            assigned_tracks.add(tid)
            assigned_dets.add(i)
        for i, center in enumerate(centers):
            if i in assigned_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"center": center, "bbox": boxes[i], "area": areas[i], "missing": 0}
            assigned_tracks.add(tid)
        for tid in list(self.tracks):
            if tid not in assigned_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]
        visible = []
        for tid, tr in self.tracks.items():
            if tr["missing"] == 0:
                visible.append(Detection(tid, tr["bbox"], tr["center"], tr["area"]))
        return visible


class HandLandmarks:
    def __init__(self, model_path: Path):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.30,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        hands: list[dict[str, Any]] = []
        h, w = bgr.shape[:2]
        for i, landmarks in enumerate(result.hand_landmarks):
            pts = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks], dtype=np.float32)
            handedness = "hand"
            confidence = 0.0
            if i < len(result.handedness) and result.handedness[i]:
                handedness = result.handedness[i][0].category_name.lower()
                confidence = float(result.handedness[i][0].score)
            pinch = (pts[4, :2] + pts[8, :2]) / 2
            palm_scale = max(1.0, float(np.linalg.norm(pts[5, :2] - pts[17, :2])))
            pinch_ratio = float(np.linalg.norm(pts[4, :2] - pts[8, :2]) / palm_scale)
            hands.append(
                {
                    "label": handedness,
                    "confidence": confidence,
                    "landmarks": pts,
                    "pinch": pinch,
                    "pinch_ratio": pinch_ratio,
                    "centroid": pts[:, :2].mean(axis=0),
                }
            )
        return hands

    def close(self) -> None:
        self.landmarker.close()


def hand_arm_exclusion(shape: tuple[int, int], hands: list[dict[str, Any]]) -> np.ndarray:
    """Mask hand plus the forearm ray so sleeve graphics are not mistaken for parts."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand in hands:
        pts = hand["landmarks"][:, :2]
        palm_width = max(18.0, float(np.linalg.norm(pts[5] - pts[17])))
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        wrist = pts[0]
        palm = pts[[5, 9, 13, 17]].mean(axis=0)
        direction = wrist - palm
        norm = float(np.linalg.norm(direction))
        if norm < 1:
            continue
        far = wrist + direction / norm * math.hypot(w, h) * 1.2
        cv2.line(mask, tuple(wrist.astype(int)), tuple(far.astype(int)), 255, int(palm_width * 1.65), cv2.LINE_AA)
    return cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)


def find_bin_and_cubes(frame: np.ndarray, hands: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray, list[tuple[int, int, int, int]], list[float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([32, 45, 35]), np.array([98, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    g_contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bin_box = None
    if g_contours:
        candidates = [c for c in g_contours if cv2.contourArea(c) > frame.shape[0] * frame.shape[1] * 0.001]
        if candidates:
            bin_box = cv2.boundingRect(max(candidates, key=cv2.contourArea))

    low_red = cv2.inRange(hsv, np.array([0, 85, 70]), np.array([15, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([165, 80, 65]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(low_red, high_red)
    if hands:
        red[hand_arm_exclusion(red.shape, hands) > 0] = 0
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    areas: list[float] = []
    image_area = frame.shape[0] * frame.shape[1]
    for c in contours:
        area = float(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        if area < image_area * 0.00008 or area > image_area * 0.008:
            continue
        if w < 7 or h < 7 or max(w / max(h, 1), h / max(w, 1)) > 4.0:
            continue
        boxes.append((x, y, w, h))
        areas.append(area)
    return green, bin_box, red, boxes, areas


class ActionCompiler:
    """Hysteretic 2D contact-state compiler; no metric-3D claims."""

    def __init__(self, fps: float, diagonal: float):
        self.fps = fps
        self.diagonal = diagonal
        self.latched = False
        self.touched_bin = False
        self.active_hand_label: str | None = None
        self.contact_frames = 0
        self.away_frames = 0
        self.phase = "OBSERVE"
        self.cycle = 0
        self.events: list[dict[str, Any]] = []
        self.phase_start_frame = 0
        self.phase_history = deque(maxlen=max(3, int(fps * 0.25)))

    def _set_phase(self, phase: str, frame_index: int, evidence: dict[str, Any]) -> None:
        self.phase_history.append(phase)
        if len(self.phase_history) < self.phase_history.maxlen:
            return
        stable = max(set(self.phase_history), key=self.phase_history.count)
        if stable == self.phase:
            return
        if self.phase != "OBSERVE":
            self.events.append(
                {
                    "phase": self.phase.lower(),
                    "start_frame": self.phase_start_frame,
                    "end_frame": frame_index - 1,
                    "start_s": round(self.phase_start_frame / self.fps, 3),
                    "end_s": round((frame_index - 1) / self.fps, 3),
                    "cycle": self.cycle,
                    "evidence": evidence,
                }
            )
        self.phase = stable
        self.phase_start_frame = frame_index

    def update(self, frame_index: int, hands: list[dict[str, Any]], cubes: list[Detection], bin_box: tuple[int, int, int, int] | None) -> tuple[str, dict[str, Any]]:
        evidence: dict[str, Any] = {"hand_count": len(hands), "visible_cube_count": len(cubes), "bin_visible": bool(bin_box)}
        if not hands:
            candidate = "VERIFY" if self.latched and self.touched_bin else "OBSERVE"
            self._set_phase(candidate, frame_index, evidence)
            return self.phase, evidence

        outside = [d for d in cubes if not d.inside_bin]
        chosen = None
        nearest_cube = None
        nearest_distance = float("inf")
        for hand in hands:
            for cube in outside:
                distance = math.dist(tuple(hand["pinch"]), cube.center)
                if distance < nearest_distance:
                    nearest_distance, chosen, nearest_cube = distance, hand, cube
        if chosen is None:
            chosen = min(hands, key=lambda h: h["pinch"][1])
        p = tuple(float(v) for v in chosen["pinch"])
        evidence.update(
            hand=chosen["label"],
            hand_confidence=round(float(chosen["confidence"]), 3),
            pinch_ratio=round(float(chosen["pinch_ratio"]), 3),
            nearest_cube_distance_norm=None if nearest_cube is None else round(nearest_distance / self.diagonal, 4),
        )
        near_cube = nearest_cube is not None and nearest_distance < self.diagonal * 0.075
        over_bin = False
        distance_to_bin = None
        if bin_box:
            x, y, w, h = bin_box
            bx, by = x + w / 2, y + h / 2
            distance_to_bin = math.dist(p, (bx, by)) / self.diagonal
            margin_x, margin_y = 0.12 * w, 0.15 * h
            over_bin = (x - margin_x <= p[0] <= x + w + margin_x and y - margin_y <= p[1] <= y + h + margin_y)
        evidence["over_bin"] = over_bin
        evidence["distance_to_bin_norm"] = None if distance_to_bin is None else round(distance_to_bin, 4)

        if not self.latched:
            if near_cube:
                self.contact_frames += 1
                candidate = "CONTACT" if self.contact_frames >= max(2, int(self.fps * 0.08)) else "APPROACH"
                if self.contact_frames >= max(3, int(self.fps * 0.14)):
                    self.latched = True
                    self.active_hand_label = chosen["label"]
                    self.touched_bin = False
                    self.away_frames = 0
                    candidate = "GRASP"
            else:
                self.contact_frames = max(0, self.contact_frames - 1)
                candidate = "APPROACH" if nearest_cube is not None else "OBSERVE"
        else:
            if over_bin:
                self.touched_bin = True
                self.away_frames = 0
                candidate = "INSERT"
            elif self.touched_bin:
                self.away_frames += 1
                candidate = "RELEASE" if self.away_frames < max(3, int(self.fps * 0.28)) else "RETRACT"
                if self.away_frames >= max(4, int(self.fps * 0.36)):
                    self.cycle += 1
                    self.latched = False
                    self.touched_bin = False
                    self.active_hand_label = None
                    self.contact_frames = 0
                    candidate = "VERIFY"
            else:
                candidate = "TRANSFER"
        self._set_phase(candidate, frame_index, evidence)
        evidence["completed_cycles"] = self.cycle
        return self.phase, evidence

    def finish(self, final_frame: int) -> list[dict[str, Any]]:
        if self.phase != "OBSERVE":
            self.events.append(
                {
                    "phase": self.phase.lower(),
                    "start_frame": self.phase_start_frame,
                    "end_frame": final_frame,
                    "start_s": round(self.phase_start_frame / self.fps, 3),
                    "end_s": round(final_frame / self.fps, 3),
                    "cycle": self.cycle,
                    "evidence": {"final": True},
                }
            )
        return self.events


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def alpha_rect(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], alpha: float) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, p1, p2, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def draw_hand(image: np.ndarray, hand: dict[str, Any]) -> None:
    pts = hand["landmarks"][:, :2].astype(np.int32)
    overlay = image.copy()
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(overlay, hull, COLORS["hand_fill"])
    cv2.addWeighted(overlay, 0.20, image, 0.80, 0, dst=image)
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), COLORS["hand"], 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(image, tuple(p), 3, COLORS["ink"], -1, cv2.LINE_AA)
    x, y, w, h = cv2.boundingRect(pts)
    cv2.putText(image, f'{hand["label"].upper()} HAND', (x, max(28, y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLORS["ink"], 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, hands: list[dict[str, Any]], cubes: list[Detection], bin_box: tuple[int, int, int, int] | None, phase: str, evidence: dict[str, Any], time_s: float) -> np.ndarray:
    h, w = frame.shape[:2]
    alpha_rect(frame, (0, 0), (w, 94), COLORS["panel"], 0.86)
    alpha_rect(frame, (0, h - 74), (w, h), COLORS["panel"], 0.82)
    cv2.putText(frame, "VIDEO-TO-ROBOT  /  OBSERVATION LAYER", (28, 32), cv2.FONT_HERSHEY_DUPLEX, 0.64, COLORS["muted"], 1, cv2.LINE_AA)
    cv2.putText(frame, phase, (28, 72), cv2.FONT_HERSHEY_DUPLEX, 1.12, COLORS["active"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"{time_s:06.2f}s", (w - 142, 38), cv2.FONT_HERSHEY_DUPLEX, 0.72, COLORS["ink"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"OBSERVED  hands {len(hands)}  |  cubes {len(cubes)}  |  completed transfers {evidence.get('completed_cycles', 0)}", (28, h - 42), cv2.FONT_HERSHEY_DUPLEX, 0.60, COLORS["ink"], 1, cv2.LINE_AA)
    cv2.putText(frame, "2D evidence only  /  metric 6D requires calibrated RGB-D + CAD", (28, h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.47, COLORS["muted"], 1, cv2.LINE_AA)
    if bin_box:
        x, y, bw, bh = bin_box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), COLORS["bin"], -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, dst=frame)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLORS["bin"], 3, cv2.LINE_AA)
        cv2.putText(frame, "TARGET BIN", (x, max(110, y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.55, COLORS["bin"], 1, cv2.LINE_AA)
    for cube in cubes:
        x, y, cw, ch = cube.bbox
        cv2.rectangle(frame, (x, y), (x + cw, y + ch), COLORS["cube"], 2, cv2.LINE_AA)
        cv2.putText(frame, f"C{cube.track_id}", (x, max(112, y - 7)), cv2.FONT_HERSHEY_DUPLEX, 0.47, COLORS["cube"], 1, cv2.LINE_AA)
    for hand in hands:
        draw_hand(frame, hand)
    return frame


def mux_audio(original: Path, silent: Path, final: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        silent.replace(final)
        return
    command = [
        ffmpeg, "-y", "-i", str(silent), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(final),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    silent.unlink(missing_ok=True)


def process_video(path: Path, output_dir: Path, hand_model: Path, width: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    fps = source_fps if 5 <= source_fps <= 120 else 24.0
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_w = min(width, source_w)
    render_h = int(round(source_h * render_w / source_w))
    if render_h % 2:
        render_h += 1
    annotated_dir = output_dir / "annotated"
    tracks_dir = output_dir / "tracks"
    timelines_dir = output_dir / "timelines"
    for d in (annotated_dir, tracks_dir, timelines_dir):
        d.mkdir(parents=True, exist_ok=True)
    silent_path = annotated_dir / f"{path.stem}_annotated_silent.mp4"
    final_path = annotated_dir / f"{path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    hands_model = HandLandmarks(hand_model)
    cube_tracker = CentroidTracker(max_distance=render_w * 0.075, max_missing=max(10, int(fps * 0.8)))
    compiler = ActionCompiler(fps, math.hypot(render_w, render_h))
    frame_records: list[dict[str, Any]] = []
    frame_index = 0
    hand_frames = 0
    bin_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_AREA)
            timestamp_ms = int(round(frame_index * 1000 / fps))
            hands = hands_model.detect(frame, timestamp_ms)
            _, bin_box, _, cube_boxes, cube_areas = find_bin_and_cubes(frame, hands)
            cubes = cube_tracker.update(cube_boxes, cube_areas)
            if bin_box:
                bx, by, bw, bh = bin_box
                for cube in cubes:
                    cx, cy = cube.center
                    cube.inside_bin = bx <= cx <= bx + bw and by <= cy <= by + bh
            phase, evidence = compiler.update(frame_index, hands, cubes, bin_box)
            hand_frames += int(bool(hands))
            bin_frames += int(bool(bin_box))
            frame_records.append(
                {
                    "frame": frame_index,
                    "time_s": round(frame_index / fps, 4),
                    "phase": phase.lower(),
                    "hands": [
                        {
                            "label": h["label"],
                            "confidence": round(float(h["confidence"]), 4),
                            "pinch_xy_px": [round(float(v), 2) for v in h["pinch"]],
                            "pinch_ratio": round(float(h["pinch_ratio"]), 4),
                            "landmarks_xy_px": [[round(float(x), 2), round(float(y), 2)] for x, y in h["landmarks"][:, :2]],
                        }
                        for h in hands
                    ],
                    "bin_bbox_px": list(bin_box) if bin_box else None,
                    "cubes": [asdict(c) for c in cubes],
                    "evidence": evidence,
                }
            )
            writer.write(draw_overlay(frame, hands, cubes, bin_box, phase, evidence, frame_index / fps))
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        hands_model.close()
    events = compiler.finish(max(0, frame_index - 1))
    with (tracks_dir / f"{path.stem}.json").open("w", encoding="utf-8") as handle:
        json.dump({"video": path.name, "fps": fps, "frames": frame_records}, handle, indent=2)
    timeline = {
        "video": path.name,
        "source_kind": "sequential_phone_demonstration",
        "coordinate_frame": "image_pixels",
        "metric_3d_available": False,
        "events": events,
        "automatic_completed_cycles": compiler.cycle,
    }
    with (timelines_dir / f"{path.stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(timeline, handle, indent=2)
    mux_audio(path, silent_path, final_path)
    return {
        "video": path.name,
        "frames": frame_index,
        "duration_s": round(frame_index / fps, 3),
        "fps": round(fps, 4),
        "hand_frame_coverage": round(hand_frames / max(1, frame_index), 4),
        "bin_frame_coverage": round(bin_frames / max(1, frame_index), 4),
        "automatic_completed_cycles": compiler.cycle,
        "event_count": len(events),
        "annotated_video": str(final_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, default=Path(__file__).parent / "models" / "hand_landmarker.task")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    videos = sorted(args.input_dir.glob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No mp4 files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}", flush=True)
        summaries.append(process_video(video, args.output_dir, args.hand_model, args.width))
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"method": "mediapipe_hands+deterministic_object_masks+hysteretic_contact_graph", "videos": summaries}, handle, indent=2)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
