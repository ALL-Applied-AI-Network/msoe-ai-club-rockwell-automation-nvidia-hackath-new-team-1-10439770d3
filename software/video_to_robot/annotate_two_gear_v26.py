"""V26 two-gear annotation renderer with reviewed post-placement tracking.

The general assembly renderer intentionally uses one reviewed destination point
after a part is placed.  In ``PXL_20261031_165150721`` the demonstrator nudges
the gearbox base while meshing the second gear and while seating the cover.  A
static destination ring therefore drifts away from the visible gear even though
the underlying annotation is otherwise correct.

This V26-only entry point adds an optional ``seat_track_xy`` list to each gear:

    [{"time": 6.85, "xy": [991, 679]}, ...]

Points are expressed in the native 1920x1080 source coordinate system and are
linearly interpolated for every rendered frame.  Other videos and the existing
V24/V25 outputs retain the original renderer and behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import annotate_newest_assemblies as base


ORIGINAL_DRAW_PART_GUIDES = base.draw_part_guides


def tracked_point(part: dict[str, Any], time_s: float, image: np.ndarray) -> np.ndarray | None:
    """Interpolate a reviewed gear centre in render coordinates."""
    rows = part.get("seat_track_xy") or []
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: float(row["time"]))
    if time_s <= float(ordered[0]["time"]):
        native = np.asarray(ordered[0]["xy"], float)
    elif time_s >= float(ordered[-1]["time"]):
        native = np.asarray(ordered[-1]["xy"], float)
    else:
        native = np.asarray(ordered[-1]["xy"], float)
        for left, right in zip(ordered, ordered[1:]):
            t0, t1 = float(left["time"]), float(right["time"])
            if t0 <= time_s <= t1:
                alpha = (time_s - t0) / max(1e-9, t1 - t0)
                p0 = np.asarray(left["xy"], float)
                p1 = np.asarray(right["xy"], float)
                native = p0 * (1.0 - alpha) + p1 * alpha
                break
    return native * np.asarray((image.shape[1] / 1920.0, image.shape[0] / 1080.0))


def tracked_scalar(rows: list[dict[str, Any]], time_s: float, fallback: float) -> float:
    """Interpolate a reviewed scalar track (used for changing guide radius)."""
    if not rows:
        return fallback
    ordered = sorted(rows, key=lambda row: float(row["time"]))
    if time_s <= float(ordered[0]["time"]):
        return float(ordered[0]["value"])
    if time_s >= float(ordered[-1]["time"]):
        return float(ordered[-1]["value"])
    for left, right in zip(ordered, ordered[1:]):
        t0, t1 = float(left["time"]), float(right["time"])
        if t0 <= time_s <= t1:
            alpha = (time_s - t0) / max(1e-9, t1 - t0)
            return float(left["value"]) * (1.0 - alpha) + float(right["value"]) * alpha
    return fallback


def track_hidden(part: dict[str, Any], time_s: float) -> bool:
    """Suppress a ring only while the cover fully hides the referenced gear."""
    return any(float(row[0]) <= time_s <= float(row[1]) for row in part.get("seat_hidden_ranges", []))


def draw_part_guides_v26(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    spec: dict,
    state: dict,
    time_s: float,
    sources: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    live_hands: list[dict[str, Any]],
) -> None:
    """Render the original guide, replacing only seated gear centres."""
    if spec.get("v26_reviewed_seat_tracking") is not True or spec.get("kind") != "gearbox":
        ORIGINAL_DRAW_PART_GUIDES(image, masks, spec, state, time_s, sources, targets, live_hands)
        return

    part_defs = base.part_map(spec)
    if spec.get("palette") == "mint_yellow":
        base_center = tuple(np.round(sources["B0"]).astype(int)) if "B0" in sources else None
        cover_center = base.paint_mask(image, masks["yellow"], base.AMBER, 0.06, 2)
    else:
        base_center = base.paint_mask(image, masks["blue"], base.BLUE, 0.08, 2)
        cover_center = base.paint_mask(image, masks["orange"], base.AMBER, 0.06, 2)
    base.paint_mask(image, masks["green"], base.GREEN, 0.09, 1)
    if base_center:
        label = "B0  BASE" if spec.get("compact_hud") else "B0  CHASSIS"
        base.pill(image, (base_center[0] - 58, max(120, base_center[1] + 58)), label, base.BLUE)

    gear_ids = [part["id"] for part in spec["parts"] if part["id"].startswith("G")]
    for gear_id in gear_ids:
        part = part_defs[gear_id]
        step = next(step for step in spec["steps"] if step["part"] == gear_id)
        if time_s < step["pick"]:
            center = sources.get(gear_id)
            status, colour = "QUEUED", base.AMBER
        elif time_s < step["place"]:
            preferred_label = str(step.get("hand_label", "")).lower()
            preferred_hands = [
                hand for hand in live_hands
                if str(hand.get("label", "")).lower() == preferred_label
            ] if preferred_label else []
            hand = base.choose_hand(preferred_hands or live_hands, targets.get(gear_id))
            if hand is not None:
                offset_native = np.asarray(part.get("pinch_offset_xy", [0, 0]), float)
                offset = offset_native * np.asarray((image.shape[1] / 1920.0, image.shape[0] / 1080.0))
                center = hand["pinch"] + offset
            else:
                center = sources.get(gear_id)
            status, colour = "IN TRANSFER", base.CYAN
        else:
            # V26 correction: track the actual visible seated gear/base motion,
            # rather than freezing the circle at target_xy and later jumping to
            # a second static verification coordinate.
            center = tracked_point(part, time_s, image)
            if center is None:
                verify_target = part.get("verify_target_xy") if time_s >= spec.get("verify_start", float("inf")) else None
                center = base.scale_point(
                    verify_target,
                    image.shape[1] / 1920.0,
                    image.shape[0] / 1080.0,
                ) if verify_target is not None else targets.get(gear_id)
            status, colour = "SEATED", base.GREEN

        if center is None:
            continue
        if status == "SEATED" and track_hidden(part, time_s):
            # A tracking ring over an opaque cover would imply fabricated
            # visual evidence. It disappears briefly, then returns as a small
            # halo around the genuinely visible tab/gear fragment.
            continue
        if status == "QUEUED":
            guide_radius = part.get("source_radius", part.get("radius", 55))
        elif status == "SEATED":
            guide_radius = part.get("target_radius", part.get("radius", 55))
            guide_radius = tracked_scalar(part.get("seat_radius_track", []), time_s, float(guide_radius))
        else:
            guide_radius = part.get("move_radius", part.get("target_radius", part.get("radius", 55)))
        radius = max(18, int(guide_radius * image.shape[1] / 1920.0))
        cv2.circle(
            image,
            tuple(np.round(center).astype(int)),
            radius,
            colour,
            2 if status != "QUEUED" else 1,
            cv2.LINE_AA,
        )
        gear_label = f"{gear_id}  {part['name']}" if spec.get("compact_hud") else f"{gear_id}  {part['name']}  /  {status}"
        base.pill(image, (int(center[0] - radius), int(center[1] - radius - 8)), gear_label, colour, 0.38)

    if cover_center:
        cover_step = next(step for step in spec["steps"] if step["part"] == "C1")
        status = "SEATED" if time_s >= cover_step["place"] else "ACTIVE" if time_s >= cover_step["start"] else "QUEUED"
        colour = base.GREEN if status == "SEATED" else base.CYAN if status == "ACTIVE" else base.AMBER
        cover_label = "C1  COVER" if spec.get("compact_hud") else f"C1  COVER  /  {status}"
        base.pill(image, (cover_center[0] - 76, max(120, cover_center[1] - 45)), cover_label, colour)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/two_gear_annotation_v26_20261031"))
    ap.add_argument("--specs", type=Path, default=Path(__file__).parent / "two_gear_annotation_v26.json")
    ap.add_argument("--hand-model", type=Path, default=Path(__file__).parent / "models" / "hand_landmarker.task")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--only", default="PXL_20261031_165150721")
    args = ap.parse_args()

    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    annotated = args.output_dir / "annotated"
    tracks = args.output_dir / "tracks"
    results = []
    base.draw_part_guides = draw_part_guides_v26
    try:
        for stem, spec in specs.items():
            if selected and stem not in selected:
                continue
            source = args.input_dir / f"{stem}.mp4"
            if not source.is_file():
                raise FileNotFoundError(source)
            print(f"[annotate-v26] {source.name}", flush=True)
            results.append(base.process_video(
                source,
                annotated / f"{stem}_assembly_intelligence_v26.mp4",
                tracks / f"{stem}_tracks_v26.json",
                args.hand_model,
                spec,
                args.width,
            ))
    finally:
        base.draw_part_guides = ORIGINAL_DRAW_PART_GUIDES

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
