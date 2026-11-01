"""Render reviewed assembly guides over the newest human-demonstration videos.

The renderer deliberately separates reviewed semantics from model-derived evidence:
reviewed step windows and part identities come from newest_assembly_annotations.json;
MediaPipe supplies every-frame 21-point hands; HSV masks supply visible printed-part
regions.  Ghost hands are not clip art: each is sampled from the demonstrator's actual
future contact frame and previewed until the live hand reaches that state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from run_pipeline import HAND_CONNECTIONS, HandLandmarks, alpha_rect, mux_audio


INK = (244, 246, 247)
MUTED = (167, 178, 190)
PANEL = (18, 23, 30)
CYAN = (238, 196, 48)
CYAN_DIM = (190, 144, 38)
AMBER = (50, 171, 248)
GREEN = (102, 211, 116)
BLUE = (244, 155, 69)
RED = (84, 91, 235)
WHITE = (255, 255, 255)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    rw = min(width, w)
    rh = int(round(h * rw / w))
    if rh % 2:
        rh += 1
    return cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)


def clean_mask(mask: np.ndarray, close: int = 7, opening: int = 3) -> np.ndarray:
    if opening:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((opening, opening), np.uint8))
    if close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    return mask


def colour_masks(frame: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "green": clean_mask(cv2.inRange(hsv, np.array([35, 45, 48]), np.array([95, 255, 255])), 7, 3),
        "blue": clean_mask(cv2.inRange(hsv, np.array([88, 50, 30]), np.array([140, 255, 255])), 11, 5),
        "orange": clean_mask(cv2.inRange(hsv, np.array([2, 75, 55]), np.array([28, 255, 255])), 11, 5),
        # Keep the neon-yellow cover while rejecting skin and warm sleeve shadows.
        "yellow": clean_mask(cv2.inRange(hsv, np.array([25, 80, 100]), np.array([38, 255, 255])), 9, 3),
    }


def contours(mask: np.ndarray, minimum: float = 100.0) -> list[np.ndarray]:
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted((c for c in found if cv2.contourArea(c) >= minimum), key=cv2.contourArea, reverse=True)


def mask_centres(mask: np.ndarray, minimum: float = 100.0) -> list[tuple[float, float, float]]:
    output = []
    for contour in contours(mask, minimum):
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            continue
        output.append((m["m10"] / m["m00"], m["m01"] / m["m00"], cv2.contourArea(contour)))
    return output


def serialize_hand(hand: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": hand["label"],
        "confidence": float(hand["confidence"]),
        "landmarks": hand["landmarks"].astype(float).tolist(),
        "pinch": hand["pinch"].astype(float).tolist(),
        "pinch_ratio": float(hand["pinch_ratio"]),
        "centroid": hand["centroid"].astype(float).tolist(),
    }


def restore_hand(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "landmarks": np.asarray(record["landmarks"], np.float32),
        "pinch": np.asarray(record["pinch"], np.float32),
        "centroid": np.asarray(record["centroid"], np.float32),
    }


def dashed_line(image: np.ndarray, a: np.ndarray, b: np.ndarray, colour, thickness=2, dash=9, gap=6) -> None:
    vector = b.astype(float) - a.astype(float)
    length = float(np.linalg.norm(vector))
    if length < 1:
        return
    direction = vector / length
    cursor = 0.0
    while cursor < length:
        p = a + direction * cursor
        q = a + direction * min(length, cursor + dash)
        cv2.line(image, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), colour, thickness, cv2.LINE_AA)
        cursor += dash + gap


def draw_live_hand(image: np.ndarray, hand: dict[str, Any]) -> None:
    pts = hand["landmarks"][:, :2].astype(np.int32)
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, cv2.convexHull(pts), CYAN_DIM)
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0, dst=image)
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), CYAN, 2, cv2.LINE_AA)
    for index, point in enumerate(pts):
        cv2.circle(image, tuple(point), 4 if index in (4, 8) else 2, AMBER if index in (4, 8) else WHITE, -1, cv2.LINE_AA)
    pinch = np.round(hand["pinch"]).astype(int)
    cv2.circle(image, tuple(pinch), 8, AMBER, 2, cv2.LINE_AA)


def draw_ghost_hand(image: np.ndarray, hand: dict[str, Any], caption: str, pulse: float) -> None:
    pts = hand["landmarks"][:, :2].astype(np.int32)
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, cv2.convexHull(pts), CYAN)
    cv2.addWeighted(overlay, 0.08, image, 0.92, 0, dst=image)
    for a, b in HAND_CONNECTIONS:
        dashed_line(image, pts[a], pts[b], CYAN, 2)
    for point in pts:
        cv2.circle(image, tuple(point), 3, CYAN, 1, cv2.LINE_AA)
    pinch = np.round(hand["pinch"]).astype(int)
    cv2.circle(image, tuple(pinch), int(15 + 5 * pulse), CYAN, 2, cv2.LINE_AA)
    x, y, _, _ = cv2.boundingRect(pts)
    cv2.putText(image, caption, (x, max(150, y - 12)), cv2.FONT_HERSHEY_DUPLEX, 0.48, CYAN, 1, cv2.LINE_AA)


def paint_mask(image: np.ndarray, mask: np.ndarray, colour, alpha: float, thickness: int = 2) -> tuple[int, int] | None:
    cs = contours(mask, image.shape[0] * image.shape[1] * 0.00015)
    if not cs:
        return None
    overlay = image.copy()
    overlay[mask > 0] = colour
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    cv2.drawContours(image, cs, -1, colour, thickness, cv2.LINE_AA)
    largest = cs[0]
    x, y, w, h = cv2.boundingRect(largest)
    return x + w // 2, y + h // 2


def part_map(spec: dict) -> dict[str, dict]:
    return {part["id"]: part for part in spec["parts"]}


def active_state(time_s: float, spec: dict) -> dict[str, Any]:
    steps = spec["steps"]
    completed = sum(1 for step in steps if time_s >= step["end"])
    for index, step in enumerate(steps):
        if step["start"] <= time_s < step["end"]:
            if time_s < step["pick"]:
                phase, ghost_kind = "APPROACH / PICK", "pick"
            elif time_s < step["place"]:
                phase, ghost_kind = "TRANSFER / ALIGN", "place"
            else:
                phase, ghost_kind = "SEAT / VERIFY", "place"
            return {"index": index, "step": step, "phase": phase, "ghost_kind": ghost_kind, "completed": completed}
    if time_s >= spec.get("verify_start", 1e9):
        return {"index": len(steps), "step": None, "phase": "ASSEMBLY COMPLETE / VERIFY", "ghost_kind": None, "completed": len(steps)}
    upcoming = next((i for i, step in enumerate(steps) if time_s < step["start"]), 0)
    return {"index": upcoming, "step": None, "phase": "OBSERVE / INVENTORY", "ghost_kind": None, "completed": completed}


def scale_point(point: list[float] | None, sx: float, sy: float) -> np.ndarray | None:
    if point is None:
        return None
    return np.asarray((point[0] * sx, point[1] * sy), float)


def choose_hand(hands: list[dict[str, Any]], anchor: np.ndarray | None = None) -> dict[str, Any] | None:
    if not hands:
        return None
    if anchor is None:
        return max(hands, key=lambda item: item.get("confidence", 0.0))
    return min(hands, key=lambda item: float(np.linalg.norm(item["pinch"] - anchor)))


def nearest_hand_frame(records: list[list[dict[str, Any]]], target: int, anchor: np.ndarray | None, radius: int = 14) -> dict[str, Any] | None:
    candidates = []
    for delta in range(radius + 1):
        for index in ({target} if delta == 0 else {target - delta, target + delta}):
            if 0 <= index < len(records) and records[index]:
                hand = choose_hand([restore_hand(item) for item in records[index]], anchor)
                if hand is not None:
                    distance = 0.0 if anchor is None else float(np.linalg.norm(hand["pinch"] - anchor))
                    candidates.append((distance + delta * 0.35, hand))
        if candidates and delta >= 4:
            break
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def source_centres_from_first(frame: np.ndarray, spec: dict) -> dict[str, np.ndarray]:
    masks = colour_masks(frame)
    found = mask_centres(masks["green"], frame.shape[0] * frame.shape[1] * 0.00025)
    found = sorted(found, key=(lambda p: p[0]) if spec.get("source_sort") == "x" else (lambda p: p[1]))
    gear_ids = [part["id"] for part in spec["parts"] if part["id"].startswith("G")]
    return {gear_id: np.asarray(found[index][:2], float) for index, gear_id in enumerate(gear_ids) if index < len(found)}


def draw_target(image: np.ndarray, point: np.ndarray, label: str, pulse: float) -> None:
    p = tuple(np.round(point).astype(int))
    radius = int(18 + pulse * 6)
    cv2.circle(image, p, radius, CYAN, 2, cv2.LINE_AA)
    cv2.circle(image, p, 5, AMBER, -1, cv2.LINE_AA)
    cv2.line(image, (p[0], p[1] - 72), (p[0], p[1] - radius), CYAN, 1, cv2.LINE_AA)
    cv2.putText(image, label, (p[0] + 14, max(150, p[1] - 48)), cv2.FONT_HERSHEY_DUPLEX, 0.45, CYAN, 1, cv2.LINE_AA)


def pill(image: np.ndarray, origin: tuple[int, int], text: str, colour, scale=0.43) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    alpha_rect(image, (x - 5, y - th - 7), (x + tw + 7, y + 6), PANEL, 0.84)
    cv2.rectangle(image, (x - 5, y - th - 7), (x + tw + 7, y + 6), colour, 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_part_guides(image: np.ndarray, masks: dict[str, np.ndarray], spec: dict, state: dict, time_s: float,
                     sources: dict[str, np.ndarray], targets: dict[str, np.ndarray], live_hands: list[dict[str, Any]]) -> None:
    kind = spec["kind"]
    part_defs = part_map(spec)
    if kind == "gearbox":
        if spec.get("palette") == "mint_yellow":
            base_center = tuple(np.round(sources["B0"]).astype(int)) if "B0" in sources else None
            cover_center = paint_mask(image, masks["yellow"], AMBER, 0.06, 2)
        else:
            base_center = paint_mask(image, masks["blue"], BLUE, 0.08, 2)
            cover_center = paint_mask(image, masks["orange"], AMBER, 0.06, 2)
        paint_mask(image, masks["green"], GREEN, 0.09, 1)
        if base_center:
            base_label = "B0  BASE" if spec.get("compact_hud") else "B0  CHASSIS"
            pill(image, (base_center[0] - 58, max(120, base_center[1] + 58)), base_label, BLUE)
        # Derive the gear inventory from the reviewed specification.  The same
        # renderer is therefore valid for both the original two-gear teaching
        # demonstration and the later four-gear progression.
        gear_ids = [part["id"] for part in spec["parts"] if part["id"].startswith("G")]
        for gear_index, gear_id in enumerate(gear_ids):
            part = part_defs[gear_id]
            step = next(step for step in spec["steps"] if step["part"] == gear_id)
            if time_s < step["pick"]:
                center = sources.get(gear_id)
                status, colour = "QUEUED", AMBER
            elif time_s < step["place"]:
                # Keep the moving guide attached to the hand that actually picked
                # this part.  Anchoring only to the destination can select the
                # demonstrator's other hand when both hands enter the frame.
                preferred_label = str(step.get("hand_label", "")).lower()
                preferred_hands = [
                    hand for hand in live_hands
                    if str(hand.get("label", "")).lower() == preferred_label
                ] if preferred_label else []
                hand = choose_hand(preferred_hands or live_hands, targets.get(gear_id))
                if hand is not None:
                    # The MediaPipe pinch point sits on the raised grasp tab,
                    # not the gear's pitch-circle centre.  A reviewed per-part
                    # image-plane offset keeps the outer guide on the gear body.
                    offset_native = np.asarray(part.get("pinch_offset_xy", [0, 0]), float)
                    offset = offset_native * np.asarray((image.shape[1] / 1920.0, image.shape[0] / 1080.0))
                    center = hand["pinch"] + offset
                else:
                    center = sources.get(gear_id)
                status, colour = "IN TRANSFER", CYAN
            else:
                # The completed assembly is nudged several pixels while the
                # drop-on cover is seated.  Use the reviewed post-cover shaft
                # centres during functional verification so the rings remain
                # concentric with the now-covered gears.
                verify_target = part.get("verify_target_xy") if time_s >= spec.get("verify_start", float("inf")) else None
                center = scale_point(
                    verify_target,
                    image.shape[1] / 1920.0,
                    image.shape[0] / 1080.0,
                ) if verify_target is not None else targets.get(gear_id)
                status, colour = "SEATED", GREEN
            if center is None:
                continue
            if status == "QUEUED":
                guide_radius = part.get("source_radius", part.get("radius", 55))
            elif status == "SEATED":
                guide_radius = part.get("target_radius", part.get("radius", 55))
            else:
                guide_radius = part.get("move_radius", part.get("target_radius", part.get("radius", 55)))
            radius = max(18, int(guide_radius * image.shape[1] / 1920.0))
            cv2.circle(image, tuple(np.round(center).astype(int)), radius, colour, 2 if status != "QUEUED" else 1, cv2.LINE_AA)
            gear_label = f"{gear_id}  {part['name']}" if spec.get("compact_hud") else f"{gear_id}  {part['name']}  /  {status}"
            pill(image, (int(center[0] - radius), int(center[1] - radius - 8)), gear_label, colour, 0.38)
        if cover_center:
            cover_step = next(step for step in spec["steps"] if step["part"] == "C1")
            status = "SEATED" if time_s >= cover_step["place"] else "ACTIVE" if time_s >= cover_step["start"] else "QUEUED"
            colour = GREEN if status == "SEATED" else CYAN if status == "ACTIVE" else AMBER
            cover_label = "C1  COVER" if spec.get("compact_hud") else f"C1  COVER  /  {status}"
            pill(image, (cover_center[0] - 76, max(120, cover_center[1] - 45)), cover_label, colour)
    else:
        blue_cs = sorted(
            contours(masks["blue"], image.shape[0] * image.shape[1] * 0.00025)[:4],
            key=lambda c: cv2.boundingRect(c)[0],
        )[:2]
        carrier_centres: list[tuple[str, np.ndarray]] = []
        for index, contour in enumerate(blue_cs):
            x, y, w, h = cv2.boundingRect(contour)
            colour = BLUE
            overlay = image.copy(); cv2.drawContours(overlay, [contour], -1, colour, -1)
            cv2.addWeighted(overlay, 0.08, image, 0.92, 0, dst=image)
            cv2.drawContours(image, [contour], -1, colour, 2, cv2.LINE_AA)
            carrier_id = "L0" if index == 0 else "R0"
            carrier_centres.append((carrier_id, np.asarray((x + w / 2, y + h / 2), float)))
            pill(image, (x, max(150, y - 8)), ("L0  LEFT CARRIER" if index == 0 else "R0  RIGHT CARRIER"), colour, 0.38)
            # The push-push receiver is a distinct functional feature.  Pulse the
            # active carrier interface instead of implying the whole rail locks.
            active_part = state["step"]["part"] if state["step"] else ""
            active_side = "L0" if active_part in ("L0", "FL") else "R0" if active_part in ("R0", "FR") else ""
            if carrier_id == active_side:
                lp = (int(x + w / 2), int(y + h * 0.34))
                pulse = 0.5 + 0.5 * math.sin(time_s * math.tau * 1.4)
                cv2.circle(image, lp, int(14 + 6 * pulse), CYAN, 2, cv2.LINE_AA)
                pill(image, (lp[0] + 18, lp[1] - 8), "PUSH-PUSH LATCH", CYAN, 0.34)
        green_cs = sorted(
            contours(masks["green"], image.shape[0] * image.shape[1] * 0.00018)[:4],
            key=lambda c: cv2.boundingRect(c)[0],
        )
        for contour in green_cs[:2]:
            x, y, w, h = cv2.boundingRect(contour)
            overlay = image.copy(); cv2.drawContours(overlay, [contour], -1, GREEN, -1)
            cv2.addWeighted(overlay, 0.10, image, 0.90, 0, dst=image)
            cv2.drawContours(image, [contour], -1, GREEN, 2, cv2.LINE_AA)
            centre = np.asarray((x + w / 2, y + h / 2), float)
            if carrier_centres:
                nearest = min(carrier_centres, key=lambda item: float(np.linalg.norm(item[1] - centre)))[0]
                finger_id = "FL" if nearest == "L0" else "FR"
            else:
                finger_id = "FL" if centre[0] < image.shape[1] / 2 else "FR"
            pill(image, (x, max(150, y - 8)), f"{finger_id}  TOOL FINGER", GREEN, 0.38)


def draw_hud(image: np.ndarray, spec: dict, state: dict, time_s: float, fps: float, hand_count: int) -> None:
    h, w = image.shape[:2]
    if spec.get("compact_hud"):
        top_h = 92
        alpha_rect(image, (0, 0), (w, top_h), PANEL, 0.86)
        alpha_rect(image, (0, h - 46), (w, h), PANEL, 0.84)
        cv2.putText(image, spec["title"], (24, 25), cv2.FONT_HERSHEY_DUPLEX, 0.50, MUTED, 1, cv2.LINE_AA)
        cv2.putText(image, state["phase"], (24, 58), cv2.FONT_HERSHEY_DUPLEX, 0.76, CYAN, 2, cv2.LINE_AA)
        labels = spec["sequence_labels"]
        x0, y0 = 24, 84
        usable = w - 48
        gap = 6
        segment = max(18, int((usable - gap * (len(labels) - 1)) / len(labels)))
        for index, label in enumerate(labels):
            x = x0 + index * (segment + gap)
            if index < state["completed"]:
                colour = GREEN
            elif index == min(state["index"], len(labels) - 1):
                colour = CYAN
            else:
                colour = (75, 82, 92)
            cv2.rectangle(image, (x, y0), (x + segment, y0 + 5), colour, -1)
            if segment > 110:
                cv2.putText(image, f"{index + 1:02d} {label}", (x, y0 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.27, colour, 1, cv2.LINE_AA)
        cv2.putText(image, f"LIVE HANDS  {hand_count}   |   GHOST = FUTURE CONTACT POSE", (24, h - 17), cv2.FONT_HERSHEY_DUPLEX, 0.39, INK, 1, cv2.LINE_AA)
        return
    top_h = 128 if w >= 900 else 156
    alpha_rect(image, (0, 0), (w, top_h), PANEL, 0.90)
    alpha_rect(image, (0, h - 72), (w, h), PANEL, 0.88)
    cv2.putText(image, spec["title"], (24, 28), cv2.FONT_HERSHEY_DUPLEX, 0.57 if w >= 900 else 0.47, MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, state["phase"], (24, 69), cv2.FONT_HERSHEY_DUPLEX, 0.92 if w >= 900 else 0.72, CYAN, 2, cv2.LINE_AA)
    step = state["step"]
    part = step["part"] if step else "—"
    destination = part_map(spec).get(part, {}).get("role", "final state")
    action = step.get("action", state["phase"]) if step else "POST-CONDITION REVIEW"
    cv2.putText(image, f"ACTIVE  {part}   |   {action}   |   DESTINATION  {destination.upper()}", (24, 99), cv2.FONT_HERSHEY_DUPLEX, 0.46 if w >= 900 else 0.36, INK, 1, cv2.LINE_AA)
    labels = spec["sequence_labels"]
    x0, y0 = 24, top_h - 13
    usable = w - 48
    gap = 5
    segment = max(18, int((usable - gap * (len(labels) - 1)) / len(labels)))
    for index, label in enumerate(labels):
        x = x0 + index * (segment + gap)
        if index < state["completed"]:
            colour = GREEN
        elif index == min(state["index"], len(labels) - 1):
            colour = CYAN
        else:
            colour = (75, 82, 92)
        cv2.rectangle(image, (x, y0 - 5), (x + segment, y0), colour, -1)
        rail_text = f"{index + 1:02d} {label}" if w >= 900 else f"{index + 1:02d} {label[:7]}"
        rail_scale = 0.28 if w >= 900 else 0.20
        if segment > (95 if w >= 900 else 54):
            cv2.putText(image, rail_text, (x, y0 - 12), cv2.FONT_HERSHEY_DUPLEX, rail_scale, colour, 1, cv2.LINE_AA)
    cv2.putText(image, f"LIVE HANDS  {hand_count}   |   REVIEWED SEQUENCE   |   GHOST = FUTURE CONTACT POSE", (24, h - 39), cv2.FONT_HERSHEY_DUPLEX, 0.43 if w >= 900 else 0.31, INK, 1, cv2.LINE_AA)
    cv2.putText(image, f"2D VIDEO EVIDENCE   •   {spec['view'].upper()}   •   {time_s:06.2f}s   •   FRAME {round(time_s * fps):04d}", (24, h - 15), cv2.FONT_HERSHEY_DUPLEX, 0.38 if w >= 900 else 0.28, MUTED, 1, cv2.LINE_AA)


def process_video(source: Path, destination: Path, track_path: Path, model: Path, spec: dict, width: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first_raw = cap.read()
    if not ok:
        raise RuntimeError(f"No frames in {source}")
    # Deliver consistent review masters: 1280-wide landscape and 720-wide
    # portrait.  ``width`` remains the landscape master width so callers only
    # need one quality control knob.
    target_width = min(width, 720) if source_h > source_w else width
    first = resize_frame(first_raw, target_width)
    render_h, render_w = first.shape[:2]
    sx, sy = render_w / source_w, render_h / source_h
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Pass one: track every hand and cache the actual future poses used by ghosts.
    tracker = HandLandmarks(model)
    records: list[list[dict[str, Any]]] = []
    try:
        frame_index = 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = resize_frame(raw, target_width)
            hands = tracker.detect(frame, int(round(frame_index * 1000 / fps)))
            records.append([serialize_hand(hand) for hand in hands])
            frame_index += 1
    finally:
        tracker.close(); cap.release()

    sources = source_centres_from_first(first, spec) if spec["kind"] == "gearbox" else {}
    definitions = part_map(spec)
    for part_id, definition in definitions.items():
        explicit = scale_point(definition.get("source_xy"), sx, sy)
        if explicit is not None:
            sources[part_id] = explicit

    ghosts: dict[tuple[int, str], dict[str, Any] | None] = {}
    targets: dict[str, np.ndarray] = {}
    for index, step in enumerate(spec["steps"]):
        definition = definitions.get(step["part"], {})
        source_anchor = sources.get(step["part"])
        explicit_target = scale_point(definition.get("target_xy"), sx, sy)
        pick_frame = min(len(records) - 1, max(0, round(step["pick"] * fps)))
        place_frame = min(len(records) - 1, max(0, round(step["place"] * fps)))
        ghosts[(index, "pick")] = nearest_hand_frame(records, pick_frame, source_anchor)
        ghosts[(index, "place")] = nearest_hand_frame(records, place_frame, explicit_target)
        if explicit_target is not None:
            targets[step["part"]] = explicit_target
        elif ghosts[(index, "place")] is not None:
            targets[step["part"]] = np.asarray(ghosts[(index, "place")]["pinch"], float)

    destination.parent.mkdir(parents=True, exist_ok=True)
    track_path.parent.mkdir(parents=True, exist_ok=True)
    silent = destination.with_name(destination.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    cap = cv2.VideoCapture(str(source))
    frame_log = []
    frame_index = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = resize_frame(raw, target_width)
        time_s = frame_index / fps
        hands = [restore_hand(item) for item in records[frame_index]]
        state = active_state(time_s, spec)
        masks = colour_masks(frame)
        pulse = 0.5 + 0.5 * math.sin(time_s * math.tau * 1.2)
        draw_part_guides(frame, masks, spec, state, time_s, sources, targets, hands)
        step = state["step"]
        ghost = None
        target = None
        if step is not None and step.get("ghost", True):
            ghost = ghosts.get((state["index"], state["ghost_kind"]))
            if state["ghost_kind"] == "pick":
                target = sources.get(step["part"])
                caption = "NEXT HAND POSE / PICK"
            else:
                target = targets.get(step["part"])
                caption = "NEXT HAND POSE / PLACE"
            if ghost is not None:
                draw_ghost_hand(frame, ghost, caption, pulse)
            if target is None and ghost is not None:
                target = np.asarray(ghost["pinch"], float)
            if target is not None:
                draw_target(frame, target, "PICK TARGET" if state["ghost_kind"] == "pick" else "PLACEMENT AXIS", pulse)
        for hand in hands:
            draw_live_hand(frame, hand)
        draw_hud(frame, spec, state, time_s, fps, len(hands))
        writer.write(frame)
        frame_log.append({
            "frame": frame_index, "time_s": round(time_s, 4), "phase": state["phase"],
            "active_part": None if step is None else step["part"], "completed_steps": state["completed"],
            "ghost_kind": state["ghost_kind"], "ghost_available": ghost is not None,
            "hands": records[frame_index],
        })
        frame_index += 1
    cap.release(); writer.release()
    mux_audio(source, silent, destination)
    track_path.write_text(json.dumps({
        "source": str(source.resolve()), "output": str(destination.resolve()), "reviewed_spec": spec,
        "fps": fps, "frame_size": [render_w, render_h], "frames": frame_log,
    }, indent=2), encoding="utf-8")
    return {
        "source": source.name, "output": str(destination.resolve()), "tracks": str(track_path.resolve()),
        "frames": frame_index, "fps": fps, "duration_s": round(frame_index / fps, 3),
        "hand_coverage": round(sum(bool(row) for row in records) / max(1, len(records)), 4),
        "reviewed_steps": len(spec["steps"]), "ghost_contacts": sum(value is not None for value in ghosts.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/newest_assembly_annotations_20261028"))
    ap.add_argument("--specs", type=Path, default=Path(__file__).parent / "newest_assembly_annotations.json")
    ap.add_argument("--hand-model", type=Path, default=Path(__file__).parent / "models" / "hand_landmarker.task")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--only", default="", help="Optional comma-separated source stems")
    args = ap.parse_args()
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    annotated = args.output_dir / "annotated"
    tracks = args.output_dir / "tracks"
    results = []
    for stem, spec in specs.items():
        if selected and stem not in selected:
            continue
        source = args.input_dir / f"{stem}.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        print(f"[annotate] {source.name}", flush=True)
        results.append(process_video(source, annotated / f"{stem}_assembly_intelligence.mp4", tracks / f"{stem}_tracks.json", args.hand_model, spec, args.width))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "run_summary.json"
    if selected and summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        by_source = {row["source"]: row for row in prior}
        by_source.update({row["source"]: row for row in results})
        summary = [by_source[name] for name in sorted(by_source)]
    else:
        summary = results
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
