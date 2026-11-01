"""Create a mechanism-focused explainer for the physical push-push fingers.

This renderer is intentionally separate from the generic assembly annotations.  It
uses reviewed event windows, MediaPipe hand landmarks, and colour evidence to name
the visible hardware while a clearly marked *conceptual* inset explains the hidden
push-push latch.  It never claims that the internal latch geometry is visible in the
recording.
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


# OpenCV colours are BGR.
PANEL = (20, 24, 30)
PANEL_2 = (31, 37, 44)
INK = (246, 247, 248)
MUTED = (171, 181, 190)
CYAN = (235, 198, 47)
BLUE = (246, 158, 64)
GREEN = (102, 213, 118)
AMBER = (57, 178, 247)
GRAY = (205, 210, 216)
DIM = (76, 84, 94)


PHASES: list[dict[str, Any]] = [
    {
        "start": 0.0, "end": 1.8, "title": "BASELINE Finger_Y INSTALLED",
        "body": "Receiver_Y is fixed to the SMC jaw; the lower blue Finger_Y is the removable insert.",
        "state": "locked", "stage": 2, "side": "left", "evidence": "VISIBLE BASELINE CONFIGURATION",
    },
    {
        "start": 1.8, "end": 4.9, "title": "SECOND PUSH / BLUE BASELINE FINGER",
        "body": "Push to the stop again: the latch peg leaves the detent and enters the exit branch.",
        "state": "push2", "stage": 3, "side": "left", "evidence": "OBSERVED RELEASE INPUT",
    },
    {
        "start": 4.9, "end": 5.8, "title": "BASELINE FINGER RELEASED",
        "body": "The return spring presents the blue Finger_Y insert so it can slide off Receiver_Y.",
        "state": "released", "stage": 4, "side": "left", "evidence": "OBSERVED WITHDRAWAL",
    },
    {
        "start": 5.8, "end": 6.5, "title": "EMPTY Receiver_Y",
        "body": "With the finger removed, the return / eject spring is briefly visible at the interface.",
        "state": "ready", "stage": 0, "side": "left", "evidence": "VISIBLE EMPTY RECEIVER",
    },
    {
        "start": 6.5, "end": 9.6, "title": "REINSERT + FIRST PUSH",
        "body": "Align the blue Finger_Y tongue, slide it in, then push to route the peg toward lock.",
        "state": "push1", "stage": 1, "side": "left", "evidence": "OBSERVED REINSERTION",
    },
    {
        "start": 9.6, "end": 11.5, "title": "BASELINE FINGER LOCKED",
        "body": "Let go: the return spring seats the latch peg in the cam-track detent.",
        "state": "locked", "stage": 2, "side": "left", "evidence": "OBSERVED RETENTION",
    },
    {
        "start": 11.5, "end": 14.0, "title": "ONE CAM TRACK / TWO PUSHES",
        "body": "Push 1 locks in the detent. Pulling stays retained. Push 2 selects the release path.",
        "state": "locked", "stage": 2, "side": "left", "evidence": "CAD-GROUNDED SCHEMATIC",
    },
    {
        "start": 14.0, "end": 17.5, "title": "ALIGN GREEN TOOL FINGER / LEFT",
        "body": "The green part is an alternate Finger_Y tool insert; guide its tongue into Receiver_Y.",
        "state": "ready", "stage": 0, "side": "left", "evidence": "OBSERVED ALIGNMENT",
    },
    {
        "start": 17.5, "end": 18.9, "title": "FIRST PUSH / LEFT",
        "body": "Push to the stop; the Latch_shuttle peg enters the locking branch of Finger_Y's cam track.",
        "state": "push1", "stage": 1, "side": "left", "evidence": "OBSERVED INSERTION",
    },
    {
        "start": 18.9, "end": 20.5, "title": "LEFT GREEN FINGER LOCKED",
        "body": "Hand clear: the peg is seated in the detent. Pulling alone does not release the finger.",
        "state": "locked", "stage": 2, "side": "left", "evidence": "LOCKED STATE",
    },
    {
        "start": 20.5, "end": 24.5, "title": "ALIGN GREEN TOOL FINGER / RIGHT",
        "body": "The second Receiver_Y accepts the same standardized, interchangeable Finger_Y interface.",
        "state": "ready", "stage": 0, "side": "right", "evidence": "OBSERVED ALIGNMENT",
    },
    {
        "start": 24.5, "end": 31.25, "title": "FIRST PUSH / RIGHT",
        "body": "A full inward stroke routes the second latch peg into the cam-track detent.",
        "state": "push1", "stage": 1, "side": "right", "evidence": "OBSERVED INSERTION",
    },
    {
        "start": 31.25, "end": 33.5, "title": "BOTH GREEN FINGERS LOCKED",
        "body": "Both alternate Finger_Y inserts are now retained by their Receiver_Y latch interfaces.",
        "state": "locked", "stage": 2, "side": "right", "evidence": "LOCKED STATE",
    },
    {
        "start": 33.5, "end": 42.0, "title": "LOCKED FOR TOOL USE",
        "body": "Working loads pull outward; the peg remains in the detent until another inward push.",
        "state": "pull", "stage": 2, "side": "both", "evidence": "MECHANISM EXPLANATION",
    },
    {
        "start": 42.0, "end": 49.8, "title": "INTERCHANGEABLE END EFFECTOR",
        "body": "Swap Finger_Y variants to change contact geometry, bit, socket, probe, or tool function.",
        "state": "locked", "stage": 2, "side": "both", "evidence": "VISIBLE INSTALLED PAIR",
    },
    {
        "start": 49.8, "end": 53.2, "title": "SECOND PUSH / LEFT",
        "body": "Push to the stop again: the peg leaves the detent and enters the cam track's exit branch.",
        "state": "push2", "stage": 3, "side": "left", "evidence": "OBSERVED RELEASE INPUT",
    },
    {
        "start": 53.2, "end": 54.0, "title": "LEFT FINGER RELEASED",
        "body": "The return spring presents Finger_Y; withdraw it by hand and replace it immediately.",
        "state": "released", "stage": 4, "side": "left", "evidence": "OBSERVED REMOVAL",
    },
    {
        "start": 54.0, "end": 55.9, "title": "SECOND PUSH / RIGHT",
        "body": "The same second-push path releases the opposite Latch_shuttle peg.",
        "state": "push2", "stage": 3, "side": "right", "evidence": "OBSERVED RELEASE INPUT",
    },
    {
        "start": 55.9, "end": 57.0, "title": "RIGHT FINGER RELEASED",
        "body": "The spring presents the released Finger_Y; withdraw it and install the next tool.",
        "state": "released", "stage": 4, "side": "right", "evidence": "OBSERVED REMOVAL",
    },
    {
        "start": 57.0, "end": 1e9, "title": "BOTH RECEIVERS EMPTY / READY",
        "body": "No screws at tool change: align, push once to lock, push again to release, then slide off.",
        "state": "ready", "stage": 4, "side": "both", "evidence": "CYCLE COMPLETE",
    },
]


CYCLE = [
    ("ALIGN", "Finger enters receiver"),
    ("PUSH 1", "Toggle to locked"),
    ("LOCKED", "Pull remains retained"),
    ("PUSH 2", "Toggle to release"),
    ("SWAP", "Remove and replace"),
]


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    rw = min(width, w)
    rh = int(round(h * rw / w))
    if rh % 2:
        rh += 1
    return cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)


def clean_mask(mask: np.ndarray, opening: int, closing: int) -> np.ndarray:
    if opening:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((opening, opening), np.uint8))
    if closing:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing, closing), np.uint8))
    return mask


def colour_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = clean_mask(cv2.inRange(hsv, np.array([88, 50, 28]), np.array([140, 255, 255])), 3, 9)
    green = clean_mask(cv2.inRange(hsv, np.array([35, 45, 40]), np.array([95, 255, 255])), 3, 7)
    return blue, green


def find_contours(mask: np.ndarray, minimum: float) -> list[np.ndarray]:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted((c for c in cs if cv2.contourArea(c) >= minimum), key=cv2.contourArea, reverse=True)


def component_contours(frame: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    h, w = frame.shape[:2]
    area = h * w
    blue, green = colour_masks(frame)
    blue_candidates = find_contours(blue, area * 0.00045)[:12]
    receiver_candidates = []
    for contour in blue_candidates:
        x, y, bw, bh = cv2.boundingRect(contour)
        cy = y + bh / 2
        if 0.20 * w < x + bw / 2 < 0.77 * w and cy < 0.64 * h and bh > 0.07 * h:
            receiver_candidates.append(contour)
    receivers = sorted(receiver_candidates[:4], key=lambda c: cv2.boundingRect(c)[0])[:2]

    # During the opening proof cycle the removable blue baseline Finger_Y is
    # physically withdrawn into the lower half of the image.  Keep that loose
    # part separate from the two receiver bodies so callouts cannot jump from
    # the fixed hardware to the demonstrator's hand.
    loose_blue = []
    for contour in blue_candidates:
        x, y, bw, bh = cv2.boundingRect(contour)
        cx, cy = x + bw / 2, y + bh / 2
        if 0.10 * w < cx < 0.76 * w and cy >= 0.54 * h and max(bw, bh) > 0.045 * h:
            loose_blue.append(contour)
    loose_blue = sorted(loose_blue, key=cv2.contourArea, reverse=True)[:1]

    finger_candidates = []
    for contour in find_contours(green, area * 0.00035)[:10]:
        x, y, bw, bh = cv2.boundingRect(contour)
        if 0.18 * w < x + bw / 2 < 0.80 * w and y < 0.94 * h and max(bw, bh) > 0.055 * h:
            finger_candidates.append(contour)
    fingers = sorted(finger_candidates[:4], key=lambda c: cv2.boundingRect(c)[0])[:2]
    return receivers, fingers, loose_blue


def phase_at(time_s: float) -> tuple[int, dict[str, Any]]:
    for index, phase in enumerate(PHASES):
        if phase["start"] <= time_s < phase["end"]:
            return index, phase
    return len(PHASES) - 1, PHASES[-1]


def contour_center(contour: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    return np.asarray((x + w / 2, y + h / 2), dtype=float)


def paint_contour(image: np.ndarray, contour: np.ndarray, colour: tuple[int, int, int], alpha: float) -> None:
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, colour, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    cv2.drawContours(image, [contour], -1, colour, 2, cv2.LINE_AA)


def draw_arrow(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], colour, thickness: int = 2) -> None:
    cv2.arrowedLine(image, start, end, colour, thickness, cv2.LINE_AA, tipLength=0.12)


def label_card(
    image: np.ndarray,
    y: int,
    code: str,
    title: str,
    subtitle: str,
    colour: tuple[int, int, int],
    anchor: np.ndarray | None,
) -> None:
    x, width, height = 22, 354, 58
    alpha_rect(image, (x, y), (x + width, y + height), PANEL, 0.86)
    cv2.rectangle(image, (x, y), (x + 4, y + height), colour, -1)
    cv2.putText(image, code, (x + 16, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.38, colour, 1, cv2.LINE_AA)
    cv2.putText(image, title, (x + 68, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.46, INK, 1, cv2.LINE_AA)
    cv2.putText(image, subtitle, (x + 16, y + 44), cv2.FONT_HERSHEY_DUPLEX, 0.33, MUTED, 1, cv2.LINE_AA)
    if anchor is not None:
        a = tuple(np.round(anchor).astype(int))
        elbow = (x + width + 28, y + height // 2)
        cv2.line(image, (x + width, y + height // 2), elbow, colour, 1, cv2.LINE_AA)
        draw_arrow(image, elbow, a, colour, 1)
        cv2.circle(image, a, 5, colour, 2, cv2.LINE_AA)


def draw_hand_skeleton(image: np.ndarray, hand: dict[str, Any]) -> None:
    pts = hand["landmarks"][:, :2].astype(np.int32)
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, cv2.convexHull(pts), CYAN)
    cv2.addWeighted(overlay, 0.07, image, 0.93, 0, dst=image)
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), CYAN, 1, cv2.LINE_AA)
    for index, point in enumerate(pts):
        cv2.circle(image, tuple(point), 3 if index in (4, 8) else 2, AMBER if index in (4, 8) else INK, -1, cv2.LINE_AA)
    pinch = tuple(np.round(hand["pinch"]).astype(int))
    cv2.circle(image, pinch, 7, AMBER, 2, cv2.LINE_AA)


def draw_lock_icon(image: np.ndarray, center: tuple[int, int], colour) -> None:
    x, y = center
    cv2.rectangle(image, (x - 10, y), (x + 10, y + 16), colour, 2, cv2.LINE_AA)
    cv2.ellipse(image, (x, y), (7, 9), 180, 0, 180, colour, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y + 7), 2, colour, -1, cv2.LINE_AA)


def draw_spring(image: np.ndarray, x0: int, x1: int, y: int, colour) -> None:
    points = [(x0, y)]
    turns = 7
    for i in range(1, turns * 2):
        x = int(round(x0 + (x1 - x0) * i / (turns * 2)))
        points.append((x, y + (-6 if i % 2 else 6)))
    points.append((x1, y))
    cv2.polylines(image, [np.asarray(points, np.int32)], False, colour, 1, cv2.LINE_AA)


def draw_mechanism_inset(image: np.ndarray, state: str, pulse: float) -> None:
    h, w = image.shape[:2]
    px, py, pw, ph = w - 382, h - 276, 360, 198
    alpha_rect(image, (px, py), (px + pw, py + ph), PANEL, 0.91)
    cv2.rectangle(image, (px, py), (px + pw, py + ph), DIM, 1, cv2.LINE_AA)
    cv2.putText(image, "CAD-GROUNDED SCHEMATIC", (px + 16, py + 23), cv2.FONT_HERSHEY_DUPLEX, 0.38, AMBER, 1, cv2.LINE_AA)
    cv2.putText(image, "CAM TRACK + LATCH PEG", (px + 16, py + 44), cv2.FONT_HERSHEY_DUPLEX, 0.48, INK, 1, cv2.LINE_AA)
    cv2.putText(image, "Hidden interface — schematic, not live X-ray", (px + 16, py + 62), cv2.FONT_HERSHEY_DUPLEX, 0.29, MUTED, 1, cv2.LINE_AA)

    # Finger_Y tongue carries the cam track.  The red peg belongs to the
    # Latch_shuttle and follows a different branch on alternating pushes.
    tx0, ty0, tw, th = px + 79, py + 76, 238, 72
    cv2.rectangle(image, (tx0, ty0), (tx0 + tw, ty0 + th), GREEN, -1)
    cv2.rectangle(image, (tx0, ty0), (tx0 + tw, ty0 + th), INK, 1, cv2.LINE_AA)
    cv2.putText(image, "Finger_Y TONGUE / CAM TRACK", (tx0 + 21, ty0 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.29, GREEN, 1, cv2.LINE_AA)

    entry = np.asarray((tx0 + tw + 3, ty0 + 34), int)
    upper_stop = np.asarray((tx0 + 29, ty0 + 22), int)
    detent = np.asarray((tx0 + 112, ty0 + 39), int)
    lower_stop = np.asarray((tx0 + 29, ty0 + 54), int)
    exit_pt = np.asarray((tx0 + tw + 3, ty0 + 55), int)
    upper_path = np.asarray([entry, (tx0 + 179, ty0 + 20), (tx0 + 150, ty0 + 20),
                             (tx0 + 132, ty0 + 35), detent, upper_stop], np.int32)
    lower_path = np.asarray([detent, (tx0 + 91, ty0 + 52), lower_stop,
                             (tx0 + 167, ty0 + 57), exit_pt], np.int32)
    cv2.polylines(image, [upper_path], False, (232, 234, 235), 7, cv2.LINE_AA)
    cv2.polylines(image, [lower_path], False, (232, 234, 235), 7, cv2.LINE_AA)
    cv2.polylines(image, [upper_path], False, PANEL_2, 1, cv2.LINE_AA)
    cv2.polylines(image, [lower_path], False, PANEL_2, 1, cv2.LINE_AA)

    peg_by_state = {
        "ready": entry,
        "push1": upper_stop,
        "locked": detent,
        "pull": detent,
        "push2": lower_stop,
        "released": exit_pt,
    }
    peg = peg_by_state.get(state, entry)
    cv2.circle(image, tuple(peg), 8, (54, 54, 232), -1, cv2.LINE_AA)
    cv2.circle(image, tuple(peg), int(11 + pulse * 4), CYAN if state in ("push1", "push2") else INK, 1, cv2.LINE_AA)
    cv2.putText(image, "LATCH PEG", (px + 16, ty0 + 31), cv2.FONT_HERSHEY_DUPLEX, 0.29, (80, 90, 235), 1, cv2.LINE_AA)
    draw_arrow(image, (px + 65, ty0 + 34), (int(peg[0] - 8), int(peg[1])), (80, 90, 235), 1)

    # Return/eject spring is shown separately because it provides the outward
    # stroke after the second push; it is briefly visible in the live video.
    draw_spring(image, tx0 + 20, tx0 + 92, ty0 + th + 14, GRAY)
    cv2.putText(image, "RETURN / EJECT SPRING", (tx0 + 102, ty0 + th + 18), cv2.FONT_HERSHEY_DUPLEX, 0.28, GRAY, 1, cv2.LINE_AA)
    if state in ("locked", "pull"):
        draw_lock_icon(image, (tx0 + tw - 18, ty0 + th + 7), GREEN)

    captions = {
        "ready": "ALIGN / PEG ENTERS TRACK",
        "push1": "PUSH 1 → FIRST STOP",
        "locked": "LET GO → PEG SEATS IN DETENT",
        "pull": "PULL → PEG REMAINS IN DETENT",
        "push2": "PUSH 2 → EXIT BRANCH",
        "released": "SPRING PRESENTS FINGER → WITHDRAW",
    }
    caption = captions.get(state, captions["ready"])
    colour = GREEN if state in ("locked", "pull", "released") else CYAN if state in ("push1", "push2") else MUTED
    cv2.putText(image, caption, (px + 16, py + ph - 9), cv2.FONT_HERSHEY_DUPLEX, 0.34, colour, 1, cv2.LINE_AA)


def draw_cycle_rail(image: np.ndarray, stage: int) -> None:
    h, w = image.shape[:2]
    x0, y0 = 24, 110
    gap = 7
    width = int((w - 48 - gap * (len(CYCLE) - 1)) / len(CYCLE))
    for index, (label, _) in enumerate(CYCLE):
        x = x0 + index * (width + gap)
        if index < stage:
            colour = GREEN
        elif index == stage:
            colour = CYAN
        else:
            colour = DIM
        cv2.putText(image, f"{index + 1:02d}  {label}", (x, y0 - 9), cv2.FONT_HERSHEY_DUPLEX, 0.32, colour, 1, cv2.LINE_AA)
        cv2.rectangle(image, (x, y0), (x + width, y0 + 5), colour, -1)


def draw_top_hud(image: np.ndarray, phase: dict[str, Any], time_s: float) -> None:
    h, w = image.shape[:2]
    alpha_rect(image, (0, 0), (w, 124), PANEL, 0.90)
    cv2.putText(image, "PUSH-PUSH QUICK-SWAP  /  Receiver_Y + Finger_Y", (24, 25), cv2.FONT_HERSHEY_DUPLEX, 0.52, MUTED, 1, cv2.LINE_AA)
    cv2.putText(image, phase["title"], (24, 59), cv2.FONT_HERSHEY_DUPLEX, 0.84, CYAN, 2, cv2.LINE_AA)
    cv2.putText(image, phase["body"], (24, 84), cv2.FONT_HERSHEY_DUPLEX, 0.42, INK, 1, cv2.LINE_AA)
    cv2.putText(image, f"{time_s:05.1f}s", (w - 96, 27), cv2.FONT_HERSHEY_DUPLEX, 0.50, INK, 1, cv2.LINE_AA)
    draw_cycle_rail(image, int(phase["stage"]))


def draw_bottom_hud(image: np.ndarray, phase: dict[str, Any], hand_count: int) -> None:
    h, w = image.shape[:2]
    alpha_rect(image, (0, h - 61), (w, h), PANEL, 0.88)
    cv2.putText(image, f"STATE  {phase['state'].upper()}   |   SIDE  {phase['side'].upper()}   |   HANDS  {hand_count}",
                (24, h - 34), cv2.FONT_HERSHEY_DUPLEX, 0.40, INK, 1, cv2.LINE_AA)
    cv2.putText(image, f"{phase['evidence']}   •   VISIBLE PARTS TRACKED IN COLOUR   •   INTERNAL LATCH CALLOUT IS CONCEPTUAL",
                (24, h - 13), cv2.FONT_HERSHEY_DUPLEX, 0.32, MUTED, 1, cv2.LINE_AA)


def draw_motion_cue(image: np.ndarray, anchor: np.ndarray | None, state: str) -> None:
    if anchor is None:
        return
    x, y = np.round(anchor).astype(int)
    if state in ("push1", "push2"):
        draw_arrow(image, (x, y + 74), (x, y - 32), CYAN, 3)
        text = "PUSH 1 / LOCK" if state == "push1" else "PUSH 2 / RELEASE"
        cv2.putText(image, text, (x + 13, y + 58), cv2.FONT_HERSHEY_DUPLEX, 0.39, CYAN, 1, cv2.LINE_AA)
    elif state == "released":
        draw_arrow(image, (x, y - 22), (x, y + 78), GREEN, 3)
        cv2.putText(image, "REMOVE / SWAP", (x + 13, y + 65), cv2.FONT_HERSHEY_DUPLEX, 0.39, GREEN, 1, cv2.LINE_AA)


def render_frame(
    frame: np.ndarray,
    hands: list[dict[str, Any]],
    phase: dict[str, Any],
    time_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    receivers, fingers, loose_blue = component_contours(frame)
    receiver_records = []
    finger_records = []
    for contour in receivers:
        paint_contour(frame, contour, BLUE, 0.10)
        x, y, w, h = cv2.boundingRect(contour)
        receiver_records.append({"bbox": [x, y, w, h], "center": [x + w / 2, y + h / 2]})
    for contour in fingers:
        paint_contour(frame, contour, GREEN, 0.12)
        x, y, w, h = cv2.boundingRect(contour)
        finger_records.append({"bbox": [x, y, w, h], "center": [x + w / 2, y + h / 2]})
    for contour in loose_blue:
        paint_contour(frame, contour, BLUE, 0.12)

    receiver_centres = [contour_center(c) for c in receivers]
    finger_centres = [contour_center(c) for c in fingers]
    left_receiver = receiver_centres[0] if receiver_centres else None
    right_receiver = receiver_centres[-1] if len(receiver_centres) >= 2 else None
    receiver_mid = np.mean(receiver_centres, axis=0) if receiver_centres else None
    gripper_anchor = None if receiver_mid is None else receiver_mid + np.asarray((0, -70), float)
    baseline_anchor = contour_center(loose_blue[0]) if loose_blue else None

    if phase["side"] == "left":
        active_receiver = left_receiver
        active_finger = finger_centres[0] if time_s >= 14.0 and finger_centres else None
    elif phase["side"] == "right":
        active_receiver = right_receiver if right_receiver is not None else left_receiver
        active_finger = finger_centres[-1] if time_s >= 14.0 and finger_centres else None
    else:
        active_receiver = receiver_mid
        active_finger = np.mean(finger_centres, axis=0) if time_s >= 14.0 and finger_centres else None

    # The interface node is estimated from the visible receiver box; the latch itself
    # is internal and therefore labelled as a conceptual location.
    interface_anchor = None
    if active_receiver is not None and receivers:
        nearest = min(receivers, key=lambda c: float(np.linalg.norm(contour_center(c) - active_receiver)))
        x, y, w, h = cv2.boundingRect(nearest)
        interface_anchor = np.asarray((x + w / 2, y + h * 0.70), float)
        pulse = 0.5 + 0.5 * math.sin(time_s * math.tau * 1.35)
        cv2.circle(frame, tuple(np.round(interface_anchor).astype(int)), int(14 + 6 * pulse), CYAN, 2, cv2.LINE_AA)
    else:
        pulse = 0.5 + 0.5 * math.sin(time_s * math.tau * 1.35)

    # Restrained articulated overlay: visible only while a person interacts with the mechanism.
    if time_s >= 12.0 or hands:
        for hand in hands:
            draw_hand_skeleton(frame, hand)

    label_card(frame, 144, "H0", "SMC GRIPPER JAW", "Fixed parallel-jaw end-effector hardware", GRAY, gripper_anchor)
    label_card(frame, 208, "R0", "Receiver_Y", "Blue housing remains bolted to the jaw", BLUE, active_receiver)
    if time_s < 14.0:
        label_card(frame, 272, "B1", "BASELINE Finger_Y", "Lower blue insert is removable", BLUE,
                   baseline_anchor if baseline_anchor is not None else interface_anchor)
    elif active_finger is not None:
        label_card(frame, 272, "F0", "ALTERNATE Finger_Y", "Green interchangeable tool finger", GREEN, active_finger)
    if interface_anchor is not None:
        label_card(frame, 336, "L0", "Latch_shuttle + PEG", "Peg follows the cam track inside Finger_Y", AMBER, interface_anchor)

    motion_anchor = active_finger if active_finger is not None else baseline_anchor if baseline_anchor is not None else interface_anchor
    draw_motion_cue(frame, motion_anchor, phase["state"])
    draw_mechanism_inset(frame, phase["state"], pulse)
    draw_top_hud(frame, phase, time_s)
    draw_bottom_hud(frame, phase, len(hands))

    return frame, {
        "receiver_components": receiver_records,
        "finger_components": finger_records,
        "loose_baseline_finger": None if baseline_anchor is None else baseline_anchor.astype(float).tolist(),
        "active_receiver": None if active_receiver is None else active_receiver.astype(float).tolist(),
        "active_finger": None if active_finger is None else active_finger.astype(float).tolist(),
        "interface_callout": None if interface_anchor is None else interface_anchor.astype(float).tolist(),
    }


def process(source: Path, output: Path, tracks: Path, hand_model: Path, width: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_w = min(width, source_w)
    render_h = int(round(source_h * render_w / source_w))
    if render_h % 2:
        render_h += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    tracks.parent.mkdir(parents=True, exist_ok=True)
    silent = output.with_name(output.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    tracker = HandLandmarks(hand_model)
    rows: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = resize_frame(raw, render_w)
            time_s = frame_index / fps
            hands = tracker.detect(frame, int(round(frame_index * 1000 / fps)))
            phase_index, phase = phase_at(time_s)
            frame, evidence = render_frame(frame, hands, phase, time_s)
            writer.write(frame)
            rows.append({
                "frame": frame_index,
                "time_s": round(time_s, 4),
                "phase_index": phase_index,
                "phase": phase["title"],
                "mechanism_state": phase["state"],
                "side": phase["side"],
                "hand_count": len(hands),
                **evidence,
            })
            frame_index += 1
    finally:
        tracker.close()
        cap.release()
        writer.release()

    mux_audio(source, silent, output)
    payload = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "frame_size": [render_w, render_h],
        "fps": fps,
        "frames": frame_index,
        "duration_s": round(frame_index / fps, 3),
        "reviewed_phases": PHASES,
        "scope": "Visible 2D evidence plus a clearly labelled CAD-grounded cam-track schematic; not live X-ray imagery.",
        "frame_records": rows,
    }
    tracks.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {key: value for key, value in payload.items() if key != "frame_records"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/push_push_explainer_20261028"))
    ap.add_argument("--hand-model", type=Path, default=Path(__file__).parent / "models" / "hand_landmarker.task")
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()
    output = args.output_dir / "PXL_20261028_203509927_push_push_mechanism_explainer.mp4"
    tracks = args.output_dir / "PXL_20261028_203509927_push_push_mechanism_tracks.json"
    result = process(args.source, output, tracks, args.hand_model, args.width)
    (args.output_dir / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
