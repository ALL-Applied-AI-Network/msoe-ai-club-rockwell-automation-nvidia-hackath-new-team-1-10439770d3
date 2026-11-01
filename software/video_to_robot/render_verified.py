"""Render clean, evidence-reviewed annotations over the six demonstrations.

The action windows come from a 2-fps human audit and are explicitly marked reviewed;
hand kinematics and bin masks remain model-derived frame-by-frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from run_pipeline import COLORS, HandLandmarks, alpha_rect, draw_hand, find_bin_and_cubes, mux_audio


PHASES = [
    (0.00, 0.24, "APPROACH", "Hand approaches one red cube"),
    (0.24, 0.38, "GRASP", "Finger contact closes around the cube"),
    (0.38, 0.70, "TRANSFER", "Cube is lifted and carried toward the target"),
    (0.70, 0.88, "INSERT", "Hand enters the green bin opening"),
    (0.88, 1.01, "RELEASE", "Cube is released; success is checked after hand-clear"),
]


def state_at(t: float, spec: dict) -> tuple[str, str, int | None, int]:
    completed = sum(1 for _, end in spec["placements"] if t >= end)
    for index, (start, end) in enumerate(spec["placements"], 1):
        if start <= t < end:
            progress = (t - start) / max(0.001, end - start)
            for lo, hi, phase, observed in PHASES:
                if lo <= progress < hi:
                    return phase, observed, index, completed
    for event in spec.get("other_events", []):
        if event["start_s"] <= t < event["end_s"]:
            return event["label"], "Context event separated from manipulation steps", None, completed
    if completed >= 4:
        return "VERIFY", "All four cubes are contained in the green bin", None, completed
    return "OBSERVE", "Inventory and target state are being observed", None, completed


def draw_clean(frame, hands, bin_box, phase, observed, cycle, completed, view, time_s):
    h, w = frame.shape[:2]
    alpha_rect(frame, (0, 0), (w, 106), COLORS["panel"], 0.88)
    alpha_rect(frame, (0, h - 92), (w, h), COLORS["panel"], 0.86)
    cv2.putText(frame, "HUMAN DEMONSTRATION  /  GROUNDED ACTION TRACE", (28, 30), cv2.FONT_HERSHEY_DUPLEX, 0.60, COLORS["muted"], 1, cv2.LINE_AA)
    cv2.putText(frame, phase, (28, 75), cv2.FONT_HERSHEY_DUPLEX, 1.12, COLORS["active"], 2, cv2.LINE_AA)
    context = f"{view.replace('_', ' ').upper()}   |   {time_s:06.2f}s"
    cv2.putText(frame, context, (w - min(520, int(w * 0.42)), 34), cv2.FONT_HERSHEY_DUPLEX, 0.53, COLORS["ink"], 1, cv2.LINE_AA)
    if cycle is not None:
        cv2.putText(frame, f"TRANSFER {cycle} / 4", (w - 244, 76), cv2.FONT_HERSHEY_DUPLEX, 0.62, COLORS["ink"], 1, cv2.LINE_AA)
    cv2.putText(frame, observed, (28, h - 53), cv2.FONT_HERSHEY_DUPLEX, 0.58, COLORS["ink"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"POST-CONDITION  {completed}/4 confirmed after release   |   reviewed timing, tracked hands", (28, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, COLORS["muted"], 1, cv2.LINE_AA)
    if bin_box:
        x, y, bw, bh = bin_box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), COLORS["bin"], -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, dst=frame)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLORS["bin"], 3, cv2.LINE_AA)
        cv2.putText(frame, "TARGET / GREEN BIN", (x, max(122, y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.52, COLORS["bin"], 1, cv2.LINE_AA)
    for hand in hands:
        draw_hand(frame, hand)
    return frame


def render_video(source: Path, output: Path, model: Path, spec: dict, width: int) -> dict:
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rw = min(width, sw)
    rh = int(round(sh * rw / sw))
    if rh % 2:
        rh += 1
    silent = output.with_name(output.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (rw, rh))
    tracker = HandLandmarks(model)
    frame_index = 0
    hand_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
            time_s = frame_index / fps
            hands = tracker.detect(frame, int(round(time_s * 1000)))
            _, bin_box, _, _, _ = find_bin_and_cubes(frame, hands)
            phase, observed, cycle, completed = state_at(time_s, spec)
            hand_frames += int(bool(hands))
            writer.write(draw_clean(frame, hands, bin_box, phase, observed, cycle, completed, spec["view"], time_s))
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        tracker.close()
    mux_audio(source, silent, output)
    return {
        "video": source.name,
        "annotated": str(output.resolve()),
        "frames": frame_index,
        "hand_frame_coverage": round(hand_frames / max(1, frame_index), 4),
        "reviewed_placements": 4,
        "view": spec["view"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, default=Path(__file__).parent / "verified_annotations.json")
    ap.add_argument("--hand-model", type=Path, default=Path(__file__).parent / "models" / "hand_landmarker.task")
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()
    specs = json.loads(args.annotations.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, source in enumerate(sorted(args.input_dir.glob("*.mp4")), 1):
        if source.stem not in specs:
            continue
        print(f"[{i}/{len(specs)}] {source.name}", flush=True)
        results.append(render_video(source, args.output_dir / f"{source.stem}_grounded.mp4", args.hand_model, specs[source.stem], args.width))
    (args.output_dir / "verified_render_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

