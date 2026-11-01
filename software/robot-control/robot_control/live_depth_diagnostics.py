"""Passive RGB-D snapshots and table-plane diagnostics for supervised runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base, load


CAPTURE_ROOT = Path(os.environ.get("ATEAM_LIVE_CAPTURE_DIR", "run-logs/live-current"))


def _safe_name(value: object) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return text[:80] or "unknown"


def _write_rgb(path: Path, image: object) -> None:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] >= 3:
        # The participant API documents RGB arrays; OpenCV writes BGR.
        cv2.imwrite(str(path), cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR))


def _write_depth_preview(path: Path, depth: np.ndarray) -> None:
    valid = depth > 0
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi > lo:
            preview[valid] = np.clip((depth[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), preview)


def _fit_table_plane(points: np.ndarray) -> dict:
    """Fit the dominant near-horizontal surface without using object truth."""

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    finite = np.all(np.isfinite(points), axis=1)
    # The declared cell workspace; broad Z limits intentionally admit a bad
    # table/floor offset so it can be reported instead of silently discarded.
    workspace = (
        finite
        & (points[:, 0] >= 0.20)
        & (points[:, 0] <= 0.65)
        & (points[:, 1] >= -0.30)
        & (points[:, 1] <= 0.30)
        & (points[:, 2] >= -0.40)
        & (points[:, 2] <= 0.20)
    )
    pts = points[workspace]
    if len(pts) < 200:
        return {"ok": False, "reason": "fewer than 200 workspace depth points", "points": int(len(pts))}

    # A 2 mm Z histogram finds the dominant horizontal plane even when bins and
    # cubes occupy part of the image. Fit only a narrow band around that mode.
    edges = np.arange(-0.40, 0.202, 0.002)
    counts, _ = np.histogram(pts[:, 2], bins=edges)
    index = int(np.argmax(counts))
    mode_z = float((edges[index] + edges[index + 1]) / 2.0)
    band = pts[np.abs(pts[:, 2] - mode_z) <= 0.008]
    if len(band) < 150:
        return {"ok": False, "reason": "dominant plane band too sparse", "points": int(len(pts))}

    design = np.column_stack([band[:, 0], band[:, 1], np.ones(len(band))])
    a, b, c = np.linalg.lstsq(design, band[:, 2], rcond=None)[0]
    residual = band[:, 2] - design @ np.array([a, b, c])
    centre_z = float(a * 0.425 + c)
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    offset_mm = float((centre_z - SURFACE_Z) * 1000.0)
    return {
        "ok": True,
        "workspace_points": int(len(pts)),
        "plane_points": int(len(band)),
        "mode_z_base_m": mode_z,
        "fitted_centre_z_base_m": centre_z,
        "expected_table_z_base_m": float(SURFACE_Z),
        "table_offset_mm": offset_mm,
        "rms_residual_mm": float(np.sqrt(np.mean(residual * residual)) * 1000.0),
        "tilt_deg": tilt_deg,
        "normal_base": normal.tolist(),
        "classification": "table_aligned" if abs(offset_mm) <= 6.0 and tilt_deg <= 2.0 else "needs_depth_or_extrinsic_adjustment",
    }


class LiveDepthDiagnostics:
    """Save a rolling snapshot while the supervised controller is holding."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = float(interval_s)
        self.last_capture = 0.0
        self.last_error: str | None = None
        self.cameras = load()

    def capture_if_due(self, obs: dict) -> None:
        now = time.monotonic()
        if now - self.last_capture < self.interval_s:
            return
        self.last_capture = now
        CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            report = {
                "captured_unix_s": time.time(),
                "trial": obs.get("trial"),
                "t": obs.get("t"),
                "tcp_pose": np.asarray(obs.get("tcp_pose", []), dtype=float).tolist(),
                "joints": np.asarray(obs.get("joints", []), dtype=float).tolist(),
            }
            for name, image in (obs.get("images") or {}).items():
                _write_rgb(CAPTURE_ROOT / f"latest_{_safe_name(name)}.jpg", image)

            for name, value in (obs.get("depth") or {}).items():
                depth = np.asarray(value, dtype=np.uint16)
                np.save(CAPTURE_ROOT / f"latest_{_safe_name(name)}_depth_mm.npy", depth)
                _write_depth_preview(CAPTURE_ROOT / f"latest_{_safe_name(name)}_depth.png", depth)
                report[f"{name}_depth"] = {
                    "shape": list(depth.shape),
                    "valid_fraction": float(np.mean(depth > 0)),
                    "minimum_valid_mm": int(depth[depth > 0].min()) if np.any(depth > 0) else None,
                    "maximum_valid_mm": int(depth.max()) if np.any(depth > 0) else None,
                }

            table_depth = (obs.get("depth") or {}).get("table")
            if table_depth is not None:
                points = depth_to_base(self.cameras["table_depth"], table_depth, stride=2)
                report["table_plane"] = _fit_table_plane(points)

            wrist_depth = (obs.get("depth") or {}).get("wrist")
            joints = obs.get("joints")
            if wrist_depth is not None and joints is not None:
                points = depth_to_base(self.cameras["wrist_depth"], wrist_depth, stride=2, joints=joints)
                report["wrist_visible_plane"] = _fit_table_plane(points)

            temporary = CAPTURE_ROOT / "latest_report.json.tmp"
            temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            os.replace(temporary, CAPTURE_ROOT / "latest_report.json")
            self.last_error = None
        except Exception as exc:  # Diagnostics must never interrupt safe holding.
            message = f"{type(exc).__name__}: {exc}"
            if message != self.last_error:
                print("[A-Team depth] " + json.dumps({"event": "capture_error", "error": message}), flush=True)
                self.last_error = message
