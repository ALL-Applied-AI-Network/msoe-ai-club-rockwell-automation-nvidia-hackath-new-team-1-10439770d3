"""Simulation-gated candidate policy for a calibrated one-wall bin lift.

This is intentionally separate from the active competition ``policy.py``.
It requests every RGB view plus both organizer-provided depth streams, fuses
independent table RGB/depth geometry, uses wrist depth for close-range evidence,
and executes a monotonic proof-lift/full-lift/hold sequence.  There is no
automatic lower or release; failed evidence holds position and fails closed.
"""

from __future__ import annotations

import json
from statistics import median

from .adaptive_types import PolicyConfiguration
from .bin_vision import BinVisionError, locate_green_bin
from .depth_vision import DepthVisionError, detect_depth_clusters
from .phase_controller import PhaseController, bin_lift_phases
from .sensor_fusion import (
    LiftEvidenceTracker,
    SensorFusionError,
    fuse_bin_pose,
    wrist_presence_crosscheck,
)
from .side_vision import SideLiftTracker, SideVisionError, locate_green_region


CONFIGURATION = PolicyConfiguration(
    cameras=("table", "side", "wrist"),
    depth=("table", "wrist"),
    image_size=(848, 480),
    image_format="jpeg",
    action="tcp",
    prefetch=False,
    dt_s=0.1,
)


class AdaptiveBinLiftPolicy:
    CAMERAS = list(CONFIGURATION.cameras)
    DEPTH = list(CONFIGURATION.depth)
    IMAGE_SIZE = CONFIGURATION.image_size
    IMAGE_FORMAT = CONFIGURATION.image_format
    ACTION = CONFIGURATION.action
    PREFETCH = CONFIGURATION.prefetch
    DT = CONFIGURATION.dt_s

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint
        self.reset()

    def reset(self) -> None:
        self.controller: PhaseController | None = None
        self.fused_pose = None
        self.tracker: LiftEvidenceTracker | None = None
        self.side_tracker: SideLiftTracker | None = None
        self.evidence_samples: dict[str, list[float]] = {}
        self.side_evidence_samples: dict[str, list[float]] = {}
        self.wrist_presence_samples: dict[str, list[bool]] = {}
        self.audit: list[dict] = []
        self.controller_history_cursor = 0
        self.initialization_error: str | None = None

    def _record_event(self, event: dict) -> None:
        """Keep an in-memory audit and mirror it to stdout for run logging."""

        self.audit.append(event)
        print("[A-Team adaptive] " + json.dumps(event, sort_keys=True), flush=True)

    @staticmethod
    def _hold_observed_pose(obs: dict) -> list[float]:
        """Fail closed by commanding the pose and gripper already observed."""

        tcp_pose = obs.get("tcp_pose")
        joints = obs.get("joints")
        if joints is None:
            joints = []
        gripper = float(joints[6]) if len(joints) >= 7 else 0.0
        if tcp_pose is None or len(tcp_pose) < 6:
            # The organizer's local mock has no forward kinematics, so its
            # first observation deliberately reports tcp_pose=None.  Seed the
            # mock once with a conservative in-bounds pose; never use this
            # fallback for a physical trial.
            trial_id = str((obs.get("trial") or {}).get("trial_id", ""))
            if trial_id.startswith("mock-"):
                return [300.0, 0.0, -90.0, -180.0, 0.0, 0.0, gripper]
            raise SensorFusionError("cannot hold safely without an observed tcp_pose")
        return [float(value) for value in tcp_pose[:6]] + [gripper]

    @staticmethod
    def _instruction(obs: dict) -> str:
        trial = obs.get("trial") or {}
        return str(trial.get("instruction", ""))

    def _clusters(self, obs: dict, streams=("table", "wrist")):
        depth = obs.get("depth") or {}
        joints = obs.get("joints")
        clusters = []
        errors = []
        for stream in streams:
            frame = depth.get(stream)
            if frame is None:
                errors.append(f"{stream}:missing")
                continue
            try:
                clusters.extend(
                    detect_depth_clusters(
                        frame,
                        stream,
                        joints=joints if stream == "wrist" else None,
                        stride=2,
                        minimum_height_m=0.006,
                    )
                )
            except DepthVisionError as exc:
                errors.append(f"{stream}:{exc}")
        return clusters, errors

    def _initialize(self, obs: dict) -> None:
        images = obs.get("images") or {}
        table_rgb = images.get("table")
        side_rgb = images.get("side")
        if table_rgb is None:
            raise SensorFusionError("table RGB is required for bin orientation")
        if side_rgb is None:
            raise SensorFusionError("side RGB is required for independent lift evidence")
        rgb = locate_green_bin(table_rgb)
        side_region = locate_green_region(side_rgb)
        # The fixed table depth camera owns metric translation.  Wrist depth is
        # intentionally excluded here: close to the fingers it can see only a
        # partial bin footprint and bias a fit even though it remains valuable
        # for independent retention evidence later.
        clusters, errors = self._clusters(obs, streams=("table",))
        fused = fuse_bin_pose(rgb, clusters)
        x_mm, y_mm, _ = fused.wall_center_base_mm
        self.controller = PhaseController(
            bin_lift_phases(x_mm, y_mm, fused.gripper_yaw_deg, grip_z_mm=-298.1),
            position_tolerance_mm=3.0,
            rotation_tolerance_deg=2.0,
            settled_observations=2,
        )
        self.fused_pose = fused
        self.tracker = LiftEvidenceTracker(
            fused.center_base_mm[:2], fused.center_base_mm[2]
        )
        self.side_tracker = SideLiftTracker(side_region)
        event = {
            "event": "initialized",
            "instruction": self._instruction(obs),
            "fused_pose": fused.to_dict(),
            "depth_warnings": errors,
            "side_region": side_region.to_dict(),
        }
        self._record_event(event)

    def _verification(self, obs: dict) -> dict[str, bool | None]:
        if self.controller is None or self.tracker is None or self.side_tracker is None:
            return {}
        key = self.controller.phase.verification
        if not key:
            return {}
        table_clusters, table_errors = self._clusters(obs, streams=("table",))
        if not table_clusters:
            self._record_event(
                {
                    "event": "table_depth_missing",
                    "phase": self.controller.phase.name,
                    "errors": table_errors,
                }
            )
            return {key: None}
        try:
            lift_mm = self.tracker.observe(table_clusters)
        except SensorFusionError as exc:
            self._record_event(
                {"event": "tracking_rejected", "phase": self.controller.phase.name, "error": str(exc)}
            )
            return {key: None}
        samples = self.evidence_samples.setdefault(key, [])
        samples.append(float(lift_mm))
        samples[:] = samples[-5:]

        wrist_clusters, wrist_errors = self._clusters(obs, streams=("wrist",))
        wrist_present, wrist_distance_mm = wrist_presence_crosscheck(
            wrist_clusters, self.tracker.center_xy_mm
        )
        wrist_samples = self.wrist_presence_samples.setdefault(key, [])
        wrist_samples.append(bool(wrist_present))
        wrist_samples[:] = wrist_samples[-5:]
        try:
            side_region = locate_green_region(obs.get("images", {}).get("side"))
            side_lift_px = self.side_tracker.observe(side_region)
        except (SideVisionError, TypeError) as exc:
            self._record_event(
                {"event": "side_tracking_rejected", "phase": self.controller.phase.name, "error": str(exc)}
            )
            return {key: None}
        side_samples = self.side_evidence_samples.setdefault(key, [])
        side_samples.append(float(side_lift_px))
        side_samples[:] = side_samples[-5:]
        if len(samples) < 3 or len(side_samples) < 3 or len(wrist_samples) < 3:
            return {key: None}
        required = {"proof_lift": 2.0, "full_lift": 8.0, "retained": 8.0}[key]
        required_side_px = {"proof_lift": 2.0, "full_lift": 6.0, "retained": 6.0}[key]
        measured = float(median(samples[-3:]))
        measured_side = float(median(side_samples[-3:]))
        wrist_votes = sum(wrist_samples[-3:])
        passed = (
            measured >= required
            and measured_side >= required_side_px
            and wrist_votes >= 2
        )
        self._record_event(
            {
                "event": "verification",
                "phase": self.controller.phase.name,
                "gate": key,
                "median_lift_mm": measured,
                "required_lift_mm": required,
                "median_side_lift_px": measured_side,
                "required_side_lift_px": required_side_px,
                "wrist_presence_votes": wrist_votes,
                "wrist_last_distance_mm": wrist_distance_mm,
                "wrist_errors": wrist_errors,
                "passed": passed,
            }
        )
        return {key: passed}

    def act(self, obs: dict):
        if self.initialization_error is not None:
            return self._hold_observed_pose(obs)
        if self.controller is None:
            try:
                self._initialize(obs)
            except (BinVisionError, DepthVisionError, ImportError, SensorFusionError, SideVisionError, TypeError, ValueError) as exc:
                self.initialization_error = f"{type(exc).__name__}: {exc}"
                event = {"event": "initialization_rejected", "error": self.initialization_error}
                self._record_event(event)
                return self._hold_observed_pose(obs)
        output = self.controller.act(obs, self._verification(obs))
        new_history = self.controller.history[self.controller_history_cursor :]
        for entry in new_history:
            self._record_event({"event": "controller", **entry})
        self.controller_history_cursor = len(self.controller.history)
        return output
