"""Fail-closed candidate for the physically calibrated west-wall bin lift.

This module is intentionally not imported by the active ``policy.py``.  It is
the replay/simulation candidate derived from physical run r20261022-a78baa.
It moves only after three stable table-camera detections match the one wall and
yaw that were physically validated.  It never lowers or releases after close.

No position-only policy can prove zero collision risk on the real cell: the API
does not expose finger contact or force/torque.  The purpose of this candidate
is to remove the known deterministic collision causes and to fail closed when
the visual evidence differs from the validated case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from statistics import median

from .bin_vision import BinVisionError, locate_green_bin, select_wall_facing
from .phase_controller import MotionPhase, PhaseController, PhaseExecutionError


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "calibration" / "real_finger_wall_grasp.json"
SIMULATION_ONLY_LOCK = ROOT / "SIMULATION_ONLY.lock"

DT_S = 0.1
TRANSIT_Z_MM = -235.0
MINIMUM_DETECTION_CONFIDENCE = 0.55
MAXIMUM_WALL_YAW_ERROR_DEG = 2.0
MAXIMUM_EDGE_LENGTH_ERROR_MM = 8.0
INITIAL_STABLE_OBSERVATIONS = 3
MAXIMUM_INITIAL_XY_SPREAD_MM = 1.5
MAXIMUM_INITIAL_YAW_SPREAD_DEG = 0.75
MAXIMUM_REVALIDATION_XY_ERROR_MM = 1.5
MAXIMUM_REVALIDATION_YAW_ERROR_DEG = 0.75


class CalibratedPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibratedWallTarget:
    tcp_x_mm: float
    tcp_y_mm: float
    yaw_deg: float
    wall_x_mm: float
    wall_y_mm: float
    wall_edge_length_mm: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _parallel_jaw_angle_delta(left_deg: float, right_deg: float) -> float:
    """Signed yaw delta for geometry that repeats every 180 degrees."""

    return (float(left_deg) - float(right_deg) + 90.0) % 180.0 - 90.0


def _load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibratedPolicyError(f"cannot load physical calibration: {exc}") from exc
    if payload.get("status") != "physically_validated_grasp_and_lift":
        raise CalibratedPolicyError("physical wall-grasp calibration is not validated")
    return payload


def target_from_table_rgb(rgb, calibration: dict | None = None) -> CalibratedWallTarget:
    """Map the explicitly validated west wall to the physical TCP target."""

    calibration = calibration or _load_calibration()
    detection = locate_green_bin(rgb)
    if detection.confidence < MINIMUM_DETECTION_CONFIDENCE:
        raise CalibratedPolicyError(
            f"green-bin confidence {detection.confidence:.3f} is below "
            f"{MINIMUM_DETECTION_CONFIDENCE:.3f}"
        )

    # The successful grasp did not use the old policy's south/-Y wall.  It used
    # the fitted west edge, whose outward normal is robot -X.
    try:
        wall = select_wall_facing(detection, (-1.0, 0.0), minimum_alignment=0.98)
    except BinVisionError as exc:
        raise CalibratedPolicyError(f"validated west wall is unavailable: {exc}") from exc
    reference_yaw = float(calibration["rgb_fitted_wall_yaw_deg"])
    yaw_error = abs(_parallel_jaw_angle_delta(wall.gripper_yaw_deg, reference_yaw))
    if yaw_error > MAXIMUM_WALL_YAW_ERROR_DEG:
        raise CalibratedPolicyError(
            f"west-wall yaw differs from the validated pose by {yaw_error:.2f} deg"
        )

    reference_length = float(calibration["rgb_fitted_wall_edge_length_mm"])
    length_error = abs(float(wall.edge_length_mm) - reference_length)
    if length_error > MAXIMUM_EDGE_LENGTH_ERROR_MM:
        raise CalibratedPolicyError(
            f"west-wall fitted length differs from calibration by {length_error:.1f} mm"
        )

    offset_x, offset_y = [
        float(value) for value in calibration["empirical_tcp_minus_rgb_wall_xy_mm"]
    ]
    wall_x, wall_y, _ = wall.midpoint_base_mm
    tcp_x = float(wall_x + offset_x)
    tcp_y = float(wall_y + offset_y)
    yaw = float(
        wall.gripper_yaw_deg
        + float(calibration["controller_yaw_minus_rgb_wall_yaw_deg"])
    )
    if not (260.0 <= tcp_x <= 550.0 and -180.0 <= tcp_y <= 180.0):
        raise CalibratedPolicyError(
            f"calibrated TCP XY ({tcp_x:.1f}, {tcp_y:.1f}) is outside the candidate envelope"
        )
    return CalibratedWallTarget(
        tcp_x_mm=tcp_x,
        tcp_y_mm=tcp_y,
        yaw_deg=yaw,
        wall_x_mm=float(wall_x),
        wall_y_mm=float(wall_y),
        wall_edge_length_mm=float(wall.edge_length_mm),
        confidence=float(detection.confidence),
    )


def _duration(start, target, translation_speed_mm_s: float, rotation_speed_deg_s: float) -> float:
    distance = math.dist([float(value) for value in start[:3]], target[:3])
    rotation = max(
        abs((float(target[index]) - float(start[index]) + 180.0) % 360.0 - 180.0)
        for index in range(3, 6)
    )
    return max(0.5, distance / translation_speed_mm_s, rotation / rotation_speed_deg_s)


def calibrated_lift_phases(
    observed: list[float], target: CalibratedWallTarget, calibration: dict
) -> list[MotionPhase]:
    """Build a vertical-only insertion and monotonic post-close lift."""

    open_pose = [float(value) for value in calibration["tcp_pose"]]
    closed_pose = [float(value) for value in calibration["validation"]["closed_tcp_pose"]]
    proof_z = float(closed_pose[2] + calibration["validation"]["proof_lift_mm"])
    orientation = (-180.0, 0.0, float(target.yaw_deg))
    approach = (target.tcp_x_mm, target.tcp_y_mm, TRANSIT_Z_MM, *orientation, 0.0)
    preinsert = (target.tcp_x_mm, target.tcp_y_mm, float(open_pose[2]), *orientation, 0.0)
    descended = (target.tcp_x_mm, target.tcp_y_mm, float(closed_pose[2]), *orientation, 0.0)
    closed = (*descended[:6], 1.0)
    proof = (target.tcp_x_mm, target.tcp_y_mm, proof_z, *orientation, 1.0)

    phases = [
        MotionPhase(
            "high_approach",
            approach,
            _duration(observed, approach, 15.0, 8.0),
            verification="approach_alignment",
            maximum_corrections=8,
        ),
        MotionPhase(
            "validated_preinsert",
            preinsert,
            _duration(approach, preinsert, 8.0, 8.0),
            verification="preinsert_alignment",
            maximum_corrections=8,
        ),
        MotionPhase("slow_validated_descent", descended, 3.0, maximum_corrections=6),
        MotionPhase("close_at_validated_pose", closed, 1.0, maximum_corrections=6),
        MotionPhase("proof_lift_only", proof, 3.0, maximum_corrections=6),
        MotionPhase("hold_closed", proof, 0.5, maximum_corrections=2),
    ]

    # Static invariants: no XY motion once below transit, no target deeper than
    # the successful close, and no downward motion after the gripper closes.
    deepest_validated_z = float(closed_pose[2])
    for phase in phases:
        if phase.target[2] < deepest_validated_z - 1e-6:
            raise CalibratedPolicyError(
                f"phase {phase.name} exceeds validated depth {deepest_validated_z:.3f} mm"
            )
    manipulation = phases[1:]
    if any(
        abs(phase.target[0] - target.tcp_x_mm) > 1e-6
        or abs(phase.target[1] - target.tcp_y_mm) > 1e-6
        for phase in manipulation
    ):
        raise CalibratedPolicyError("manipulation phases must remain vertical")
    post_close_z = [phase.target[2] for phase in phases[3:]]
    if any(right < left - 1e-6 for left, right in zip(post_close_z, post_close_z[1:])):
        raise CalibratedPolicyError("post-close phases must never move downward")
    return phases


class CalibratedWestWallLiftPolicy:
    """Replay/simulation candidate; physical use requires an explicit arm flag."""

    CAMERAS = ["table", "side", "wrist"]
    DEPTH = []
    IMAGE_SIZE = (1280, 720)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DT = DT_S

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint
        self.calibration = _load_calibration()
        self.reset()

    def reset(self) -> None:
        self.samples: list[CalibratedWallTarget] = []
        self.target: CalibratedWallTarget | None = None
        self.controller: PhaseController | None = None
        self.latched_error: str | None = None
        self.audit: list[dict] = []
        self.history_cursor = 0

    def _record(self, event: dict) -> None:
        self.audit.append(event)
        print("[A-Team calibrated-wall] " + json.dumps(event, sort_keys=True), flush=True)

    @staticmethod
    def _observed(obs: dict) -> list[float]:
        tcp = obs.get("tcp_pose")
        joints = obs.get("joints")
        if tcp is None or len(tcp) < 6 or joints is None or len(joints) < 7:
            trial_id = str((obs.get("trial") or {}).get("trial_id", ""))
            if trial_id.startswith("mock-"):
                return [300.0, 0.0, -90.0, -180.0, 0.0, 89.49, 0.0]
            raise CalibratedPolicyError("observed TCP and gripper state are required")
        return [float(value) for value in tcp[:6]] + [float(joints[6])]

    @staticmethod
    def _is_mock(obs: dict) -> bool:
        return str((obs.get("trial") or {}).get("trial_id", "")).startswith("mock-")

    def _physical_is_armed(self) -> bool:
        return (
            os.environ.get("ATEAM_ALLOW_CALIBRATED_PHYSICAL") == "YES"
            and not SIMULATION_ONLY_LOCK.exists()
        )

    def _hold(self, observed: list[float]) -> list[float]:
        return [float(value) for value in observed]

    def _observe_target(self, obs: dict) -> CalibratedWallTarget:
        table = (obs.get("images") or {}).get("table")
        if table is None:
            raise CalibratedPolicyError("table RGB is missing")
        return target_from_table_rgb(table, self.calibration)

    def _initialize(self, obs: dict, observed: list[float]) -> None:
        sample = self._observe_target(obs)
        self.samples.append(sample)
        self.samples[:] = self.samples[-INITIAL_STABLE_OBSERVATIONS:]
        self._record({"event": "initialization_sample", "target": sample.to_dict()})
        if len(self.samples) < INITIAL_STABLE_OBSERVATIONS:
            return

        xs = [item.tcp_x_mm for item in self.samples]
        ys = [item.tcp_y_mm for item in self.samples]
        yaws = [item.yaw_deg for item in self.samples]
        if max(xs) - min(xs) > MAXIMUM_INITIAL_XY_SPREAD_MM:
            raise CalibratedPolicyError("initial wall X is not stable")
        if max(ys) - min(ys) > MAXIMUM_INITIAL_XY_SPREAD_MM:
            raise CalibratedPolicyError("initial wall Y is not stable")
        if max(yaws) - min(yaws) > MAXIMUM_INITIAL_YAW_SPREAD_DEG:
            raise CalibratedPolicyError("initial wall yaw is not stable")

        self.target = CalibratedWallTarget(
            tcp_x_mm=float(median(xs)),
            tcp_y_mm=float(median(ys)),
            yaw_deg=float(median(yaws)),
            wall_x_mm=float(median([item.wall_x_mm for item in self.samples])),
            wall_y_mm=float(median([item.wall_y_mm for item in self.samples])),
            wall_edge_length_mm=float(
                median([item.wall_edge_length_mm for item in self.samples])
            ),
            confidence=float(median([item.confidence for item in self.samples])),
        )
        phases = calibrated_lift_phases(observed, self.target, self.calibration)
        self.controller = PhaseController(
            phases,
            position_tolerance_mm=1.0,
            rotation_tolerance_deg=0.75,
            settled_observations=3,
        )
        self._record(
            {
                "event": "initialized",
                "target": self.target.to_dict(),
                "phases": [
                    {"name": phase.name, "target": list(phase.target)} for phase in phases
                ],
            }
        )

    def _verification(self, obs: dict) -> dict[str, bool | None]:
        if self.controller is None or self.target is None:
            return {}
        key = self.controller.phase.verification
        if not key:
            return {}
        try:
            current = self._observe_target(obs)
        except (BinVisionError, CalibratedPolicyError, TypeError, ValueError) as exc:
            self._record({"event": "alignment_evidence_missing", "gate": key, "error": str(exc)})
            return {key: None}
        xy_error = math.dist(
            [current.tcp_x_mm, current.tcp_y_mm],
            [self.target.tcp_x_mm, self.target.tcp_y_mm],
        )
        yaw_error = abs(_parallel_jaw_angle_delta(current.yaw_deg, self.target.yaw_deg))
        passed = (
            xy_error <= MAXIMUM_REVALIDATION_XY_ERROR_MM
            and yaw_error <= MAXIMUM_REVALIDATION_YAW_ERROR_DEG
        )
        self._record(
            {
                "event": "alignment_verification",
                "gate": key,
                "xy_error_mm": xy_error,
                "yaw_error_deg": yaw_error,
                "passed": passed,
            }
        )
        return {key: passed}

    def act(self, obs: dict):
        observed = self._observed(obs)
        if self.latched_error is not None:
            return self._hold(observed)
        if not self._is_mock(obs) and not self._physical_is_armed():
            if not self.audit:
                self._record(
                    {
                        "event": "physical_motion_locked",
                        "reason": "explicit arm flag and removal of SIMULATION_ONLY.lock are required",
                    }
                )
            return self._hold(observed)

        try:
            if self.controller is None:
                self._initialize(obs, observed)
                if self.controller is None:
                    return self._hold(observed)
            output = self.controller.act(obs, self._verification(obs))
            for entry in self.controller.history[self.history_cursor :]:
                self._record({"event": "controller", **entry})
            self.history_cursor = len(self.controller.history)
            return output
        except (BinVisionError, CalibratedPolicyError, PhaseExecutionError, TypeError, ValueError) as exc:
            self.latched_error = f"{type(exc).__name__}: {exc}"
            self._record({"event": "failed_closed", "error": self.latched_error})
            return self._hold(observed)
