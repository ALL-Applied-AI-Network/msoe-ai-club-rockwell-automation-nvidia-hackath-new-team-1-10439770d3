"""Drop-in policy for the official SIDC Lab Physical AI client.

The policy only consumes a plan that a local human approved through the A-Team
CLI. It defaults to mock mode. Physical mode remains locked unless the operator
sets ATEAM_EXECUTION_MODE=physical during an approved test slot.
"""

from __future__ import annotations

import math
import os
import sys
import json
from pathlib import Path


MVP_ROOT = Path(os.environ.get("ATEAM_MVP_ROOT", Path(__file__).resolve().parent))
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from robot_control.safety import as_tcp_action  # noqa: E402
from robot_control.safety import validate_path  # noqa: E402
from robot_control.store import PlanStore  # noqa: E402
from robot_control.bin_vision import locate_green_bin  # noqa: E402


DT_SECONDS = 0.1
MAX_TRANSLATION_STEP_MM = 2.0
MAX_ROTATION_STEP_DEG = 1.0
MAX_CHUNK_ACTIONS = 100
# The first physical commissioning run settled 2.53-2.54 mm from waypoint A
# in both recorded episodes. A 5 mm gate is still conservative for the
# non-contact commissioning path and prevents that repeatable servo/calibration
# residual from blocking the next waypoint. Precision manipulation should use
# a task-specific tolerance after physical calibration.
POSITION_TOLERANCE_MM = 3.0
ROTATION_TOLERANCE_DEG = 2.0
GRIPPER_TOLERANCE = 0.25


def vision_one_wall_waypoints(detection) -> list[dict]:
    """Build the bounded path from a calibrated table-camera detection.

    The failed q77a4ef run showed up to ~4 mm of downward tracking overshoot.
    A -294 mm target inserts about 14 mm below the nominal -280.1 mm wall top,
    leaving roughly 11 mm to the pocket floor before overshoot and ~7 mm after
    the observed worst case. The 10 mm lift is deliberately small.
    """
    x_mm, y_mm, _ = detection.south_wall_center_base_mm
    yaw_deg = detection.gripper_yaw_deg
    approach_z = -235.0
    grip_z = -294.0
    lifted_z = -284.0
    return validate_path(
        [
            {"x_mm": x_mm, "y_mm": y_mm, "z_mm": approach_z, "yaw_deg": yaw_deg, "gripper": 0},
            {"x_mm": x_mm, "y_mm": y_mm, "z_mm": grip_z, "yaw_deg": yaw_deg, "gripper": 0},
            {"x_mm": x_mm, "y_mm": y_mm, "z_mm": grip_z, "yaw_deg": yaw_deg, "gripper": 1},
            {"x_mm": x_mm, "y_mm": y_mm, "z_mm": lifted_z, "yaw_deg": yaw_deg, "gripper": 1},
            {"x_mm": x_mm, "y_mm": y_mm, "z_mm": grip_z, "yaw_deg": yaw_deg, "gripper": 1},
            {"x_mm": x_mm, "y_mm": y_mm, "z_mm": grip_z, "yaw_deg": yaw_deg, "gripper": 0},
            {"x_mm": x_mm, "y_mm": y_mm, "z_mm": approach_z, "yaw_deg": yaw_deg, "gripper": 0},
        ]
    )


def shortest_angle_delta(start_deg: float, target_deg: float) -> float:
    """Signed shortest change from start to target, in degrees."""
    return (target_deg - start_deg + 180.0) % 360.0 - 180.0


def interpolate_tcp(start: list[float], target: list[float]) -> list[list[float]]:
    """Build one conservative 10 Hz TCP chunk toward target.

    Translation is capped at 20 mm/s and rotation at 10 deg/s. If the full
    move needs more than the cell's 100-row maximum, the next policy call
    continues from the newly observed physical TCP pose.
    """
    if len(start) < 6 or len(target) != 7:
        raise RuntimeError("TCP interpolation requires a 6D observed pose and 7D target")
    xyz_distance = math.dist(start[:3], target[:3])
    angle_deltas = [shortest_angle_delta(float(start[i]), float(target[i])) for i in range(3, 6)]
    steps = max(
        1,
        math.ceil(xyz_distance / MAX_TRANSLATION_STEP_MM),
        *(math.ceil(abs(delta) / MAX_ROTATION_STEP_DEG) for delta in angle_deltas),
    )
    emitted = min(steps, MAX_CHUNK_ACTIONS)
    gripper_now = float(start[6]) if len(start) >= 7 else float(target[6])
    rows: list[list[float]] = []
    for index in range(1, emitted + 1):
        fraction = index / steps
        row = [float(start[i]) + (float(target[i]) - float(start[i])) * fraction for i in range(3)]
        row.extend(float(start[i]) + angle_deltas[i - 3] * fraction for i in range(3, 6))
        row.append(float(target[6]) if index == steps else gripper_now)
        rows.append(row)
    return rows


def pose_errors(observed: list[float], target: list[float]) -> tuple[float, float]:
    position = math.dist([float(value) for value in observed[:3]], target[:3])
    rotation = max(abs(shortest_angle_delta(float(observed[i]), target[i])) for i in range(3, 6))
    return position, rotation


class Policy:
    # The corrected policy genuinely uses the fixed, calibrated table camera.
    # Side/wrist remain evidence until a validated verification stage exists.
    CAMERAS = ["table"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = []
    DT = DT_SECONDS

    def __init__(self, checkpoint: str | None = None):
        self.mode = os.environ.get("ATEAM_EXECUTION_MODE", "mock").strip().lower()
        if self.mode not in {"mock", "physical"}:
            raise RuntimeError("ATEAM_EXECUTION_MODE must be mock or physical")
        if self.mode == "physical" and os.environ.get("ATEAM_ALLOW_PHYSICAL") != "YES":
            raise RuntimeError("physical execution is locked; an on-site operator must arm it")
        self.store = PlanStore()
        self.plan = None
        self.actions: list[list[float]] = []
        self.target_index = 0
        self.use_bin_vision = self.mode == "physical" or os.environ.get("ATEAM_USE_BIN_VISION") == "YES"
        self.detection = None

    def reset(self) -> None:
        self.plan = self.store.claim_next(self.mode)
        if self.plan is None:
            raise RuntimeError(
                f"no human-approved {self.mode} plan is queued; draft and approve one before starting the run"
            )
        if self.use_bin_vision:
            if "vision" not in self.plan["label"].lower():
                self.store.interrupt(
                    self.plan["id"],
                    "vision policy refused a plan not explicitly labelled for vision",
                )
                raise RuntimeError("physical vision policy requires an approved vision-labelled plan")
            self.actions = []
        else:
            self.actions = [as_tcp_action(item) for item in self.plan["waypoints"]]
        self.target_index = 0
        self.detection = None

    def act(self, obs: dict):
        if self.use_bin_vision and not self.actions:
            table_image = obs.get("images", {}).get("table")
            if table_image is None:
                self.store.interrupt(self.plan["id"], "table camera missing; no motion sent")
                raise RuntimeError("table camera missing; refusing the bin approach")
            try:
                self.detection = locate_green_bin(table_image)
                waypoints = vision_one_wall_waypoints(self.detection)
            except Exception as exc:
                self.store.interrupt(
                    self.plan["id"], f"bin vision rejected the scene: {type(exc).__name__}: {exc}"
                )
                raise RuntimeError(f"bin vision rejected the scene; no motion sent: {exc}") from exc
            self.actions = [as_tcp_action(item) for item in waypoints]
            print(
                "[A-Team vision] " + json.dumps(self.detection.to_dict(), sort_keys=True),
                flush=True,
            )

        observed = obs.get("tcp_pose")
        if observed is None:
            if self.mode == "physical":
                raise RuntimeError("physical rollout did not provide tcp_pose; refusing an unobserved move")
            # The public mock intentionally has no TCP pose until its first TCP
            # command. Seed it with the first validated target in mock mode only.
            return self.actions[0]

        while self.target_index < len(self.actions):
            target = self.actions[self.target_index]
            position_error, rotation_error = pose_errors(observed, target)
            observed_gripper = float(obs.get("joints", [0.0] * 7)[6])
            gripper_error = abs(observed_gripper - float(target[6]))
            if (
                position_error > POSITION_TOLERANCE_MM
                or rotation_error > ROTATION_TOLERANCE_DEG
                or gripper_error > GRIPPER_TOLERANCE
            ):
                start = [float(value) for value in observed[:6]] + [observed_gripper]
                return interpolate_tcp(start, target)
            self.target_index += 1

        target = self.actions[-1]
        position_error, rotation_error = pose_errors(observed, target)
        if self.plan is not None and self.plan["status"] == "executing":
            self.plan = self.store.complete(
                self.plan["id"],
                f"reported TCP error {position_error:.2f} mm; rotation error {rotation_error:.2f} deg",
            )
        # A policy cannot end the rollout; holding the final target lets the
        # operator inspect and score it without further motion.
        return target
