"""Drop-in policy for the official SIDC Lab Physical AI client.

The policy only consumes a plan that a local human approved through the A-Team
CLI. It defaults to mock mode. Physical mode remains locked unless the operator
sets ATEAM_EXECUTION_MODE=physical during an approved test slot.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path


MVP_ROOT = Path(os.environ.get("ATEAM_MVP_ROOT", Path(__file__).resolve().parent))
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from robot_control.safety import as_tcp_action  # noqa: E402
from robot_control.store import PlanStore  # noqa: E402


DT_SECONDS = 0.1
MAX_TRANSLATION_STEP_MM = 2.0
MAX_ROTATION_STEP_DEG = 1.0
MAX_CHUNK_ACTIONS = 100
POSITION_TOLERANCE_MM = 2.0
ROTATION_TOLERANCE_DEG = 2.0


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
    CAMERAS = []
    IMAGE_SIZE = (320, 180)
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

    def reset(self) -> None:
        self.plan = self.store.claim_next(self.mode)
        if self.plan is None:
            raise RuntimeError(
                f"no human-approved {self.mode} plan is queued; draft and approve one before starting the run"
            )
        self.actions = [as_tcp_action(item) for item in self.plan["waypoints"]]
        self.target_index = 0

    def act(self, obs: dict):
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
            if position_error > POSITION_TOLERANCE_MM or rotation_error > ROTATION_TOLERANCE_DEG:
                observed_gripper = float(obs.get("joints", [0.0] * 7)[6])
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
