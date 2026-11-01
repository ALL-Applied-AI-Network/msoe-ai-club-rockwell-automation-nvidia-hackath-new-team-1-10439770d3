"""Hot-reloadable, bounded TCP command queue for supervised physical trials."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


COMMAND_PATH = Path(os.environ.get("ATEAM_LIVE_COMMANDS", "live_commands.json"))
DT = 0.1
CHUNK_SECONDS = 1.0
TRANSLATION_MM_PER_SECOND = 20.0
ROTATION_DEG_PER_SECOND = 10.0


class LiveCommandError(RuntimeError):
    pass


def _angle_delta(start: float, target: float) -> float:
    return (target - start + 180.0) % 360.0 - 180.0


def validate_target(values) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise LiveCommandError("each TCP target must contain [x,y,z,w,p,r,gripper]")
    target = [float(value) for value in values]
    if not all(math.isfinite(value) for value in target):
        raise LiveCommandError("TCP target contains NaN or infinity")
    bounds = ((200.0, 600.0), (-300.0, 300.0), (-305.0, 200.0))
    for axis, value, (low, high) in zip("xyz", target[:3], bounds):
        if not low <= value <= high:
            raise LiveCommandError(f"{axis}={value:.1f} is outside [{low}, {high}]")
    if any(abs(value) > 360.0 for value in target[3:6]):
        raise LiveCommandError("TCP rotations must stay within +/-360 degrees")
    if target[6] not in (0.0, 1.0):
        raise LiveCommandError("gripper must be exactly 0 (open) or 1 (closed)")
    return target


def bounded_chunk(observed: list[float], goal: list[float]) -> list[list[float]]:
    """Advance no more than one second toward a goal at conservative rates."""

    xyz_delta = [goal[index] - observed[index] for index in range(3)]
    distance = math.sqrt(sum(value * value for value in xyz_delta))
    xyz_scale = min(1.0, TRANSLATION_MM_PER_SECOND * CHUNK_SECONDS / max(distance, 1e-9))
    rotation_delta = [_angle_delta(observed[index], goal[index]) for index in range(3, 6)]
    rotation_scale = min(
        1.0,
        ROTATION_DEG_PER_SECOND * CHUNK_SECONDS / max(max(abs(v) for v in rotation_delta), 1e-9),
    )
    endpoint = [observed[index] + xyz_delta[index] * xyz_scale for index in range(3)]
    endpoint.extend(observed[index] + rotation_delta[index - 3] * rotation_scale for index in range(3, 6))
    pose_reached = distance <= 3.0 and max(abs(v) for v in rotation_delta) <= 2.0
    endpoint.append(goal[6] if pose_reached else observed[6])
    rows = int(CHUNK_SECONDS / DT)
    output = []
    for index in range(1, rows + 1):
        fraction = index / rows
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        row = [observed[i] + (endpoint[i] - observed[i]) * blend for i in range(6)]
        row.append(endpoint[6] if index == rows else observed[6])
        output.append(row)
    return output


class LiveSupervisedPolicy:
    CAMERAS = ["table", "side", "wrist"]
    DEPTH = ["table", "wrist"]
    IMAGE_SIZE = (848, 480)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = 2
    DT = DT

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint
        self.reset()

    def reset(self) -> None:
        self.revision = None
        self.commands: list[list[float]] = []
        self.settled = 0
        self.last_error = None

    @staticmethod
    def _observed(obs: dict) -> list[float]:
        tcp = obs.get("tcp_pose")
        joints = obs.get("joints")
        if tcp is None or len(tcp) < 6 or joints is None or len(joints) < 7:
            trial_id = str((obs.get("trial") or {}).get("trial_id", ""))
            if trial_id.startswith("mock-"):
                return [300.0, 0.0, -90.0, -180.0, 0.0, 0.0, 0.0]
            raise LiveCommandError("live control requires observed TCP and gripper state")
        return [float(value) for value in tcp[:6]] + [float(joints[6])]

    def _reload(self) -> None:
        try:
            payload = json.loads(COMMAND_PATH.read_text(encoding="utf-8"))
            revision = int(payload["revision"])
            if revision == self.revision:
                return
            mode = str(payload.get("mode", "hold")).lower()
            if mode not in {"hold", "execute"}:
                raise LiveCommandError("mode must be hold or execute")
            commands = [] if mode == "hold" else [validate_target(item) for item in payload.get("commands", [])]
            self.revision = revision
            self.commands = commands
            self.settled = 0
            self.last_error = None
            print(
                "[A-Team live] "
                + json.dumps({"event": "revision", "revision": revision, "mode": mode, "commands": commands}),
                flush=True,
            )
        except (OSError, ValueError, KeyError, TypeError, LiveCommandError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message != self.last_error:
                print("[A-Team live] " + json.dumps({"event": "command_rejected", "error": message}), flush=True)
                self.last_error = message
            self.commands = []

    def act(self, obs: dict):
        observed = self._observed(obs)
        self._reload()
        if not self.commands:
            return observed
        goal = self.commands[0]
        position_error = math.dist(observed[:3], goal[:3])
        rotation_error = max(abs(_angle_delta(observed[i], goal[i])) for i in range(3, 6))
        gripper_error = abs(observed[6] - goal[6])
        reached = position_error <= 3.0 and rotation_error <= 2.0 and gripper_error <= 0.25
        self.settled = self.settled + 1 if reached else 0
        if self.settled >= 2:
            print(
                "[A-Team live] "
                + json.dumps({"event": "target_reached", "revision": self.revision, "target": goal,
                              "position_error_mm": position_error, "rotation_error_deg": rotation_error}),
                flush=True,
            )
            self.commands.pop(0)
            self.settled = 0
            if not self.commands:
                return observed
            goal = self.commands[0]
        return bounded_chunk(observed, goal)

