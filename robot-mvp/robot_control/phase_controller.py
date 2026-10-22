"""Monotonic, observation-gated TCP execution for the physical-cell API.

The controller never infers progress from elapsed wall-clock time and never
returns to an earlier manipulation phase.  Robot lag can therefore cause a
hold or a correction toward the *same* target, but cannot turn a lift into the
lift/lower oscillation observed in run q77a4ef.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


class PhaseExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class MotionPhase:
    name: str
    target: tuple[float, float, float, float, float, float, float]
    duration_s: float
    verification: str | None = None
    maximum_corrections: int = 3


def _angle_delta(start: float, target: float) -> float:
    return (target - start + 180.0) % 360.0 - 180.0


def tcp_chunk(
    start: list[float] | tuple[float, ...],
    target: tuple[float, ...],
    duration_s: float,
    *,
    dt_s: float = 0.1,
    maximum_rows: int = 100,
) -> list[list[float]]:
    """Generate one bounded smooth TCP action chunk, including exact endpoint."""

    if len(start) != 7 or len(target) != 7:
        raise PhaseExecutionError("TCP start and target must each contain seven values")
    rows = max(1, min(maximum_rows, int(math.ceil(duration_s / dt_s))))
    deltas = [float(target[i]) - float(start[i]) for i in range(3)]
    deltas += [_angle_delta(float(start[i]), float(target[i])) for i in range(3, 6)]
    output: list[list[float]] = []
    for index in range(1, rows + 1):
        raw = index / rows
        blend = raw * raw * (3.0 - 2.0 * raw)
        row = [float(start[i]) + deltas[i] * blend for i in range(6)]
        # The physical gripper is binary. Switch only at the endpoint, after
        # the arm has settled at the phase pose.
        row.append(float(target[6]) if index == rows else float(start[6]))
        output.append(row)
    output[-1] = [float(value) for value in target]
    return output


class PhaseController:
    """Advance through motion phases only after pose and evidence gates pass."""

    def __init__(
        self,
        phases: list[MotionPhase],
        *,
        position_tolerance_mm: float = 3.0,
        rotation_tolerance_deg: float = 2.0,
        settled_observations: int = 2,
    ):
        if not phases:
            raise PhaseExecutionError("at least one phase is required")
        self.phases = tuple(phases)
        self.position_tolerance_mm = float(position_tolerance_mm)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.settled_observations = int(settled_observations)
        self.index = 0
        self.dispatched = False
        self.settle_count = 0
        self.corrections = 0
        self.history: list[dict] = []

    @property
    def phase(self) -> MotionPhase:
        return self.phases[min(self.index, len(self.phases) - 1)]

    @property
    def complete(self) -> bool:
        return self.index >= len(self.phases)

    def _errors(self, tcp_pose, target) -> tuple[float, float]:
        position = math.dist([float(v) for v in tcp_pose[:3]], target[:3])
        rotation = max(
            abs(_angle_delta(float(tcp_pose[i]), float(target[i])))
            for i in range(3, 6)
        )
        return position, rotation

    def act(self, observation: Mapping, verification: Mapping[str, bool | None] | None = None):
        tcp_pose = observation.get("tcp_pose")
        joints = observation.get("joints")
        if tcp_pose is None or len(tcp_pose) < 6:
            raise PhaseExecutionError("an observed 6D TCP pose is required")
        if joints is None or len(joints) < 7:
            raise PhaseExecutionError("observed joints and gripper state are required")

        if self.complete:
            return [float(v) for v in self.phases[-1].target]

        phase = self.phase
        target = phase.target
        observed = [float(v) for v in tcp_pose[:6]] + [float(joints[6])]
        position_error, rotation_error = self._errors(tcp_pose, target)
        gripper_ok = abs(observed[6] - target[6]) <= 0.25

        if not self.dispatched:
            self.dispatched = True
            self.history.append({"event": "dispatch", "phase": phase.name})
            return tcp_chunk(observed, target, phase.duration_s)

        pose_ok = (
            position_error <= self.position_tolerance_mm
            and rotation_error <= self.rotation_tolerance_deg
            and gripper_ok
        )
        self.settle_count = self.settle_count + 1 if pose_ok else 0
        if not pose_ok:
            self.corrections += 1
            if self.corrections > phase.maximum_corrections:
                raise PhaseExecutionError(
                    f"phase {phase.name} failed to settle after {self.corrections} corrections"
                )
            self.history.append(
                {
                    "event": "correct",
                    "phase": phase.name,
                    "position_error_mm": position_error,
                    "rotation_error_deg": rotation_error,
                }
            )
            return tcp_chunk(observed, target, max(0.3, phase.duration_s / 2.0))

        if self.settle_count < self.settled_observations:
            return [float(v) for v in target]

        if phase.verification:
            value = (verification or {}).get(phase.verification)
            if value is None:
                self.history.append({"event": "await_evidence", "phase": phase.name})
                return [float(v) for v in target]
            if value is False:
                raise PhaseExecutionError(
                    f"phase {phase.name} failed verification gate {phase.verification}"
                )

        self.history.append({"event": "complete", "phase": phase.name})
        self.index += 1
        self.dispatched = False
        self.settle_count = 0
        self.corrections = 0
        if self.complete:
            return [float(v) for v in target]
        return self.act(observation, verification)


def bin_lift_phases(
    x_mm: float,
    y_mm: float,
    yaw_deg: float,
    *,
    grip_z_mm: float = -298.0,
    lift_mm: float = 10.0,
) -> list[MotionPhase]:
    """A lift-and-hold proof. Lower/release are intentionally not automatic."""

    pose = (-180.0, 0.0, float(yaw_deg))
    approach = (float(x_mm), float(y_mm), -235.0, *pose, 0.0)
    grip_open = (float(x_mm), float(y_mm), float(grip_z_mm), *pose, 0.0)
    grip_closed = (*grip_open[:6], 1.0)
    proof = (float(x_mm), float(y_mm), float(grip_z_mm + 4.0), *pose, 1.0)
    lifted = (float(x_mm), float(y_mm), float(grip_z_mm + lift_mm), *pose, 1.0)
    return [
        MotionPhase("approach", approach, 2.0),
        MotionPhase("descend", grip_open, 2.0),
        MotionPhase("close", grip_closed, 0.8),
        MotionPhase("proof_lift", proof, 0.6, verification="proof_lift"),
        MotionPhase("lift", lifted, 1.0, verification="full_lift"),
        MotionPhase("hold", lifted, 0.5, verification="retained"),
    ]
