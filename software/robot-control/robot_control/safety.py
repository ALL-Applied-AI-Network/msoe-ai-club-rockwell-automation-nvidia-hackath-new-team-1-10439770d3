"""Conservative, project-level checks layered on top of the cell's safety limits."""

from __future__ import annotations

from dataclasses import dataclass
import math


class SafetyError(ValueError):
    """A proposed target violates the MVP's conservative envelope."""


@dataclass(frozen=True)
class Envelope:
    # The public cell accepts a larger box. This smaller envelope deliberately
    # keeps the first MVP high over the central work area.
    x_mm: tuple[float, float] = (350.0, 550.0)
    y_mm: tuple[float, float] = (-180.0, 180.0)
    # The corrected wall-grip target is -294 mm after the first physical run
    # showed almost 4 mm of downward tracking overshoot. Reject targets below
    # -296 mm so planned motion cannot consume the remaining floor margin.
    z_mm: tuple[float, float] = (-296.0, -180.0)
    yaw_deg: tuple[float, float] = (-180.0, 180.0)
    max_segment_mm: float = 180.0


ENVELOPE = Envelope()


def validate_waypoint(waypoint: dict) -> dict:
    """Return a normalized TCP waypoint or raise SafetyError."""
    required = ("x_mm", "y_mm", "z_mm")
    missing = [name for name in required if name not in waypoint]
    if missing:
        raise SafetyError(f"missing fields: {', '.join(missing)}")

    values = {name: float(waypoint[name]) for name in required}
    values["yaw_deg"] = float(waypoint.get("yaw_deg", 0.0))
    values["gripper"] = float(waypoint.get("gripper", 0.0))
    if not all(math.isfinite(value) for value in values.values()):
        raise SafetyError("waypoint contains NaN or infinity")

    for key, limits in (
        ("x_mm", ENVELOPE.x_mm),
        ("y_mm", ENVELOPE.y_mm),
        ("z_mm", ENVELOPE.z_mm),
        ("yaw_deg", ENVELOPE.yaw_deg),
    ):
        lo, hi = limits
        if not lo <= values[key] <= hi:
            raise SafetyError(f"{key}={values[key]} is outside the MVP envelope [{lo}, {hi}]")
    if values["gripper"] not in (0.0, 1.0):
        raise SafetyError("gripper must be 0 (open) or 1 (closed)")

    return {
        "x_mm": values["x_mm"],
        "y_mm": values["y_mm"],
        "z_mm": values["z_mm"],
        "yaw_deg": values["yaw_deg"],
        "gripper": int(values["gripper"]),
    }


def validate_path(waypoints: list[dict]) -> list[dict]:
    if not 1 <= len(waypoints) <= 20:
        raise SafetyError("a plan must contain between 1 and 20 waypoints")
    clean = [validate_waypoint(item) for item in waypoints]
    for index, (left, right) in enumerate(zip(clean, clean[1:]), start=1):
        distance = math.dist(
            (left["x_mm"], left["y_mm"], left["z_mm"]),
            (right["x_mm"], right["y_mm"], right["z_mm"]),
        )
        if distance > ENVELOPE.max_segment_mm:
            raise SafetyError(
                f"segment {index}->{index + 1} is {distance:.1f} mm; "
                f"limit is {ENVELOPE.max_segment_mm:.1f} mm"
            )
    return clean


def as_tcp_action(waypoint: dict) -> list[float]:
    item = validate_waypoint(waypoint)
    return [
        item["x_mm"],
        item["y_mm"],
        item["z_mm"],
        -180.0,
        0.0,
        item["yaw_deg"],
        float(item["gripper"]),
    ]
