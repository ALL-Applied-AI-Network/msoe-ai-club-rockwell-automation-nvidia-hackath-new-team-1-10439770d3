from __future__ import annotations

import math

from policy import (
    MAX_ROTATION_STEP_DEG,
    MAX_TRANSLATION_STEP_MM,
    POSITION_TOLERANCE_MM,
    Policy,
    interpolate_tcp,
)
from robot_control.store import PlanStore


def test_interpolate_tcp_limits_each_10hz_step():
    start = [450.0, -70.0, -230.0, -180.0, 0.0, 0.0]
    target = [450.0, 70.0, -230.0, -180.0, 0.0, 20.0, 0.0]
    rows = interpolate_tcp(start, target)

    assert rows[-1] == target
    previous = start
    for row in rows:
        assert math.dist(previous[:3], row[:3]) <= MAX_TRANSLATION_STEP_MM + 1e-9
        assert max(abs(row[i] - previous[i]) for i in range(3, 6)) <= MAX_ROTATION_STEP_DEG + 1e-9
        previous = row


def test_policy_advances_and_completes_a_smoothed_plan(tmp_path, monkeypatch):
    db = tmp_path / "plans.sqlite3"
    monkeypatch.setenv("ATEAM_ROBOT_DB", str(db))
    monkeypatch.setenv("ATEAM_EXECUTION_MODE", "mock")
    store = PlanStore(db)
    plan = store.create(
        "test",
        [
            {"x_mm": 450, "y_mm": -70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
            {"x_mm": 450, "y_mm": 70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
        ],
    )
    store.approve(plan["id"], "mock", "test")

    policy = Policy()
    policy.reset()

    first = policy.act({"tcp_pose": [450, -80, -230, -180, 0, 0]})
    assert first[-1][:3] == [450.0, -70.0, -230.0]
    assert all(math.dist(left[:3], right[:3]) <= MAX_TRANSLATION_STEP_MM + 1e-9
               for left, right in zip([[450, -80, -230, -180, 0, 0, 0], *first], first))

    second = policy.act({"tcp_pose": [450, -70, -230, -180, 0, 0]})
    assert second[-1][:3] == [450.0, 70.0, -230.0]
    assert len(second) == 70

    hold = policy.act({"tcp_pose": [450, 70, -230, -180, 0, 0]})
    assert hold == [450.0, 70.0, -230.0, -180.0, 0.0, 0.0, 0.0]
    assert store.get(plan["id"])["status"] == "completed"


def test_physical_commissioning_residual_advances_to_second_waypoint(tmp_path, monkeypatch):
    """Regression test for the 2.54 mm residual measured on the real FANUC."""
    assert POSITION_TOLERANCE_MM >= 2.54
    db = tmp_path / "plans.sqlite3"
    monkeypatch.setenv("ATEAM_ROBOT_DB", str(db))
    monkeypatch.setenv("ATEAM_EXECUTION_MODE", "physical")
    monkeypatch.setenv("ATEAM_ALLOW_PHYSICAL", "YES")
    store = PlanStore(db)
    plan = store.create(
        "physical residual",
        [
            {"x_mm": 450, "y_mm": -70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
            {"x_mm": 450, "y_mm": 70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
        ],
    )
    store.approve(plan["id"], "physical", "Test Operator")
    policy = Policy()
    policy.reset()

    actions = policy.act({"tcp_pose": [450, -70, -232.54, -180, 0, 0], "joints": [0] * 7})

    assert actions[-1][:3] == [450.0, 70.0, -230.0]
