from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import policy as policy_module

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
        "vision physical residual",
        [
            {"x_mm": 450, "y_mm": -70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
            {"x_mm": 450, "y_mm": 70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
        ],
    )
    store.approve(plan["id"], "physical", "Test Operator")
    detection = SimpleNamespace(
        south_wall_center_base_mm=(450.0, -70.0, 49.9),
        gripper_yaw_deg=0.0,
        to_dict=lambda: {"fixture": True},
    )
    monkeypatch.setattr(policy_module, "locate_green_bin", lambda image: detection)
    policy = Policy()
    policy.reset()

    actions = policy.act(
        {
            "tcp_pose": [450, -70, -237.54, -180, 0, 0],
            "joints": [0] * 7,
            "images": {"table": np.zeros((360, 640, 3), dtype=np.uint8)},
        }
    )

    assert actions[-1][:3] == [450.0, -70.0, -294.0]


def test_stationary_gripper_change_is_not_skipped(tmp_path, monkeypatch):
    db = tmp_path / "plans.sqlite3"
    monkeypatch.setenv("ATEAM_ROBOT_DB", str(db))
    monkeypatch.setenv("ATEAM_EXECUTION_MODE", "mock")
    store = PlanStore(db)
    plan = store.create(
        "gripper regression",
        [{"x_mm": 450, "y_mm": 0, "z_mm": -230, "yaw_deg": 0, "gripper": 1}],
    )
    store.approve(plan["id"], "mock", "test")
    policy = Policy()
    policy.reset()

    actions = policy.act(
        {"tcp_pose": [450, 0, -230, -180, 0, 0], "joints": [0] * 7}
    )

    assert actions == [[450.0, 0.0, -230.0, -180.0, 0.0, 0.0, 1.0]]


def test_physical_vision_refuses_to_move_without_table_image(tmp_path, monkeypatch):
    db = tmp_path / "plans.sqlite3"
    monkeypatch.setenv("ATEAM_ROBOT_DB", str(db))
    monkeypatch.setenv("ATEAM_EXECUTION_MODE", "physical")
    monkeypatch.setenv("ATEAM_ALLOW_PHYSICAL", "YES")
    store = PlanStore(db)
    plan = store.create(
        "vision missing-camera regression",
        [{"x_mm": 450, "y_mm": 0, "z_mm": -235, "yaw_deg": 0, "gripper": 0}],
    )
    store.approve(plan["id"], "physical", "Test Operator")
    policy = Policy()
    policy.reset()

    with pytest.raises(RuntimeError, match="table camera missing"):
        policy.act({"tcp_pose": [450, 0, -200, -180, 0, 0], "joints": [0] * 7})

    assert store.get(plan["id"])["status"] == "interrupted"
