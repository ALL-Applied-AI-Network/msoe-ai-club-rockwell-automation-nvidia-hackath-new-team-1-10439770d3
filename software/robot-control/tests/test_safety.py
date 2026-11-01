import pytest

from robot_control.safety import SafetyError, as_tcp_action, validate_path


def test_safe_demo_converts_to_tcp():
    item = {"x_mm": 450, "y_mm": 0, "z_mm": -230, "yaw_deg": 15, "gripper": 0}
    assert as_tcp_action(item) == [450.0, 0.0, -230.0, -180.0, 0.0, 15.0, 0.0]


def test_low_target_is_rejected():
    with pytest.raises(SafetyError, match="z_mm"):
        validate_path([{"x_mm": 450, "y_mm": 0, "z_mm": -330}])


def test_corrected_wall_grip_depth_is_allowed_but_old_depth_is_rejected():
    assert validate_path([{"x_mm": 450, "y_mm": 0, "z_mm": -294}])[0]["z_mm"] == -294
    with pytest.raises(SafetyError, match="z_mm"):
        validate_path([{"x_mm": 450, "y_mm": 0, "z_mm": -300}])


def test_long_segment_is_rejected():
    with pytest.raises(SafetyError, match="segment"):
        validate_path(
            [
                {"x_mm": 350, "y_mm": -180, "z_mm": -230},
                {"x_mm": 550, "y_mm": 180, "z_mm": -230},
            ]
        )
