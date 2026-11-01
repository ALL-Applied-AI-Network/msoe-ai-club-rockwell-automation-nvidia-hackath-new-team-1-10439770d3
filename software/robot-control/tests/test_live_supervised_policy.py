import json

import pytest

import robot_control.live_supervised_policy as live


def test_target_validation_keeps_commands_inside_bounded_workspace():
    assert live.validate_target([383, -39, -235, -180, 0, 1.5, 0])[-1] == 0
    with pytest.raises(live.LiveCommandError):
        live.validate_target([383, -39, -320, -180, 0, 1.5, 0])


def test_bounded_chunk_limits_one_second_motion():
    start = [300, 0, -90, -180, 0, 0, 0]
    goal = [450, 0, -235, -180, 0, 0, 0]
    chunk = live.bounded_chunk(start, goal)
    assert len(chunk) == 10
    displacement = sum((chunk[-1][i] - start[i]) ** 2 for i in range(3)) ** 0.5
    assert displacement <= 20.0001
    assert chunk[-1] != goal


def test_revision_can_replace_queue_with_hold(tmp_path, monkeypatch):
    command_path = tmp_path / "commands.json"
    monkeypatch.setattr(live, "COMMAND_PATH", command_path)
    command_path.write_text(json.dumps({"revision": 1, "mode": "execute", "commands": [[300, 0, -90, -180, 0, 0, 0]]}))
    policy = live.LiveSupervisedPolicy()
    policy._reload()
    assert policy.commands
    command_path.write_text(json.dumps({"revision": 2, "mode": "hold", "commands": []}))
    policy._reload()
    assert policy.commands == []
