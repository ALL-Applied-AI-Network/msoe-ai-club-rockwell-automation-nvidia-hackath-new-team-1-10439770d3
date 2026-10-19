"""Measured FANUC ER-4iA point-to-point smoke test for Isaac Sim 6.0.

The controller demo uses UF0 millimetres. Physical calibration places UF0
330 mm above the robot base, so controller Z=-230 mm becomes base Z=+100 mm.
This script moves between the two resulting TCP targets with an open gripper,
then verifies the measured articulation state through the calibrated FK model.
It controls simulation only; it contains no physical-controller transport.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/point_to_point_report.json")
parser.add_argument("--physics-only", action="store_true")
args = parser.parse_args()
output_path = Path(args.out).resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp


app = SimulationApp({"headless": True})
report = {
    "task_complete": False,
    "frame": "FANUC robot base; metres; controller UF0 Z plus 0.330 m",
    "controller_targets_mm_deg": [
        [450.0, -70.0, -230.0, -180.0, 0.0, 0.0, 0.0],
        [450.0, 70.0, -230.0, -180.0, 0.0, 0.0, 0.0],
    ],
    "waypoints": [],
}

try:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    import cell
    import fk
    import ik
    import scene_config as C

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage)
    cell.enable_ccd(world)
    world.reset()

    arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    arm_indices = [names.index(f"J{i}") for i in range(1, 7)]
    all_indices = {name: i for i, name in enumerate(names)}
    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])

    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    tool_offset = tuple(np.asarray(C.TOOL_AXIS, float) * cell.tip_offset())
    approach = (0.0, 0.0, -1.0)
    targets = (
        np.array((0.450, -0.070, 0.100), float),
        np.array((0.450, 0.070, 0.100), float),
    )

    q_home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
    solutions = []
    seed = q_home
    for target in targets:
        solved, converged = ik.solve(
            seed, target, approach, limits, tool_offset=tool_offset
        )
        if not converged:
            raise RuntimeError(f"IK did not converge for {target.tolist()}")
        solutions.append(solved)
        seed = solved

    def full_target(q6):
        command = arm.get_joint_positions().copy()
        command[arm_indices] = q6
        for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
            command[all_indices[joint]] = value
        return command

    def command(q6):
        arm.apply_action(ArticulationAction(joint_positions=full_target(q6)))

    def advance(start, finish, frames=180):
        for frame in range(1, frames + 1):
            t = frame / frames
            smooth = t * t * (3.0 - 2.0 * t)
            command(start + (finish - start) * smooth)
            world.step(render=not args.physics_only)
        for _ in range(60):
            command(finish)
            world.step(render=not args.physics_only)

    arm.set_joint_positions(full_target(q_home))
    for _ in range(60):
        command(q_home)
        world.step(render=not args.physics_only)

    previous = q_home
    for index, (target, solution) in enumerate(zip(targets, solutions), start=1):
        advance(previous, solution)
        measured = arm.get_joint_positions()[arm_indices]
        _, measured_tcp = fk.tool_pose(measured, tool_offset)
        position_error_mm = float(np.linalg.norm(measured_tcp - target) * 1000.0)
        joint_error_deg = np.abs(np.rad2deg(measured - solution))
        waypoint = {
            "index": index,
            "target_tcp_m": target.tolist(),
            "measured_tcp_m": measured_tcp.tolist(),
            "position_error_mm": position_error_mm,
            "commanded_joints_deg": np.rad2deg(solution).tolist(),
            "measured_joints_deg": np.rad2deg(measured).tolist(),
            "max_joint_error_deg": float(np.max(joint_error_deg)),
        }
        waypoint["verified"] = (
            waypoint["position_error_mm"] <= 5.0
            and waypoint["max_joint_error_deg"] <= 2.0
        )
        report["waypoints"].append(waypoint)
        previous = solution

    report["task_complete"] = all(w["verified"] for w in report["waypoints"])
    if not report["task_complete"]:
        raise RuntimeError("Measured point-to-point tolerances were not met")
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
    raise
finally:
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[point-to-point] report written to {output_path}")
    print(json.dumps(report, indent=2))
    app.close()
