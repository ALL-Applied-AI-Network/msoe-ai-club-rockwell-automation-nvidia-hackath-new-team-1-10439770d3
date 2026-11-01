"""Kinematic feasibility gate for a nominal ER-4iA eye-in-hand orbit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--fanuc-source", default="tmp/fanuc_source_extract/fanuc_er4ia_sim")
parser.add_argument("--radius-m", type=float, default=0.30)
parser.add_argument("--object-x-m", type=float, default=0.27)
parser.add_argument("--target-z-m", type=float, default=0.035)
parser.add_argument("--azimuths", type=int, default=18)
parser.add_argument("--elevations", nargs="+", type=float, default=(18, 32, 46, 60, 74))
parser.add_argument("--camera-offset-m", type=float, default=0.0)
parser.add_argument("--out")
args = parser.parse_args()

source = Path(args.fanuc_source).resolve()
sys.path.insert(0, str(source))
import ik  # noqa: E402
import scene_config as C  # noqa: E402

limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
ik.set_tool_axis((1.0, 0.0, 0.0))
target = np.asarray((args.object_x_m, 0.0, args.target_z_m))
offset = np.asarray((args.camera_offset_m, 0.0, 0.0))

records = []
q_previous = home
seed_bank = [
    home,
    np.deg2rad((0, 30, -20, 0, -80, 0)),
    np.deg2rad((90, 20, 0, 0, -70, 0)),
    np.deg2rad((-90, 20, 0, 0, -70, 0)),
]
for elevation_deg in args.elevations:
    elevation = np.deg2rad(elevation_deg)
    for azimuth_index in range(args.azimuths):
        azimuth = 2 * np.pi * azimuth_index / args.azimuths
        eye = target + np.asarray((
            args.radius_m * np.cos(elevation) * np.cos(azimuth),
            args.radius_m * np.cos(elevation) * np.sin(azimuth),
            args.radius_m * np.sin(elevation),
        ))
        approach = target - eye
        approach /= np.linalg.norm(approach)
        solved, q_best = False, None
        for seed in [q_previous, *seed_bank]:
            q_try, ok = ik.solve(seed, eye, approach, limits, tool_offset=offset, iters=400, damping=0.035)
            if ok:
                solved, q_best = True, q_try
                break
        if solved:
            q_previous = q_best
        records.append({
            "azimuth_deg": float(np.rad2deg(azimuth)),
            "elevation_deg": float(elevation_deg),
            "eye_base_m": eye.tolist(),
            "ik": solved,
            "joint_deg": np.rad2deg(q_best).tolist() if solved else None,
        })

result = {
    "robot": "FANUC ER-4iA kinematic model",
    "radius_m": args.radius_m,
    "object_center_base_m": target.tolist(),
    "camera_offset_m": args.camera_offset_m,
    "poses": len(records),
    "ik_successes": sum(r["ik"] for r in records),
    "ik_success_fraction": sum(r["ik"] for r in records) / len(records),
    "by_elevation": {
        str(e): {
            "successes": sum(r["ik"] for r in records if r["elevation_deg"] == e),
            "poses": sum(1 for r in records if r["elevation_deg"] == e),
        } for e in args.elevations
    },
    "records": records,
    "caveat": "kinematic gate only; does not certify self/table collision or cable clearance",
}
payload = json.dumps(result, indent=2)
if args.out:
    Path(args.out).write_text(payload, encoding="utf-8")
print(payload)
