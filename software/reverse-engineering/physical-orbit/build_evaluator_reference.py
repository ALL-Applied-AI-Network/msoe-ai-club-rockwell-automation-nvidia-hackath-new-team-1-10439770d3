"""Build the CAD-nominal evaluator scene *after* reconstruction inputs freeze.

This script is intentionally outside the remote reconstruction bundle.  It is
used only for post-hoc scoring and clearly records the assumed physical state:
measured base plus one measured gear on the left shaft at x=-28 mm, z=9 mm.
The printed physical part can deviate from the nominal STL, so this reference is
CAD nominal—not metrology-grade scan ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gear", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gear-x-mm", type=float, default=-28.0)
    parser.add_argument("--gear-y-mm", type=float, default=0.0)
    parser.add_argument("--gear-z-mm", type=float, default=9.0)
    parser.add_argument("--gear-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    base = trimesh.load_mesh(args.base, force="mesh", process=True)
    gear = trimesh.load_mesh(args.gear, force="mesh", process=True)
    transform = trimesh.transformations.rotation_matrix(np.deg2rad(args.gear_yaw_deg), [0, 0, 1])
    transform[:3, 3] = [args.gear_x_mm, args.gear_y_mm, args.gear_z_mm]
    gear.apply_transform(transform)
    scene_mesh = trimesh.util.concatenate([base, gear])
    args.out.mkdir(parents=True, exist_ok=True)
    scene_mesh.export(args.out / "cad_nominal_base_plus_one_gear.ply")
    scene_mesh.export(args.out / "cad_nominal_base_plus_one_gear.obj")
    scene_mesh.export(args.out / "cad_nominal_base_plus_one_gear.stl")
    manifest = {
        "role": "post-hoc evaluator only; never provided to reconstruction candidates",
        "base_source": str(args.base.resolve()),
        "gear_source": str(args.gear.resolve()),
        "gear_transform_mm_deg": {
            "x": args.gear_x_mm,
            "y": args.gear_y_mm,
            "z": args.gear_z_mm,
            "yaw": args.gear_yaw_deg,
        },
        "assumption": "one gear seated on the left shaft; yaw is physically symmetric and may be optimized for scoring",
        "limitation": "CAD nominal reference does not include FDM print variation or unknown gear tooth phase",
        "bounds_mm": scene_mesh.bounds.tolist(),
        "extents_mm": scene_mesh.extents.tolist(),
        "vertices": int(len(scene_mesh.vertices)),
        "faces": int(len(scene_mesh.faces)),
        "watertight_components": [bool(base.is_watertight), bool(gear.is_watertight)],
    }
    (args.out / "reference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
