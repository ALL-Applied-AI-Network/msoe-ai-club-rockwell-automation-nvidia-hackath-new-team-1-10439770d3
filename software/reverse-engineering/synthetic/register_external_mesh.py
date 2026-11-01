"""Register an independently produced mesh with the common evaluator.

This utility performs format normalization only.  It never reads the withheld
source mesh and, by default, applies no geometric cleanup or alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def as_mesh(value) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Scene):
        values = [g for g in value.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not values:
            raise ValueError("input scene has no triangle meshes")
        return trimesh.util.concatenate(values)
    if not isinstance(value, trimesh.Trimesh):
        raise TypeError(type(value).__name__)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--runtime-s", type=float)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--translate", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    args = parser.parse_args()

    source = Path(args.input).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mesh = as_mesh(trimesh.load(source, process=False))
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * args.scale + np.asarray(args.translate)
    outputs = {}
    for suffix in ("ply", "obj", "stl"):
        path = out / f"{args.method}.{suffix}"
        mesh.export(path)
        outputs[suffix] = path.name
    report = {
        "method": args.method,
        "status": "completed",
        "runtime_s": args.runtime_s,
        "source_mesh_access": False,
        "normalization_only": True,
        "input_mesh": str(source),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "outputs": outputs,
    }
    (out / f"{args.method}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
