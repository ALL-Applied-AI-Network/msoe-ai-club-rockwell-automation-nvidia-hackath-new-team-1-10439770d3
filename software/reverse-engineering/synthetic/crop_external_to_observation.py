"""Normalize an external reconstruction using observation-derived bounds only.

The script never reads ``evaluator_only``.  It uses the RGB-D TSDF envelope as
an independently reconstructed conservative object volume, removes triangles
outside that volume, and exports common evaluator formats.  This is needed for
methods such as 2DGS that can also reconstruct the masked white background.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def mesh_from(path: Path) -> trimesh.Trimesh:
    value = trimesh.load(path, process=False)
    if isinstance(value, trimesh.Scene):
        value = trimesh.util.concatenate(tuple(value.geometry.values()))
    if not isinstance(value, trimesh.Trimesh):
        raise TypeError(type(value).__name__)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True, help="Observation-only TSDF envelope")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--runtime-s", type=float)
    parser.add_argument("--padding-mm", type=float, default=5.0)
    parser.add_argument(
        "--axis-map", default="xyz", choices=("xyz", "zxy"),
        help="zxy undoes NeuS2/Instant-NGP's xyz<-yzx internal axis cycle",
    )
    args = parser.parse_args()

    source = mesh_from(Path(args.input).resolve())
    reference = mesh_from(Path(args.reference).resolve())
    vertices = np.asarray(source.vertices, dtype=np.float64).copy()
    if args.axis_map == "zxy":
        vertices = vertices[:, (2, 0, 1)]

    padding = args.padding_mm / 1000.0
    lower = np.asarray(reference.bounds[0]) - padding
    upper = np.asarray(reference.bounds[1]) + padding
    inside = np.all((vertices >= lower) & (vertices <= upper), axis=1)
    faces = np.asarray(source.faces)
    keep = np.all(inside[faces], axis=1)
    cropped = trimesh.Trimesh(vertices=vertices, faces=faces[keep], process=False)
    cropped.remove_unreferenced_vertices()

    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": args.method,
        "runtime_s": args.runtime_s,
        "status": "completed" if len(cropped.faces) >= 100 else "failed",
        "source_mesh_access": False,
        "normalization": "observation-derived TSDF AABB crop",
        "axis_map": args.axis_map,
        "padding_mm": args.padding_mm,
        "observation_bounds_m": [lower.tolist(), upper.tolist()],
        "input_vertices": int(len(source.vertices)),
        "input_faces": int(len(source.faces)),
        "retained_vertices": int(len(cropped.vertices)),
        "retained_faces": int(len(cropped.faces)),
    }
    if report["status"] == "completed":
        outputs = {}
        for suffix in ("ply", "obj", "stl"):
            path = output / f"{args.method}.{suffix}"
            cropped.export(path)
            outputs[suffix] = path.name
        report["outputs"] = outputs
        report["vertices"] = int(len(cropped.vertices))
        report["faces"] = int(len(cropped.faces))
    else:
        report["error"] = "fewer than 100 faces intersect the observation-derived object volume"
    (output / f"{args.method}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
