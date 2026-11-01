"""Create observation-derived Poisson and ball-pivoting meshes from a point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def clean(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--density-quantile", type=float, default=0.03)
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.input))
    if len(cloud.points) < 100:
        raise RuntimeError("Point cloud has fewer than 100 points")
    points = np.asarray(cloud.points)
    tree = o3d.geometry.KDTreeFlann(cloud)
    neighbor_distances = []
    for index in np.linspace(0, len(points) - 1, min(len(points), 5000), dtype=int):
        _, _, squared = tree.search_knn_vector_3d(cloud.points[index], min(6, len(points)))
        if len(squared) > 1:
            neighbor_distances.append(np.sqrt(squared[-1]))
    spacing = float(np.median(neighbor_distances))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(4 * spacing, 1e-9), max_nn=40))
    try:
        cloud.orient_normals_consistent_tangent_plane(min(30, max(3, len(points) // 20)))
    except RuntimeError:
        cloud.normalize_normals()
    bounds = cloud.get_axis_aligned_bounding_box()
    bounds = bounds.scale(1.05, bounds.get_center())
    args.out.mkdir(parents=True, exist_ok=True)

    poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=args.poisson_depth, linear_fit=True)
    densities = np.asarray(densities)
    if len(densities) == len(poisson.vertices):
        poisson.remove_vertices_by_mask(densities < np.quantile(densities, args.density_quantile))
    poisson = poisson.crop(bounds)
    poisson = clean(poisson)
    poisson.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.out / "poisson.ply"), poisson, write_ascii=False)
    o3d.io.write_triangle_mesh(str(args.out / "poisson.obj"), poisson, write_ascii=False)
    o3d.io.write_triangle_mesh(str(args.out / "poisson.stl"), poisson, write_ascii=False)

    radii = o3d.utility.DoubleVector([1.5 * spacing, 2.5 * spacing, 4.0 * spacing])
    ball = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
    ball = clean(ball)
    ball.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.out / "ball_pivoting.ply"), ball, write_ascii=False)
    o3d.io.write_triangle_mesh(str(args.out / "ball_pivoting.obj"), ball, write_ascii=False)

    manifest = {
        "input": str(args.input.resolve()),
        "input_points": int(len(cloud.points)),
        "median_knn_spacing": spacing,
        "poisson": {
            "depth": args.poisson_depth,
            "density_quantile": args.density_quantile,
            "vertices": int(len(poisson.vertices)),
            "triangles": int(len(poisson.triangles)),
            "watertight": bool(poisson.is_watertight()),
        },
        "ball_pivoting": {
            "radii": list(radii),
            "vertices": int(len(ball.vertices)),
            "triangles": int(len(ball.triangles)),
            "watertight": bool(ball.is_watertight()),
        },
        "cad_used": False,
    }
    (args.out / "meshing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
