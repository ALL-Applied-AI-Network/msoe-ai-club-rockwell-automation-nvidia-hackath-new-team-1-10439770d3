"""Extract the blue housing and green gear from a colored reconstruction.

This is an observation-only post-process: it uses the colors visible in the
input orbit, never CAD geometry.  A largest-component DBSCAN pass removes
isolated chromatic reconstruction noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--eps-scale", type=float, default=3.5)
    parser.add_argument(
        "--dbscan-sample-limit",
        type=int,
        default=250000,
        help="Cluster at most this many color-selected points, then propagate the largest component to all points",
    )
    args = parser.parse_args()

    geometry = trimesh.load(args.input, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    colors = np.asarray(geometry.colors if hasattr(geometry, "colors") else geometry.visual.vertex_colors)
    if colors.ndim != 2 or len(colors) != len(vertices) or colors.shape[1] < 3:
        raise RuntimeError("Input has no per-point RGB colors")
    rgb = colors[:, :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue, saturation, value = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    blue = (hue >= 88) & (hue <= 145) & (saturation >= 45) & (value >= 18)
    green = (hue >= 32) & (hue < 88) & (saturation >= 45) & (value >= 25)
    selected = blue | green
    color_vertices, color_rgb = vertices[selected], rgb[selected]
    if len(color_vertices) < 50:
        raise RuntimeError(f"Color filter retained only {len(color_vertices)} points")

    sample = color_vertices
    if len(sample) > 20000:
        rng = np.random.default_rng(20261027)
        sample = sample[rng.choice(len(sample), 20000, replace=False)]
    distances = cKDTree(sample).query(sample, k=min(6, len(sample)), workers=-1)[0][:, -1]
    median_spacing = float(np.median(distances[distances > 0]))
    eps = max(median_spacing * args.eps_scale, 1e-9)
    cluster_vertices = color_vertices
    cluster_indices = np.arange(len(color_vertices))
    if args.dbscan_sample_limit > 0 and len(cluster_vertices) > args.dbscan_sample_limit:
        rng = np.random.default_rng(20261028)
        cluster_indices = np.sort(
            rng.choice(len(cluster_vertices), args.dbscan_sample_limit, replace=False)
        )
        cluster_vertices = cluster_vertices[cluster_indices]
        # Subsampling increases nearest-neighbor spacing.  Re-estimate epsilon
        # on the actual clustering set rather than reusing the full-cloud value.
        spacing_sample = cluster_vertices
        if len(spacing_sample) > 20000:
            spacing_sample = spacing_sample[
                rng.choice(len(spacing_sample), 20000, replace=False)
            ]
        distances = cKDTree(spacing_sample).query(
            spacing_sample, k=min(6, len(spacing_sample)), workers=-1
        )[0][:, -1]
        median_spacing = float(np.median(distances[distances > 0]))
        eps = max(median_spacing * args.eps_scale, 1e-9)

    labels = DBSCAN(eps=eps, min_samples=4, n_jobs=-1).fit_predict(cluster_vertices)
    valid = labels >= 0
    if np.any(valid):
        ids, counts = np.unique(labels[valid], return_counts=True)
        keep_label = ids[np.argmax(counts)]
        cluster_keep = labels == keep_label
        if len(cluster_vertices) == len(color_vertices):
            keep = cluster_keep
        else:
            # Propagate membership to the full-resolution cloud using a tight
            # nearest-neighbor tube around the sampled largest component.
            distance = cKDTree(cluster_vertices[cluster_keep]).query(
                color_vertices, workers=-1
            )[0]
            keep = distance <= (2.0 * eps)
    else:
        keep = np.ones(len(color_vertices), dtype=bool)
        keep_label = -1
    result = trimesh.points.PointCloud(color_vertices[keep], colors=color_rgb[keep])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)
    manifest = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "input_points": int(len(vertices)),
        "color_selected_points": int(len(color_vertices)),
        "largest_component_points": int(np.count_nonzero(keep)),
        "median_knn_spacing": median_spacing,
        "dbscan_eps": eps,
        "dbscan_label": int(keep_label),
        "dbscan_input_points": int(len(cluster_vertices)),
        "filter": "HSV blue/green visible-object colors followed by largest DBSCAN component",
        "cad_used": False,
    }
    manifest_path = args.manifest or args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
