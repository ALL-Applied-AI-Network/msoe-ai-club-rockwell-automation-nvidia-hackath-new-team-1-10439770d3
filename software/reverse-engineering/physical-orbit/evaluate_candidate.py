"""Similarity-align and score an RGB-only reconstruction against held-back CAD.

The evaluator is run only after a candidate is frozen.  It reports the scale
required to align the monocular result rather than pretending raw RGB provides
known metric scale.  Symmetric accuracy/completeness and F-scores are computed
from independently sampled surfaces.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def as_mesh_or_points(path: Path):
    geometry = trimesh.load(path, force=None, process=True)
    if isinstance(geometry, trimesh.Scene):
        meshes = [item for item in geometry.geometry.values() if isinstance(item, trimesh.Trimesh)]
        clouds = [item for item in geometry.geometry.values() if isinstance(item, trimesh.points.PointCloud)]
        if meshes:
            return trimesh.util.concatenate(meshes)
        if clouds:
            vertices = np.concatenate([cloud.vertices for cloud in clouds], axis=0)
            return trimesh.points.PointCloud(vertices)
        raise RuntimeError(f"No mesh or point cloud geometry in {path}")
    return geometry


def sample_geometry(geometry, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
        points, _ = trimesh.sample.sample_surface(geometry, count, seed=seed)
    else:
        points = np.asarray(geometry.vertices, dtype=np.float64)
        if len(points) > count:
            points = points[rng.choice(len(points), count, replace=False)]
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 50:
        raise RuntimeError("Candidate has fewer than 50 finite points")
    return points


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes


def robust_extent(points: np.ndarray) -> np.ndarray:
    return np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0)


def umeyama(source: np.ndarray, target: np.ndarray, allow_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    src, dst = source - src_mean, target - dst_mean
    covariance = dst.T @ src / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(src * src, axis=1)))
    scale = float(np.sum(singular * sign) / max(variance, 1e-12)) if allow_scale else 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def symmetric_distances(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dst_tree, src_tree = cKDTree(target), cKDTree(source)
    return dst_tree.query(source, workers=-1)[0], src_tree.query(target, workers=-1)[0]


def refine_similarity(points: np.ndarray, target: np.ndarray, initial_a: np.ndarray, initial_b: np.ndarray, iterations: int = 25) -> tuple[np.ndarray, np.ndarray, float]:
    a, b = initial_a.copy(), initial_b.copy()
    previous = np.inf
    for _ in range(iterations):
        transformed = points @ a.T + b
        dst_tree = cKDTree(target)
        forward_d, forward_i = dst_tree.query(transformed, workers=-1)
        src_tree = cKDTree(transformed)
        reverse_d, reverse_i = src_tree.query(target, workers=-1)
        f_keep = forward_d <= np.percentile(forward_d, 80)
        r_keep = reverse_d <= np.percentile(reverse_d, 80)
        source_pairs = np.concatenate([transformed[f_keep], transformed[reverse_i[r_keep]]], axis=0)
        target_pairs = np.concatenate([target[forward_i[f_keep]], target[r_keep]], axis=0)
        delta_scale, delta_rotation, delta_translation = umeyama(source_pairs, target_pairs, allow_scale=True)
        delta_scale = float(np.clip(delta_scale, 0.97, 1.03))
        delta_a = delta_scale * delta_rotation
        a = delta_a @ a
        b = delta_a @ b + delta_translation
        transformed = points @ a.T + b
        forward, reverse = symmetric_distances(transformed, target)
        score = 0.5 * (np.mean(forward) + np.mean(reverse))
        if abs(previous - score) < 1e-7:
            break
        previous = score
    return a, b, float(previous)


def best_alignment(candidate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    candidate_center, candidate_axes = principal_axes(candidate)
    reference_center, reference_axes = principal_axes(reference)
    candidate_local = (candidate - candidate_center) @ candidate_axes
    reference_local = (reference - reference_center) @ reference_axes
    candidate_scale_extent = np.linalg.norm(robust_extent(candidate_local))
    reference_scale_extent = np.linalg.norm(robust_extent(reference_local))
    initial_scale = reference_scale_extent / max(candidate_scale_extent, 1e-12)
    best = None
    for signs in itertools.product([-1.0, 1.0], repeat=3):
        sign_matrix = np.diag(signs)
        rotation = reference_axes @ sign_matrix @ candidate_axes.T
        if np.linalg.det(rotation) < 0:
            continue
        a0 = initial_scale * rotation
        b0 = reference_center - a0 @ candidate_center
        a, b, score = refine_similarity(candidate, reference, a0, b0)
        if best is None or score < best[0]:
            best = (score, a, b, signs)
    if best is None:
        raise RuntimeError("Could not initialize a proper similarity transform")
    score, a, b, signs = best
    return a, b, {"initial_scale": initial_scale, "pca_signs": signs, "icp_chamfer_mm": score}


def fscore(accuracy: np.ndarray, completeness: np.ndarray, threshold: float) -> dict:
    precision = float(np.mean(accuracy <= threshold))
    recall = float(np.mean(completeness <= threshold))
    value = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "fscore": value}


def mesh_component_count(mesh: trimesh.Trimesh) -> int | None:
    """Count face components without materializing every component submesh.

    ``Trimesh.split`` duplicates large vertex-attribute arrays and metadata.
    On learned Poisson meshes with roughly 1.5 million faces, that reporting
    step can use more memory than the actual geometric evaluation.
    """
    if len(mesh.faces) == 0:
        return 0
    try:
        labels = trimesh.graph.connected_component_labels(
            edges=mesh.face_adjacency,
            node_count=len(mesh.faces),
        )
        return int(labels.max()) + 1 if len(labels) else int(len(mesh.faces))
    except (MemoryError, ValueError, RuntimeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--method", required=True)
    parser.add_argument("--runtime-s", type=float)
    args = parser.parse_args()

    candidate_geometry = as_mesh_or_points(args.candidate)
    reference_geometry = as_mesh_or_points(args.reference)
    candidate_points = sample_geometry(candidate_geometry, args.samples, seed=20261027)
    reference_points = sample_geometry(reference_geometry, args.samples, seed=20261028)
    a, b, alignment = best_alignment(candidate_points, reference_points)
    aligned_points = candidate_points @ a.T + b
    accuracy, completeness = symmetric_distances(aligned_points, reference_points)
    scale = float(np.cbrt(abs(np.linalg.det(a))))
    aligned_geometry = candidate_geometry.copy()
    transform = np.eye(4)
    transform[:3, :3] = a
    transform[:3, 3] = b
    aligned_geometry.apply_transform(transform)
    args.out.mkdir(parents=True, exist_ok=True)
    aligned_path = args.out / "aligned_candidate.ply"
    aligned_geometry.export(aligned_path)

    candidate_extents = np.percentile(aligned_points, 98, axis=0) - np.percentile(aligned_points, 2, axis=0)
    reference_extents = np.percentile(reference_points, 98, axis=0) - np.percentile(reference_points, 2, axis=0)
    thresholds = {str(value): fscore(accuracy, completeness, value) for value in (1.0, 2.0, 5.0)}
    metrics = {
        "method": args.method,
        "candidate": str(args.candidate.resolve()),
        "reference": str(args.reference.resolve()),
        "aligned_output": str(aligned_path.resolve()),
        "runtime_s": args.runtime_s,
        "sample_count_each": args.samples,
        "similarity_alignment": {
            **alignment,
            "scale_to_reference_mm": scale,
            "matrix": transform.tolist(),
            "warning": "Post-hoc evaluation transform; scale/pose were not supplied to reconstruction.",
        },
        "geometry_mm": {
            "accuracy_mean": float(np.mean(accuracy)),
            "accuracy_median": float(np.median(accuracy)),
            "accuracy_p95": float(np.percentile(accuracy, 95)),
            "completeness_mean": float(np.mean(completeness)),
            "completeness_median": float(np.median(completeness)),
            "completeness_p95": float(np.percentile(completeness, 95)),
            "symmetric_chamfer_mean": float(0.5 * (np.mean(accuracy) + np.mean(completeness))),
            "fscore": thresholds,
            "candidate_robust_extents": candidate_extents.tolist(),
            "reference_robust_extents": reference_extents.tolist(),
            "extent_absolute_error": np.abs(candidate_extents - reference_extents).tolist(),
        },
        "topology": {
            "kind": type(candidate_geometry).__name__,
            "vertices": int(len(candidate_geometry.vertices)),
            "faces": int(len(candidate_geometry.faces)) if isinstance(candidate_geometry, trimesh.Trimesh) else 0,
            "watertight": bool(candidate_geometry.is_watertight) if isinstance(candidate_geometry, trimesh.Trimesh) else False,
            "components": mesh_component_count(candidate_geometry) if isinstance(candidate_geometry, trimesh.Trimesh) else None,
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
