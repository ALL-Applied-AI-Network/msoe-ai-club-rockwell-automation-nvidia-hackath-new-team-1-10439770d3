"""Evaluator-only surface comparison and method ranking.

The supplied source STL files are multi-component and non-watertight, so the
primary metrics are symmetric surface accuracy/completeness and F-score.
Watertightness is reported but cannot be the dominant score when the source
itself is not watertight.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reconstructions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=20261024)
    return parser.parse_args()


def as_mesh(value) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Scene):
        geometries = [g for g in value.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("mesh scene contains no triangle geometry")
        return trimesh.util.concatenate(geometries)
    if not isinstance(value, trimesh.Trimesh):
        raise TypeError(type(value))
    return value


def sample(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # trimesh currently uses NumPy's module-level RNG for surface selection.
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    points, faces = trimesh.sample.sample_surface(mesh, count)
    normals = np.asarray(mesh.face_normals)[faces]
    return points, normals


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def component_count(mesh: trimesh.Trimesh) -> int:
    """Count face-connected components without materializing mesh copies.

    ``mesh.split()`` duplicates vertices/faces for every component and can use
    tens of GiB on the high-resolution neural-SDF meshes.  Component labels on
    the face-adjacency graph provide the same count with bounded memory.
    """
    if len(mesh.faces) == 0:
        return 0
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency,
        node_count=len(mesh.faces),
    )
    return int(len(np.unique(labels)))


def score_mesh(gt: trimesh.Trimesh, recon: trimesh.Trimesh, count: int, rng: np.random.Generator) -> dict:
    gt_points, gt_normals = sample(gt, count, rng)
    recon_points, recon_normals = sample(recon, count, rng)
    gt_tree = cKDTree(gt_points)
    recon_tree = cKDTree(recon_points)
    accuracy, nearest_gt = gt_tree.query(recon_points, workers=-1)
    completeness, nearest_recon = recon_tree.query(gt_points, workers=-1)

    normal_a = np.abs(np.einsum("ij,ij->i", recon_normals, gt_normals[nearest_gt]))
    normal_c = np.abs(np.einsum("ij,ij->i", gt_normals, recon_normals[nearest_recon]))
    metrics = {
        "accuracy_mean_mm": float(np.mean(accuracy) * 1000.0),
        "accuracy_median_mm": float(np.median(accuracy) * 1000.0),
        "accuracy_p95_mm": percentile(accuracy, 95) * 1000.0,
        "accuracy_p99_mm": percentile(accuracy, 99) * 1000.0,
        "completeness_mean_mm": float(np.mean(completeness) * 1000.0),
        "completeness_median_mm": float(np.median(completeness) * 1000.0),
        "completeness_p95_mm": percentile(completeness, 95) * 1000.0,
        "completeness_p99_mm": percentile(completeness, 99) * 1000.0,
        "symmetric_chamfer_mean_mm": float((np.mean(accuracy) + np.mean(completeness)) * 500.0),
        "normal_consistency": float(0.5 * (np.mean(normal_a) + np.mean(normal_c))),
    }
    for threshold_mm in (0.5, 1.0, 2.0, 5.0):
        threshold = threshold_mm / 1000.0
        precision = float(np.mean(accuracy <= threshold))
        recall = float(np.mean(completeness <= threshold))
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        key = str(threshold_mm).replace(".", "p")
        metrics[f"precision_{key}mm"] = precision
        metrics[f"recall_{key}mm"] = recall
        metrics[f"fscore_{key}mm"] = fscore

    bottom = gt_points[:, 2] <= (float(gt.bounds[0, 2]) + 0.001)
    metrics["bottom_sample_fraction"] = float(np.mean(bottom))
    metrics["bottom_recall_1mm"] = float(np.mean(completeness[bottom] <= 0.001)) if bottom.any() else None
    metrics["bottom_completeness_mean_mm"] = float(np.mean(completeness[bottom]) * 1000.0) if bottom.any() else None
    gt_extents = np.asarray(gt.extents)
    recon_extents = np.asarray(recon.extents)
    metrics["extent_error_mm_xyz"] = (np.abs(recon_extents - gt_extents) * 1000.0).tolist()
    metrics["extent_error_mean_mm"] = float(np.mean(np.abs(recon_extents - gt_extents)) * 1000.0)
    return metrics


def ranking_score(row: dict) -> float:
    # Accuracy/completeness dominate. Runtime and watertightness are tie
    # breakers because the supplied evaluator source is itself non-watertight.
    f1 = float(row.get("fscore_1p0mm", 0.0))
    f2 = float(row.get("fscore_2p0mm", 0.0))
    normal = float(row.get("normal_consistency", 0.0))
    chamfer = float(row.get("symmetric_chamfer_mean_mm", 100.0))
    bottom = float(row.get("bottom_recall_1mm") or 0.0)
    runtime = float(row.get("runtime_s", 9999.0))
    return (
        45.0 * f1 + 25.0 * f2 + 12.0 * normal + 8.0 * bottom
        + 8.0 * np.exp(-chamfer / 2.0) + 1.0 * float(bool(row.get("watertight")))
        + 1.0 * np.exp(-runtime / 30.0)
    )


def main() -> int:
    args = parse_args()
    root = Path(args.dataset).resolve()
    recon_root = Path(args.reconstructions).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    eval_dir = root / "evaluator_only"
    source_npz = eval_dir / "source_mesh.npz"
    if not source_npz.exists():
        raise FileNotFoundError(source_npz)
    source = np.load(source_npz)
    gt = trimesh.Trimesh(vertices=source["vertices"], faces=source["faces"], process=False)
    rng = np.random.default_rng(args.seed)
    rows = []
    for report_path in sorted(recon_root.glob("*.json")):
        if report_path.name == "reconstruction_summary.json":
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or "outputs" not in report:
            rows.append({"method": report.get("method", report_path.stem), "status": report.get("status", "failed"), "error": report.get("error")})
            continue
        method = report["method"]
        mesh_path = recon_root / report["outputs"]["ply"]
        start = time.perf_counter()
        try:
            recon = as_mesh(trimesh.load(mesh_path, process=False))
            metrics = score_mesh(gt, recon, args.samples, rng)
            row = {
                "method": method,
                "status": "scored",
                "runtime_s": report.get("runtime_s"),
                "vertices": int(len(recon.vertices)),
                "faces": int(len(recon.faces)),
                "watertight": bool(recon.is_watertight),
                "components": component_count(recon),
                "evaluation_s": time.perf_counter() - start,
                **metrics,
            }
            row["ranking_score"] = ranking_score(row)
        except Exception as exc:
            row = {"method": method, "status": "evaluation_failed", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)

    ranked = sorted([r for r in rows if r.get("status") == "scored"], key=lambda r: r["ranking_score"], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    failures = [r for r in rows if r.get("status") != "scored"]
    result = {
        "source": {
            "vertices": int(len(gt.vertices)), "faces": int(len(gt.faces)),
            "watertight": bool(gt.is_watertight),
            "components": component_count(gt),
            "extents_mm": (np.asarray(gt.extents) * 1000.0).tolist(),
        },
        "metric_contract": {
            "primary": "bidirectional sampled surface F-score and distance",
            "thresholds_mm": [0.5, 1.0, 2.0, 5.0],
            "ranking_note": "source is non-watertight; watertightness is a one-point tie-breaker",
            "samples_per_surface": args.samples,
        },
        "ranking": ranked,
        "failures": failures,
    }
    (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if ranked:
        columns = list(ranked[0].keys())
        with (out / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ranked)
    print(json.dumps({"source": result["source"], "ranking": [
        {"rank": r["rank"], "method": r["method"], "score": r["ranking_score"],
         "f1_1mm": r["fscore_1p0mm"], "chamfer_mm": r["symmetric_chamfer_mean_mm"],
         "runtime_s": r["runtime_s"]} for r in ranked
    ], "failures": failures}, indent=2))
    return 0 if ranked else 2


if __name__ == "__main__":
    raise SystemExit(main())
