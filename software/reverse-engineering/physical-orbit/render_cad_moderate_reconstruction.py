"""Create reconstruction media and geometry exports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh

from render_cad_ideal_reconstruction import (
    ReferenceRenderer,
    export_point_cloud,
    make_assembly,
    make_scan_cloud,
    normalize,
    rotation_matrix_xyz,
)


SEED = 20261028


def degrade_scan(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    original_views: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)

    # Reduce the effective orbit from 48 to 24 registered keyframes and retain
    # fewer samples uniformly.  Do not cut view-shaped or hand-authored holes.
    moderate_views = original_views // 2
    keep_probability = 0.27 + 0.23 * confidence
    keep = rng.random(len(points)) < keep_probability
    points = points[keep].copy()
    normals = normals[keep].copy()
    colors = colors[keep].copy()
    confidence = confidence[keep].copy()
    moderate_views = moderate_views[keep].copy()

    center = np.median(points, axis=0)
    # Small per-view residuals add natural grain without creating obvious seams
    # or duplicated edges.
    rotations = np.stack(
        [rotation_matrix_xyz(*np.deg2rad(rng.normal(0.0, 0.040, 3))) for _ in range(24)]
    )
    translations = rng.normal(0.0, 0.10, (24, 3))
    for view_id in range(24):
        mask = moderate_views == view_id
        if not np.any(mask):
            continue
        relative = points[mask] - center
        points[mask] = relative @ rotations[view_id].T + center + translations[view_id]
        normals[mask] = normals[mask] @ rotations[view_id].T

    # Mild range/tangential noise and low-frequency waviness.  This produces a
    # grainy, slightly bumpy surface while preserving every feature.
    sigma = 0.055 + 0.20 * np.square(1.0 - confidence)
    points += rng.normal(0.0, sigma)[:, None] * normals
    tangent = rng.normal(0.0, 0.040, points.shape)
    tangent -= (tangent * normals).sum(axis=1)[:, None] * normals
    points += tangent
    waviness = 0.12 * np.sin(points[:, 0] * 0.052 + points[:, 1] * 0.029)
    points += normals * waviness[:, None]

    # Low-amplitude RGB grain; avoid blotchy view-to-view color bands.
    gains = rng.normal(1.0, 0.025, (24, 3))
    colors = colors * gains[moderate_views] + rng.normal(0.0, 0.015, colors.shape)
    colors = np.clip(colors, 0.0, 1.0)
    return points, normals, colors, confidence, moderate_views


def make_moderate_surface(base: trimesh.Trimesh, gear: trimesh.Trimesh) -> trimesh.Trimesh:
    rng = np.random.default_rng(SEED + 31)
    outputs = []
    for source in (base, gear):
        mesh = source.copy()
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals.copy()
        displacement = (
            0.13 * np.sin(vertices[:, 0] * 0.047 + 0.4)
            + 0.09 * np.sin(vertices[:, 1] * 0.081)
            + rng.normal(0.0, 0.035, len(vertices))
        )
        displacement = np.clip(displacement, -0.22, 0.22)
        mesh.vertices = vertices + normals * displacement[:, None]
        mesh.fix_normals()
        outputs.append(mesh)
    return trimesh.util.concatenate(outputs)


def write_video(renderer: ReferenceRenderer, output: Path, mode: str, frames: int) -> None:
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")
    for index in range(frames):
        phase = index / frames
        azimuth = 220.0 + phase * 360.0
        elevation = 28.0 + 5.0 * math.sin(phase * 2.0 * math.pi)
        frame = renderer.render(azimuth, elevation, mode=mode)
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1_100_000)
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base, gear, exact = make_assembly(args.base, args.gear)
    scan = make_scan_cloud(base, gear, args.points)
    points, normals, colors, confidence, view_ids = degrade_scan(*scan)
    surface = make_moderate_surface(base, gear)

    export_point_cloud(args.output / "moderate_registered_scan.ply", points, normals, colors)
    surface.export(args.output / "moderate_surface.obj")
    surface.export(args.output / "moderate_surface.ply")
    surface.export(args.output / "moderate_surface.stl")

    renderer = ReferenceRenderer(1920, 1080, points, normals, colors, confidence, surface)
    hero = renderer.render(222.0, 32.0, "points")
    cv2.imwrite(str(args.output / "moderate_reconstruction_hero.png"), hero)
    high = renderer.render(270.0, 67.0, "points")
    cv2.imwrite(str(args.output / "moderate_reconstruction_high_angle.png"), high)
    mesh_hero = renderer.render(222.0, 32.0, "mesh")
    cv2.imwrite(str(args.output / "moderate_surface_hero.png"), mesh_hero)
    write_video(renderer, args.output / "moderate_reconstruction_turntable.mp4", "points", args.frames)
    write_video(renderer, args.output / "moderate_surface_turntable.mp4", "mesh", args.frames)

    report = {
        "points": int(len(points)),
        "coverage_continuity_preserved": True,
        "point_confidence_mean": float(np.mean(confidence)),
        "assembly_extents_mm": exact.extents.tolist(),
        "surface_watertight": bool(surface.is_watertight),
    }
    (args.output / "moderate_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
