"""Render a lightweight turntable video from an exported triangle mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


def as_mesh(value) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Scene):
        return trimesh.util.concatenate(tuple(value.geometry.values()))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--points", type=int, default=220000)
    parser.add_argument("--label", default="reconstruction")
    args = parser.parse_args()

    mesh = as_mesh(trimesh.load(args.mesh, process=False))
    np.random.seed(20261024)
    points, face_ids = trimesh.sample.sample_surface(mesh, args.points)
    normals = np.asarray(mesh.face_normals)[face_ids]
    center = np.asarray(mesh.bounds).mean(axis=0)
    points = points - center
    scale = max(float(np.max(np.ptp(points, axis=0))), 1e-9)
    points /= scale

    width, height = 1280, 720
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    elevation = np.deg2rad(24.0)
    ce, se = np.cos(elevation), np.sin(elevation)
    rx = np.asarray(((1, 0, 0), (0, ce, -se), (0, se, ce)), dtype=np.float64)
    light = np.asarray((0.25, -0.35, 0.90), dtype=np.float64)
    light /= np.linalg.norm(light)
    for index in range(args.frames):
        angle = 2.0 * np.pi * index / args.frames
        ca, sa = np.cos(angle), np.sin(angle)
        rz = np.asarray(((ca, -sa, 0), (sa, ca, 0), (0, 0, 1)), dtype=np.float64)
        rotation = rz @ rx
        p = points @ rotation
        n = normals @ rotation
        x = np.rint(width * 0.5 + p[:, 0] * 0.84 * width).astype(np.int32)
        y = np.rint(height * 0.53 - p[:, 2] * 0.84 * width).astype(np.int32)
        valid = (x >= 2) & (x < width - 2) & (y >= 2) & (y < height - 2)
        ids = np.flatnonzero(valid)
        # Painter order, back to front. Two-pixel splats make thin surfaces legible.
        ids = ids[np.argsort(p[ids, 1])[::-1]]
        shade = np.clip(0.28 + 0.72 * np.abs(n @ light), 0.0, 1.0)
        image = np.full((height, width, 3), (28, 30, 34), dtype=np.uint8)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            values = shade[ids]
            image[y[ids] + dy, x[ids] + dx, 0] = (70 + 45 * values).astype(np.uint8)
            image[y[ids] + dy, x[ids] + dx, 1] = (125 + 90 * values).astype(np.uint8)
            image[y[ids] + dy, x[ids] + dx, 2] = (195 + 55 * values).astype(np.uint8)
        cv2.putText(image, args.label, (36, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (235, 238, 242), 2, cv2.LINE_AA)
        cv2.putText(image, f"view {index + 1:03d}/{args.frames}", (36, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180, 185, 192), 2, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    print(f"wrote {args.frames} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
