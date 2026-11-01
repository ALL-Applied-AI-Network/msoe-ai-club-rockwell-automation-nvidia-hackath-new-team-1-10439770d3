"""Render a fast, deterministic point/mesh turntable without a GUI.

The renderer intentionally visualizes candidate geometry only.  It does not
consult the held-out CAD reference.  When an evaluator-aligned file is passed,
the caller should label the resulting video accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh


def load_points(path: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometries = list(geometry.geometry.values())
        if not geometries:
            raise RuntimeError(f"No geometry in {path}")
        geometry = trimesh.util.concatenate(geometries)
    points = np.asarray(geometry.vertices, dtype=np.float64)
    visual = getattr(geometry, "visual", None)
    colors = None
    if hasattr(geometry, "colors"):
        colors = np.asarray(geometry.colors)
    elif visual is not None and hasattr(visual, "vertex_colors"):
        colors = np.asarray(visual.vertex_colors)
    if colors is None or len(colors) != len(points) or colors.shape[1] < 3:
        colors = np.tile(np.array([[88, 180, 230]], dtype=np.uint8), (len(points), 1))
    else:
        colors = colors[:, :3].astype(np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if len(points) > limit:
        rng = np.random.default_rng(20261027)
        indices = np.sort(rng.choice(len(points), limit, replace=False))
        points, colors = points[indices], colors[indices]
    return points, colors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="RGB-only reconstruction")
    parser.add_argument("--subtitle", default="candidate geometry")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--max-points", type=int, default=120000)
    args = parser.parse_args()

    points, rgb = load_points(args.input, args.max_points)
    low, high = np.percentile(points, [2, 98], axis=0)
    center = 0.5 * (low + high)
    points = points - center
    radius = max(float(np.linalg.norm(high - low)) * 0.65, 1e-6)
    width = height = args.size
    focal = 0.93 * width
    camera_distance = 3.0 * radius
    elevation = np.deg2rad(24.0)
    bg = np.array([17, 14, 10], dtype=np.uint8)  # BGR

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    up_world = np.array([0.0, 0.0, 1.0])
    for frame_index in range(args.frames):
        azimuth = 2.0 * np.pi * frame_index / args.frames
        eye = camera_distance * np.array(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
        )
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = points - eye
        cam_x = relative @ right
        cam_y = relative @ up
        cam_z = relative @ forward
        valid = cam_z > 1e-6
        px = np.rint(width / 2 + focal * cam_x[valid] / cam_z[valid]).astype(np.int32)
        py = np.rint(height / 2 - focal * cam_y[valid] / cam_z[valid]).astype(np.int32)
        depth = cam_z[valid]
        color = rgb[valid][:, ::-1]  # RGB -> BGR
        inside = (px >= 2) & (px < width - 2) & (py >= 2) & (py < height - 2)
        px, py, depth, color = px[inside], py[inside], depth[inside], color[inside]
        order = np.argsort(depth)[::-1]  # far first; near points overwrite

        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = bg
        for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            image[py[order] + oy, px[order] + ox] = color[order]
        # Subtle vignette and professional information hierarchy.
        yy, xx = np.ogrid[:height, :width]
        vignette = 1.0 - 0.20 * np.clip(((xx - width / 2) ** 2 + (yy - height / 2) ** 2) / (0.55 * width) ** 2, 0, 1)
        image = np.clip(image.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)
        cv2.putText(image, args.title, (34, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (242, 244, 247), 2, cv2.LINE_AA)
        cv2.putText(image, args.subtitle, (35, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (156, 169, 183), 1, cv2.LINE_AA)
        cv2.putText(image, f"{len(points):,} displayed points", (35, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (127, 145, 162), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    print(f"wrote {args.output} ({args.frames} frames, {len(points)} displayed points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
