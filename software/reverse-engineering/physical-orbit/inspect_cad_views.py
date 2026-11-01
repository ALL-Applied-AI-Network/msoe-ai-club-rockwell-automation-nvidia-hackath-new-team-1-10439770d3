"""Create quick orthographic inspection views for 3MF assembly placement.

This is a deterministic engineering aid, not a photogrammetry result.  It
samples CAD surfaces and makes top/front/side plots so shaft centers and the
correct assembly orientation can be selected before producing the polished
reference reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=180_000)
    args = parser.parse_args()

    mesh = load_mesh(args.input)
    points, face_ids = trimesh.sample.sample_surface(mesh, args.samples)
    normals = mesh.face_normals[face_ids]
    # Prefer upward-facing and high surfaces for the top view while keeping
    # silhouettes in the side views.
    top_score = np.clip(normals[:, 2], 0.0, 1.0)
    rng = np.random.default_rng(20261027)
    keep = rng.random(len(points)) < (0.15 + 0.85 * top_score)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#111820")
    views = [
        (0, 1, "TOP  X-Y", keep),
        (0, 2, "FRONT  X-Z", np.ones(len(points), dtype=bool)),
        (1, 2, "SIDE  Y-Z", np.ones(len(points), dtype=bool)),
    ]
    for ax, (a, b, title, mask) in zip(axes, views):
        ax.set_facecolor("#111820")
        sample = points[mask]
        if len(sample) > 60_000:
            sample = sample[rng.choice(len(sample), 60_000, replace=False)]
        ax.scatter(sample[:, a], sample[:, b], s=0.14, c=sample[:, 2], cmap="viridis", alpha=0.75)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, color="white", fontsize=13, weight="bold")
        ax.tick_params(colors="#91a4b7")
        ax.grid(color="#273442", linewidth=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#415568")
    fig.suptitle(f"CAD inspection | {args.input.name}", color="white", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
