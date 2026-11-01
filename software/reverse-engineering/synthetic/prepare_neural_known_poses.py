"""Prepare an Instant-NGP/NeuS2-style masked dataset from observations only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_classical import intrinsics, load_manifest, read_rgb_depth, select_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--views", type=int, default=72)
    args = parser.parse_args()

    root = Path(args.dataset).resolve()
    observations = root / "observations"
    out = Path(args.out).resolve()
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(root)
    _, calib = intrinsics(manifest)
    upright = select_frames(manifest["frames"], 0, include_flipped=False)
    indices = np.linspace(0, len(upright) - 1, min(args.views, len(upright))).round().astype(int)
    frames = []
    for image_id, source_index in enumerate(indices):
        frame = upright[int(source_index)]
        rgb, _, mask = read_rgb_depth(observations, frame)
        # Alpha is consumed by Instant-NGP/NeuS2. Neuralangelo's loader keeps
        # RGB only, so also replace the calibrated background with white to
        # prevent the table from becoming part of the learned object surface.
        masked_rgb = rgb.copy()
        masked_rgb[~mask] = 255
        rgba = np.dstack((masked_rgb, mask.astype(np.uint8) * 255))
        name = f"{image_id:04d}.png"
        cv2.imwrite(str(image_dir / name), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        # Isaac/USD uses row vectors with camera +X right, +Y up and -Z
        # forward, which becomes the OpenGL camera-to-world convention after
        # transposition. The object/canonical frame is the reconstruction world.
        transform = np.asarray(frame["camera_to_object"], dtype=np.float64).T
        frames.append({"file_path": f"images/{name}", "transform_matrix": transform.tolist()})

    payload = {
        "camera_model": "OPENCV",
        "w": int(calib["width"]), "h": int(calib["height"]),
        "fl_x": float(calib["fx"]), "fl_y": float(calib["fy"]),
        "sk_x": 0.0, "sk_y": 0.0,
        "cx": float(calib["cx"]), "cy": float(calib["cy"]),
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "p1": 0.0, "p2": 0.0, "is_fisheye": False,
        "camera_angle_x": float(2.0 * np.arctan(calib["width"] / (2.0 * calib["fx"]))),
        "camera_angle_y": float(2.0 * np.arctan(calib["height"] / (2.0 * calib["fy"]))),
        "aabb_scale": 1,
        # These are fixed scan-workspace bounds, not bounds read from the
        # withheld source mesh.  Neuralangelo uses them to normalize cameras
        # and to bound mesh extraction.
        "aabb_range": [[-0.16, 0.16], [-0.08, 0.08], [-0.01, 0.10]],
        "sphere_center": [0.0, 0.0, 0.0],
        "sphere_radius": 0.5,
        "frames": frames,
        "provenance": {
            "source_mesh_access": False,
            "poses": "reported_FANUC_FK_plus_hand_eye",
            "segmentation": "calibrated_RGB_background_mask",
        },
    }
    (out / "transforms.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"prepared {len(frames)} masked known-pose frames at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
