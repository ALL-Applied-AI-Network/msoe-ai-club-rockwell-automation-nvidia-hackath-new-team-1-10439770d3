"""Prepare a masked COLMAP text model from reported orbit poses.

This uses only the observation contract. A downsampled noisy-depth point cloud
initializes 2DGS, while COLMAP MVS receives the same fixed camera model/poses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_classical import (  # noqa: E402
    fused_points, intrinsics, load_manifest, read_rgb_depth,
    select_frames, usd_row_to_opencv_object,
)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--views", type=int, default=54)
parser.add_argument("--point-voxel-mm", type=float, default=1.2)
args = parser.parse_args()

root = Path(args.dataset).resolve()
obs = root / "observations"
out = Path(args.out).resolve()
images_dir = out / "images"
sparse = out / "sparse" / "0"
images_dir.mkdir(parents=True, exist_ok=True)
sparse.mkdir(parents=True, exist_ok=True)
manifest = load_manifest(root)
intrinsic, calib = intrinsics(manifest)
upright = select_frames(manifest["frames"], 0, include_flipped=False)
indices = np.linspace(0, len(upright) - 1, min(args.views, len(upright))).round().astype(int)
frames = [upright[int(i)] for i in indices]

(sparse / "cameras.txt").write_text(
    f"# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    f"1 PINHOLE {calib['width']} {calib['height']} {calib['fx']:.12f} {calib['fy']:.12f} "
    f"{calib['cx']:.12f} {calib['cy']:.12f}\n",
    encoding="utf-8",
)
image_lines = ["# Image list", "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
for image_id, frame in enumerate(frames, 1):
    rgb, _, mask = read_rgb_depth(obs, frame)
    masked = rgb.copy()
    masked[~mask] = 255
    name = f"{image_id:04d}.png"
    cv2.imwrite(str(images_dir / name), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    object_from_camera = usd_row_to_opencv_object(np.asarray(frame["camera_to_object"]))
    camera_from_object = np.linalg.inv(object_from_camera)
    quat_xyzw = Rotation.from_matrix(camera_from_object[:3, :3]).as_quat()
    qx, qy, qz, qw = quat_xyzw
    tx, ty, tz = camera_from_object[:3, 3]
    image_lines.append(
        f"{image_id} {qw:.14g} {qx:.14g} {qy:.14g} {qz:.14g} "
        f"{tx:.14g} {ty:.14g} {tz:.14g} 1 {name}"
    )
    image_lines.append("")
(sparse / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")

cloud = fused_points(obs, frames, intrinsic, args.point_voxel_mm / 1000.0)
xyz = np.asarray(cloud.points)
colours = np.clip(np.asarray(cloud.colors) * 255.0, 0, 255).astype(np.uint8)
point_lines = ["# 3D point list", "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]"]
for point_id, (point, colour) in enumerate(zip(xyz, colours), 1):
    point_lines.append(
        f"{point_id} {point[0]:.12g} {point[1]:.12g} {point[2]:.12g} "
        f"{int(colour[0])} {int(colour[1])} {int(colour[2])} 0.001"
    )
(sparse / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
(out / "input_summary.txt").write_text(
    f"views={len(frames)}\npoints={len(xyz)}\nsource_mesh_access=false\n"
    "poses=reported_FANUC_FK_plus_hand_eye\npoint_initialization=noisy_RGBD\n",
    encoding="utf-8",
)
print(f"prepared {len(frames)} known-pose views and {len(xyz)} RGB-D seed points at {out}")
