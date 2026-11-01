"""Export calibrated FANUC/FK wrist-camera pose priors for the orbit endpoints.

These priors form a separate telemetry-assisted benchmark track.  They are not
ground truth: physical testing measured roughly 10--14 mm local residuals, so
every reconstruction using them must refine poses by image bundle adjustment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_geometry(path: Path):
    spec = importlib.util.spec_from_file_location("physical_cell_geometry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", default="orbit_endpoints_18")
    args = parser.parse_args()

    telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
    camera_data = json.loads(args.cameras.read_text(encoding="utf-8"))["wrist"]
    geometry = load_geometry(args.geometry)
    records = [record for record in telemetry["records"] if str(record["set"]) == args.set]
    image_paths = sorted(args.images.glob("*.png"))
    if len(records) != len(image_paths):
        raise RuntimeError(f"Telemetry/image count mismatch: {len(records)} versus {len(image_paths)}")

    source_width, source_height = camera_data["size"]
    width, height = 640, 360
    scale_x, scale_y = width / source_width, height / source_height
    if abs(scale_x - scale_y) > 1e-9:
        raise RuntimeError("Expected isotropic wrist-image resizing")
    fx = float(camera_data["fx"]) * scale_x
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    k1 = float(camera_data["k1"])
    t_flange_cam = np.asarray(camera_data["T_flange_cam"], dtype=np.float64)
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])

    args.out.mkdir(parents=True, exist_ok=True)
    colmap = args.out / "colmap_prior"
    colmap.mkdir(parents=True, exist_ok=True)
    frames = []
    camera_priors = []
    image_lines = ["# FANUC FK/hand-eye pose priors; refine with bundle adjustment", "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "# POINTS2D[] is intentionally empty"]
    for image_id, (record, image_path) in enumerate(zip(records, image_paths), start=1):
        joints = np.asarray(record["joint_deg_and_gripper"][:6], dtype=np.float64)
        t_base_flange = np.asarray(geometry.flange_pose(joints), dtype=np.float64)
        c2w_cv = t_base_flange @ t_flange_cam
        c2w_gl = c2w_cv @ opencv_to_opengl
        w2c = np.linalg.inv(c2w_cv)
        qx, qy, qz, qw = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        tx, ty, tz = w2c[:3, 3]
        image_lines.extend([
            f"{image_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {tx:.12g} {ty:.12g} {tz:.12g} 1 {image_path.name}",
            "",
        ])
        frames.append({"file_path": f"images/{image_path.name}", "transform_matrix": c2w_gl.tolist()})
        camera_priors.append(
            {
                "image": image_path.name,
                "clean_frame": record["clean_frame"],
                "clean_time_s": record["clean_time_s"],
                "joints_deg": joints.tolist(),
                "T_base_camera_opencv": c2w_cv.tolist(),
                "T_base_camera_opengl": c2w_gl.tolist(),
            }
        )

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "k1": k1,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "aabb_scale": 1,
        "pose_source": "FANUC FK @ calibrated T_flange_cam; prior only",
        "frames": frames,
    }
    (args.out / "transforms_fanuc_prior.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    (args.out / "camera_priors.json").write_text(
        json.dumps(
            {
                "intrinsics_640x360": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1},
                "transform_definition": "T_base_camera maps OpenCV camera coordinates (+x right,+y down,+z forward) to robot-base metres",
                "known_residual_warning": "Use as a bundle-adjustment prior; wrist local residuals measured about 10--14 mm.",
                "frames": camera_priors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (colmap / "cameras.txt").write_text(
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        f"1 SIMPLE_RADIAL {width} {height} {fx:.12g} {cx:.12g} {cy:.12g} {k1:.12g}\n",
        encoding="utf-8",
    )
    (colmap / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (colmap / "points3D.txt").write_text("# Empty prior model\n", encoding="utf-8")
    print(f"Exported {len(frames)} pose priors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
