"""Render evaluator-separated RGB-D orbit scans in Isaac Sim 6.

The source STL is used only in this capture process.  The observation folder
contains no mesh vertices, object transforms from USD, or semantic labels.
Reported camera/object poses are deliberately perturbed to emulate hand-eye
and absolute-kinematic error.  Exact poses go to ``evaluator_only``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--stl", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--azimuths", type=int, default=18)
parser.add_argument("--elevations", nargs="+", type=float, default=(18, 32, 46, 60, 74))
parser.add_argument("--radius-m", type=float, default=0.42)
parser.add_argument("--seed", type=int, default=20261024)
parser.add_argument("--dual-pass", action="store_true")
parser.add_argument("--photometric-keyframes", type=int, default=12)
args = parser.parse_args()

SOURCE = Path(args.stl).resolve()
ROOT = Path(args.out).resolve()
OBS = ROOT / "observations"
EVAL = ROOT / "evaluator_only"
OBS.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": True, "width": int(args.width), "height": int(args.height)}
)

import cv2
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera


def load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    binary = False
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        binary = 84 + 50 * count == len(data)
    if binary:
        dtype = np.dtype(
            [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
        )
        triangles = np.frombuffer(data, dtype=dtype, offset=84, count=count)["vertices"].astype(np.float64)
    else:
        vertices = []
        for line in data.decode("ascii", errors="ignore").splitlines():
            words = line.strip().split()
            if len(words) == 4 and words[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in words[1:]))
        triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    flat = triangles.reshape(-1, 3) / 1000.0
    bounds = np.asarray((flat.min(axis=0), flat.max(axis=0)))
    flat[:, :2] -= bounds.mean(axis=0)[:2]
    flat[:, 2] -= bounds[0, 2]
    faces = np.arange(len(flat), dtype=np.int32).reshape(-1, 3)
    return flat, faces


def matrix_to_list(matrix: Gf.Matrix4d) -> list[list[float]]:
    return [[float(matrix[r][c]) for c in range(4)] for r in range(4)]


def perturb_pose(matrix: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply bounded FANUC absolute/hand-eye error to a USD row transform."""
    translation = np.clip(rng.normal(0.0, 0.00028, 3), -0.0008, 0.0008)
    angles = np.clip(rng.normal(0.0, math.radians(0.07), 3), -math.radians(0.2), math.radians(0.2))
    cx, sx = math.cos(angles[0]), math.sin(angles[0])
    cy, sy = math.cos(angles[1]), math.sin(angles[1])
    cz, sz = math.cos(angles[2]), math.sin(angles[2])
    rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    out = matrix.copy()
    out[:3, :3] = out[:3, :3] @ (rx @ ry @ rz)
    out[3, :3] += translation
    return out, {
        "translation_mm": (translation * 1000.0).tolist(),
        "rotation_deg_xyz": np.degrees(angles).tolist(),
    }


def noisy_depth(depth: np.ndarray, rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    value = np.asarray(depth, dtype=np.float32).copy()
    valid = np.isfinite(value) & (value > 0.01) & (value < 4.0)
    sigma = 0.00035 + 0.0008 * np.square(np.nan_to_num(value, nan=0.0))
    value[valid] += rng.normal(0.0, sigma[valid])
    value[valid] = np.round(value[valid] / 0.0005) * 0.0005
    # A deterministic blue-object mask is a legal calibrated-background cue,
    # not a semantic label.  It is used only to model edge dropout here.
    part = (rgb[..., 2] > 1.12 * rgb[..., 1]) & (rgb[..., 2] > 1.4 * rgb[..., 0])
    edge = cv2.morphologyEx(part.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    random_dropout = rng.random(value.shape) < 0.006
    edge_dropout = edge & (rng.random(value.shape) < 0.28)
    value[random_dropout | edge_dropout | ~valid] = np.nan
    return value, {
        "quantization_mm": 0.5,
        "base_sigma_mm": 0.35,
        "range_squared_sigma_coefficient_mm": 0.8,
        "random_dropout_fraction": float(np.mean(random_dropout)),
        "edge_dropout_fraction": float(np.mean(edge_dropout)),
    }


def object_pose(pass_name: str, height: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if pass_name == "flipped":
        matrix[:3, :3] = np.diag((1.0, -1.0, -1.0))
        matrix[3, 2] = height
    return matrix


def set_mesh_pose(op, matrix: np.ndarray) -> None:
    value = Gf.Matrix4d()
    for r in range(4):
        for c in range(4):
            value[r, c] = float(matrix[r, c])
    op.Set(value)


def main() -> int:
    rng = np.random.default_rng(args.seed)
    vertices, faces = load_stl(SOURCE)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    target_z = float(extents[2] * 0.48)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    table.CreateDisplayColorAttr([Gf.Vec3f(0.40, 0.41, 0.43)])
    tx = UsdGeom.Xformable(table.GetPrim())
    tx.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.0125))
    tx.AddScaleOp().Set(Gf.Vec3f(1.2, 1.2, 0.025))

    mesh = UsdGeom.Mesh.Define(stage, "/World/ScanObject")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.055, 0.30, 0.88)])
    mesh_pose_op = UsdGeom.Xformable(mesh.GetPrim()).AddTransformOp()
    mesh_pose_op.Set(Gf.Matrix4d(1.0))

    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
    key_lights = []
    for index, position in enumerate(((0.35, 0.0, 0.35), (-0.35, 0.0, 0.35), (0.0, 0.35, 0.32), (0.0, -0.35, 0.32))):
        light = UsdLux.SphereLight.Define(stage, f"/World/PhotoLight{index}")
        light.CreateRadiusAttr(0.025)
        light.CreateIntensityAttr(0.0)
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*position))
        key_lights.append(light)

    camera_schema = UsdGeom.Camera.Define(stage, "/World/OrbitCamera")
    camera_schema.CreateFocalLengthAttr(18.0)
    camera_schema.CreateHorizontalApertureAttr(20.955)
    camera_schema.CreateVerticalApertureAttr(20.955 * args.height / args.width)
    camera_schema.CreateClippingRangeAttr(Gf.Vec2f(0.01, 3.0))
    camera_xform = UsdGeom.Xformable(camera_schema.GetPrim()).AddTransformOp()
    camera_xform.Set(Gf.Matrix4d(1.0))
    camera = Camera("/World/OrbitCamera", resolution=(args.width, args.height))

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    for _ in range(80):
        world.step(render=True)

    # World.reset()/sensor initialization can invalidate previously-held USD
    # schema/op wrappers even though the stage prims still exist. Reacquire
    # every handle that is changed during capture from the live stage.
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ScanObject"))
    mesh_xform = UsdGeom.Xformable(mesh.GetPrim())
    mesh_xform.ClearXformOpOrder()
    mesh_pose_op = mesh_xform.AddTransformOp()
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Dome"))
    key_lights = [UsdLux.SphereLight(stage.GetPrimAtPath(f"/World/PhotoLight{i}")) for i in range(4)]
    camera_prim = stage.GetPrimAtPath("/World/OrbitCamera")
    camera_live_xform = UsdGeom.Xformable(camera_prim)
    camera_live_xform.ClearXformOpOrder()
    camera_xform = camera_live_xform.AddTransformOp()

    passes = ["upright"] + (["flipped"] if args.dual_pass else [])
    observation_frames = []
    evaluator_frames = []
    preview_frames = []
    start = time.perf_counter()
    frame_index = 0
    photo_stride = max(1, int(math.ceil(args.azimuths * len(args.elevations) / max(1, args.photometric_keyframes))))

    for pass_name in passes:
        canonical_to_world = object_pose(pass_name, float(extents[2]))
        set_mesh_pose(mesh_pose_op, canonical_to_world)
        for _ in range(4):
            world.step(render=True)
        for elevation_deg in args.elevations:
            elevation = math.radians(float(elevation_deg))
            for azimuth_index in range(args.azimuths):
                azimuth = 2.0 * math.pi * azimuth_index / args.azimuths
                eye = np.asarray((
                    args.radius_m * math.cos(elevation) * math.cos(azimuth),
                    args.radius_m * math.cos(elevation) * math.sin(azimuth),
                    args.radius_m * math.sin(elevation) + target_z,
                ))
                view = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(*eye), Gf.Vec3d(0.0, 0.0, target_z), Gf.Vec3d(0.0, 0.0, 1.0)
                )
                camera_xform.Set(view.GetInverse())
                # The first changed viewpoint needs extra RTX history/shader
                # settling; subsequent viewpoints need fewer frames.
                settle_frames = 14 if frame_index == 0 else 5
                for _ in range(settle_frames):
                    world.step(render=True)
                rgba = camera.get_rgba()
                depth = camera.get_depth()
                if rgba is None or depth is None:
                    raise RuntimeError("orbit camera returned no RGB-D frame")
                rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
                clean = np.asarray(depth, dtype=np.float32)
                noisy, noise_meta = noisy_depth(clean, rgb, rng)
                actual = np.asarray(matrix_to_list(view.GetInverse()), dtype=np.float64)
                reported, pose_error = perturb_pose(actual, rng)
                world_to_canonical = np.linalg.inv(canonical_to_world)
                camera_to_canonical_actual = actual @ world_to_canonical
                camera_to_canonical_reported = reported @ world_to_canonical

                stem = f"{pass_name}_{frame_index:04d}"
                cv2.imwrite(str(OBS / f"{stem}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                np.savez_compressed(OBS / f"{stem}_depth_noisy.npz", depth=noisy)
                # Clean depth is evaluator-only and used to bound the sensor ceiling.
                np.savez_compressed(EVAL / f"{stem}_depth_clean.npz", depth=clean)
                preview_frames.append(rgb)

                frame_record = {
                    "id": stem,
                    "pass": pass_name,
                    "rgb": f"{stem}_rgb.png",
                    "depth": f"{stem}_depth_noisy.npz",
                    "camera_to_object": camera_to_canonical_reported.tolist(),
                    "eye_object_m": camera_to_canonical_reported[3, :3].tolist(),
                    "elevation_deg": float(elevation_deg),
                    "azimuth_deg": float(math.degrees(azimuth)),
                    "pose_prior": "FANUC_FK_plus_hand_eye",
                    "pose_prior_error_bound_mm": 0.8,
                    "pose_error_hidden": True,
                    "noise": noise_meta,
                }
                observation_frames.append(frame_record)
                evaluator_frames.append({
                    "id": stem,
                    "camera_to_object_actual": camera_to_canonical_actual.tolist(),
                    "reported_pose_error": pose_error,
                    "canonical_to_world": canonical_to_world.tolist(),
                })

                if frame_index % photo_stride == 0:
                    dome.GetIntensityAttr().Set(80.0)
                    for light_index, light in enumerate(key_lights):
                        for other_index, other in enumerate(key_lights):
                            other.GetIntensityAttr().Set(22000.0 if other_index == light_index else 0.0)
                        for _ in range(2):
                            world.step(render=True)
                        lit = camera.get_rgba()[:, :, :3].astype(np.uint8)
                        name = f"{stem}_light{light_index}.png"
                        cv2.imwrite(str(OBS / name), cv2.cvtColor(lit, cv2.COLOR_RGB2BGR))
                        frame_record.setdefault("photometric_rgb", []).append(name)
                    for light in key_lights:
                        light.GetIntensityAttr().Set(0.0)
                    dome.GetIntensityAttr().Set(650.0)
                    for _ in range(2):
                        world.step(render=True)
                frame_index += 1

    camera_meta = {
        "resolution": [args.width, args.height],
        "focal_length_mm": 18.0,
        "horizontal_aperture_mm": 20.955,
        "vertical_aperture_mm": 20.955 * args.height / args.width,
        "depth_type": "distance_to_image_plane_m",
        "camera_axes": "USD_row_vector_x_right_y_up_minus_z_forward",
    }
    observation_manifest = {
        "schema": "fanuc_isaac_reverse_engineering_v1",
        "source_name": SOURCE.stem,
        "source_mesh_withheld": True,
        "camera": camera_meta,
        "frames": observation_frames,
        "allowed_inputs": ["rgb", "noisy_depth", "reported_FK_hand_eye_pose", "multi_light_rgb"],
    }
    evaluator_manifest = {
        "source_stl": str(SOURCE),
        "canonical_bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "canonical_extents_m": extents.tolist(),
        "frames": evaluator_frames,
    }
    (OBS / "manifest.json").write_text(json.dumps(observation_manifest, indent=2), encoding="utf-8")
    (EVAL / "manifest.json").write_text(json.dumps(evaluator_manifest, indent=2), encoding="utf-8")
    np.savez_compressed(EVAL / "source_mesh.npz", vertices=vertices, faces=faces)

    # A contact sheet is cheap, deterministic visual evidence even when ffmpeg
    # is unavailable in the container.
    sample_count = min(30, len(preview_frames))
    indices = np.linspace(0, len(preview_frames) - 1, sample_count).astype(int)
    thumbs = [cv2.resize(preview_frames[i], (240, 180)) for i in indices]
    rows = [np.concatenate(thumbs[i:i + 6], axis=1) for i in range(0, len(thumbs), 6)]
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(ROOT / "orbit_contact_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    summary = {
        "task_complete": True,
        "frames": len(observation_frames),
        "passes": passes,
        "capture_wall_s": time.perf_counter() - start,
        "extents_mm": (extents * 1000.0).tolist(),
        "observations": str(OBS),
        "evaluator_only": str(EVAL),
    }
    (ROOT / "capture_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
