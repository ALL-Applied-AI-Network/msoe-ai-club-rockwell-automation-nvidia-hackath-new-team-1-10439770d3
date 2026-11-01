"""Capture randomized truck-task observations in Isaac Sim.

This is the only benchmark stage allowed to author simulator state.  It writes
sensor observations and evaluator labels into sibling directories.  Policies
are later launched with the observations directory only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--scenes", type=int, default=48)
parser.add_argument("--seed", type=int, default=20261024)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--active-wrist-scan", action="store_true")
parser.add_argument("--active-oblique-scan", action="store_true")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(args.out).resolve()
OBS_ROOT = OUT / "observations"
EVAL_ROOT = OUT / "evaluator_only"
OBS_ROOT.mkdir(parents=True, exist_ok=True)
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": bool(args.headless),
        "width": int(args.width),
        "height": int(args.height),
    }
)

import cv2
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.camera import Camera

import cell
import ik
import scene_config as C


COLOURS = {
    "red": np.asarray((0.82, 0.035, 0.035), dtype=np.float64),
    "orange": np.asarray((0.95, 0.27, 0.025), dtype=np.float64),
    "blue": np.asarray((0.025, 0.18, 0.90), dtype=np.float64),
}
ASSET_DIR = ROOT / "assets" / "measured"
TABLE_Z = C.TABLE_TOP_Z + C.SURFACE_DZ
TRUCK_DECK_CENTRE_LOCAL = np.asarray((0.08129196, 0.15421628), dtype=np.float64)
TRUCK_DECK_LENGTH = 0.18085018
TRUCK_DECK_WIDTH = 0.04709631
TRUCK_DECK_HEIGHT = 0.0335
TRUCK_DECK_LOCAL_YAW_DEG = -41.29227386
VARIATIONS = (
    "nominal",
    "photometric",
    "depth_noise",
    "calibration_drift",
    "clutter_occlusion",
    "combined",
)


def rotate_xy(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray((c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]))


def load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    binary = False
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        binary = 84 + 50 * count == len(data)
    if binary:
        dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        )
        triangles = np.frombuffer(data, dtype=dtype, offset=84, count=count)["vertices"].astype(np.float64)
    else:
        vertices = []
        for line in data.decode("ascii", errors="ignore").splitlines():
            words = line.strip().split()
            if len(words) == 4 and words[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in words[1:]))
        triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    flat = triangles.reshape(-1, 3)
    indices = np.arange(len(flat), dtype=np.int32).reshape(-1, 3)
    return flat, indices


def cube_transform(vertices: np.ndarray) -> np.ndarray:
    out = vertices.copy()
    out[:, 0] = (vertices[:, 0] - 6.0) / 1000.0
    out[:, 1] = (vertices[:, 1] - 6.0) / 1000.0
    out[:, 2] = (vertices[:, 2] + 12.0) / 1000.0
    return out


def bin_transform(vertices: np.ndarray) -> np.ndarray:
    out = np.empty_like(vertices, dtype=np.float64)
    out[:, 0] = (vertices[:, 0] + 21.0) / 1000.0
    out[:, 1] = -(vertices[:, 2] + 21.0) / 1000.0
    out[:, 2] = vertices[:, 1] / 1000.0
    return out


def truck_transform(vertices: np.ndarray) -> np.ndarray:
    return vertices / 1000.0


def add_mesh(stage, path: str, stl: Path, transform, colour) -> UsdGeom.Mesh:
    vertices, faces = load_stl(stl)
    vertices = transform(vertices)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
    return mesh


def create_root(stage, path: str):
    root = UsdGeom.Xform.Define(stage, path)
    xform = UsdGeom.Xformable(root)
    translate = xform.AddTranslateOp()
    rotate = xform.AddRotateZOp()
    scale = xform.AddScaleOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    rotate.Set(0.0)
    scale.Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return root, translate, rotate, scale


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return x @ y @ z


def perturb_calibration(calibration: dict, rng: np.random.Generator, severe: bool) -> tuple[dict, dict]:
    reported = json.loads(json.dumps(calibration))
    translation_sigma = 0.0025 if severe else 0.00045
    angle_sigma_deg = 0.38 if severe else 0.07
    translation = rng.normal(0.0, translation_sigma, size=3)
    angles_deg = rng.normal(0.0, angle_sigma_deg, size=3)
    matrix = np.asarray(reported["camera_to_world"], dtype=np.float64)
    matrix[:3, :3] = matrix[:3, :3] @ rotation_matrix_xyz(*np.deg2rad(angles_deg))
    matrix[3, :3] += translation
    reported["camera_to_world"] = matrix.tolist()
    error = {
        "translation_mm": (translation * 1000.0).tolist(),
        "rotation_deg_xyz": angles_deg.tolist(),
    }
    return reported, error


def camera_calibration(stage, path: str, resolution: tuple[int, int]) -> dict:
    prim = stage.GetPrimAtPath(path)
    camera = UsdGeom.Camera(prim)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    matrix = [[float(transform[row][column]) for column in range(4)] for row in range(4)]
    return {
        "resolution": [int(resolution[0]), int(resolution[1])],
        "focal_length_mm": float(camera.GetFocalLengthAttr().Get()),
        "horizontal_aperture_mm": float(camera.GetHorizontalApertureAttr().Get()),
        "vertical_aperture_mm": float(camera.GetVerticalApertureAttr().Get()),
        "camera_to_world": matrix,
        "table_top_z": float(TABLE_Z),
    }


def transform_rgb(rgb: np.ndarray, rng: np.random.Generator, variation: str) -> tuple[np.ndarray, dict]:
    severe = variation in ("photometric", "combined")
    if severe:
        exposure = float(rng.uniform(0.62, 1.42))
        gamma = float(rng.uniform(0.72, 1.38))
        white_balance = rng.uniform(0.78, 1.22, size=3)
        blur = int(rng.choice((0, 0, 3, 5)))
        noise_sigma = float(rng.uniform(1.0, 7.0))
    else:
        exposure = float(rng.uniform(0.92, 1.08))
        gamma = float(rng.uniform(0.94, 1.06))
        white_balance = rng.uniform(0.96, 1.04, size=3)
        blur = 0
        noise_sigma = float(rng.uniform(0.0, 1.2))
    image = rgb.astype(np.float32) / 255.0
    image = np.clip(image * exposure * white_balance.reshape(1, 1, 3), 0.0, 1.0)
    image = np.power(image, gamma)
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    if noise_sigma > 0:
        image += rng.normal(0.0, noise_sigma / 255.0, size=image.shape).astype(np.float32)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image, {
        "exposure": exposure,
        "gamma": gamma,
        "white_balance_rgb": white_balance.tolist(),
        "blur_kernel": blur,
        "rgb_noise_sigma": noise_sigma,
    }


def transform_depth(depth: np.ndarray, rng: np.random.Generator, variation: str) -> tuple[np.ndarray, dict]:
    severe = variation in ("depth_noise", "combined")
    sigma = float(rng.uniform(0.0010, 0.0030) if severe else rng.uniform(0.00005, 0.00035))
    dropout = float(rng.uniform(0.01, 0.07) if severe else rng.uniform(0.0, 0.003))
    quantization = float(rng.choice((0.0005, 0.0010, 0.0020)) if severe else 0.00025)
    result = np.asarray(depth, dtype=np.float32).copy()
    finite = np.isfinite(result) & (result > 0)
    result[finite] += rng.normal(0.0, sigma, size=int(finite.sum())).astype(np.float32)
    result[finite] = np.round(result[finite] / quantization) * quantization
    dropped = rng.random(result.shape) < dropout
    result[dropped] = np.nan
    return result.astype(np.float16), {
        "depth_noise_sigma_mm": sigma * 1000.0,
        "depth_dropout_fraction": dropout,
        "depth_quantization_mm": quantization * 1000.0,
    }


def sample_layout(rng: np.random.Generator, scene_index: int) -> dict:
    variation = VARIATIONS[scene_index % len(VARIATIONS)]
    truck_origin = np.asarray((0.351, -0.085), dtype=np.float64) + rng.uniform((-0.018, -0.018), (0.018, 0.018))
    truck_root_yaw = float(rng.uniform(-14.0, 14.0))
    deck_centre = truck_origin + rotate_xy(TRUCK_DECK_CENTRE_LOCAL, math.radians(truck_root_yaw))
    deck_yaw = TRUCK_DECK_LOCAL_YAW_DEG + truck_root_yaw
    long_axis = np.asarray((math.cos(math.radians(deck_yaw)), math.sin(math.radians(deck_yaw))))

    bin_slots = [np.asarray((0.265, -0.135)), np.asarray((0.385, -0.137)), np.asarray((0.505, -0.135))]
    rng.shuffle(bin_slots)
    bins = {}
    for colour, base in zip(COLOURS, bin_slots):
        bins[colour] = {
            "xy": (base + rng.uniform((-0.014, -0.010), (0.014, 0.010))).tolist(),
            "yaw_deg": float(rng.uniform(-43.0, 43.0)),
            "scale_xy": float(rng.uniform(0.988, 1.014)),
        }

    grid = [np.asarray((x, y)) for y in (-0.300, -0.253, -0.206) for x in (0.235, 0.300, 0.365, 0.430)]
    rng.shuffle(grid)
    cubes = []
    index = 0
    for colour in COLOURS:
        for _ in range(4):
            base = grid[index]
            cubes.append(
                {
                    "name": f"Cube{index:02d}",
                    "colour": colour,
                    "xy": (base + rng.uniform((-0.011, -0.010), (0.011, 0.010))).tolist(),
                    "yaw_deg": float(rng.uniform(-45.0, 45.0)),
                    "scale": float(rng.uniform(0.975, 1.025)),
                }
            )
            index += 1
    rng.shuffle(cubes)
    zones = {
        colour: (deck_centre + offset * long_axis).tolist()
        for colour, offset in zip(COLOURS, (-0.057, 0.0, 0.057))
    }
    return {
        "variation": variation,
        "truck": {
            "origin_xy": truck_origin.tolist(),
            "root_yaw_deg": truck_root_yaw,
            "deck_centre_xy": deck_centre.tolist(),
            "deck_yaw_deg": deck_yaw,
            "deck_z": TABLE_Z + TRUCK_DECK_HEIGHT,
            "zones": zones,
        },
        "bins": bins,
        "cubes": cubes,
    }


def main() -> int:
    started = time.time()
    rng = np.random.default_rng(args.seed)
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)

    truck_root, truck_translate, truck_rotate, truck_scale = create_root(stage, "/World/MeasuredTruckBenchmark")
    truck_mesh = add_mesh(stage, "/World/MeasuredTruckBenchmark/Visual", ASSET_DIR / "truck.stl", truck_transform, (0.14, 0.16, 0.19))

    bin_nodes = {}
    for colour in COLOURS:
        root, translate, rotate, scale = create_root(stage, f"/World/BenchmarkBin_{colour}")
        mesh = add_mesh(stage, f"/World/BenchmarkBin_{colour}/Visual", ASSET_DIR / "bin_50x50x54.stl", bin_transform, COLOURS[colour])
        bin_nodes[colour] = (translate, rotate, scale, mesh)

    cube_nodes = {}
    for index in range(12):
        root, translate, rotate, scale = create_root(stage, f"/World/BenchmarkCube{index:02d}")
        mesh = add_mesh(stage, f"/World/BenchmarkCube{index:02d}/Visual", ASSET_DIR / "cube_12mm.stl", cube_transform, (0.5, 0.5, 0.5))
        cube_nodes[f"Cube{index:02d}"] = (translate, rotate, scale, mesh)

    distractors = []
    for index in range(5):
        root, translate, rotate, scale = create_root(stage, f"/World/BenchmarkClutter{index}")
        mesh = cell.box(
            stage,
            f"/World/BenchmarkClutter{index}/Visual",
            (0.026, 0.010, 0.026),
            (0.0, 0.0, 0.013),
            (0.10 + 0.02 * index, 0.105, 0.11),
            collision=False,
        )
        distractors.append((translate, rotate, scale, mesh))

    resolution = (int(args.width), int(args.height))
    camera_resolutions = {
        "table": resolution,
        "side": resolution,
        # Preserve the supplied physical wrist-camera aspect ratio/FOV.  It is
        # a third independent observation, not a copy of an overhead view.
        "wrist": (1280, 720),
    }
    cameras = {
        "table": Camera(prim_path=cell.TABLE_CAM_PATH, resolution=camera_resolutions["table"]),
        "side": Camera(prim_path=cell.SIDE_CAM_PATH, resolution=camera_resolutions["side"]),
        "wrist": Camera(prim_path=cell.WRIST_CAM_PATH, resolution=camera_resolutions["wrist"]),
    }
    world.reset()
    for camera in cameras.values():
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()

    arm = SingleArticulation(C.ROBOT_PRIM, name="benchmark_er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    joints = arm.get_joint_positions().copy()
    for joint, degrees in C.CAPTURE_POSE_DEG.items():
        joints[names.index(joint)] = math.radians(degrees)
    for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
        joints[names.index(joint)] = value
    arm.set_joint_positions(joints)

    for _ in range(100):
        world.step(render=True)

    camera_paths = {
        "table": cell.TABLE_CAM_PATH,
        "side": cell.SIDE_CAM_PATH,
        "wrist": cell.WRIST_CAM_PATH,
    }
    active_wrist_joint_sets = []
    if args.active_wrist_scan or args.active_oblique_scan:
        arm_indices = [names.index(f"J{i}") for i in range(1, 7)]
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool_offset = tuple(np.asarray(C.TOOL_AXIS, float) * cell.tip_offset())
        seed_joints = np.asarray([joints[index] for index in arm_indices], float)
        scan_specs = [
            (
                "wrist_scan_near",
                np.asarray((0.285, -0.255, 0.285)),
                np.asarray((0.0, 0.0, -1.0)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
            # Verified from the preceding near-scan joint state: 0.043 mm
            # Cartesian residual with the shipped ER-4iA kinematics/limits.
            (
                "wrist_scan_far",
                np.asarray((0.410, -0.160, 0.360)),
                np.asarray((0.0, 0.0, -1.0)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
        ]
        if args.active_oblique_scan:
            # High west-side 24-degree view, aimed through the full three-bin
            # task stays visible.  It was verified from the far-scan joint
            # state at 0.020 mm Cartesian residual with the
            # shipped ER-4iA kinematics/limits.  It exposes vertical object
            # edges that remain hidden in repeated overhead views.
            scan_specs.append(
                (
                    "wrist_scan_oblique_west",
                    np.asarray((0.250, -0.220, 0.380)),
                    np.asarray((0.4067, 0.0, -0.9135)),
                    np.asarray((0.0, 1.0, 0.0)),
                )
            )
        for scan_name, target, tool_direction, spread in scan_specs:
            solved, converged = ik.solve(
                seed_joints,
                target,
                tool_direction,
                limits,
                tool_offset=tool_offset,
                spread_dir=spread,
                iters=700,
                damping=0.02,
            )
            if not converged:
                raise RuntimeError(f"IK failed for active wrist scan {scan_name}")
            full = joints.copy()
            full[arm_indices] = solved
            active_wrist_joint_sets.append((scan_name, full))
            seed_joints = solved
    scene_records = []
    for scene_index in range(args.scenes):
        scene_id = f"scene_{scene_index:04d}"
        layout = sample_layout(rng, scene_index)
        variation = layout["variation"]
        severe_colour = variation in ("photometric", "combined")

        truck = layout["truck"]
        truck_translate.Set(Gf.Vec3d(float(truck["origin_xy"][0]), float(truck["origin_xy"][1]), float(TABLE_Z)))
        truck_rotate.Set(float(truck["root_yaw_deg"]))
        truck_scale.Set(Gf.Vec3f(1.0, 1.0, 1.0))

        for colour, item in layout["bins"].items():
            translate, rotate, scale, mesh = bin_nodes[colour]
            translate.Set(Gf.Vec3d(float(item["xy"][0]), float(item["xy"][1]), float(TABLE_Z)))
            rotate.Set(float(item["yaw_deg"]))
            scale.Set(Gf.Vec3f(float(item["scale_xy"]), float(item["scale_xy"]), 1.0))
            colour_value = np.clip(COLOURS[colour] * rng.uniform(0.78, 1.18, size=3), 0.01, 1.0) if severe_colour else COLOURS[colour]
            mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*colour_value)])

        for item in layout["cubes"]:
            translate, rotate, scale, mesh = cube_nodes[item["name"]]
            translate.Set(Gf.Vec3d(float(item["xy"][0]), float(item["xy"][1]), float(TABLE_Z)))
            rotate.Set(float(item["yaw_deg"]))
            scale.Set(Gf.Vec3f(float(item["scale"]), float(item["scale"]), float(item["scale"])))
            colour_value = np.clip(COLOURS[item["colour"]] * rng.uniform(0.76, 1.20, size=3), 0.01, 1.0) if severe_colour else COLOURS[item["colour"]]
            mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*colour_value)])

        clutter_active = variation in ("clutter_occlusion", "combined")
        clutter_truth = []
        for index, (translate, rotate, scale, _) in enumerate(distractors):
            if clutter_active:
                anchor = layout["cubes"][index % len(layout["cubes"])]
                xy = np.asarray(anchor["xy"]) + rng.uniform((-0.020, -0.010), (0.020, 0.010))
                z = TABLE_Z
                translate.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), float(z)))
                rotate.Set(float(rng.uniform(-90.0, 90.0)))
                scale.Set(Gf.Vec3f(float(rng.uniform(0.8, 1.4)), 1.0, float(rng.uniform(0.7, 1.4))))
                clutter_truth.append({"xy": xy.tolist(), "anchor": anchor["name"]})
            else:
                translate.Set(Gf.Vec3d(-3.0 - index, -3.0, -3.0))

        for _ in range(4):
            world.step(render=True)

        scene_obs = OBS_ROOT / scene_id
        scene_obs.mkdir(parents=True, exist_ok=True)
        camera_manifest = {}
        calibration_errors = {}
        sensor_effects = {}
        actual_calibrations = {}

        def capture_named(name, camera, prim_path, camera_resolution):
            rgba = camera.get_rgba()
            depth = camera.get_depth()
            if rgba is None or depth is None:
                raise RuntimeError(f"{name} camera did not produce RGB/depth for {scene_id}")
            rgb, rgb_effects = transform_rgb(rgba[:, :, :3].astype(np.uint8), rng, variation)
            noisy_depth, depth_effects = transform_depth(depth, rng, variation)
            actual_calibration = camera_calibration(stage, prim_path, camera_resolution)
            actual_calibrations[name] = actual_calibration
            calibration, error = perturb_calibration(
                actual_calibration,
                rng,
                variation in ("calibration_drift", "combined"),
            )
            rgb_name = f"{name}_rgb.jpg"
            depth_name = f"{name}_depth.npz"
            cv2.imwrite(str(scene_obs / rgb_name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
            np.savez_compressed(scene_obs / depth_name, depth=noisy_depth)
            camera_manifest[name] = {
                "rgb": rgb_name,
                "depth": depth_name,
                "calibration": calibration,
            }
            calibration_errors[name] = error
            sensor_effects[name] = {**rgb_effects, **depth_effects}

        for name, camera in cameras.items():
            capture_named(name, camera, camera_paths[name], camera_resolutions[name])

        for scan_name, scan_joints in active_wrist_joint_sets:
            arm.set_joint_positions(scan_joints)
            for _ in range(12):
                world.step(render=True)
            capture_named(
                scan_name,
                cameras["wrist"],
                camera_paths["wrist"],
                camera_resolutions["wrist"],
            )
        if active_wrist_joint_sets:
            arm.set_joint_positions(joints)
            for _ in range(12):
                world.step(render=True)

        robot_state = {
            "joint_names": names,
            "joint_positions_rad": [float(value) for value in joints],
            "gripper_command_per_jaw_m": float(C.JAW_OPEN),
            "timestamp_s": float(scene_index / 10.0),
        }
        observation_manifest = {
            "scene_id": scene_id,
            "cameras": camera_manifest,
            "robot_state": robot_state,
            "variation": {
                "name": variation,
                "sensor_effects": sensor_effects,
                "contains_unlabelled_clutter": bool(clutter_active),
            },
            "allowed_inputs": ["rgb", "depth", "reported_camera_calibration", "joint_state", "gripper_state", "timestamp"],
        }
        (scene_obs / "observation.json").write_text(json.dumps(observation_manifest, indent=2), encoding="utf-8")

        evaluator = {
            "scene_id": scene_id,
            "variation": variation,
            "objects": [
                {
                    "name": item["name"],
                    "category": "cube",
                    "colour": item["colour"],
                    "xy": item["xy"],
                    "yaw_deg": item["yaw_deg"],
                    "size_m": 0.012 * item["scale"],
                }
                for item in layout["cubes"]
            ]
            + [
                {
                    "name": f"Bin_{colour}",
                    "category": "bin",
                    "colour": colour,
                    "xy": item["xy"],
                    "yaw_deg": item["yaw_deg"],
                    "size_m": 0.050 * item["scale_xy"],
                }
                for colour, item in layout["bins"].items()
            ]
            + [
                {
                    "name": "Truck",
                    "category": "truck",
                    "colour": "truck",
                    "xy": truck["deck_centre_xy"],
                    "yaw_deg": truck["deck_yaw_deg"],
                    "deck_centre_xy": truck["deck_centre_xy"],
                    "deck_yaw_deg": truck["deck_yaw_deg"],
                    "deck_z": truck["deck_z"],
                    "deck_extents_m": [TRUCK_DECK_LENGTH, TRUCK_DECK_WIDTH],
                    "zones": truck["zones"],
                }
            ],
            "clutter": clutter_truth,
            "actual_camera_calibration": actual_calibrations,
            "reported_calibration_errors": calibration_errors,
        }
        (EVAL_ROOT / f"{scene_id}.json").write_text(json.dumps(evaluator, indent=2), encoding="utf-8")
        scene_records.append({"scene_id": scene_id, "variation": variation})
        print(f"[capture] {scene_id} {variation}", flush=True)

    manifest = {
        "task": "truck: sort 4 red + 4 orange + 4 blue cubes; load 3 matching bins longitudinally",
        "seed": int(args.seed),
        "scene_count": int(args.scenes),
        "resolution": {name: list(value) for name, value in camera_resolutions.items()},
        "variations": list(VARIATIONS),
        "active_wrist_scan": bool(args.active_wrist_scan),
        "active_oblique_scan": bool(args.active_oblique_scan),
        "scenes": scene_records,
        "policy_boundary": {
            "observations": str(OBS_ROOT),
            "evaluator_labels": str(EVAL_ROOT),
            "simulator_truth_allowed_for_policy": False,
        },
        "elapsed_s": time.time() - started,
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
