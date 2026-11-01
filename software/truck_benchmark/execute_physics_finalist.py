"""Observation-only, open-loop PhysX gate for truck-task finalists.

The controller is intentionally blind to USD/object transforms.  Simulator
state is used while authoring the randomized scene and by the evaluator only
after every robot command has finished.  Before motion the selected policy
receives two RGB-D images plus reported calibration and writes an immutable
plan.  This file is an integration harness, not a physical-robot driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", required=True)
parser.add_argument("--policy", default="hybrid_depth_anchor_hue_model_table_inventory")
parser.add_argument(
    "--controller",
    choices=(
        "direct_nominal",
        "geometry_lever",
        "calibrated_guarded",
        "vision_replan",
        "vision_recovery",
        "vision_recovery_v2",
        "vision_recovery_v3",
    ),
    default="calibrated_guarded",
)
parser.add_argument(
    "--variation",
    choices=("nominal", "photometric", "depth_noise", "calibration_drift", "combined"),
    default="nominal",
)
parser.add_argument("--seed", type=int, default=20261024)
parser.add_argument("--render-video", action="store_true")
parser.add_argument("--fps", type=int, default=12)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets" / "measured"
OUT = Path(args.out_dir).resolve()
FRAMES = OUT / "frames"
OUT.mkdir(parents=True, exist_ok=True)
if args.render_video:
    FRAMES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

print("[physics-finalist] starting Isaac Sim", flush=True)
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1920, "height": 1080})


def rotate_xy(vector, angle_rad):
    vector = np.asarray(vector, dtype=float)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray((c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]))


def ease(value):
    return value * value * (3.0 - 2.0 * value)


def turn(start, finish, t):
    start, finish = np.asarray(start, float), np.asarray(finish, float)
    angle = math.atan2(start[0] * finish[1] - start[1] * finish[0], float(np.dot(start[:2], finish[:2])))
    return np.asarray((
        start[0] * math.cos(angle * t) - start[1] * math.sin(angle * t),
        start[0] * math.sin(angle * t) + start[1] * math.cos(angle * t),
        0.0,
    ))


def quat_yaw_tilt(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return yaw, math.degrees(math.acos(float(np.clip(up_z, -1.0, 1.0))))


def load_stl(path):
    data = Path(path).read_bytes()
    binary = len(data) >= 84 and 84 + 50 * struct.unpack_from("<I", data, 80)[0] == len(data)
    if binary:
        count = struct.unpack_from("<I", data, 80)[0]
        record = np.dtype([
            ("normal", "<f4", (3,)), ("v0", "<f4", (3,)), ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)), ("attribute", "<u2"),
        ])
        triangles = np.frombuffer(data, dtype=record, offset=84, count=count)
        points = np.stack((triangles["v0"], triangles["v1"], triangles["v2"]), axis=1)
    else:
        values = []
        for line in data.decode("ascii", errors="ignore").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                values.append([float(value) for value in fields[1:]])
        points = np.asarray(values, np.float32).reshape(-1, 3, 3)
    flat = points.reshape(-1, 3)
    vertices, inverse = np.unique(np.round(flat, 5), axis=0, return_inverse=True)
    return vertices.astype(float), inverse.reshape(-1, 3).astype(np.int32)


report = {
    "task_complete": False,
    "policy": args.policy,
    "controller": args.controller,
    "variation": args.variation,
    "seed": args.seed,
    "boundary": {
        "policy_inputs": ["table_rgbd", "side_rgbd", "wrist_rgbd", "reported_calibration", "joint_state", "timestamp"],
        "simulator_pose_available_to_controller": False,
        "plan_frozen_before_motion": args.controller not in ("vision_replan", "vision_recovery", "vision_recovery_v2", "vision_recovery_v3"),
        "camera_only_replanning": args.controller in ("vision_replan", "vision_recovery", "vision_recovery_v2", "vision_recovery_v3"),
        "residual_cube_recovery": args.controller in ("vision_recovery", "vision_recovery_v2", "vision_recovery_v3"),
        "recovery_pass_limit": 2 if args.controller in ("vision_recovery_v2", "vision_recovery_v3") else 1,
        "loaded_bin_transport_guard": args.controller == "vision_recovery_v3",
        "camera_refresh_render_steps": 3,
        "evaluator_reads_state_after_motion_only": True,
    },
    "actions": [],
    "evaluation": {},
    "error": None,
}
exit_code = 0

try:
    import cv2
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera

    import cell
    import fk
    import ik
    import scene_config as C
    from truck_benchmark.camera_geometry import CameraModel
    from truck_benchmark.contract import assert_policy_source_clean
    from truck_benchmark.final_fusion import final_policy_registry
    from truck_benchmark.policies import CameraObservation, SceneObservation, policy_registry

    assert_policy_source_clean(ROOT / "truck_benchmark" / "policies.py")
    assert_policy_source_clean(ROOT / "truck_benchmark" / "final_fusion.py")
    registry = policy_registry()
    registry.update(final_policy_registry())
    if args.policy not in registry:
        raise RuntimeError(f"Unknown policy {args.policy!r}; choices={sorted(registry)}")

    COLOURS = {
        "red": (0.82, 0.035, 0.035),
        "orange": (0.95, 0.27, 0.025),
        "blue": (0.025, 0.18, 0.90),
    }
    CUBE_EDGE = 0.012
    BIN_OUTER, BIN_HEIGHT, BIN_POCKET, BIN_FLOOR = 0.050, 0.054, 0.035, 0.0265
    BIN_WALL = (BIN_OUTER - BIN_POCKET) / 2.0
    TABLE_Z = C.TABLE_TOP_Z + C.SURFACE_DZ
    DECK_LOCAL_CENTRE = np.asarray((0.08129196, 0.15421628))
    DECK_LENGTH, DECK_WIDTH, DECK_HEIGHT = 0.18085018, 0.04709631, 0.0335
    DECK_LOCAL_YAW = -41.29227386
    PHYSICS_HZ = 60
    RENDER_FPS = float(args.fps)
    STEPS_PER_FRAME = max(1, round(PHYSICS_HZ / RENDER_FPS))
    rng = np.random.default_rng(args.seed)

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / RENDER_FPS)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    cell.enable_ccd(world)

    def add_mesh(path, stl, transform, colour):
        vertices, faces = load_stl(stl)
        vertices = transform(vertices)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        mesh.CreateDoubleSidedAttr(True)
        return mesh.GetPrim()

    def cube_transform(vertices):
        out = vertices.copy()
        out[:, 0] = (vertices[:, 0] - 6.0) / 1000.0
        out[:, 1] = (vertices[:, 1] - 6.0) / 1000.0
        out[:, 2] = (vertices[:, 2] + 12.0) / 1000.0
        return out

    def bin_transform(vertices):
        out = np.empty_like(vertices, dtype=float)
        out[:, 0] = (vertices[:, 0] + 21.0) / 1000.0
        out[:, 1] = -(vertices[:, 2] + 21.0) / 1000.0
        out[:, 2] = vertices[:, 1] / 1000.0
        return out

    def physics_material(path, static=0.9, dynamic=0.8):
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()
        api = UsdPhysics.MaterialAPI.Apply(prim)
        api.CreateStaticFrictionAttr().Set(static)
        api.CreateDynamicFrictionAttr().Set(dynamic)
        api.CreateRestitutionAttr().Set(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr().Set("average")
        return prim

    plastic = physics_material("/World/Physics/MeasuredPlastic")

    def bind_physics(prim):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(plastic),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        cell.small_contact_offsets(prim)

    # Scene construction is allowed to know the sampled state.  That state is
    # never inserted into the observation or action plan.
    truck_origin = np.asarray((0.351, -0.085)) + rng.uniform((-0.012, -0.012), (0.012, 0.012))
    truck_root_yaw = float(rng.uniform(-9.0, 9.0))
    truck_root = UsdGeom.Xform.Define(stage, "/World/MeasuredTruck")
    truck_xf = UsdGeom.Xformable(truck_root)
    truck_xf.AddTranslateOp().Set(Gf.Vec3d(float(truck_origin[0]), float(truck_origin[1]), float(TABLE_Z)))
    truck_xf.AddRotateZOp().Set(truck_root_yaw)
    add_mesh("/World/MeasuredTruck/Visual", ASSET_DIR / "truck.stl", lambda value: value / 1000.0, (0.14, 0.16, 0.19))
    deck = cell.box(
        stage, "/World/MeasuredTruck/DeckCollider", (DECK_LENGTH, DECK_WIDTH, 0.006),
        (float(DECK_LOCAL_CENTRE[0]), float(DECK_LOCAL_CENTRE[1]), 0.0305),
        (0.18, 0.20, 0.22), collision=True, rot=(0.0, 0.0, DECK_LOCAL_YAW),
    )
    UsdGeom.Imageable(deck).CreateVisibilityAttr().Set("invisible")
    bind_physics(deck)

    wrappers = {}
    truth_meta = {}
    bin_slots = [np.asarray((0.265, -0.135)), np.asarray((0.385, -0.137)), np.asarray((0.505, -0.135))]
    rng.shuffle(bin_slots)
    bin_layout = {}
    for colour, anchor in zip(COLOURS, bin_slots):
        bin_layout[colour] = {
            "xy": anchor + rng.uniform((-0.010, -0.008), (0.010, 0.008)),
            "yaw": float(rng.uniform(-34.0, 34.0)),
        }

    cube_grid = [np.asarray((x, y)) for y in (-0.300, -0.253, -0.206) for x in (0.235, 0.300, 0.365, 0.430)]
    rng.shuffle(cube_grid)
    cube_layout = []
    cube_index = 0
    for colour in COLOURS:
        for _ in range(4):
            cube_layout.append({
                "name": f"Cube{cube_index:02d}",
                "colour": colour,
                "xy": cube_grid[cube_index] + rng.uniform((-0.008, -0.008), (0.008, 0.008)),
                "yaw": float(rng.uniform(-43.0, 43.0)),
            })
            cube_index += 1

    def add_cube(item):
        path = f"/World/{item['name']}"
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(item["xy"][0]), float(item["xy"][1]), float(TABLE_Z + 0.0004)))
        xf.AddRotateZOp().Set(item["yaw"])
        add_mesh(path + "/MeasuredVisual", ASSET_DIR / "cube_12mm.stl", cube_transform, COLOURS[item["colour"]])
        collider = cell.box(stage, path + "/Collider", (CUBE_EDGE,) * 3, (0, 0, CUBE_EDGE / 2), COLOURS[item["colour"]], collision=True)
        UsdGeom.Imageable(collider).CreateVisibilityAttr().Set("invisible")
        bind_physics(collider)
        cell.make_dynamic(root, 0.006, collide_self=False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateEnableCCDAttr().Set(True)
        wrappers[item["name"]] = SingleXFormPrim(path, name=item["name"])
        truth_meta[item["name"]] = {
            "category": "cube",
            "colour": item["colour"],
            "initial_xy_m": np.asarray(item["xy"], float).tolist(),
            "initial_yaw_deg": float(item["yaw"]),
        }

    def add_bin(colour, item):
        name, path = f"Bin_{colour}", f"/World/Bin_{colour}"
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(item["xy"][0]), float(item["xy"][1]), float(TABLE_Z + 0.0003)))
        xf.AddRotateZOp().Set(item["yaw"])
        add_mesh(path + "/MeasuredVisual", ASSET_DIR / "bin_50x50x54.stl", bin_transform, COLOURS[colour])

        def part(tag, size, position):
            prim = cell.box(stage, path + "/" + tag, size, position, COLOURS[colour], collision=True)
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
            bind_physics(prim)

        part("Floor", (BIN_OUTER, BIN_OUTER, BIN_FLOOR), (0, 0, BIN_FLOOR / 2))
        straight_h = BIN_HEIGHT - BIN_FLOOR - 0.002
        straight_z = BIN_FLOOR + straight_h / 2
        part("WallSouth", (BIN_OUTER, BIN_WALL, straight_h), (0, -BIN_OUTER / 2 + BIN_WALL / 2, straight_z))
        part("WallNorth", (BIN_OUTER, BIN_WALL, straight_h), (0, BIN_OUTER / 2 - BIN_WALL / 2, straight_z))
        part("WallWest", (BIN_WALL, BIN_POCKET, straight_h), (-BIN_OUTER / 2 + BIN_WALL / 2, 0, straight_z))
        part("WallEast", (BIN_WALL, BIN_POCKET, straight_h), (BIN_OUTER / 2 - BIN_WALL / 2, 0, straight_z))
        for tag, size, position in (
            ("LipSouth", (BIN_OUTER, 0.0015, 0.002), (0, -0.02425, 0.053)),
            ("LipNorth", (BIN_OUTER, 0.0015, 0.002), (0, 0.02425, 0.053)),
            ("LipWest", (0.0015, 0.047, 0.002), (-0.02425, 0, 0.053)),
            ("LipEast", (0.0015, 0.047, 0.002), (0.02425, 0, 0.053)),
        ):
            part(tag, size, position)
        cell.make_dynamic(root, 0.133, collide_self=False)
        UsdPhysics.MassAPI.Apply(root).CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0232))
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateEnableCCDAttr().Set(True)
        wrappers[name] = SingleXFormPrim(path, name=name)
        truth_meta[name] = {
            "category": "bin",
            "colour": colour,
            "initial_xy_m": np.asarray(item["xy"], float).tolist(),
            "initial_yaw_deg": float(item["yaw"]),
        }

    for item in cube_layout:
        add_cube(item)
    for colour, item in bin_layout.items():
        add_bin(colour, item)

    # RGB-D cameras provide the only object-localization inputs.
    resolution = (1920, 1080)
    sensor_resolutions = {"table": resolution, "side": resolution, "wrist": (1280, 720)}
    sensor_cameras = {
        "table": Camera(cell.TABLE_CAM_PATH, resolution=sensor_resolutions["table"]),
        "side": Camera(cell.SIDE_CAM_PATH, resolution=sensor_resolutions["side"]),
        "wrist": Camera(cell.WRIST_CAM_PATH, resolution=sensor_resolutions["wrist"]),
    }
    showcase_path = "/World/BenchmarkShowcaseCamera"
    cell.camera(stage, showcase_path, {
        "eye": (0.735, -0.690, 1.430), "look_at": (0.385, -0.055, TABLE_Z + 0.050),
        "up": (0, 0, 1), "resolution": (1280, 720), "focal_length": 19.0, "horiz_aperture": 20.955,
    })
    showcase = Camera(showcase_path, resolution=(1280, 720)) if args.render_video else None

    world.reset()
    for camera in sensor_cameras.values():
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
    if showcase is not None:
        showcase.initialize()

    arm = SingleArticulation(C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    dof_names = list(arm.dof_names)
    arm_indices = [dof_names.index(f"J{i}") for i in range(1, 7)]
    dof_index = {name: index for index, name in enumerate(dof_names)}
    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    tool_offset = tuple(np.asarray(C.TOOL_AXIS, float) * cell.tip_offset())
    down = np.asarray((0.0, 0.0, -1.0))
    home = np.asarray((0.300, -0.020, 0.245))
    q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
    q, ok = ik.solve(q, home, down, limits, tool_offset=tool_offset, iters=600, damping=0.02)
    if not ok:
        raise RuntimeError("Home IK failed")

    def full_target(q6, jaw):
        current = arm.get_joint_positions().copy()
        current[arm_indices] = q6
        for joint, value in cell.jaw_targets(jaw).items():
            current[dof_index[joint]] = value
        return current

    arm.set_joint_positions(full_target(q, C.JAW_OPEN))
    for _ in range(110):
        world.step(render=True)

    frame_number = 0

    def capture(label):
        nonlocal_marker = None
        global frame_number
        if showcase is None:
            return
        rgba = showcase.get_rgba()
        if rgba is None:
            return
        image = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (1280, 74), (12, 14, 18), -1)
        cv2.putText(image, label, (22, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"OBSERVATION-ONLY {args.policy} | {args.controller}", (22, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 225, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(FRAMES / f"f{frame_number:05d}.png"), image)
        frame_number += 1

    active_scan_enabled = (
        "active_wrist" in args.policy
        or "task_hybrid" in args.policy
        or args.policy == "L_online_consensus_inventory_fusion"
    )
    active_scan_joint_sets = []
    if active_scan_enabled:
        scan_specs = [
            (
                "wrist_scan_near",
                np.asarray((0.285, -0.255, 0.285)),
                np.asarray((0.0, 0.0, -1.0)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
            (
                "wrist_scan_far",
                np.asarray((0.410, -0.160, 0.360)),
                np.asarray((0.0, 0.0, -1.0)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
            (
                "wrist_scan_oblique_west",
                np.asarray((0.250, -0.220, 0.380)),
                np.asarray((0.4067, 0.0, -0.9135)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
        ]
        scan_seed = q.copy()
        for scan_name, target, tool_direction, spread in scan_specs:
            solved, converged = ik.solve(
                scan_seed,
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
            active_scan_joint_sets.append((scan_name, solved.copy()))
            scan_seed = solved

    active_scan_motion_count = 0

    def drive_scan_joints(target_q, label):
        """Move through a real joint trajectory; never teleport for sensing."""
        global active_scan_motion_count
        measured = arm.get_joint_positions().copy()
        start_q = measured[arm_indices]
        steps = 72
        for index in range(steps):
            t = ease((index + 1) / steps)
            commanded = start_q + (np.asarray(target_q, float) - start_q) * t
            arm.apply_action(ArticulationAction(joint_positions=full_target(commanded, C.JAW_OPEN)))
            world.step(render=args.render_video and index % STEPS_PER_FRAME == STEPS_PER_FRAME - 1)
            if args.render_video and index % STEPS_PER_FRAME == STEPS_PER_FRAME - 1:
                capture(label)
        active_scan_motion_count += 1

    def calibration(path, camera_resolution):
        prim = stage.GetPrimAtPath(path)
        camera = UsdGeom.Camera(prim)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        matrix = np.asarray([[float(transform[row][column]) for column in range(4)] for row in range(4)])
        if args.variation in ("calibration_drift", "combined"):
            matrix[3, :3] += rng.normal(0.0, 0.0022, size=3)
        return CameraModel.from_json({
            "resolution": list(camera_resolution),
            "focal_length_mm": float(camera.GetFocalLengthAttr().Get()),
            "horizontal_aperture_mm": float(camera.GetHorizontalApertureAttr().Get()),
            "vertical_aperture_mm": float(camera.GetVerticalApertureAttr().Get()),
            "camera_to_world": matrix.tolist(),
            "table_top_z": float(TABLE_Z),
        })

    sensor_paths = {"table": cell.TABLE_CAM_PATH, "side": cell.SIDE_CAM_PATH, "wrist": cell.WRIST_CAM_PATH}
    observation_index = 0

    def make_observation(label):
        nonlocal_marker = None
        global observation_index
        # Isaac camera render products are not refreshed by physics-only
        # stepping.  Force fresh sensor frames before every observation so a
        # mid-task policy cannot accidentally consume its initial image.
        for _ in range(3):
            world.step(render=True)
        camera_observations = {}
        observation_dir = OUT / "observations" / f"{observation_index + 1:02d}_{label}"

        def capture_sensor(name, camera, path, camera_resolution):
            rgba, depth = camera.get_rgba(), camera.get_depth()
            if rgba is None or depth is None:
                raise RuntimeError(f"{name} RGB-D unavailable")
            rgb = rgba[:, :, :3].astype(np.uint8)
            depth = np.asarray(depth, np.float32).copy()
            if args.variation in ("photometric", "combined"):
                scale = rng.uniform(0.65, 1.38)
                balance = rng.uniform(0.80, 1.20, size=3)
                rgb = np.clip(rgb.astype(np.float32) * scale * balance.reshape(1, 1, 3), 0, 255).astype(np.uint8)
                rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
            if args.variation in ("depth_noise", "combined"):
                valid = np.isfinite(depth) & (depth > 0)
                depth[valid] += rng.normal(0.0, 0.0018, size=int(valid.sum())).astype(np.float32)
                depth[rng.random(depth.shape) < 0.035] = np.nan
            model = calibration(path, camera_resolution)
            captured = CameraObservation(
                name=name,
                rgb_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                depth=depth,
                camera=model,
            )
            if args.controller in ("vision_recovery_v2", "vision_recovery_v3"):
                # Retain the exact observations consumed by the policy.  The
                # files contain no simulator transforms or evaluator labels;
                # they make missed recoveries reproducible offline.
                observation_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(observation_dir / f"{name}_rgb.png"), captured.rgb_bgr)
                np.savez_compressed(observation_dir / f"{name}_depth.npz", depth=depth)
                (observation_dir / f"{name}_calibration.json").write_text(json.dumps({
                    "resolution": [model.width, model.height],
                    "focal_length_mm": model.focal_length_mm,
                    "horizontal_aperture_mm": model.horizontal_aperture_mm,
                    "vertical_aperture_mm": model.vertical_aperture_mm,
                    "camera_to_world": model.camera_to_world.tolist(),
                    "table_top_z": model.table_z,
                }, indent=2), encoding="utf-8")
            return captured

        for name, camera in sensor_cameras.items():
            camera_observations[name] = capture_sensor(
                name,
                camera,
                sensor_paths[name],
                sensor_resolutions[name],
            )

        if active_scan_joint_sets:
            home_q = q.copy()
            for scan_name, scan_joints in active_scan_joint_sets:
                drive_scan_joints(scan_joints, f"ACTIVE PERCEPTION: {scan_name}")
                for _ in range(3):
                    world.step(render=True)
                camera_observations[scan_name] = capture_sensor(
                    scan_name,
                    sensor_cameras["wrist"],
                    sensor_paths["wrist"],
                    sensor_resolutions["wrist"],
                )
            drive_scan_joints(home_q, "ACTIVE PERCEPTION: return home")
        observation_index += 1
        return SceneObservation(
            scene_id=f"physics_{args.seed}_{observation_index}_{label}",
            cameras=camera_observations,
            robot_state={
                "joint_names": dof_names,
                "joint_positions_rad": arm.get_joint_positions().tolist(),
                "gripper_command_per_jaw_m": float(C.JAW_OPEN),
                "timestamp_s": float(observation_index),
            },
            variation={"name": args.variation},
        )

    observation = make_observation("initial")
    prediction = registry[args.policy].predict(observation)
    detections = prediction["detections"]
    inventory = {
        colour: {
            "cubes": sorted(
                [item for item in detections if item.get("category") == "cube" and item.get("colour") == colour],
                key=lambda item: (item["xy"][1], item["xy"][0]),
            )[:4],
            "bins": sorted(
                [item for item in detections if item.get("category") == "bin" and item.get("colour") == colour],
                key=lambda item: item.get("confidence", 0.0), reverse=True,
            )[:1],
        }
        for colour in COLOURS
    }
    trucks = sorted(
        [item for item in detections if item.get("category") == "truck"],
        key=lambda item: item.get("confidence", 0.0), reverse=True,
    )
    complete = all(len(value["cubes"]) == 4 and len(value["bins"]) == 1 for value in inventory.values()) and len(trucks) >= 1
    if not complete:
        raise RuntimeError("Policy failed inventory gate before motion")

    truck = trucks[0]
    deck_centre = np.asarray(truck.get("deck_centre_xy", truck["xy"]), float)
    deck_yaw_deg = float(truck.get("deck_yaw_deg", truck["yaw_deg"]))
    deck_yaw = math.radians(deck_yaw_deg)
    long_axis = np.asarray((math.cos(deck_yaw), math.sin(deck_yaw)))
    zones = {colour: deck_centre + offset * long_axis for colour, offset in zip(COLOURS, (-0.057, 0.0, 0.057))}

    # Freeze a serializable plan before the first command.  It contains only
    # perception output and declared task geometry.
    plan = {
        "scene_id": observation.scene_id,
        "policy": args.policy,
        "controller": args.controller,
        "perception_latency_ms": prediction["latency_ms"],
        "inventory": inventory,
        "truck": truck,
        "zones": {key: value.tolist() for key, value in zones.items()},
    }
    plan_text = json.dumps(plan, sort_keys=True, indent=2)
    (OUT / "frozen_plan.json").write_text(plan_text, encoding="utf-8")
    report["frozen_plan_sha256"] = hashlib.sha256(plan_text.encode()).hexdigest()
    report["perception_latency_ms"] = prediction["latency_ms"]
    motion_started = time.perf_counter()

    path_length = 0.0
    action_count = 0
    current_tip = home.copy()
    current_spread = np.asarray((0.0, 1.0, 0.0))
    report["gripper_checks"] = []
    report["camera_retries"] = []

    def measured_jaw_gap_mm():
        measured = arm.get_joint_positions()
        return float((
            2.0 * C.JAW_OPEN
            - abs(float(measured[dof_index["jaw_l"]]))
            - abs(float(measured[dof_index["jaw_r"]]))
        ) * 1000.0)

    def solve(target, spread):
        solution, converged = ik.solve(q, target, down, limits, tool_offset=tool_offset, spread_dir=spread, iters=500, damping=0.02)
        _, actual = fk.tool_pose(solution, tool_offset)
        if not converged or np.linalg.norm(actual - target) > 0.0015:
            raise RuntimeError(f"IK rejected target; error_mm={np.linalg.norm(actual-target)*1000:.3f}")
        return solution

    def command(label, finish, seconds, jaw_start, jaw_finish, finish_spread):
        nonlocal_marker = None
        global q, current_tip, current_spread, path_length, action_count
        start, spread_start = current_tip.copy(), current_spread.copy()
        frames = max(2, round(seconds * RENDER_FPS))
        path_length += float(np.linalg.norm(finish - start))
        action_count += 1
        for index in range(frames):
            t = ease((index + 1) / frames)
            target = start + (finish - start) * t
            spread = turn(spread_start, finish_spread, t)
            q = solve(target, spread)
            arm.apply_action(ArticulationAction(joint_positions=full_target(q, jaw_start + (jaw_finish - jaw_start) * t)))
            for step_index in range(STEPS_PER_FRAME):
                world.step(render=args.render_video and step_index == STEPS_PER_FRAME - 1)
            capture(label)
        report["actions"].append({"label": label, "target_tip_m": finish.tolist(), "duration_s": seconds})
        current_tip, current_spread = finish.copy(), finish_spread.copy()

    def dwell(label, seconds, jaw):
        global action_count
        action_count += 1
        frames = max(1, round(seconds * RENDER_FPS))
        for _ in range(frames):
            arm.apply_action(ArticulationAction(joint_positions=full_target(q, jaw)))
            for step_index in range(STEPS_PER_FRAME):
                world.step(render=args.render_video and step_index == STEPS_PER_FRAME - 1)
            capture(label)
        report["actions"].append({"label": label, "target_tip_m": current_tip.tolist(), "duration_s": seconds})

    def command_wall_pivot(label, centre_xy, centre_yaw_start, centre_yaw_finish, radius, seconds, jaw):
        """Rotate a one-wall grasp about the bin centre, not the TCP.

        A plain wrist rotation pivots the 50 mm bin about the contacted wall and
        sweeps its centre through a roughly 21 mm arc.  That motion was enough
        to throw top-layer cubes over the 3.5 mm remaining rim.  Moving the TCP
        around the opposite arc keeps the payload centre stationary while the
        held wall and gripper rotate together.
        """
        global q, current_tip, current_spread, path_length, action_count
        frames = max(2, round(seconds * RENDER_FPS))
        delta = math.atan2(
            math.sin(centre_yaw_finish - centre_yaw_start),
            math.cos(centre_yaw_finish - centre_yaw_start),
        )
        previous = current_tip.copy()
        action_count += 1
        for index in range(frames):
            t = ease((index + 1) / frames)
            yaw = centre_yaw_start + delta * t
            normal = np.asarray((math.sin(yaw), -math.cos(yaw)))
            spread = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
            target = np.asarray((
                float(centre_xy[0] + normal[0] * radius),
                float(centre_xy[1] + normal[1] * radius),
                float(current_tip[2]),
            ))
            path_length += float(np.linalg.norm(target - previous))
            q = solve(target, spread)
            arm.apply_action(ArticulationAction(joint_positions=full_target(q, jaw)))
            for step_index in range(STEPS_PER_FRAME):
                world.step(render=args.render_video and step_index == STEPS_PER_FRAME - 1)
            capture(label)
            previous = target
        current_tip = previous.copy()
        current_spread = np.asarray((-math.sin(centre_yaw_finish), math.cos(centre_yaw_finish), 0.0))
        report["actions"].append({
            "label": label,
            "target_tip_m": current_tip.tolist(),
            "duration_s": seconds,
            "motion": "bin-centre-preserving wall arc",
            "yaw_delta_deg": math.degrees(delta),
            "radius_m": radius,
        })

    cube_jaw = max(C.JAW_CLOSED, CUBE_EDGE / 2.0 - C.FINGER_INNER_TIP - 0.00050)
    cube_release = min(C.JAW_OPEN, cube_jaw + 0.0007)
    # Reserve a clear south-wall finger corridor before the later one-wall bin
    # grasp.  Four floor-level cells would put the southern cubes directly in
    # the path of the finger that enters the pocket.  Two north-biased columns
    # stacked two high keep every cube face at least 12.5 mm from that wall.
    # Recovery v2 uses the tightest non-overlapping 12 mm column spacing.
    # This raises lateral wall margin from 4.0 mm to 5.25 mm, reducing the
    # chance that a top cube reaches the rim while the filled bin rotates.
    packing_x = 0.00625 if args.controller in ("vision_recovery_v2", "vision_recovery_v3") else 0.0075
    packing = [
        (-packing_x, 0.0060, 0),
        (packing_x, 0.0060, 0),
        (-packing_x, 0.0060, 1),
        (packing_x, 0.0060, 1),
    ]
    guarded = args.controller in ("calibrated_guarded", "vision_replan", "vision_recovery", "vision_recovery_v2", "vision_recovery_v3")
    cube_attempts = []

    for colour in COLOURS:
        bin_detection = inventory[colour]["bins"][0]
        bin_xy = np.asarray(bin_detection["xy"], float)
        bin_yaw = math.radians(float(bin_detection["yaw_deg"]))
        bin_spread = np.asarray((-math.sin(bin_yaw), math.cos(bin_yaw), 0.0))
        for number, (cube, slot) in enumerate(zip(inventory[colour]["cubes"], packing), start=1):
            cube_xy = np.asarray(cube["xy"], float)
            cube_attempts.append({
                "colour": colour,
                "number": number,
                "initial_xy": cube_xy.copy(),
                "slot": tuple(slot),
            })
            cube_yaw = math.radians(float(cube["yaw_deg"]))
            cube_spread = np.asarray((-math.sin(cube_yaw), math.cos(cube_yaw), 0.0))
            source = np.asarray((cube_xy[0], cube_xy[1], C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE))
            source_above = source + np.asarray((0.0, 0.0, 0.085 if guarded else 0.065))
            slot_xy = bin_xy + rotate_xy(np.asarray(slot[:2]), bin_yaw)
            destination = np.asarray((
                slot_xy[0], slot_xy[1],
                C.SURFACE_DZ + BIN_FLOOR + C.CUBE_TIP_CLEARANCE + slot[2] * CUBE_EDGE,
            ))
            destination_above = destination + np.asarray((0.0, 0.0, 0.085 if guarded else 0.065))
            label = f"{colour} cube {number}/4"
            command(label + " approach", source_above, 0.70 if guarded else 0.55, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
            command(label + " descend", source, 0.55 if guarded else 0.42, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
            dwell(label + " close", 0.65 if guarded else 0.30, cube_jaw)
            close_gap = measured_jaw_gap_mm()
            close_check = {
                "label": label,
                "kind": "cube_close",
                "jaw_gap_mm": close_gap,
                "empty_command_gap_mm": float(2.0 * cube_jaw * 1000.0),
                "attempt": 1,
            }
            report["gripper_checks"].append(close_check)
            empty_close_threshold = float(2.0 * cube_jaw * 1000.0 + 0.35)
            if args.controller in ("vision_replan", "vision_recovery", "vision_recovery_v2", "vision_recovery_v3") and close_gap <= empty_close_threshold:
                # The measured jaw travel says the jaws reached their empty
                # target rather than being stopped by a cube.  Retreat before
                # looking again so the wrist/tool cannot occlude the cameras.
                dwell(label + " retry open", 0.35, C.JAW_OPEN)
                command(label + " retry rise", source_above, 0.60, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                command(label + " retry camera clear", home, 0.85, C.JAW_OPEN, C.JAW_OPEN, np.asarray((0.0, 1.0, 0.0)))
                dwell(label + " retry camera settle", 0.40, C.JAW_OPEN)
                retry_observation = make_observation(f"retry_{colour}_{number}")
                retry_prediction = registry[args.policy].predict(retry_observation)
                retry_candidates = sorted(
                    [
                        item for item in retry_prediction["detections"]
                        if item.get("category") == "cube"
                        and item.get("colour") == colour
                    ],
                    key=lambda item: float(np.linalg.norm(np.asarray(item["xy"], float) - cube_xy)),
                )
                retry_doc = {
                    "label": label,
                    "scene_id": retry_observation.scene_id,
                    "latency_ms": retry_prediction["latency_ms"],
                    "empty_close_threshold_mm": empty_close_threshold,
                    "first_close_gap_mm": close_gap,
                    "candidate_count": len(retry_candidates),
                    "selected": None,
                }
                if retry_candidates:
                    nearest = retry_candidates[0]
                    shift = float(np.linalg.norm(np.asarray(nearest["xy"], float) - cube_xy))
                    if shift <= 0.040:
                        retry_doc["selected"] = nearest
                        retry_doc["recenter_distance_mm"] = shift * 1000.0
                        cube_xy = np.asarray(nearest["xy"], float)
                        cube_yaw = math.radians(float(nearest["yaw_deg"]))
                    else:
                        cube_yaw += math.pi / 2.0
                        retry_doc["fallback"] = "nearest candidate too far; rotate grasp axis 90 degrees"
                else:
                    cube_yaw += math.pi / 2.0
                    retry_doc["fallback"] = "no refreshed cube candidate; rotate grasp axis 90 degrees"
                cube_spread = np.asarray((-math.sin(cube_yaw), math.cos(cube_yaw), 0.0))
                source = np.asarray((cube_xy[0], cube_xy[1], C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE))
                source_above = source + np.asarray((0.0, 0.0, 0.085))
                retry_path = OUT / f"retry_{colour}_{number}.json"
                retry_text = json.dumps(retry_doc, sort_keys=True, indent=2)
                retry_path.write_text(retry_text, encoding="utf-8")
                retry_doc["sha256"] = hashlib.sha256(retry_text.encode()).hexdigest()
                report["camera_retries"].append(retry_doc)
                report["perception_latency_ms"] += retry_prediction["latency_ms"]
                command(label + " retry approach", source_above, 0.75, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                command(label + " retry descend", source, 0.55, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                dwell(label + " retry close", 0.65, cube_jaw)
                close_gap = measured_jaw_gap_mm()
                report["gripper_checks"].append({
                    "label": label,
                    "kind": "cube_close",
                    "jaw_gap_mm": close_gap,
                    "empty_command_gap_mm": float(2.0 * cube_jaw * 1000.0),
                    "attempt": 2,
                })
                if close_gap <= empty_close_threshold:
                    raise RuntimeError(
                        f"camera-only cube retry still measured an empty close: {label}; gap_mm={close_gap:.3f}"
                    )
            command(label + " lift", source_above, 0.75 if guarded else 0.50, cube_jaw, cube_jaw, cube_spread)
            command(label + " carry", destination_above, 1.10 if guarded else 0.65, cube_jaw, cube_jaw, bin_spread)
            command(label + " lower", destination, 0.70 if guarded else 0.45, cube_jaw, cube_jaw, bin_spread)
            dwell(label + " release", 0.50 if guarded else 0.38, cube_release)
            command(label + " withdraw", destination_above, 0.65 if guarded else 0.42, cube_release, cube_release, bin_spread)
            dwell(label + " fully open", 0.20, C.JAW_OPEN)

    if args.controller in ("vision_replan", "vision_recovery", "vision_recovery_v2", "vision_recovery_v3"):
        # The objects may have nudged their bins during twelve deliveries.
        # Clear the fixed cameras, take a fresh RGB-D observation, and update
        # only bins/truck.  No USD transform or evaluator value is consulted.
        command("mid-task camera clear", home, 1.20, C.JAW_OPEN, C.JAW_OPEN, np.asarray((0.0, 1.0, 0.0)))
        dwell("mid-task camera settle", 0.45, C.JAW_OPEN)
        refreshed_observation = make_observation("before_bin_transport")
        refreshed_prediction = registry[args.policy].predict(refreshed_observation)
        refreshed = refreshed_prediction["detections"]
        refreshed_for_recovery = list(refreshed)
        recovery_rgb_policy_name = "locked_calibration__rgb_hue_nearest_r1.00_k21_model_fit_inventory"
        if args.controller in ("vision_recovery_v2", "vision_recovery_v3") and recovery_rgb_policy_name in registry:
            recovery_rgb = registry[recovery_rgb_policy_name].predict(refreshed_observation)
            report["perception_latency_ms"] += recovery_rgb["latency_ms"]
            rgb_cubes = [
                item for item in recovery_rgb["detections"]
                if item.get("category") == "cube" and float(item.get("fit_iou", 0.0)) >= 0.50
            ]
            for candidate in rgb_cubes:
                if not any(
                    item.get("category") == "cube"
                    and item.get("colour") == candidate.get("colour")
                    and float(np.linalg.norm(np.asarray(item["xy"], float) - np.asarray(candidate["xy"], float))) < 0.010
                    for item in refreshed_for_recovery
                ):
                    refreshed_for_recovery.append(candidate)
            report["recovery_detector_diagnostics"] = {
                "rgb_policy": recovery_rgb_policy_name,
                "rgb_latency_ms": recovery_rgb["latency_ms"],
                "primary_cube_count": sum(item.get("category") == "cube" for item in refreshed),
                "admitted_rgb_cube_count": len(rgb_cubes),
            }
        replan = {
            "scene_id": refreshed_observation.scene_id,
            "latency_ms": refreshed_prediction["latency_ms"],
            "updated": {},
            "fallbacks": [],
        }
        def angular_delta_deg(first, second):
            return (float(second) - float(first) + 180.0) % 360.0 - 180.0

        def plausible_update(previous, candidate, max_shift_m, max_yaw_deg, deck=False):
            previous_xy = previous.get("deck_centre_xy", previous["xy"]) if deck else previous["xy"]
            candidate_xy = candidate.get("deck_centre_xy", candidate["xy"]) if deck else candidate["xy"]
            previous_yaw = previous.get("deck_yaw_deg", previous["yaw_deg"]) if deck else previous["yaw_deg"]
            candidate_yaw = candidate.get("deck_yaw_deg", candidate["yaw_deg"]) if deck else candidate["yaw_deg"]
            shift = float(np.linalg.norm(np.asarray(candidate_xy, float) - np.asarray(previous_xy, float)))
            yaw_delta = abs(angular_delta_deg(previous_yaw, candidate_yaw))
            return shift <= max_shift_m and yaw_delta <= max_yaw_deg, shift, yaw_delta

        for colour in COLOURS:
            candidates = sorted(
                [item for item in refreshed if item.get("category") == "bin" and item.get("colour") == colour],
                key=lambda item: item.get("confidence", 0.0),
                reverse=True,
            )
            if candidates:
                accepted, shift, yaw_delta = plausible_update(
                    inventory[colour]["bins"][0], candidates[0], 0.025, 20.0
                )
                replan.setdefault("gates", {})[f"bin_{colour}"] = {
                    "accepted": accepted,
                    "shift_mm": shift * 1000.0,
                    "yaw_delta_deg": yaw_delta,
                }
                if accepted:
                    inventory[colour]["bins"] = [candidates[0]]
                    replan["updated"][f"bin_{colour}"] = candidates[0]
                else:
                    replan["fallbacks"].append(f"bin_{colour}:implausible_refresh")
            else:
                replan["fallbacks"].append(f"bin_{colour}")
        refreshed_trucks = sorted(
            [item for item in refreshed if item.get("category") == "truck"],
            key=lambda item: item.get("confidence", 0.0),
            reverse=True,
        )
        if refreshed_trucks:
            accepted, shift, yaw_delta = plausible_update(truck, refreshed_trucks[0], 0.015, 8.0, deck=True)
            replan.setdefault("gates", {})["truck"] = {
                "accepted": accepted,
                "shift_mm": shift * 1000.0,
                "yaw_delta_deg": yaw_delta,
            }
            if accepted:
                truck = refreshed_trucks[0]
                replan["updated"]["truck"] = truck
            else:
                replan["fallbacks"].append("truck:implausible_refresh")
        else:
            replan["fallbacks"].append("truck")

        if args.controller in ("vision_recovery", "vision_recovery_v2", "vision_recovery_v3"):
            # A failed top-down pickup generally leaves the cube close to its
            # original table pose.  Use only the refreshed RGB-D detections to
            # find same-colour cubes still outside their bin, match each to the
            # nearest original attempt, and retry its reserved packing slot.
            used_attempts = set()
            residuals = []
            for colour in COLOURS:
                current_bin = inventory[colour]["bins"][0]
                bin_xy = np.asarray(current_bin["xy"], float)
                candidates = [
                    item for item in refreshed_for_recovery
                    if item.get("category") == "cube"
                    and item.get("colour") == colour
                    and float(np.linalg.norm(np.asarray(item["xy"], float) - bin_xy)) > 0.032
                ]
                attempts = [item for item in cube_attempts if item["colour"] == colour]
                for candidate in sorted(candidates, key=lambda item: item.get("confidence", 0.0), reverse=True):
                    candidate_xy = np.asarray(candidate["xy"], float)
                    available = [
                        item for item in attempts
                        if (item["colour"], item["number"]) not in used_attempts
                    ]
                    if not available:
                        break
                    attempt = min(
                        available,
                        key=lambda item: float(np.linalg.norm(candidate_xy - item["initial_xy"])),
                    )
                    source_shift = float(np.linalg.norm(candidate_xy - attempt["initial_xy"]))
                    if source_shift > 0.060:
                        continue
                    used_attempts.add((attempt["colour"], attempt["number"]))
                    residuals.append((candidate, attempt, source_shift))

            recovery_log = []
            for candidate, attempt, source_shift in residuals:
                colour = attempt["colour"]
                current_bin = inventory[colour]["bins"][0]
                bin_xy = np.asarray(current_bin["xy"], float)
                bin_yaw = math.radians(float(current_bin["yaw_deg"]))
                bin_spread = np.asarray((-math.sin(bin_yaw), math.cos(bin_yaw), 0.0))
                cube_xy = np.asarray(candidate["xy"], float)
                cube_yaw = math.radians(float(candidate["yaw_deg"]))
                cube_spread = np.asarray((-math.sin(cube_yaw), math.cos(cube_yaw), 0.0))
                source = np.asarray((cube_xy[0], cube_xy[1], C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE))
                source_above = source + np.asarray((0.0, 0.0, 0.090))
                slot = attempt["slot"]
                # If an early bottom-layer delivery was missed, later cubes
                # have already fallen into the vacant support cell.  Driving a
                # delayed recovery back to z=0 ploughs into that collapsed
                # stack.  Rebuild from the exposed top layer instead.
                recovery_slot = (slot[0], slot[1], max(1, slot[2]))
                slot_xy = bin_xy + rotate_xy(np.asarray(recovery_slot[:2]), bin_yaw)
                destination = np.asarray((
                    slot_xy[0], slot_xy[1],
                    C.SURFACE_DZ + BIN_FLOOR + C.CUBE_TIP_CLEARANCE + recovery_slot[2] * CUBE_EDGE,
                ))
                destination_above = destination + np.asarray((0.0, 0.0, 0.090))
                label = f"{colour} residual cube {attempt['number']}/4"
                command(label + " approach", source_above, 0.75, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                command(label + " descend", source, 0.60, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                dwell(label + " close", 0.70, cube_jaw)
                recovery_gap = measured_jaw_gap_mm()
                command(label + " lift", source_above, 0.80, cube_jaw, cube_jaw, cube_spread)
                command(label + " carry", destination_above, 1.10, cube_jaw, cube_jaw, bin_spread)
                command(label + " lower", destination, 0.75, cube_jaw, cube_jaw, bin_spread)
                dwell(label + " release", 0.55, cube_release)
                command(label + " withdraw", destination_above, 0.70, cube_release, cube_release, bin_spread)
                dwell(label + " fully open", 0.20, C.JAW_OPEN)
                recovery_log.append({
                    "colour": colour,
                    "attempt": attempt["number"],
                    "source_xy": cube_xy.tolist(),
                    "source_match_shift_mm": source_shift * 1000.0,
                    "jaw_gap_mm": recovery_gap,
                    "original_slot": list(slot),
                    "recovery_slot": list(recovery_slot),
                })
            report["residual_cube_recovery"] = recovery_log

            if recovery_log:
                # The retry can nudge a bin.  Refresh once more, but only
                # accept small physically plausible bin changes.  The truck is
                # static, so its already-gated pose remains frozen.
                command("post-recovery camera clear", home, 1.10, C.JAW_OPEN, C.JAW_OPEN, np.asarray((0.0, 1.0, 0.0)))
                dwell("post-recovery camera settle", 0.45, C.JAW_OPEN)
                post_observation = make_observation("after_cube_recovery")
                post_prediction = registry[args.policy].predict(post_observation)
                report["perception_latency_ms"] += post_prediction["latency_ms"]
                post_detections = post_prediction["detections"]
                post_recovery_detections = list(post_detections)
                if args.controller in ("vision_recovery_v2", "vision_recovery_v3") and recovery_rgb_policy_name in registry:
                    post_rgb = registry[recovery_rgb_policy_name].predict(post_observation)
                    report["perception_latency_ms"] += post_rgb["latency_ms"]
                    post_rgb_cubes = [
                        item for item in post_rgb["detections"]
                        if item.get("category") == "cube" and float(item.get("fit_iou", 0.0)) >= 0.50
                    ]
                    for candidate in post_rgb_cubes:
                        if not any(
                            item.get("category") == "cube"
                            and item.get("colour") == candidate.get("colour")
                            and float(np.linalg.norm(np.asarray(item["xy"], float) - np.asarray(candidate["xy"], float))) < 0.010
                            for item in post_recovery_detections
                        ):
                            post_recovery_detections.append(candidate)
                    report.setdefault("recovery_detector_diagnostics", {})["post_rgb_latency_ms"] = post_rgb["latency_ms"]
                post_log = {"scene_id": post_observation.scene_id, "latency_ms": post_prediction["latency_ms"], "bins": {}}
                for colour in COLOURS:
                    candidates = sorted(
                        [item for item in post_detections if item.get("category") == "bin" and item.get("colour") == colour],
                        key=lambda item: item.get("confidence", 0.0),
                        reverse=True,
                    )
                    if not candidates:
                        post_log["bins"][colour] = {"accepted": False, "reason": "not_detected"}
                        continue
                    accepted, shift, yaw_delta = plausible_update(
                        inventory[colour]["bins"][0], candidates[0], 0.025, 20.0
                    )
                    post_log["bins"][colour] = {
                        "accepted": accepted,
                        "shift_mm": shift * 1000.0,
                        "yaw_delta_deg": yaw_delta,
                    }
                    if accepted:
                        inventory[colour]["bins"] = [candidates[0]]
                report["post_recovery_replan"] = post_log

                if args.controller in ("vision_recovery_v2", "vision_recovery_v3"):
                    # A first recovery grasp can move a marginally localized
                    # cube without capturing it.  The original controller
                    # stopped at that point.  Re-observe after the motion,
                    # require a same-colour residual close to the just-used
                    # source, then make one bounded re-centred retry with the
                    # orthogonal grasp axis.  This remains observation-only:
                    # no evaluator/object transform participates in the gate.
                    second_pass = []
                    claimed_candidates = set()
                    for first in recovery_log:
                        colour = first["colour"]
                        current_bin = inventory[colour]["bins"][0]
                        bin_xy = np.asarray(current_bin["xy"], float)
                        prior_source = np.asarray(first["source_xy"], float)
                        candidates = [
                            item for item in post_recovery_detections
                            if item.get("category") == "cube"
                            and item.get("colour") == colour
                            and float(np.linalg.norm(np.asarray(item["xy"], float) - bin_xy)) > 0.032
                            and id(item) not in claimed_candidates
                        ]
                        if not candidates:
                            continue
                        candidate = min(
                            candidates,
                            key=lambda item: float(np.linalg.norm(np.asarray(item["xy"], float) - prior_source)),
                        )
                        shift = float(np.linalg.norm(np.asarray(candidate["xy"], float) - prior_source))
                        if shift > 0.040:
                            continue
                        claimed_candidates.add(id(candidate))
                        attempt = next(
                            item for item in cube_attempts
                            if item["colour"] == colour and item["number"] == first["attempt"]
                        )
                        bin_yaw = math.radians(float(current_bin["yaw_deg"]))
                        bin_spread = np.asarray((-math.sin(bin_yaw), math.cos(bin_yaw), 0.0))
                        cube_xy = np.asarray(candidate["xy"], float)
                        cube_yaw = math.radians(float(candidate["yaw_deg"])) + math.pi / 2.0
                        cube_spread = np.asarray((-math.sin(cube_yaw), math.cos(cube_yaw), 0.0))
                        source = np.asarray((cube_xy[0], cube_xy[1], C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE))
                        source_above = source + np.asarray((0.0, 0.0, 0.090))
                        slot = attempt["slot"]
                        recovery_slot = (slot[0], slot[1], max(1, slot[2]))
                        slot_xy = bin_xy + rotate_xy(np.asarray(recovery_slot[:2]), bin_yaw)
                        destination = np.asarray((
                            slot_xy[0], slot_xy[1],
                            C.SURFACE_DZ + BIN_FLOOR + C.CUBE_TIP_CLEARANCE + recovery_slot[2] * CUBE_EDGE,
                        ))
                        destination_above = destination + np.asarray((0.0, 0.0, 0.090))
                        recovery_jaw = max(C.JAW_CLOSED, cube_jaw - 0.00025)
                        label = f"{colour} residual cube {attempt['number']}/4 pass 2"
                        command(label + " approach", source_above, 0.80, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                        command(label + " descend", source, 0.65, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                        dwell(label + " close", 0.75, recovery_jaw)
                        second_gap = measured_jaw_gap_mm()
                        command(label + " lift", source_above, 0.85, recovery_jaw, recovery_jaw, cube_spread)
                        command(label + " carry", destination_above, 1.15, recovery_jaw, recovery_jaw, bin_spread)
                        command(label + " lower", destination, 0.80, recovery_jaw, recovery_jaw, bin_spread)
                        dwell(label + " release", 0.60, cube_release)
                        command(label + " withdraw", destination_above, 0.75, cube_release, cube_release, bin_spread)
                        dwell(label + " fully open", 0.20, C.JAW_OPEN)
                        second_pass.append({
                            "colour": colour,
                            "attempt": attempt["number"],
                            "source_xy": cube_xy.tolist(),
                            "motion_since_pass_1_mm": shift * 1000.0,
                            "jaw_gap_mm": second_gap,
                            "jaw_target_per_finger_mm": recovery_jaw * 1000.0,
                            "grasp_axis_offset_deg": 90.0,
                            "original_slot": list(slot),
                            "recovery_slot": list(recovery_slot),
                        })
                    report["residual_cube_recovery_pass_2"] = second_pass
                    # Recovery placement can leave a cube rocking on the
                    # reconstructed stack.  Clear the cameras/tool and allow
                    # gravity and contacts to settle before the loaded bin is
                    # grasped by a single wall.
                    command("post-recovery transport clear", home, 1.20, C.JAW_OPEN, C.JAW_OPEN, np.asarray((0.0, 1.0, 0.0)))
                    dwell("post-recovery payload settle", 2.00, C.JAW_OPEN)
        deck_centre = np.asarray(truck.get("deck_centre_xy", truck["xy"]), float)
        deck_yaw_deg = float(truck.get("deck_yaw_deg", truck["yaw_deg"]))
        deck_yaw = math.radians(deck_yaw_deg)
        long_axis = np.asarray((math.cos(deck_yaw), math.sin(deck_yaw)))
        zones = {colour: deck_centre + offset * long_axis for colour, offset in zip(COLOURS, (-0.057, 0.0, 0.057))}
        replan["zones"] = {key: value.tolist() for key, value in zones.items()}
        replan_text = json.dumps(replan, sort_keys=True, indent=2)
        (OUT / "midtask_replan.json").write_text(replan_text, encoding="utf-8")
        report["midtask_replan_sha256"] = hashlib.sha256(replan_text.encode()).hexdigest()
        report["midtask_replan"] = replan

    release_bias = {
        "red": np.asarray((0.00207, -0.00046)),
        "orange": np.asarray((0.00033, 0.00251)),
        "blue": np.asarray((0.00260, -0.00075)),
    }
    release_yaw_bias = {"red": 0.645, "orange": -1.288, "blue": 0.635}
    lever = 0.02125 if args.controller == "direct_nominal" else 0.02465
    insertion = 0.020 if args.controller == "vision_recovery_v3" else 0.018
    report["loaded_bin_grip_checks"] = []
    for colour in ("blue", "red", "orange"):
        detection = inventory[colour]["bins"][0]
        target_yaw_deg = deck_yaw_deg
        zone = zones[colour].copy()
        if guarded:
            zone += rotate_xy(release_bias[colour], deck_yaw)
            target_yaw_deg += release_yaw_bias[colour]
        target_yaw = math.radians(target_yaw_deg)
        target_normal = np.asarray((math.sin(target_yaw), -math.cos(target_yaw)))
        target_spread = np.asarray((-math.sin(target_yaw), math.cos(target_yaw), 0.0))
        label = f"{colour} filled bin"

        # v3 reproduces the validated one-wall experiment's conservative
        # 10 mm micro-lift before committing to transport.  Jaw separation is
        # checked both immediately and under load: an empty jaw can report a
        # plausible command but cannot retain the 5.5--10 mm wall-contact gap.
        max_grasp_attempts = 2 if args.controller == "vision_recovery_v3" else 1
        grasp_insertion = insertion
        for grasp_attempt in range(1, max_grasp_attempts + 1):
            bin_xy = np.asarray(detection["xy"], float)
            source_yaw = math.radians(float(detection["yaw_deg"]))
            source_normal = np.asarray((math.sin(source_yaw), -math.cos(source_yaw)))
            source_spread = np.asarray((-math.sin(source_yaw), math.cos(source_yaw), 0.0))
            wall_xy = bin_xy + source_normal * (BIN_OUTER / 2.0 - BIN_WALL / 2.0)
            source = np.asarray((wall_xy[0], wall_xy[1], C.SURFACE_DZ + BIN_HEIGHT - grasp_insertion))
            source_above = source + np.asarray((0.0, 0.0, 0.095))
            proof_height = 0.010 if args.controller == "vision_recovery_v3" else 0.016
            proof = source + np.asarray((0.0, 0.0, proof_height))
            carry = source + np.asarray((0.0, 0.0, 0.110))

            suffix = "" if grasp_attempt == 1 else f" retry {grasp_attempt}"
            command(label + suffix + " approach", source_above, 0.90 if args.controller == "vision_recovery_v3" else 0.75, C.JAW_OPEN, C.JAW_OPEN, source_spread)
            command(label + suffix + " insert", source, 0.75 if args.controller == "vision_recovery_v3" else 0.55, C.JAW_OPEN, C.JAW_OPEN, source_spread)
            dwell(label + suffix + " close", 0.90 if args.controller == "vision_recovery_v3" else 0.65, C.JAW_CLOSED)
            close_gap = measured_jaw_gap_mm()
            check = {
                "label": label,
                "attempt": grasp_attempt,
                "phase": "close",
                "jaw_gap_mm": close_gap,
                "accepted_contact_gap_mm": [5.5, 10.0],
                "insertion_mm": grasp_insertion * 1000.0,
            }
            report["loaded_bin_grip_checks"].append(check)
            report["gripper_checks"].append({
                "label": label,
                "kind": "bin_wall_close",
                "jaw_gap_mm": close_gap,
                "accepted_contact_gap_mm": [5.5, 10.0],
                "attempt": grasp_attempt,
            })
            command(
                label + suffix + " proof lift",
                proof,
                1.50 if args.controller == "vision_recovery_v3" else (1.00 if guarded else 0.55),
                C.JAW_CLOSED,
                C.JAW_CLOSED,
                source_spread,
            )
            if guarded:
                dwell(label + suffix + " proof dwell", 1.20 if args.controller == "vision_recovery_v3" else 0.65, C.JAW_CLOSED)
            proof_gap = measured_jaw_gap_mm()
            proof_ok = 5.5 <= proof_gap <= 10.0
            report["loaded_bin_grip_checks"].append({
                "label": label,
                "attempt": grasp_attempt,
                "phase": "proof_hold",
                "jaw_gap_mm": proof_gap,
                "accepted": proof_ok,
                "proof_lift_mm": proof_height * 1000.0,
            })
            if args.controller != "vision_recovery_v3" or proof_ok:
                break
            if grasp_attempt >= max_grasp_attempts:
                raise RuntimeError(f"{label} lost wall contact during proof lift; gap_mm={proof_gap:.3f}")

            # Release vertically, clear all fixed cameras, and re-localize the
            # same-colour bin.  The retry is accepted only near the previous
            # pose and uses 2 mm more insertion; no evaluator pose is read.
            dwell(label + " failed proof open", 0.50, C.JAW_OPEN)
            command(label + " failed proof rise", source_above, 1.00, C.JAW_OPEN, C.JAW_OPEN, source_spread)
            command(label + " recovery camera clear", home, 1.20, C.JAW_OPEN, C.JAW_OPEN, np.asarray((0.0, 1.0, 0.0)))
            dwell(label + " recovery camera settle", 0.55, C.JAW_OPEN)
            bin_observation = make_observation(f"retry_bin_{colour}")
            bin_prediction = registry[args.policy].predict(bin_observation)
            report["perception_latency_ms"] += bin_prediction["latency_ms"]
            bin_candidates = sorted(
                [item for item in bin_prediction["detections"] if item.get("category") == "bin" and item.get("colour") == colour],
                key=lambda item: float(np.linalg.norm(np.asarray(item["xy"], float) - bin_xy)),
            )
            if not bin_candidates:
                raise RuntimeError(f"{label} proof retry could not re-detect the bin")
            shift = float(np.linalg.norm(np.asarray(bin_candidates[0]["xy"], float) - bin_xy))
            if shift > 0.060:
                raise RuntimeError(f"{label} proof retry rejected implausible {shift*1000.0:.1f} mm shift")
            detection = bin_candidates[0]
            inventory[colour]["bins"] = [detection]
            grasp_insertion = 0.022
            report["loaded_bin_grip_checks"].append({
                "label": label,
                "attempt": grasp_attempt + 1,
                "phase": "camera_recenter",
                "shift_mm": shift * 1000.0,
                "selected": detection,
            })

        target_tip_xy = zone + target_normal * lever
        target = np.asarray((
            target_tip_xy[0], target_tip_xy[1],
            float(truck.get("deck_z", TABLE_Z + DECK_HEIGHT) - C.TABLE_TOP_Z + BIN_HEIGHT - grasp_insertion + 0.0002),
        ))
        target_above = target + np.asarray((0.0, 0.0, 0.110))
        preplace = target + np.asarray((0.0, 0.0, 0.010))

        if args.controller == "vision_recovery_v3":
            staged_raise = source + np.asarray((0.0, 0.0, 0.055))
            command(label + " staged raise", staged_raise, 1.80, C.JAW_CLOSED, C.JAW_CLOSED, source_spread)
            dwell(label + " staged hold", 0.60, C.JAW_CLOSED)
            staged_gap = measured_jaw_gap_mm()
            staged_ok = 5.5 <= staged_gap <= 10.0
            report["loaded_bin_grip_checks"].append({
                "label": label,
                "phase": "staged_hold",
                "jaw_gap_mm": staged_gap,
                "accepted": staged_ok,
            })
            if not staged_ok:
                raise RuntimeError(f"{label} lost wall contact during staged raise; gap_mm={staged_gap:.3f}")
            command(label + " raise", carry, 2.20, C.JAW_CLOSED, C.JAW_CLOSED, source_spread)
            raised_gap = measured_jaw_gap_mm()
            raised_ok = 5.5 <= raised_gap <= 10.0
            report["loaded_bin_grip_checks"].append({
                "label": label,
                "phase": "raised_hold",
                "jaw_gap_mm": raised_gap,
                "accepted": raised_ok,
            })
            if not raised_ok:
                raise RuntimeError(f"{label} lost wall contact before transport; gap_mm={raised_gap:.3f}")
            # Follow the wall's circular locus so the bin rotates around its
            # own centre.  Rotating at constant TCP would swing the bin centre
            # and was observed to eject otherwise-contained cubes.
            command_wall_pivot(
                label + " centre-preserving rotate",
                bin_xy,
                source_yaw,
                target_yaw,
                BIN_OUTER / 2.0 - BIN_WALL / 2.0,
                3.00,
                C.JAW_CLOSED,
            )
            dwell(label + " rotation settle", 0.60, C.JAW_CLOSED)
            command(label + " carry", target_above, 3.60, C.JAW_CLOSED, C.JAW_CLOSED, target_spread)
        else:
            command(label + " raise", carry, 1.50 if guarded else 0.70, C.JAW_CLOSED, C.JAW_CLOSED, source_spread)
            command(label + " carry", target_above, 3.20 if guarded else 0.95, C.JAW_CLOSED, C.JAW_CLOSED, target_spread)
        command(label + " pre-place", preplace, 2.00 if guarded else 0.85, C.JAW_CLOSED, C.JAW_CLOSED, target_spread)
        command(label + " final lower", target, 1.20 if guarded else 0.65, C.JAW_CLOSED, C.JAW_CLOSED, target_spread)
        dwell(label + " supported", 0.80 if guarded else 0.35, C.JAW_CLOSED)
        dwell(label + " full release", 0.70, C.JAW_OPEN)
        if guarded:
            dwell(label + " release dwell", 0.45, C.JAW_OPEN)
        command(label + " vertical withdraw", target_above, 0.85, C.JAW_OPEN, C.JAW_OPEN, target_spread)

    command("final retreat", home, 0.70, C.JAW_OPEN, C.JAW_OPEN, np.asarray((0.0, 1.0, 0.0)))
    for _ in range(round(1.0 * RENDER_FPS)):
        world.step(render=args.render_video)
        capture("Final evaluator snapshot")

    if args.controller in ("vision_recovery_v2", "vision_recovery_v3"):
        # Final observation before the evaluator boundary.  v2 records this
        # packet only; v3 may use it for one bounded spill recovery after all
        # bins are supported on the truck.
        final_observation = make_observation("after_bin_transport")
        diagnostic_names = [
            args.policy,
            "locked_calibration__rgb_hue_nearest_r1.00_k21_model_fit_inventory",
            "locked_calibration__hybrid_depth_anchor_hue_model_table_inventory_truckachromatic",
        ]
        diagnostics = {}
        for policy_name in dict.fromkeys(diagnostic_names):
            if policy_name not in registry:
                continue
            diagnostic = registry[policy_name].predict(final_observation)
            diagnostics[policy_name] = {
                "latency_ms": diagnostic["latency_ms"],
                "detections": diagnostic["detections"],
            }
        report["final_camera_diagnostics"] = {
            "scene_id": final_observation.scene_id,
            "policies": diagnostics,
        }

        if args.controller == "vision_recovery_v3":
            primary = diagnostics.get(args.policy, {}).get("detections", [])
            rgb_name = "locked_calibration__rgb_hue_nearest_r1.00_k21_model_fit_inventory"
            rgb_final = diagnostics.get(rgb_name, {}).get("detections", [])
            final_cubes = [item for item in primary if item.get("category") == "cube"]
            for candidate in rgb_final:
                if candidate.get("category") != "cube" or float(candidate.get("fit_iou", 0.0)) < 0.50:
                    continue
                if not any(
                    item.get("colour") == candidate.get("colour")
                    and float(np.linalg.norm(np.asarray(item["xy"], float) - np.asarray(candidate["xy"], float))) < 0.010
                    for item in final_cubes
                ):
                    final_cubes.append(candidate)

            final_bin_candidates = [item for item in rgb_final if item.get("category") == "bin"]
            final_recovery_log = []
            recovered_colours = set()
            for candidate in sorted(final_cubes, key=lambda item: item.get("confidence", 0.0), reverse=True):
                colour = candidate.get("colour")
                if colour not in COLOURS or colour in recovered_colours:
                    continue
                # A deterministic destination is available only when this is
                # a cube already associated with a camera-confirmed recovery
                # attempt.  That history tells us exactly which top cell was
                # vacated; arbitrary probing of a full bin is deliberately
                # rejected.
                history = [item for item in report.get("residual_cube_recovery", []) if item.get("colour") == colour]
                if not history:
                    continue
                planned_centre = np.asarray(zones[colour], float)
                bins = sorted(
                    [item for item in final_bin_candidates if item.get("colour") == colour],
                    key=lambda item: float(np.linalg.norm(np.asarray(item["xy"], float) - planned_centre)),
                )
                if not bins:
                    continue
                final_bin = bins[0]
                bin_shift = float(np.linalg.norm(np.asarray(final_bin["xy"], float) - planned_centre))
                if bin_shift > 0.030:
                    continue
                cube_xy = np.asarray(candidate["xy"], float)
                observed_bin_xy = np.asarray(final_bin["xy"], float)
                if float(np.linalg.norm(cube_xy - observed_bin_xy)) <= 0.032:
                    continue

                recovery_slot = tuple(history[-1].get("recovery_slot", (0.0, 0.006, 1)))
                # The RGB contour confirms that the colored bin is present,
                # but once loaded it can merge with the truck/cubes and bias
                # its fitted centre by ~20 mm.  The supported placement pose
                # is already known from our own command and was sub-millimetre
                # accurate in the failing scene, so use that calibrated pose
                # for insertion rather than the merged contour geometry.
                bin_xy = planned_centre
                bin_yaw = math.radians(float(deck_yaw_deg + release_yaw_bias[colour]))
                cube_yaw = math.radians(float(candidate["yaw_deg"]))
                cube_spread = np.asarray((-math.sin(cube_yaw), math.cos(cube_yaw), 0.0))
                bin_spread = np.asarray((-math.sin(bin_yaw), math.cos(bin_yaw), 0.0))
                source = np.asarray((cube_xy[0], cube_xy[1], C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE))
                source_above = source + np.asarray((0.0, 0.0, 0.090))
                slot_xy = bin_xy + rotate_xy(np.asarray(recovery_slot[:2]), bin_yaw)
                deck_relative_z = float(truck.get("deck_z", TABLE_Z + DECK_HEIGHT) - C.TABLE_TOP_Z)
                destination = np.asarray((
                    slot_xy[0],
                    slot_xy[1],
                    deck_relative_z + BIN_FLOOR + C.CUBE_TIP_CLEARANCE + recovery_slot[2] * CUBE_EDGE,
                ))
                destination_above = destination + np.asarray((0.0, 0.0, 0.090))
                label = f"{colour} final observed spill"
                command(label + " approach", source_above, 0.85, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                command(label + " descend", source, 0.70, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                dwell(label + " close", 0.80, cube_jaw)
                gap = measured_jaw_gap_mm()
                accepted = gap > float(2.0 * cube_jaw * 1000.0 + 0.35)
                entry = {
                    "colour": colour,
                    "source_xy": cube_xy.tolist(),
                    "source": candidate,
                    "bin": final_bin,
                    "planned_bin_gate_mm": bin_shift * 1000.0,
                    "recovery_slot": list(recovery_slot),
                    "jaw_gap_mm": gap,
                    "accepted_contact": accepted,
                }
                if not accepted:
                    dwell(label + " empty release", 0.45, C.JAW_OPEN)
                    command(label + " empty withdraw", source_above, 0.70, C.JAW_OPEN, C.JAW_OPEN, cube_spread)
                    final_recovery_log.append(entry)
                    continue
                command(label + " lift", source_above, 0.90, cube_jaw, cube_jaw, cube_spread)
                command(label + " carry", destination_above, 1.40, cube_jaw, cube_jaw, bin_spread)
                command(label + " lower", destination, 0.90, cube_jaw, cube_jaw, bin_spread)
                dwell(label + " release", 0.65, cube_release)
                command(label + " withdraw", destination_above, 0.80, cube_release, cube_release, bin_spread)
                dwell(label + " fully open", 0.25, C.JAW_OPEN)
                recovered_colours.add(colour)
                final_recovery_log.append(entry)
            report["final_observed_spill_recovery"] = final_recovery_log
            if final_recovery_log:
                command("post-spill final retreat", home, 1.00, C.JAW_OPEN, C.JAW_OPEN, np.asarray((0.0, 1.0, 0.0)))
                dwell("post-spill payload settle", 2.00, C.JAW_OPEN)

    # ---------------- evaluator boundary: no values below feed the controller ----------------
    def evaluator_pose(name):
        position, quaternion = wrappers[name].get_world_pose()
        yaw, tilt = quat_yaw_tilt(quaternion)
        return np.asarray(position, float), yaw, tilt

    actual_deck_centre = truck_origin + rotate_xy(DECK_LOCAL_CENTRE, math.radians(truck_root_yaw))
    actual_deck_yaw_deg = DECK_LOCAL_YAW + truck_root_yaw
    actual_deck_yaw = math.radians(actual_deck_yaw_deg)
    actual_long = np.asarray((math.cos(actual_deck_yaw), math.sin(actual_deck_yaw)))
    actual_lateral = np.asarray((-math.sin(actual_deck_yaw), math.cos(actual_deck_yaw)))
    actual_zones = {colour: actual_deck_centre + offset * actual_long for colour, offset in zip(COLOURS, (-0.057, 0.0, 0.057))}
    bin_results = {}
    cube_results = []
    bins_on_deck = 0
    cubes_correct = 0
    for colour in COLOURS:
        bin_position, bin_yaw, bin_tilt = evaluator_pose(f"Bin_{colour}")
        zone_error = float(np.linalg.norm(bin_position[:2] - actual_zones[colour]) * 1000.0)
        yaw_error = (bin_yaw - actual_deck_yaw_deg + 180.0) % 360.0 - 180.0
        rel = bin_position[:2] - actual_deck_centre
        on_deck = bool(
            abs(float(np.dot(rel, actual_long))) <= DECK_LENGTH / 2.0 - BIN_OUTER / 2.0 + 0.006
            and abs(float(np.dot(rel, actual_lateral))) <= 0.005
            and abs(float(bin_position[2] - (TABLE_Z + DECK_HEIGHT))) <= 0.008
            and bin_tilt <= 8.0
        )
        bins_on_deck += int(on_deck)
        bin_results[colour] = {
            "position_m": bin_position.tolist(), "yaw_deg": bin_yaw, "tilt_deg": bin_tilt,
            "zone_error_mm": zone_error, "yaw_error_deg": yaw_error, "on_deck": on_deck,
        }
        for name, meta in truth_meta.items():
            if meta["category"] != "cube" or meta["colour"] != colour:
                continue
            cube_position, _, cube_tilt = evaluator_pose(name)
            local = rotate_xy(cube_position[:2] - bin_position[:2], -math.radians(bin_yaw))
            inside = bool(
                abs(local[0]) <= BIN_POCKET / 2.0 - CUBE_EDGE / 2.0 + 0.004
                and abs(local[1]) <= BIN_POCKET / 2.0 - CUBE_EDGE / 2.0 + 0.004
                and cube_position[2] >= bin_position[2] + BIN_FLOOR - 0.004
                and cube_position[2] <= bin_position[2] + BIN_HEIGHT + 2.5 * CUBE_EDGE
            )
            cubes_correct += int(inside)
            cube_results.append({
                "cube": name, "colour": colour, "position_m": cube_position.tolist(),
                "local_xy_m": local.tolist(), "tilt_deg": cube_tilt, "inside_matching_bin": inside,
            })

    report["evaluation"] = {
        "cubes_in_matching_bins": cubes_correct,
        "cubes_required": 12,
        "bins_on_deck": bins_on_deck,
        "bins_required": 3,
        "cube_orientation_required": False,
        "full_task_success": bool(cubes_correct == 12 and bins_on_deck == 3),
        "bins": bin_results,
        "cubes": cube_results,
        "initial_truth_evaluator_only": truth_meta,
    }
    report["action_count"] = action_count
    report["active_scan_motion_count"] = active_scan_motion_count
    report["action_count_including_active_scans"] = action_count + active_scan_motion_count
    report["tcp_path_length_m"] = path_length
    report["motion_wall_time_s"] = time.perf_counter() - motion_started
    report["frames"] = frame_number
    report["task_complete"] = True

except Exception as exc:
    import traceback

    exit_code = 1
    report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    print(report["error"]["traceback"], flush=True)
finally:
    (OUT / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "task_complete": report["task_complete"],
        "policy": report["policy"],
        "controller": report["controller"],
        "evaluation": report.get("evaluation", {}),
        "error": report["error"],
    }, indent=2), flush=True)
    app.close()

raise SystemExit(exit_code)
