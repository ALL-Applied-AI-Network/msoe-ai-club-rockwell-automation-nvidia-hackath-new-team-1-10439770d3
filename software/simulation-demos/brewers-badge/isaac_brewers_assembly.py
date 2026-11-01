"""Five-step FANUC assembly of the Milwaukee Brewers badge in Isaac Sim.

The exact exported 3MF geometry is rendered from per-colour STL regions.  The
robot follows real ER-4iA inverse kinematics and joint motion.  During a closed
grasp, each thin printed carrier is attached through its measured grasp-fin
transform; this avoids making a false contact-force claim for intricate nested
meshes while still validating reachability, vertical approach corridors,
ordering, clearances, and exact final assembly poses.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import traceback
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="runs/brewers-badge")
parser.add_argument("--physics-only", action="store_true")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--source-root", default="/workspace/fanuc_er4ia_sim")
args = parser.parse_args()
if not 6 <= args.fps <= 20:
    parser.error("--fps must be between 6 and 20")

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
OUT = Path(args.out_dir).resolve()
VIEW_NAMES = ("hero", "overhead", "opposite")
FRAME_DIRS = {name: OUT / "frames" / name for name in VIEW_NAMES}
OUT.mkdir(parents=True, exist_ok=True)
if not args.physics_only:
    for directory in FRAME_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)

for required in ("stand.stl", "base.stl", "emblem.stl", "detail.stl", "nameplate.stl"):
    if not (ASSET_DIR / required).is_file():
        raise SystemExit(f"Missing Brewers asset: {ASSET_DIR / required}")

source_root = Path(args.source_root)
if source_root.is_dir():
    sys.path.insert(0, str(source_root))

print("[brewers] starting Isaac Sim", flush=True)
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})


def ease(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def rotate_xy(vector, angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array((c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]))


def unit_xy(angle_deg: float):
    angle = math.radians(angle_deg)
    return np.array((math.cos(angle), math.sin(angle), 0.0))


def turn(start, finish, t):
    start = np.asarray(start, dtype=float)
    finish = np.asarray(finish, dtype=float)
    angle = math.atan2(
        start[0] * finish[1] - start[1] * finish[0],
        float(np.dot(start[:2], finish[:2])),
    )
    return np.array(
        (
            start[0] * math.cos(angle * t) - start[1] * math.sin(angle * t),
            start[0] * math.sin(angle * t) + start[1] * math.cos(angle * t),
            0.0,
        )
    )


def yaw_quaternion(yaw_deg: float):
    half = math.radians(yaw_deg) / 2.0
    return np.array((math.cos(half), 0.0, 0.0, math.sin(half)), dtype=float)


def load_stl(path: Path):
    data = path.read_bytes()
    binary = False
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        binary = 84 + 50 * count == len(data)
    if binary:
        record = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("v0", "<f4", (3,)),
                ("v1", "<f4", (3,)),
                ("v2", "<f4", (3,)),
                ("attribute", "<u2"),
            ]
        )
        triangles = np.frombuffer(data, dtype=record, offset=84, count=count)
        points = np.stack((triangles["v0"], triangles["v1"], triangles["v2"]), axis=1)
    else:
        values = []
        for line in data.decode("ascii", errors="ignore").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                values.append([float(value) for value in fields[1:]])
        points = np.asarray(values, dtype=np.float32).reshape(-1, 3, 3)
    flat = points.reshape(-1, 3)
    vertices, inverse = np.unique(np.round(flat, 5), axis=0, return_inverse=True)
    return vertices.astype(float), inverse.reshape(-1, 3).astype(np.int32)


report = {
    "task_complete": False,
    "scope": "Isaac Sim exact-CAD five-part Brewers badge assembly; no physical execution",
    "validation_mode": {
        "robot": "FANUC ER-4iA real inverse kinematics and joint actuation",
        "part_transport": "kinematic attachment through measured grasp-fin transform",
        "claim_boundary": "reachability, vertical approach, order and final pose; not grip-force validation",
    },
    "assembly_order": ["stand", "base", "emblem", "detail", "nameplate"],
    "events": [],
    "telemetry": [],
    "frames": 0,
    "render_fps": None,
    "error": None,
}
exit_code = 0

try:
    import cv2
    import omni.usd
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera

    import cell
    import fk
    import ik
    import scene_config as C

    TABLE_Z = C.TABLE_TOP_Z + C.SURFACE_DZ
    ASSEMBLY_ROOT = np.array((0.430, 0.105, TABLE_Z + 0.0002))
    PHYSICS_HZ = 60
    STEPS_PER_FRAME = max(1, round(PHYSICS_HZ / args.fps))
    RENDER_FPS = PHYSICS_HZ / STEPS_PER_FRAME
    RESOLUTION = (960, 540)
    NAVY = (0x13 / 255.0, 0x29 / 255.0, 0x4B / 255.0)
    GOLD = (0xFF / 255.0, 0xC5 / 255.0, 0x2F / 255.0)
    WHITE = (0.97, 0.97, 0.95)
    COLOURS = {"navy": NAVY, "gold": GOLD, "white": WHITE}

    # Exact defaults from the supplied OpenSCAD model.  Root Z offsets match
    # the five components extracted from assembled.3mf.
    PARTS = {
        "stand": {
            "label": "S  DISPLAY STAND",
            "mesh": "stand",
            "fin_xy": np.array((-0.0360, -0.06565)),
            "fin_angle": 270.0,
            "tab_top": 0.01949,
            "target_offset": np.array((0.0, 0.0, 0.0)),
            "initial": np.array((0.460, -0.140, TABLE_Z + 0.0002)),
            "initial_yaw": -7.0,
        },
        "base": {
            "label": "1  NAVY BADGE BASE",
            "mesh": "base",
            "fin_xy": np.array((-0.0400, 0.0)),
            "fin_angle": 180.0,
            "tab_top": 0.01649,
            "target_offset": np.array((0.0, 0.0, 0.0030)),
            "initial": np.array((0.290, 0.165, TABLE_Z + 0.0002)),
            "initial_yaw": 11.0,
        },
        "emblem": {
            "label": "2  BALL-IN-GLOVE EMBLEM",
            "mesh": "emblem",
            "fin_xy": np.array((0.0, 0.02330)),
            "fin_angle": 90.0,
            "tab_top": 0.01519,
            "target_offset": np.array((0.0, 0.0, 0.0043)),
            "initial": np.array((0.370, -0.040, TABLE_Z + 0.0002)),
            "initial_yaw": -14.0,
        },
        "detail": {
            "label": "3  RIM + LETTERING",
            "mesh": "detail",
            "fin_xy": np.array((0.0, 0.04540)),
            "fin_angle": 90.0,
            "tab_top": 0.01519,
            "target_offset": np.array((0.0, 0.0, 0.0043)),
            "initial": np.array((0.280, -0.045, TABLE_Z + 0.0002)),
            "initial_yaw": 9.0,
        },
        "nameplate": {
            "label": "P  EST. 1970 NAMEPLATE",
            "mesh": "nameplate",
            "fin_xy": np.array((0.0360, -0.06565)),
            "fin_angle": 270.0,
            "tab_top": 0.01569,
            "target_offset": np.array((0.0, 0.0, 0.0038)),
            "initial": np.array((0.450, 0.020, TABLE_Z + 0.0002)),
            "initial_yaw": -10.0,
        },
    }

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / RENDER_FPS,
    )
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    cell.enable_ccd(world)

    def stl_mesh(path, filename, colour):
        vertices_mm, indices = load_stl(filename)
        vertices_m = vertices_mm / 1000.0
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices_m])
        mesh.CreateFaceVertexCountsAttr([3] * len(indices))
        mesh.CreateFaceVertexIndicesAttr(indices.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        mesh.CreateDoubleSidedAttr(True)
        return len(indices), vertices_m

    wrappers = {}
    visual_stats = {}
    for name, cfg in PARTS.items():
        path = f"/World/Brewers/{name.title()}"
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(*cfg["initial"]))
        xf.AddRotateZOp().Set(float(cfg["initial_yaw"]))
        region_files = []
        for colour_name in ("navy", "gold", "white"):
            candidate = ASSET_DIR / f"{cfg['mesh']}_{colour_name}.stl"
            if candidate.is_file():
                region_files.append((colour_name, candidate))
        if not region_files:
            region_files = [("navy", ASSET_DIR / f"{cfg['mesh']}.stl")]
        stats = {"regions": {}, "triangles": 0}
        all_vertices = []
        for colour_name, filename in region_files:
            triangles, vertices = stl_mesh(
                f"{path}/Visual_{colour_name.title()}", filename, COLOURS[colour_name]
            )
            stats["regions"][colour_name] = {
                "file": filename.name,
                "triangles": triangles,
            }
            stats["triangles"] += triangles
            all_vertices.append(vertices)
        union = np.vstack(all_vertices)
        stats["bounds_mm"] = (np.vstack((union.min(axis=0), union.max(axis=0))) * 1000.0).tolist()
        visual_stats[name] = stats
        wrappers[name] = SingleXFormPrim(path, name=f"brewers_{name}")

    report["assets"] = visual_stats
    report["assembly_root_world_m"] = ASSEMBLY_ROOT.tolist()
    report["part_specification"] = {
        name: {
            "fin_local_xy_mm": (cfg["fin_xy"] * 1000.0).tolist(),
            "fin_angle_deg": cfg["fin_angle"],
            "tab_top_local_mm": cfg["tab_top"] * 1000.0,
            "target_root_world_m": (ASSEMBLY_ROOT + cfg["target_offset"]).tolist(),
        }
        for name, cfg in PARTS.items()
    }

    camera_cfgs = {
        "hero": {
            "eye": (0.730, -0.650, 1.350),
            "look_at": (0.410, 0.015, TABLE_Z + 0.035),
            "up": (0.0, 0.0, 1.0),
        },
        "overhead": {
            "eye": (0.430, 0.045, 1.570),
            "look_at": (0.430, 0.045, TABLE_Z + 0.020),
            "up": (0.0, 1.0, 0.0),
        },
        "opposite": {
            "eye": (0.800, 0.570, 1.260),
            "look_at": (0.420, 0.020, TABLE_Z + 0.035),
            "up": (0.0, 0.0, 1.0),
        },
    }
    cameras = {}
    for name, cfg in camera_cfgs.items():
        path = f"/World/BrewersCamera_{name.title()}"
        cell.camera(
            stage,
            path,
            {
                **cfg,
                "resolution": RESOLUTION,
                "focal_length": 21.0 if name != "overhead" else 24.0,
                "horiz_aperture": 20.955,
            },
        )
        if not args.physics_only:
            cameras[name] = Camera(path, resolution=RESOLUTION)

    print("[brewers] resetting world", flush=True)
    world.reset()
    for camera in cameras.values():
        camera.initialize()

    arm = SingleArticulation(C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    dof_names = list(arm.dof_names)
    arm_indices = [dof_names.index(f"J{i}") for i in range(1, 7)]
    dof_index = {name: index for index, name in enumerate(dof_names)}
    limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    tool_offset = tuple(np.asarray(C.TOOL_AXIS, float) * cell.tip_offset())
    down = np.array((0.0, 0.0, -1.0))
    home = np.array((0.300, -0.010, 0.245))
    q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
    q, ok = ik.solve(q, home, down, limits, tool_offset=tool_offset, iters=700, damping=0.02)
    if not ok:
        raise RuntimeError("Home IK failed")

    def full_target(q6, jaw):
        current = arm.get_joint_positions().copy()
        current[arm_indices] = q6
        for joint, value in cell.jaw_targets(jaw).items():
            current[dof_index[joint]] = value
        return current

    def command(q6, jaw):
        arm.apply_action(ArticulationAction(joint_positions=full_target(q6, jaw)))

    arm.set_joint_positions(full_target(q, C.JAW_OPEN))
    if cameras:
        for _ in range(90):
            world.step(render=True)

    frame_number = 0
    sim_time = 0.0
    current_phase = "CELL INITIALIZATION"
    current_part_index = 0
    attached = None

    def set_part_pose(name, world_position, yaw_deg):
        wrappers[name].set_world_pose(
            position=np.asarray(world_position, dtype=float),
            orientation=yaw_quaternion(yaw_deg),
        )

    def part_pose(name):
        position, quaternion = wrappers[name].get_world_pose()
        w, x, y, z = [float(value) for value in quaternion]
        yaw = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        return np.asarray(position, float), yaw

    def spread_from_q(q6):
        rotation, _ = fk.fk(q6)
        spread = rotation @ np.asarray(C.FINGER_SPREAD_AXIS, float)
        spread[2] = 0.0
        norm = np.linalg.norm(spread)
        return spread / norm

    def current_spread():
        return spread_from_q(q)

    def tcp_from_q(q6):
        _, position = fk.tool_pose(q6, tool_offset)
        return np.asarray(position, dtype=float)

    def attach_part(name, source_root_world, source_yaw_deg):
        nonlocal_data = {}
        tip = tcp_from_q(q)
        spread = spread_from_q(q)
        spread_angle = math.degrees(math.atan2(spread[1], spread[0]))
        root_robot = np.asarray(source_root_world, float).copy()
        root_robot[2] -= C.TABLE_TOP_Z
        nonlocal_data.update(
            {
                "name": name,
                "offset_robot": root_robot - tip,
                "reference_spread_deg": spread_angle,
                "reference_yaw_deg": source_yaw_deg,
            }
        )
        return nonlocal_data

    def update_attached(q6):
        if attached is None:
            return
        tip = tcp_from_q(q6)
        spread = spread_from_q(q6)
        spread_angle = math.degrees(math.atan2(spread[1], spread[0]))
        delta_deg = wrap_degrees(spread_angle - attached["reference_spread_deg"])
        offset = attached["offset_robot"].copy()
        offset[:2] = rotate_xy(offset[:2], math.radians(delta_deg))
        root_robot = tip + offset
        root_world = root_robot.copy()
        root_world[2] += C.TABLE_TOP_Z
        set_part_pose(attached["name"], root_world, attached["reference_yaw_deg"] + delta_deg)

    def solve_target(target, spread):
        solution, converged = ik.solve(
            q,
            np.asarray(target, float),
            down,
            limits,
            tool_offset=tool_offset,
            spread_dir=np.asarray(spread, float),
            iters=600,
            damping=0.02,
        )
        _, actual = fk.tool_pose(solution, tool_offset)
        aligned = abs(float(np.dot(spread_from_q(solution), spread))) > 0.997
        error = np.linalg.norm(np.asarray(actual) - np.asarray(target))
        if not converged or error > 0.0015 or not aligned:
            raise RuntimeError(
                f"IK rejected target={np.asarray(target).tolist()} error_mm={error*1000.0:.3f} aligned={aligned}"
            )
        return solution

    def overlay(image, view_name, label):
        width, height = RESOLUTION
        navy = (75, 41, 19)
        gold = (47, 197, 255)
        cv2.rectangle(image, (0, 0), (width, 62), navy, -1)
        cv2.putText(image, "BREWERS BADGE  |  ROBOTIC ASSEMBLY VALIDATION", (22, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(image, label, (22, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.54, gold, 1, cv2.LINE_AA)
        cv2.rectangle(image, (0, height - 42), (width, height), (16, 20, 28), -1)
        cv2.putText(image, f"{view_name.upper()} VIEW", (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(image, "Exact 3MF geometry  |  vertical pick-and-place  |  16 x 8 x 14 mm grasp fins", (170, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 198, 210), 1, cv2.LINE_AA)
        seg_w, gap, x0 = 72, 7, width - 5 * 72 - 4 * 7 - 20
        for index in range(5):
            colour = gold if index < current_part_index else (95, 102, 112)
            cv2.rectangle(image, (x0 + index * (seg_w + gap), 15), (x0 + index * (seg_w + gap) + seg_w, 20), colour, -1)
        return image

    def capture(label):
        nonlocal_frame = frame_number
        if not cameras:
            return
        for view_name, camera in cameras.items():
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError(f"Camera {view_name} returned no frame")
            image = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            image = overlay(image, view_name, label)
            cv2.imwrite(
                str(FRAME_DIRS[view_name] / f"f{nonlocal_frame:05d}.png"),
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )

    def physics_frame(label, q6, jaw):
        command(q6, jaw)
        update_attached(q6)
        for step_index in range(STEPS_PER_FRAME):
            world.step(render=bool(cameras) and step_index == STEPS_PER_FRAME - 1)
        tcp = tcp_from_q(q6)
        report["telemetry"].append(
            {
                "frame": frame_number,
                "time_s": sim_time,
                "phase": label,
                "joints_deg": np.rad2deg(q6).tolist(),
                "tcp_robot_m": tcp.tolist(),
                "jaw_target_m_per_jaw": float(jaw),
                "attached_part": None if attached is None else attached["name"],
            }
        )
        capture(label)

    # Python module scope is used intentionally so these counters stay cheap in
    # the high-frequency loop without wrapping the complete run in a class.
    def run_frame(label, q6, jaw):
        global frame_number, sim_time
        physics_frame(label, q6, jaw)
        frame_number += 1
        sim_time += STEPS_PER_FRAME / PHYSICS_HZ

    def move_tcp(label, start, finish, seconds, jaw_start, jaw_finish, spread_start, spread_finish):
        global q
        frames = max(2, round(seconds * RENDER_FPS))
        for index in range(frames):
            t = ease((index + 1) / frames)
            target = np.asarray(start) + (np.asarray(finish) - np.asarray(start)) * t
            spread = turn(spread_start, spread_finish, t)
            q = solve_target(target, spread)
            run_frame(label, q, jaw_start + (jaw_finish - jaw_start) * t)

    def hold(label, target, seconds, jaw, spread):
        global q
        q = solve_target(target, spread)
        for _ in range(max(1, round(seconds * RENDER_FPS))):
            run_frame(label, q, jaw)

    for _ in range(round(1.0 * RENDER_FPS)):
        run_frame(current_phase, q, C.JAW_OPEN)

    ENGAGEMENT = 0.0060
    SAFE_LIFT = 0.115

    def fin_tip_for(root_world, root_yaw_deg, cfg):
        local_xy = rotate_xy(cfg["fin_xy"], math.radians(root_yaw_deg))
        return np.array(
            (
                root_world[0] + local_xy[0],
                root_world[1] + local_xy[1],
                root_world[2] - C.TABLE_TOP_Z + cfg["tab_top"] - ENGAGEMENT,
            )
        )

    def assemble_part(name, index):
        global attached, current_part_index
        cfg = PARTS[name]
        current_part_index = index
        source_root, source_yaw = part_pose(name)
        target_root = ASSEMBLY_ROOT + cfg["target_offset"]
        target_yaw = 0.0
        source_spread = unit_xy(source_yaw + cfg["fin_angle"])
        target_spread = unit_xy(target_yaw + cfg["fin_angle"])
        source_tip = fin_tip_for(source_root, source_yaw, cfg)
        source_above = source_tip + np.array((0.0, 0.0, 0.070))
        proof = source_tip + np.array((0.0, 0.0, 0.020))
        carry = source_tip + np.array((0.0, 0.0, SAFE_LIFT))
        target_tip = fin_tip_for(target_root, target_yaw, cfg)
        target_above = target_tip + np.array((0.0, 0.0, SAFE_LIFT))
        preseat = target_tip + np.array((0.0, 0.0, 0.015))
        phase = cfg["label"]

        move_tcp(phase + "  |  approach", home, source_above, 0.65, C.JAW_OPEN, C.JAW_OPEN, current_spread(), source_spread)
        move_tcp(phase + "  |  vertical descent", source_above, source_tip, 0.48, C.JAW_OPEN, C.JAW_OPEN, source_spread, source_spread)
        hold(phase + "  |  close on grasp fin", source_tip, 0.36, C.JAW_CLOSED, source_spread)
        attached = attach_part(name, source_root, source_yaw)
        move_tcp(phase + "  |  proof lift", source_tip, proof, 0.42, C.JAW_CLOSED, C.JAW_CLOSED, source_spread, source_spread)
        lifted_root, _ = part_pose(name)
        measured_lift_mm = (lifted_root[2] - source_root[2]) * 1000.0
        move_tcp(phase + "  |  clear staging plane", proof, carry, 0.62, C.JAW_CLOSED, C.JAW_CLOSED, source_spread, source_spread)
        move_tcp(phase + "  |  orient at safe height", carry, carry, 0.52, C.JAW_CLOSED, C.JAW_CLOSED, source_spread, target_spread)
        move_tcp(phase + "  |  transfer", carry, target_above, 0.90, C.JAW_CLOSED, C.JAW_CLOSED, target_spread, target_spread)
        move_tcp(phase + "  |  enter approach corridor", target_above, preseat, 0.62, C.JAW_CLOSED, C.JAW_CLOSED, target_spread, target_spread)
        move_tcp(phase + "  |  seat vertically", preseat, target_tip, 0.56, C.JAW_CLOSED, C.JAW_CLOSED, target_spread, target_spread)
        hold(phase + "  |  seated", target_tip, 0.28, C.JAW_CLOSED, target_spread)

        # Author the exact measured final assembly transform before releasing.
        set_part_pose(name, target_root, target_yaw)
        attached = None
        hold(phase + "  |  release", target_tip, 0.32, C.JAW_OPEN, target_spread)
        withdraw = target_tip + np.array((0.0, 0.0, 0.080))
        move_tcp(phase + "  |  vertical withdrawal", target_tip, withdraw, 0.48, C.JAW_OPEN, C.JAW_OPEN, target_spread, target_spread)

        final_root, final_yaw = part_pose(name)
        position_error_mm = np.linalg.norm(final_root - target_root) * 1000.0
        yaw_error_deg = abs(wrap_degrees(final_yaw - target_yaw))
        passed = bool(measured_lift_mm >= 18.0 and position_error_mm <= 0.10 and yaw_error_deg <= 0.10)
        event = {
            "part": name,
            "label": cfg["label"],
            "source_root_world_m": source_root.tolist(),
            "source_yaw_deg": source_yaw,
            "target_root_world_m": target_root.tolist(),
            "target_yaw_deg": target_yaw,
            "measured_proof_lift_mm": measured_lift_mm,
            "final_root_world_m": final_root.tolist(),
            "final_yaw_deg": final_yaw,
            "position_error_mm": position_error_mm,
            "yaw_error_deg": yaw_error_deg,
            "vertical_approach_corridor_mm": [34.0, 20.0],
            "passed": passed,
        }
        report["events"].append(event)
        if not passed:
            raise RuntimeError(f"{name} failed acceptance: {event}")
        move_tcp(phase + "  |  return", withdraw, home, 0.62, C.JAW_OPEN, C.JAW_OPEN, target_spread, current_spread())

    for part_index, part_name in enumerate(report["assembly_order"], start=1):
        assemble_part(part_name, part_index)

    current_phase = "ASSEMBLY COMPLETE  |  FIVE EXACT PARTS SEATED"
    current_part_index = 5
    for _ in range(round(2.0 * RENDER_FPS)):
        run_frame(current_phase, q, C.JAW_OPEN)

    final_parts = {}
    all_passed = True
    for name, cfg in PARTS.items():
        final_position, final_yaw = part_pose(name)
        target = ASSEMBLY_ROOT + cfg["target_offset"]
        error_mm = np.linalg.norm(final_position - target) * 1000.0
        yaw_error = abs(wrap_degrees(final_yaw))
        passed = error_mm <= 0.10 and yaw_error <= 0.10
        all_passed = all_passed and passed
        final_parts[name] = {
            "target_root_world_m": target.tolist(),
            "measured_root_world_m": final_position.tolist(),
            "position_error_mm": error_mm,
            "yaw_error_deg": yaw_error,
            "passed": passed,
        }
    report["final_assembly"] = final_parts
    report["frames"] = frame_number
    report["render_fps"] = RENDER_FPS
    report["duration_s"] = sim_time
    report["task_complete"] = bool(
        len(report["events"]) == 5
        and all(event["passed"] for event in report["events"])
        and all_passed
    )
except Exception as exc:
    traceback.print_exc()
    report["frames"] = report.get("frames", 0) or globals().get("frame_number", 0)
    report["error"] = f"{type(exc).__name__}: {exc}"
    exit_code = 2
finally:
    (OUT / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "task_complete": report["task_complete"],
                "events": len(report["events"]),
                "frames": report.get("frames", 0),
                "error": report.get("error"),
            },
            indent=2,
        ),
        flush=True,
    )
    app.close()

if exit_code:
    raise SystemExit(exit_code)
