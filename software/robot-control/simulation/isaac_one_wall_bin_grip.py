"""Physics proof for gripping one wall of the official hackathon bin.

The experiment uses the starter's articulated FANUC ER-4iA and SMC gripper,
creates a dynamic compound bin at the official 50 x 50 x 55 mm dimensions,
straddles only the south wall, closes the real simulated jaws, lifts 10 mm,
holds, lowers, and releases.  It writes a machine-readable acceptance report
and optional side-camera frames.

This controls Isaac Sim only.  It contains no physical-robot transport.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from one_wall_bin_model import BinSpec, GripPlan, assess_trial, default_manifest


parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/one_wall_bin/report.json")
parser.add_argument("--frames-dir", default=None)
parser.add_argument("--mass-kg", type=float, default=0.133)
parser.add_argument("--friction", type=float, default=0.90)
parser.add_argument("--lift-mm", type=float, default=10.0)
parser.add_argument("--insertion-mm", type=float, default=14.0)
parser.add_argument(
    "--jaw-target-mm",
    type=float,
    default=0.0,
    help="per-jaw opening target in millimetres (0=fully closed, 6=fully open)",
)
parser.add_argument(
    "--jaw-max-force-n",
    type=float,
    default=180.0,
    help="maximum force of each simulated prismatic jaw drive",
)
parser.add_argument(
    "--grip-x-error-mm",
    type=float,
    default=0.0,
    help="deliberate grasp-centre error along the bin wall",
)
parser.add_argument(
    "--grip-y-error-mm",
    type=float,
    default=0.0,
    help="deliberate grasp-centre error normal to the bin wall",
)
parser.add_argument(
    "--grip-yaw-error-deg",
    type=float,
    default=0.0,
    help="deliberate wrist alignment error relative to the detected wall",
)
parser.add_argument("--bin-x-mm", type=float, default=450.0)
parser.add_argument("--bin-y-mm", type=float, default=0.0)
parser.add_argument("--bin-yaw-deg", type=float, default=0.0)
parser.add_argument("--physics-only", action="store_true")
parser.add_argument(
    "--validate-perception",
    action="store_true",
    help="render calibrated table RGB/depth and verify the fused grasp target",
)
args = parser.parse_args()

if not 0.03 <= args.mass_kg <= 0.40:
    parser.error("--mass-kg must be between 0.03 and 0.40")
if not 0.10 <= args.friction <= 2.0:
    parser.error("--friction must be between 0.10 and 2.0")
if not 1.0 <= args.lift_mm <= 10.0:
    parser.error("--lift-mm must be between 1 and 10")
if not 12.0 <= args.insertion_mm <= 23.0:
    parser.error("--insertion-mm must be between 12 and 23")
if not 0.0 <= args.jaw_target_mm <= 6.0:
    parser.error("--jaw-target-mm must be between 0 and 6")
if not 1.0 <= args.jaw_max_force_n <= 180.0:
    parser.error("--jaw-max-force-n must be between 1 and 180")
if abs(args.grip_x_error_mm) > 12.0 or abs(args.grip_y_error_mm) > 6.0:
    parser.error("grip errors exceed the bounded robustness-test envelope")
if abs(args.grip_yaw_error_deg) > 10.0:
    parser.error("grip yaw error exceeds the bounded robustness-test envelope")
if not 300.0 <= args.bin_x_mm <= 560.0:
    parser.error("--bin-x-mm must remain inside the approved work envelope")
if not -180.0 <= args.bin_y_mm <= 180.0:
    parser.error("--bin-y-mm must remain inside the approved work envelope")
if abs(args.bin_yaw_deg) > 30.0:
    parser.error("--bin-yaw-deg must be between -30 and 30 degrees")

output_path = Path(args.out).resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)
frames_dir = (
    Path(args.frames_dir).resolve()
    if args.frames_dir
    else output_path.parent / "frames"
)
if not args.physics_only or args.validate_perception:
    frames_dir.mkdir(parents=True, exist_ok=True)

spec = BinSpec()
plan = GripPlan(
    bin_x=args.bin_x_mm / 1000.0,
    bin_y=args.bin_y_mm / 1000.0,
    insertion_depth=args.insertion_mm / 1000.0,
    lift_height=args.lift_mm / 1000.0,
)

print("[one-wall-bin] importing Isaac Sim", flush=True)
from isaacsim import SimulationApp

print("[one-wall-bin] creating headless SimulationApp", flush=True)
app = SimulationApp({"headless": True})
print("[one-wall-bin] SimulationApp ready; building scene", flush=True)
report = {
    "task_complete": False,
    "experiment": "one-wall bin straddle, 10 mm micro-lift",
    "scope": "Isaac Sim physics evidence only; physical execution is not authorized by this report",
    "configuration": {
        "mass_kg": args.mass_kg,
        "friction": args.friction,
        "lift_mm": args.lift_mm,
        "insertion_mm": args.insertion_mm,
        "jaw_target_mm_per_jaw": args.jaw_target_mm,
        "jaw_max_force_n_per_finger": args.jaw_max_force_n,
        "grip_x_error_mm": args.grip_x_error_mm,
        "grip_y_error_mm": args.grip_y_error_mm,
        "grip_yaw_error_deg": args.grip_yaw_error_deg,
        "bin_x_mm": args.bin_x_mm,
        "bin_y_mm": args.bin_y_mm,
        "bin_yaw_deg": args.bin_yaw_deg,
        "rendered": (not args.physics_only) or args.validate_perception,
        "perception_validation": args.validate_perception,
    },
    "model": default_manifest(),
    "phases": {},
    "timeline_10hz": [],
}
exit_code = 0

try:
    print("[one-wall-bin] importing Isaac APIs and starter modules", flush=True)
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

    print("[one-wall-bin] constructing world and ER-4iA cell", flush=True)
    # The starter's generic 180 N drive cap is intentionally conservative for
    # motion testing, but the MHZL2-16D catalog rates external gripping at
    # 34 N per finger at 0.5 MPa and L=20 mm.  Make the cap explicit so the
    # physical-force envelope can be swept without pretending partial jaw
    # positioning is available on the binary pneumatic hardware.
    C.JAW_MAX_FORCE = float(args.jaw_max_force_n)
    PHYSICS_HZ = 60
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / 30.0,
    )
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    print("[one-wall-bin] cell built; adding dynamic bin", flush=True)

    # Use the same coefficient on the bin and soft tips so the material-combine
    # rule cannot silently turn a low-friction sweep into the nominal 1.5 value.
    soft_material = UsdPhysics.MaterialAPI.Apply(
        stage.GetPrimAtPath(cell.SOFT_TIP_MAT_PATH)
    )
    soft_material.CreateStaticFrictionAttr().Set(float(args.friction))
    soft_material.CreateDynamicFrictionAttr().Set(float(args.friction * 0.90))
    PhysxSchema.PhysxMaterialAPI.Apply(
        stage.GetPrimAtPath(cell.SOFT_TIP_MAT_PATH)
    ).CreateFrictionCombineModeAttr().Set("average")

    bin_material_path = "/World/Physics/OneWallBinMaterial"
    bin_material_prim = UsdGeom.Scope.Define(stage, bin_material_path).GetPrim()
    bin_material = UsdPhysics.MaterialAPI.Apply(bin_material_prim)
    bin_material.CreateStaticFrictionAttr().Set(float(args.friction))
    bin_material.CreateDynamicFrictionAttr().Set(float(args.friction * 0.90))
    bin_material.CreateRestitutionAttr().Set(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(
        bin_material_prim
    ).CreateFrictionCombineModeAttr().Set("average")

    root_path = "/World/TestBin"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    UsdGeom.Xformable(root).AddTranslateOp().Set(
        Gf.Vec3d(
            plan.bin_x,
            plan.bin_y,
            C.TABLE_TOP_Z + plan.table_z_robot,
        )
    )
    if args.bin_yaw_deg:
        UsdGeom.Xformable(root).AddRotateZOp().Set(float(args.bin_yaw_deg))
    cell.make_dynamic(root, args.mass_kg, collide_self=False)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateCenterOfMassAttr().Set(
        Gf.Vec3f(0.0, 0.0, float(spec.center_of_mass_z))
    )
    PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateEnableCCDAttr().Set(True)

    green = (0.08, 0.58, 0.18)

    def compound_box(name, size, pos):
        prim = cell.box(
            stage,
            f"{root_path}/{name}",
            size,
            pos,
            green,
            kinematic=True,
            collision=True,
            mat=None,
        )
        relation = UsdShade.MaterialBindingAPI.Apply(prim)
        relation.Bind(
            UsdShade.Material(bin_material_prim),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        cell.small_contact_offsets(prim)
        return prim

    floor_h = spec.floor_height
    compound_box(
        "Floor",
        (spec.outer_x, spec.outer_y, floor_h),
        (0.0, 0.0, floor_h / 2.0),
    )
    wall_z = floor_h + spec.pocket_depth / 2.0
    compound_box(
        "WallSouth",
        (spec.outer_x, spec.wall_y, spec.pocket_depth),
        (0.0, -spec.outer_y / 2.0 + spec.wall_y / 2.0, wall_z),
    )
    compound_box(
        "WallNorth",
        (spec.outer_x, spec.wall_y, spec.pocket_depth),
        (0.0, spec.outer_y / 2.0 - spec.wall_y / 2.0, wall_z),
    )
    compound_box(
        "WallWest",
        (spec.wall_x, spec.pocket_y, spec.pocket_depth),
        (-spec.outer_x / 2.0 + spec.wall_x / 2.0, 0.0, wall_z),
    )
    compound_box(
        "WallEast",
        (spec.wall_x, spec.pocket_y, spec.pocket_depth),
        (spec.outer_x / 2.0 - spec.wall_x / 2.0, 0.0, wall_z),
    )

    bin_prim = SingleXFormPrim(prim_path=root_path, name="one_wall_test_bin")
    side_camera = None
    table_rgb_camera = None
    table_depth_camera = None
    wrist_depth_camera = None
    if not args.physics_only or args.validate_perception:
        side_camera = Camera(
            prim_path=cell.SIDE_CAM_PATH,
            resolution=(848, 480) if args.validate_perception else C.SIDE_CAM["resolution"],
        )
    if args.validate_perception:
        table_rgb_camera = Camera(
            prim_path=cell.TABLE_CAM_PATH,
            resolution=(640, 360),
        )
        calibration_path = Path(__file__).resolve().parent / "calibration" / "cameras.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        depth_cfg = calibration["table_depth"]
        transform = np.asarray(depth_cfg["T_base_cam"], dtype=float)
        eye = transform[:3, 3] + np.array((0.0, 0.0, C.TABLE_TOP_Z))
        forward = transform[:3, :3] @ np.array((0.0, 0.0, 1.0))
        up = transform[:3, :3] @ np.array((0.0, -1.0, 0.0))
        depth_path = "/World/TableDepthPolicy"
        depth_camera_prim = cell.camera(
            stage,
            depth_path,
            {
                "eye": tuple(eye),
                "look_at": tuple(eye + forward),
                "up": tuple(up),
                "resolution": tuple(depth_cfg["size"]),
                "focal_length": float(
                    depth_cfg["fx"] * C.TABLE_CAM["horiz_aperture"] / depth_cfg["size"][0]
                ),
                "horiz_aperture": C.TABLE_CAM["horiz_aperture"],
            },
        )
        # USD cameras default to a centred principal point.  Reproduce the
        # organizer's measured RealSense offsets so the rendered image can be
        # back-projected through the unmodified official calibration file.
        width, height = depth_cfg["size"]
        horizontal_aperture = float(C.TABLE_CAM["horiz_aperture"])
        vertical_aperture = horizontal_aperture * height / width
        depth_camera_schema = UsdGeom.Camera(depth_camera_prim)
        depth_camera_schema.CreateHorizontalApertureOffsetAttr().Set(
            -(float(depth_cfg["cx"]) - width / 2.0)
            * horizontal_aperture
            / width
        )
        depth_camera_schema.CreateVerticalApertureOffsetAttr().Set(
            (float(depth_cfg["cy"]) - height / 2.0)
            * vertical_aperture
            / height
        )
        table_depth_camera = Camera(
            prim_path=depth_path,
            resolution=tuple(depth_cfg["size"]),
        )

        wrist_cfg = calibration["wrist_depth"]
        wrist_transform = np.asarray(wrist_cfg["T_flange_cam"], dtype=float)
        wrist_eye = wrist_transform[:3, 3]
        wrist_forward = wrist_transform[:3, :3] @ np.array((0.0, 0.0, 1.0))
        wrist_up = wrist_transform[:3, :3] @ np.array((0.0, -1.0, 0.0))
        wrist_path = f"{C.ROBOT_PRIM}/J6_link/WristDepthPolicy"
        wrist_camera_prim = cell.camera(
            stage,
            wrist_path,
            {
                "eye": tuple(wrist_eye),
                "look_at": tuple(wrist_eye + wrist_forward),
                "up": tuple(wrist_up),
                "resolution": tuple(wrist_cfg["size"]),
                "focal_length": float(
                    wrist_cfg["fx"]
                    * C.WRIST_CAM["horiz_aperture"]
                    / wrist_cfg["size"][0]
                ),
                "horiz_aperture": C.WRIST_CAM["horiz_aperture"],
            },
        )
        wrist_width, wrist_height = wrist_cfg["size"]
        wrist_horizontal_aperture = float(C.WRIST_CAM["horiz_aperture"])
        wrist_vertical_aperture = (
            wrist_horizontal_aperture * wrist_height / wrist_width
        )
        wrist_camera_schema = UsdGeom.Camera(wrist_camera_prim)
        wrist_camera_schema.CreateHorizontalApertureOffsetAttr().Set(
            -(float(wrist_cfg["cx"]) - wrist_width / 2.0)
            * wrist_horizontal_aperture
            / wrist_width
        )
        wrist_camera_schema.CreateVerticalApertureOffsetAttr().Set(
            (float(wrist_cfg["cy"]) - wrist_height / 2.0)
            * wrist_vertical_aperture
            / wrist_height
        )
        wrist_depth_camera = Camera(
            prim_path=wrist_path,
            resolution=tuple(wrist_cfg["size"]),
        )

    cell.enable_ccd(world)
    print("[one-wall-bin] resetting physics world", flush=True)
    world.reset()
    print("[one-wall-bin] physics world ready", flush=True)
    if side_camera is not None:
        side_camera.initialize()
    if table_rgb_camera is not None:
        table_rgb_camera.initialize()
    if table_depth_camera is not None:
        table_depth_camera.initialize()
        table_depth_camera.add_distance_to_image_plane_to_frame()
    if wrist_depth_camera is not None:
        wrist_depth_camera.initialize()
        wrist_depth_camera.add_distance_to_image_plane_to_frame()

    render_enabled = any(
        camera is not None
        for camera in (
            side_camera,
            table_rgb_camera,
            table_depth_camera,
            wrist_depth_camera,
        )
    )

    arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    dof_names = list(arm.dof_names)
    arm_indices = [dof_names.index(f"J{i}") for i in range(1, 7)]
    dof_index = {name: index for index, name in enumerate(dof_names)}
    limits = np.deg2rad(
        [C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]
    )
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    tool_offset = tuple(np.asarray(C.TOOL_AXIS, float) * cell.tip_offset())
    approach_axis = (0.0, 0.0, -1.0)
    yaw_error_rad = math.radians(args.bin_yaw_deg + args.grip_yaw_error_deg)
    spread_axis = (
        -math.sin(yaw_error_rad),
        math.cos(yaw_error_rad),
        0.0,
    )

    safe_tip = np.array((0.300, 0.000, 0.240), float)
    grasp_error = np.array(
        (args.grip_x_error_mm, args.grip_y_error_mm, 0.0), float
    ) / 1000.0
    bin_yaw_rad = math.radians(args.bin_yaw_deg)
    south_normal = np.array(
        (math.sin(bin_yaw_rad), -math.cos(bin_yaw_rad)), dtype=float
    )
    ground_truth_wall = (
        np.array((plan.bin_x, plan.bin_y), dtype=float)
        + south_normal * (spec.outer_y / 2.0 - spec.wall_y / 2.0)
    )
    grip_tip = np.array(plan.grip_tip(spec), float) + grasp_error
    grip_tip[:2] = ground_truth_wall + grasp_error[:2]
    approach_tip = grip_tip + np.array((0.0, 0.0, plan.approach_clearance))
    lifted_tip = grip_tip + np.array((0.0, 0.0, plan.lift_height))
    q_seed = np.deg2rad(
        [C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)]
    )
    jaw_grip = args.jaw_target_mm / 1000.0
    q, converged = ik.solve(
        q_seed,
        safe_tip,
        approach_axis,
        limits,
        tool_offset=tool_offset,
        spread_dir=spread_axis,
        iters=500,
        damping=0.02,
    )
    if not converged:
        raise RuntimeError("Safe-pose IK did not converge")

    def full_target(q6, jaw):
        current = arm.get_joint_positions().copy()
        current[arm_indices] = q6
        for joint_name, value in cell.jaw_targets(jaw).items():
            current[dof_index[joint_name]] = value
        return current

    def command(q6, jaw):
        arm.apply_action(
            ArticulationAction(joint_positions=full_target(q6, jaw))
        )

    arm.set_joint_positions(full_target(q, C.JAW_OPEN))
    for _ in range(120):
        command(q, C.JAW_OPEN)
        world.step(render=render_enabled)

    frame_number = 0
    physics_step = 0
    phase_samples = {}

    def quaternion_tilt_deg(quat):
        w, x, y, z = [float(v) for v in quat]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return math.degrees(math.acos(np.clip(up_z, -1.0, 1.0)))

    def state_sample(phase):
        global physics_step
        measured = arm.get_joint_positions()
        _, tip_robot = fk.tool_pose(measured[arm_indices], tool_offset)
        tip_world = np.asarray(tip_robot) + np.array((0.0, 0.0, C.TABLE_TOP_Z))
        bin_position, bin_orientation = bin_prim.get_world_pose()
        jaw_gap_mm = (
            2.0 * C.JAW_OPEN
            - abs(measured[dof_index["jaw_l"]])
            - abs(measured[dof_index["jaw_r"]])
        ) * 1000.0
        sample = {
            "time_s": physics_step / PHYSICS_HZ,
            "phase": phase,
            "tip_robot_m": np.asarray(tip_robot).tolist(),
            "bin_origin_world_m": np.asarray(bin_position).tolist(),
            "bin_tilt_deg": quaternion_tilt_deg(bin_orientation),
            "jaw_gap_mm": float(jaw_gap_mm),
            "bin_minus_tip_z_mm": float(
                (float(bin_position[2]) - float(tip_world[2])) * 1000.0
            ),
        }
        phase_samples.setdefault(phase, []).append(sample)
        if physics_step % 6 == 0:
            report["timeline_10hz"].append(sample)
        return sample

    def capture(phase):
        global frame_number
        if side_camera is None or args.physics_only or physics_step % 2:
            return
        rgba = side_camera.get_rgba()
        if rgba is None:
            return
        bgr = cv2.cvtColor(
            rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.putText(
            bgr,
            phase,
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (40, 240, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(frames_dir / f"f{frame_number:05d}.png"), bgr)
        frame_number += 1

    def step(phase, q6, jaw):
        global physics_step
        command(q6, jaw)
        world.step(render=render_enabled)
        physics_step += 1
        sample = state_sample(phase)
        capture(phase)
        return sample

    def move_tcp(phase, start, finish, duration_s, jaw_start, jaw_finish):
        global q
        frames = max(1, round(duration_s * PHYSICS_HZ))
        failures = 0
        for index in range(frames):
            t = (index + 1) / frames
            smooth = t * t * (3.0 - 2.0 * t)
            target = start + (finish - start) * smooth
            q_next, ok = ik.solve(
                q,
                target,
                approach_axis,
                limits,
                tool_offset=tool_offset,
                spread_dir=spread_axis,
                iters=300,
                damping=0.02,
            )
            _, actual = fk.tool_pose(q_next, tool_offset)
            error = float(np.linalg.norm(actual - target))
            if not ok or error > 0.001:
                failures += 1
                raise RuntimeError(
                    f"IK failed during {phase} at frame {index}: "
                    f"converged={ok}, error_mm={error * 1000:.3f}"
                )
            q = q_next
            jaw = jaw_start + (jaw_finish - jaw_start) * smooth
            step(phase, q, jaw)
        report["phases"][phase] = {
            "duration_s": duration_s,
            "ik_failures": failures,
            "target_tip_robot_m": finish.tolist(),
            "last_sample": phase_samples[phase][-1],
        }

    def hold(phase, target, duration_s, jaw):
        global q
        q_hold, ok = ik.solve(
            q,
            target,
            approach_axis,
            limits,
            tool_offset=tool_offset,
            spread_dir=spread_axis,
            iters=300,
            damping=0.02,
        )
        if not ok:
            raise RuntimeError(f"IK failed before {phase}")
        q = q_hold
        for _ in range(max(1, round(duration_s * PHYSICS_HZ))):
            step(phase, q, jaw)
        report["phases"][phase] = {
            "duration_s": duration_s,
            "ik_failures": 0,
            "target_tip_robot_m": target.tolist(),
            "last_sample": phase_samples[phase][-1],
        }

    # Let the dynamic bin settle on the table before establishing its baseline.
    hold("settle", safe_tip, 1.0, C.JAW_OPEN)
    initial = phase_samples["settle"][-1]
    side_lift_tracker = None
    rgb_detection = None
    if args.validate_perception:
        from robot_control.bin_vision import locate_green_bin
        from calibration.geometry import depth_to_base, load as load_calibration
        from robot_control.depth_vision import detect_depth_clusters
        from robot_control.sensor_fusion import fuse_bin_pose
        from robot_control.side_vision import SideLiftTracker, locate_green_region

        rgba = table_rgb_camera.get_rgba()
        depth_frame = table_depth_camera.get_current_frame().get(
            "distance_to_image_plane"
        )
        if rgba is None or depth_frame is None:
            raise RuntimeError("calibrated RGB/depth render was unavailable")
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
        depth_m = np.asarray(depth_frame, dtype=np.float64)
        valid_depth = np.isfinite(depth_m) & (depth_m > 0)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid_depth] = np.clip(
            np.rint(depth_m[valid_depth] * 1000.0), 1, 65535
        ).astype(np.uint16)
        rgb_detection = locate_green_bin(rgb)
        initial_side_rgba = side_camera.get_rgba()
        if initial_side_rgba is None:
            raise RuntimeError("side RGB render was unavailable")
        initial_side_region = locate_green_region(
            np.asarray(initial_side_rgba[:, :, :3], dtype=np.uint8)
        )
        side_lift_tracker = SideLiftTracker(initial_side_region)
        depth_clusters = detect_depth_clusters(
            depth_mm,
            "table",
            stride=1,
            minimum_points=30,
            minimum_height_m=0.006,
        )
        # Persist raw sensor evidence and diagnostics before fusion.  A failed
        # run must remain inspectable instead of collapsing into one exception
        # string; this is particularly useful when a foreground component is
        # rejected because it is merged, too large, or outside the workspace.
        cv2.imwrite(
            str(frames_dir / "table_rgb.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        depth_preview = np.zeros(depth_mm.shape, dtype=np.uint8)
        if np.any(depth_mm):
            low, high = np.percentile(depth_mm[depth_mm > 0], (2, 98))
            if high > low:
                scaled = np.clip((depth_mm.astype(float) - low) / (high - low), 0, 1)
                depth_preview = np.rint(255 * (1 - scaled)).astype(np.uint8)
                depth_preview[depth_mm == 0] = 0
        cv2.imwrite(
            str(frames_dir / "table_depth.png"),
            cv2.applyColorMap(depth_preview, cv2.COLORMAP_TURBO),
        )
        np.save(frames_dir / "table_depth_mm.npy", depth_mm)
        raw_points = depth_to_base(
            load_calibration()["table_depth"], depth_mm, stride=1
        )
        finite_points = raw_points[np.all(np.isfinite(raw_points), axis=1)]
        report["perception_diagnostics"] = {
            "rgb_detection": rgb_detection.to_dict(),
            "initial_side_region": initial_side_region.to_dict(),
            "depth_shape": list(depth_mm.shape),
            "valid_depth_pixels": int(np.count_nonzero(depth_mm)),
            "depth_mm_percentiles": (
                [float(value) for value in np.percentile(depth_mm[depth_mm > 0], (0, 2, 50, 98, 100))]
                if np.any(depth_mm)
                else []
            ),
            "raw_base_bounds_m": (
                {
                    "minimum": [float(value) for value in np.min(finite_points, axis=0)],
                    "maximum": [float(value) for value in np.max(finite_points, axis=0)],
                }
                if len(finite_points)
                else None
            ),
            "depth_clusters": [cluster.to_dict() for cluster in depth_clusters],
            "artifacts": {
                "rgb": str(frames_dir / "table_rgb.png"),
                "depth_preview": str(frames_dir / "table_depth.png"),
                "depth_mm": str(frames_dir / "table_depth_mm.npy"),
            },
        }
        fused = fuse_bin_pose(rgb_detection, depth_clusters)
        expected_center = np.array((plan.bin_x, plan.bin_y)) * 1000.0
        expected_wall = ground_truth_wall * 1000.0
        center_error = float(
            np.linalg.norm(np.asarray(fused.center_base_mm[:2]) - expected_center)
        )
        wall_error = float(
            np.linalg.norm(np.asarray(fused.wall_center_base_mm[:2]) - expected_wall)
        )
        report["perception"] = {
            "fused": fused.to_dict(),
            "depth_cluster_count": len(depth_clusters),
            "expected_center_base_mm": expected_center.tolist(),
            "expected_wall_center_base_mm": expected_wall.tolist(),
            "center_error_mm": center_error,
            "wall_error_mm": wall_error,
            "passed": center_error <= 2.0 and wall_error <= 2.0,
        }
        if not report["perception"]["passed"]:
            raise RuntimeError("fused table RGB/depth target exceeded 2 mm error")
        # From here on the robot is driven by the measured wall target, not by
        # the bin's simulator ground truth.  Ground truth remains evaluation
        # only and is never supplied to the policy.
        perceived_wall = np.asarray(fused.wall_center_base_mm[:2], dtype=float) / 1000.0
        grip_tip[:2] = perceived_wall
        approach_tip[:2] = perceived_wall
        lifted_tip[:2] = perceived_wall
        perceived_yaw_rad = math.radians(fused.gripper_yaw_deg)
        spread_axis = (
            -math.sin(perceived_yaw_rad),
            math.cos(perceived_yaw_rad),
            0.0,
        )
        report["perception"]["drives_motion"] = True
        report["perception"]["commanded_wall_center_base_mm"] = (
            np.asarray(perceived_wall) * 1000.0
        ).tolist()
    move_tcp("approach", safe_tip, approach_tip, 1.5, C.JAW_OPEN, C.JAW_OPEN)
    if args.validate_perception:
        wrist_frame = wrist_depth_camera.get_current_frame().get(
            "distance_to_image_plane"
        )
        if wrist_frame is None:
            raise RuntimeError("calibrated wrist-depth render was unavailable")
        wrist_m = np.asarray(wrist_frame, dtype=np.float64)
        wrist_valid = np.isfinite(wrist_m) & (wrist_m > 0)
        wrist_mm = np.zeros(wrist_m.shape, dtype=np.uint16)
        wrist_mm[wrist_valid] = np.clip(
            np.rint(wrist_m[wrist_valid] * 1000.0), 1, 65535
        ).astype(np.uint16)
        wrist_joints_deg = np.rad2deg(arm.get_joint_positions()[arm_indices])
        wrist_clusters = detect_depth_clusters(
            wrist_mm,
            "wrist",
            joints=wrist_joints_deg,
            stride=1,
            minimum_points=30,
            minimum_height_m=0.006,
        )
        wrist_candidates = [
            cluster for cluster in wrist_clusters if cluster.category_hint == "bin"
        ]
        if not wrist_candidates:
            raise RuntimeError("wrist depth did not retain a bin-scale cluster")
        wrist_bin = min(
            wrist_candidates,
            key=lambda cluster: float(
                np.linalg.norm(
                    np.asarray(cluster.centroid_base_m[:2]) * 1000.0
                    - np.asarray(fused.center_base_mm[:2])
                )
            ),
        )
        wrist_center = np.asarray(wrist_bin.centroid_base_m[:2]) * 1000.0
        wrist_vs_table = float(
            np.linalg.norm(wrist_center - np.asarray(fused.center_base_mm[:2]))
        )
        wrist_ground_truth_error = float(
            np.linalg.norm(wrist_center - expected_center)
        )
        np.save(frames_dir / "wrist_depth_mm.npy", wrist_mm)
        report["wrist_depth_validation"] = {
            "cluster_count": len(wrist_clusters),
            "clusters": [cluster.to_dict() for cluster in wrist_clusters],
            "wrist_vs_table_center_mm": wrist_vs_table,
            "ground_truth_center_error_mm_evaluation_only": wrist_ground_truth_error,
            "role": "presence/retention cross-check; not grasp translation",
            "passed": wrist_vs_table <= 15.0 and wrist_bin.confidence >= 0.45,
        }
        if not report["wrist_depth_validation"]["passed"]:
            raise RuntimeError("wrist-depth cross-check rejected the tracked bin")
    move_tcp("descend", approach_tip, grip_tip, 1.2, C.JAW_OPEN, C.JAW_OPEN)
    hold("preclose", grip_tip, 0.3, C.JAW_OPEN)
    move_tcp("close", grip_tip, grip_tip, 1.0, C.JAW_OPEN, jaw_grip)
    hold("dwell", grip_tip, 1.0, jaw_grip)
    contact = phase_samples["dwell"][-1]
    move_tcp("lift", grip_tip, lifted_tip, 1.0, jaw_grip, jaw_grip)
    hold("hold", lifted_tip, 2.0, jaw_grip)
    held = phase_samples["hold"][-1]
    if args.validate_perception:
        held_side_rgba = side_camera.get_rgba()
        if held_side_rgba is None:
            raise RuntimeError("side RGB hold render was unavailable")
        held_side_region = locate_green_region(
            np.asarray(held_side_rgba[:, :, :3], dtype=np.uint8)
        )
        side_lift_px = side_lift_tracker.observe(held_side_region)
        report["side_lift_validation"] = {
            "initial": side_lift_tracker.baseline.to_dict(),
            "held": held_side_region.to_dict(),
            "upward_motion_px": side_lift_px,
            "passed": side_lift_px >= 6.0,
        }
        if not report["side_lift_validation"]["passed"]:
            raise RuntimeError("side camera did not confirm the 10 mm lift")
    move_tcp("lower", lifted_tip, grip_tip, 1.0, jaw_grip, jaw_grip)
    hold("release", grip_tip, 0.7, C.JAW_OPEN)
    move_tcp("retreat", grip_tip, approach_tip, 1.0, C.JAW_OPEN, C.JAW_OPEN)

    initial_z = float(initial["bin_origin_world_m"][2])
    held_z = float(held["bin_origin_world_m"][2])
    measured_lift_mm = (held_z - initial_z) * 1000.0
    reference_relative_z = float(contact["bin_minus_tip_z_mm"])
    hold_slips = [
        reference_relative_z - float(sample["bin_minus_tip_z_mm"])
        for sample in phase_samples["hold"]
    ]
    lift_and_hold = phase_samples["lift"] + phase_samples["hold"]
    transient_slips = [
        reference_relative_z - float(sample["bin_minus_tip_z_mm"])
        for sample in lift_and_hold
    ]
    max_hold_slip_mm = max(0.0, max(hold_slips))
    max_transient_slip_mm = max(0.0, max(transient_slips))
    max_tilt_deg = max(
        float(sample["bin_tilt_deg"]) for sample in lift_and_hold
    )
    jaw_gap_mm = float(contact["jaw_gap_mm"])
    ik_failures = sum(
        int(phase.get("ik_failures", 0)) for phase in report["phases"].values()
    )
    acceptance = assess_trial(
        commanded_lift_mm=args.lift_mm,
        measured_lift_mm=measured_lift_mm,
        max_downward_slip_mm=max_hold_slip_mm,
        max_tilt_deg=max_tilt_deg,
        jaw_gap_mm=jaw_gap_mm,
        ik_failures=ik_failures,
    )
    report["measurements"] = {
        "initial_bin_origin_world_m": initial["bin_origin_world_m"],
        "held_bin_origin_world_m": held["bin_origin_world_m"],
        "measured_lift_mm": measured_lift_mm,
        "max_hold_downward_slip_mm": max_hold_slip_mm,
        "max_transient_downward_slip_mm": max_transient_slip_mm,
        "max_lift_and_hold_tilt_deg": max_tilt_deg,
        "contact_jaw_gap_mm": jaw_gap_mm,
        "captured_frames": frame_number,
    }
    report["acceptance"] = acceptance
    report["task_complete"] = bool(acceptance["passed"])
    if not report["task_complete"]:
        raise RuntimeError(
            "One-wall grip did not meet the pre-declared acceptance gate"
        )
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
    exit_code = 2
finally:
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[one-wall-bin] report written to {output_path}")
    print(
        json.dumps(
            {
                "task_complete": report["task_complete"],
                "measurements": report.get("measurements"),
                "acceptance": report.get("acceptance"),
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    app.close()

if exit_code:
    raise SystemExit(exit_code)
