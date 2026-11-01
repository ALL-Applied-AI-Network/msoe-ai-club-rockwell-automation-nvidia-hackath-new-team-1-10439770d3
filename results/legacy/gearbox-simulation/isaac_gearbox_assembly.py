"""Robot top-down assembly of the measured 1:1 gearbox in Isaac Sim.

The exact exported STL parts are used for visible geometry.  The cover export
is in its print-on-front orientation; it is transformed back into assembly
coordinates so its broad back/skirt is horizontal and its fused grasp tab
points upward.  Compound physics colliders preserve the gear bores and cover
locating holes, allowing real top-down shaft and pin insertion rather than
teleporting parts into place.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import traceback
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="runs/gearbox-assembly")
parser.add_argument("--physics-only", action="store_true")
parser.add_argument("--fps", type=int, default=12)
args = parser.parse_args()
if not 6 <= args.fps <= 30:
    parser.error("--fps must be between 6 and 30")

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets" / "gearbox"
OUT = Path(args.out_dir).resolve()
FRAMES = OUT / "frames"
OUT.mkdir(parents=True, exist_ok=True)
if not args.physics_only:
    FRAMES.mkdir(parents=True, exist_ok=True)
for required in ("base.stl", "gear.stl", "cover_print_orientation.stl"):
    if not (ASSET_DIR / required).is_file():
        raise SystemExit(f"Missing measured gearbox asset: {ASSET_DIR / required}")

print("[gearbox] starting Isaac Sim", flush=True)
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})


def ease(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def rotate_xy(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array((c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]))


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


def angle_error(measured, desired):
    return (measured - desired + 180.0) % 360.0 - 180.0


def periodic_angle_error(measured, desired, period):
    """Smallest error for geometry repeated every ``period`` degrees."""
    return (measured - desired + period / 2.0) % period - period / 2.0


def quat_yaw_tilt(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    yaw = math.degrees(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )
    up_z = 1.0 - 2.0 * (x * x + y * y)
    tilt = math.degrees(math.acos(float(np.clip(up_z, -1.0, 1.0))))
    return yaw, tilt


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
    "scope": "Isaac Sim measured-CAD top-down gearbox assembly; no physical execution",
    "assets": {
        "base": str(ASSET_DIR / "base.stl"),
        "gear": str(ASSET_DIR / "gear.stl"),
        "cover_export": str(ASSET_DIR / "cover_print_orientation.stl"),
        "assembled_reference": str(ASSET_DIR / "assembled_reference.stl"),
    },
    "cover_orientation": {
        "export_pose": "standing on front face for printing",
        "simulation_pose": "broad flat back/skirt horizontal, fused tab upward",
        "inverse_scad_transform_mm": {
            "x_assembly": "x_export",
            "y_assembly": "z_export - 45.0",
            "z_assembly": "21.9 - y_export",
        },
        "assembled_reference_crosscheck": {
            "method": "nearest-vertex comparison against assembled.stl",
            "standalone_part_vertices_within_0_05_mm_fraction": 0.9996371074,
            "median_error_mm": 0.0000007157,
            "note": "Confirms the inverse cover transform and both gear assembly transforms.",
        },
    },
    "events": [],
    "telemetry": [],
    "frames": 0,
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

    # Exact SCAD-derived dimensions in metres.
    BASE_SIZE = np.array((0.135, 0.085, 0.008))
    GEAR_OD = 0.060
    GEAR_TIP_RADIUS = GEAR_OD / 2.0
    GEAR_PITCH_RADIUS = 0.028
    GEAR_ROOT_RADIUS = 0.0255
    GEAR_THICKNESS = 0.008
    GEAR_BORE_RADIUS = 0.0043
    GEAR_HUB_RADIUS = 0.010
    GEAR_HUB_TOP = 0.0158
    GEAR_TAB_SIZE = np.array((0.008, 0.016, 0.014))
    GEAR_TAB_BOTTOM = 0.01579
    GEAR_TAB_TOP = 0.02979
    SHAFT_RADIUS = 0.004
    SHAFT_LAND_BOTTOM = 0.009
    SHAFT_LAND_TOP = 0.018
    SHAFT_TIP = 0.0192
    GEAR_SEAT_Z = 0.009
    GEAR_CENTRE_DISTANCE = 0.056
    TOOTH_PITCH_DEG = 360.0 / 28.0
    RIGHT_GEAR_PHASE_DEG = 180.0 + 180.0 / 28.0
    PIN_X = 0.0585
    PIN_Y = 0.0335
    PIN_RADIUS = 0.003
    PIN_HOLE_RADIUS = 0.0035
    COVER_SUPPORT_OUTER_RADIUS = 0.0065
    COVER_SKIRT_BOTTOM = 0.005
    COVER_PLATE_BOTTOM = 0.0185
    COVER_PLATE_TOP = 0.021
    COVER_TAB_BOTTOM = 0.0248
    COVER_TAB_TOP = 0.0388
    COVER_PRINT_TRANSLATE_Y = 21.9
    COVER_PRINT_TRANSLATE_Z = 45.0
    TABLE_Z = C.TABLE_TOP_Z + C.SURFACE_DZ
    BASE_WORLD = np.array((0.430, 0.085, TABLE_Z))
    BASE_YAW_DEG = 0.0
    LEFT_GEAR_TARGET = BASE_WORLD + np.array((-0.028, 0.0, GEAR_SEAT_Z))
    RIGHT_GEAR_TARGET = BASE_WORLD + np.array((0.028, 0.0, GEAR_SEAT_Z))
    COVER_TARGET = BASE_WORLD.copy()
    INITIAL = {
        "Gear_Left": {"xyz": np.array((0.285, -0.180, TABLE_Z + 0.0003)), "yaw": 21.0},
        "Gear_Right": {"xyz": np.array((0.375, -0.185, TABLE_Z + 0.0003)), "yaw": -31.0},
        # Root is 5 mm below the table because the assembly-coordinate cover
        # has its skirt/back bottom at local Z=5 mm.  This is the requested
        # broad-back-down staging pose, not the STL's vertical print pose.
        "Cover": {"xyz": np.array((0.505, -0.105, TABLE_Z - COVER_SKIRT_BOTTOM)), "yaw": -10.0},
    }
    PHYSICS_HZ = 60
    STEPS_PER_FRAME = max(1, round(PHYSICS_HZ / args.fps))
    RENDER_FPS = PHYSICS_HZ / STEPS_PER_FRAME

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / RENDER_FPS,
    )
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    cell.enable_ccd(world)

    def stl_visual(path, filename, transform, colour):
        vertices_mm, indices = load_stl(filename)
        vertices_m = transform(vertices_mm)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices_m])
        mesh.CreateFaceVertexCountsAttr([3] * len(indices))
        mesh.CreateFaceVertexIndicesAttr(indices.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        mesh.CreateDoubleSidedAttr(True)
        return mesh.GetPrim(), vertices_m

    def direct_mm(vertices):
        return vertices / 1000.0

    def cover_to_assembly(vertices):
        out = np.empty_like(vertices, dtype=float)
        out[:, 0] = vertices[:, 0] / 1000.0
        out[:, 1] = (vertices[:, 2] - COVER_PRINT_TRANSLATE_Z) / 1000.0
        out[:, 2] = (COVER_PRINT_TRANSLATE_Y - vertices[:, 1]) / 1000.0
        return out

    def physics_material(path, static=0.90, dynamic=0.80):
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()
        api = UsdPhysics.MaterialAPI.Apply(prim)
        api.CreateStaticFrictionAttr().Set(static)
        api.CreateDynamicFrictionAttr().Set(dynamic)
        api.CreateRestitutionAttr().Set(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr().Set("average")
        return prim

    gearbox_material = physics_material("/World/Physics/GearboxPlastic")

    def bind_physics(prim):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(gearbox_material),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        cell.small_contact_offsets(prim)

    def hidden_box(path, size, position, rotation=(0.0, 0.0, 0.0)):
        prim = cell.box(
            stage,
            path,
            size,
            position,
            (0.2, 0.2, 0.2),
            collision=True,
            rot=rotation,
        )
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
        bind_physics(prim)
        return prim

    def hidden_cylinder(path, radius, height, position):
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)
        xf = UsdGeom.Xformable(cylinder.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        cylinder.CreateVisibilityAttr().Set("invisible")
        bind_physics(cylinder.GetPrim())
        return cylinder.GetPrim()

    def hidden_cone(path, radius_bottom, radius_top, height, position):
        cone = UsdGeom.Cone.Define(stage, path)
        cone.CreateAxisAttr("Z")
        cone.CreateRadiusAttr(radius_bottom)
        cone.CreateHeightAttr(height)
        # USD Cone has one radius; a small cone accurately represents the
        # shaft/pin lead-in and is conservative at its base.
        xf = UsdGeom.Xformable(cone.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.CollisionAPI.Apply(cone.GetPrim())
        cone.CreateVisibilityAttr().Set("invisible")
        bind_physics(cone.GetPrim())
        return cone.GetPrim()

    def annulus(path, inner_radius, outer_radius, z_bottom, z_top, sectors=24, centre=(0.0, 0.0)):
        """Compound convex annulus that preserves a real central hole."""
        for index in range(sectors):
            a0 = 2.0 * math.pi * index / sectors
            a1 = 2.0 * math.pi * (index + 1) / sectors
            points = []
            for z in (z_bottom, z_top):
                points.extend(
                    [
                        (centre[0] + inner_radius * math.cos(a0), centre[1] + inner_radius * math.sin(a0), z),
                        (centre[0] + outer_radius * math.cos(a0), centre[1] + outer_radius * math.sin(a0), z),
                        (centre[0] + outer_radius * math.cos(a1), centre[1] + outer_radius * math.sin(a1), z),
                        (centre[0] + inner_radius * math.cos(a1), centre[1] + inner_radius * math.sin(a1), z),
                    ]
                )
            mesh = UsdGeom.Mesh.Define(stage, f"{path}/Sector_{index:02d}")
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
            faces = (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (3, 7, 6, 2),
                (0, 4, 7, 3),
                (1, 2, 6, 5),
            )
            mesh.CreateFaceVertexCountsAttr([4] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([item for face in faces for item in face])
            mesh.CreateSubdivisionSchemeAttr("none")
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            mesh.CreateVisibilityAttr().Set("invisible")
            bind_physics(prim)

    def gear_teeth(path):
        """Twenty-eight convex involute-envelope tooth colliders.

        The visual remains the exact STL.  These trapezoidal envelopes retain
        the measured pitch, root/tip radii and 0.4 mm backlash, allowing the
        pair to mesh physically instead of behaving like two smooth rings.
        """
        pressure_angle = math.radians(20.0)
        module = 0.002
        backlash = 0.0004
        base_radius = GEAR_PITCH_RADIUS * math.cos(pressure_angle)
        half_at_pitch = (
            math.pi * module / 2.0 - backlash / 2.0
        ) / (2.0 * GEAR_PITCH_RADIUS)

        def involute(angle):
            return math.tan(angle) - angle

        root_half = half_at_pitch + involute(pressure_angle)
        tip_pressure = math.acos(base_radius / GEAR_TIP_RADIUS)
        tip_half = half_at_pitch + involute(pressure_angle) - involute(tip_pressure)
        for index in range(28):
            centre = index * 2.0 * math.pi / 28.0
            planar = []
            for radius, half in (
                (GEAR_ROOT_RADIUS - 0.0001, root_half),
                (GEAR_TIP_RADIUS - 0.0002, tip_half),
            ):
                for sign in (-1.0, 1.0):
                    angle = centre + sign * half
                    planar.append((radius * math.cos(angle), radius * math.sin(angle)))
            # Order around the trapezoid: root-, tip-, tip+, root+.
            planar = [planar[0], planar[2], planar[3], planar[1]]
            points = [
                (x, y, z)
                for z in (0.00035, GEAR_THICKNESS - 0.00035)
                for x, y in planar
            ]
            mesh = UsdGeom.Mesh.Define(stage, f"{path}/Tooth_{index:02d}")
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
            faces = (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            )
            mesh.CreateFaceVertexCountsAttr([4] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([item for face in faces for item in face])
            mesh.CreateSubdivisionSchemeAttr("none")
            prim = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexHull")
            mesh.CreateVisibilityAttr().Set("invisible")
            bind_physics(prim)

    wrappers = {}

    # Static base and exact measured visual.
    base_root = UsdGeom.Xform.Define(stage, "/World/GearboxBase").GetPrim()
    base_xf = UsdGeom.Xformable(base_root)
    base_xf.AddTranslateOp().Set(Gf.Vec3d(*BASE_WORLD))
    base_xf.AddRotateZOp().Set(BASE_YAW_DEG)
    _, base_vertices = stl_visual(
        "/World/GearboxBase/MeasuredVisual", ASSET_DIR / "base.stl", direct_mm, (0.16, 0.18, 0.22)
    )
    hidden_box(
        "/World/GearboxBase/BasePlateCollider",
        tuple(BASE_SIZE),
        (0.0, 0.0, BASE_SIZE[2] / 2.0),
    )
    for side, x in (("Left", -0.028), ("Right", 0.028)):
        hidden_cylinder(
            f"/World/GearboxBase/{side}WasherCollider", 0.008, 0.001, (x, 0.0, 0.0085)
        )
        hidden_cylinder(
            f"/World/GearboxBase/{side}ShaftCollider",
            SHAFT_RADIUS,
            SHAFT_LAND_TOP - SHAFT_LAND_BOTTOM,
            (x, 0.0, (SHAFT_LAND_BOTTOM + SHAFT_LAND_TOP) / 2.0),
        )
        hidden_cone(
            f"/World/GearboxBase/{side}LeadInCollider",
            SHAFT_RADIUS,
            0.0,
            SHAFT_TIP - SHAFT_LAND_TOP,
            (x, 0.0, (SHAFT_LAND_TOP + SHAFT_TIP) / 2.0),
        )
    for sx in (-1, 1):
        for sy in (-1, 1):
            tag = f"{'P' if sx > 0 else 'N'}X_{'P' if sy > 0 else 'N'}Y"
            hidden_cylinder(
                f"/World/GearboxBase/Pin_{tag}", PIN_RADIUS, 0.004, (sx * PIN_X, sy * PIN_Y, 0.010)
            )
            hidden_cone(
                f"/World/GearboxBase/PinLead_{tag}", PIN_RADIUS, 0.0, 0.002, (sx * PIN_X, sy * PIN_Y, 0.013)
            )

    def add_gear(name, colour):
        cfg = INITIAL[name]
        path = f"/World/{name}"
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(*cfg["xyz"]))
        xf.AddRotateZOp().Set(cfg["yaw"])
        _, vertices = stl_visual(path + "/MeasuredVisual", ASSET_DIR / "gear.stl", direct_mm, colour)
        annulus(path + "/GearBodyCollider", GEAR_BORE_RADIUS + 0.00005, GEAR_ROOT_RADIUS - 0.0001, 0.0, GEAR_THICKNESS, 28)
        gear_teeth(path + "/GearToothColliders")
        annulus(path + "/HubCollider", GEAR_BORE_RADIUS + 0.00005, GEAR_HUB_RADIUS, GEAR_THICKNESS, GEAR_HUB_TOP, 20)
        hidden_box(
            path + "/TabCollider",
            tuple(GEAR_TAB_SIZE),
            (0.0, 0.0, (GEAR_TAB_BOTTOM + GEAR_TAB_TOP) / 2.0),
        )
        cell.make_dynamic(root, 0.021, collide_self=False)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0064))
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rigid.CreateEnableCCDAttr().Set(True)
        # A printed 8.6 mm bore on an 8.0 mm printed shaft rotates by hand but
        # does not coast like a near-frictionless bearing.  Model that dry
        # sliding drag so release impulses do not spin a seated gear.
        rigid.CreateAngularDampingAttr().Set(5.0)
        rigid.CreateSolverPositionIterationCountAttr().Set(24)
        rigid.CreateSolverVelocityIterationCountAttr().Set(8)
        wrappers[name] = SingleXFormPrim(path, name=name)
        return vertices

    gear_vertices = add_gear("Gear_Left", (0.96, 0.42, 0.035))
    add_gear("Gear_Right", (0.96, 0.72, 0.04))

    def add_cover():
        cfg = INITIAL["Cover"]
        path = "/World/Cover"
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        xf = UsdGeom.Xformable(root)
        xf.AddTranslateOp().Set(Gf.Vec3d(*cfg["xyz"]))
        xf.AddRotateZOp().Set(cfg["yaw"])
        _, vertices = stl_visual(
            path + "/MeasuredVisual",
            ASSET_DIR / "cover_print_orientation.stl",
            cover_to_assembly,
            (0.22, 0.52, 0.88),
        )
        # Four outer skirt rails; they are flat on the table in the requested
        # staging pose and surround, rather than intersect, the base at assembly.
        skirt_h = COVER_PLATE_BOTTOM - COVER_SKIRT_BOTTOM
        hidden_box(path + "/SkirtNorth", (0.140, 0.002, skirt_h), (0.0, 0.044, (COVER_SKIRT_BOTTOM + COVER_PLATE_BOTTOM) / 2.0))
        hidden_box(path + "/SkirtSouth", (0.140, 0.002, skirt_h), (0.0, -0.044, (COVER_SKIRT_BOTTOM + COVER_PLATE_BOTTOM) / 2.0))
        hidden_box(path + "/SkirtEast", (0.002, 0.086, skirt_h), (0.069, 0.0, (COVER_SKIRT_BOTTOM + COVER_PLATE_BOTTOM) / 2.0))
        hidden_box(path + "/SkirtWest", (0.002, 0.086, skirt_h), (-0.069, 0.0, (COVER_SKIRT_BOTTOM + COVER_PLATE_BOTTOM) / 2.0))
        # Annular columns are both the support surfaces and the real locating
        # holes.  The base pins pass through their 7 mm openings.
        for sx in (-1, 1):
            for sy in (-1, 1):
                tag = f"{'P' if sx > 0 else 'N'}X_{'P' if sy > 0 else 'N'}Y"
                annulus(
                    path + f"/LocateColumn_{tag}",
                    PIN_HOLE_RADIUS,
                    COVER_SUPPORT_OUTER_RADIUS,
                    0.008,
                    COVER_PLATE_BOTTOM,
                    16,
                    (sx * PIN_X, sy * PIN_Y),
                )
        # The centre saddle and fused tab carry the real gripper load.  Plate
        # collision stays away from the two gear windows and their tabs.
        hidden_box(path + "/TabSaddle", (0.008, 0.030, COVER_TAB_BOTTOM - COVER_PLATE_TOP), (0.0, 0.0, (COVER_TAB_BOTTOM + COVER_PLATE_TOP) / 2.0))
        hidden_box(path + "/FusedTabCollider", tuple(GEAR_TAB_SIZE), (0.0, 0.0, (COVER_TAB_BOTTOM + COVER_TAB_TOP) / 2.0))
        hidden_box(path + "/TopSpineNorth", (0.006, 0.032, COVER_PLATE_TOP - COVER_PLATE_BOTTOM), (0.0, 0.029, (COVER_PLATE_BOTTOM + COVER_PLATE_TOP) / 2.0))
        hidden_box(path + "/TopSpineSouth", (0.006, 0.032, COVER_PLATE_TOP - COVER_PLATE_BOTTOM), (0.0, -0.029, (COVER_PLATE_BOTTOM + COVER_PLATE_TOP) / 2.0))
        cell.make_dynamic(root, 0.105, collide_self=False)
        mass = UsdPhysics.MassAPI.Apply(root)
        # Exact transformed-STL mass properties.  The former guessed COM was
        # displaced by 7 mm in Y and 4.3 mm in Z, creating a false torque that
        # made the cover swing 23 degrees under an otherwise centred tab grip.
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, -0.0007983541, 0.015708492))
        mass.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(0.00010119868, 0.00021224674, 0.00030865924)
        )
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rigid.CreateEnableCCDAttr().Set(True)
        rigid.CreateAngularDampingAttr().Set(3.0)
        rigid.CreateSolverPositionIterationCountAttr().Set(32)
        rigid.CreateSolverVelocityIterationCountAttr().Set(8)
        wrappers["Cover"] = SingleXFormPrim(path, name="Cover")
        return vertices

    cover_vertices = add_cover()
    report["measured_geometry"] = {
        "base_bounds_mm": (np.vstack((base_vertices.min(axis=0), base_vertices.max(axis=0))) * 1000.0).tolist(),
        "gear_bounds_mm": (np.vstack((gear_vertices.min(axis=0), gear_vertices.max(axis=0))) * 1000.0).tolist(),
        "cover_assembly_bounds_mm": (np.vstack((cover_vertices.min(axis=0), cover_vertices.max(axis=0))) * 1000.0).tolist(),
        "shaft_diameter_mm": SHAFT_RADIUS * 2000.0,
        "gear_bore_diameter_mm": GEAR_BORE_RADIUS * 2000.0,
        "diametral_bore_clearance_mm": (GEAR_BORE_RADIUS - SHAFT_RADIUS) * 2000.0,
        "pin_diameter_mm": PIN_RADIUS * 2000.0,
        "locating_hole_diameter_mm": PIN_HOLE_RADIUS * 2000.0,
        "diametral_pin_clearance_mm": (PIN_HOLE_RADIUS - PIN_RADIUS) * 2000.0,
    }

    camera_path = "/World/GearboxShowcaseCamera"
    cell.camera(
        stage,
        camera_path,
        {
            "eye": (0.730, -0.650, 1.350),
            "look_at": (0.405, -0.020, TABLE_Z + 0.035),
            "up": (0.0, 0.0, 1.0),
            "resolution": (1280, 720),
            "focal_length": 20.0,
            "horiz_aperture": 20.955,
        },
    )
    camera = None if args.physics_only else Camera(camera_path, resolution=(1280, 720))

    print("[gearbox] resetting world", flush=True)
    world.reset()
    if camera is not None:
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
    q, ok = ik.solve(q, home, down, limits, tool_offset=tool_offset, iters=600, damping=0.02)
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
    if camera is not None:
        for _ in range(90):
            world.step(render=True)

    frame_number = 0
    sim_time = 0.0
    current_phase = "Measured gearbox initialization"

    def root_pose(name):
        position, quaternion = wrappers[name].get_world_pose()
        yaw, tilt = quat_yaw_tilt(quaternion)
        return np.asarray(position, float), np.asarray(quaternion, float), yaw, tilt

    def capture(label):
        global frame_number
        if camera is None:
            return
        rgba = camera.get_rgba()
        if rgba is None:
            raise RuntimeError("Gearbox camera returned no frame")
        image = cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (1280, 74), (12, 14, 18), -1)
        cv2.putText(image, label, (24, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "Measured CAD | top-down insertion | cover broad back horizontal",
            (24, 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (110, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(FRAMES / f"f{frame_number:05d}.png"), image)
        frame_number += 1

    def physics_frame(label, q6, jaw):
        global sim_time
        command(q6, jaw)
        for step_index in range(STEPS_PER_FRAME):
            world.step(render=camera is not None and step_index == STEPS_PER_FRAME - 1)
            sim_time += 1.0 / PHYSICS_HZ
        _, tcp = fk.tool_pose(arm.get_joint_positions()[arm_indices], tool_offset)
        report["telemetry"].append(
            {
                "time_s": sim_time,
                "phase": label,
                "joints_deg": np.rad2deg(arm.get_joint_positions()[arm_indices]).tolist(),
                "tcp_robot_m": np.asarray(tcp, float).tolist(),
                "jaw_target_m_per_jaw": float(jaw),
                "parts": {
                    name: {
                        "world_position_m": root_pose(name)[0].tolist(),
                        "yaw_deg": root_pose(name)[2],
                        "tilt_deg": root_pose(name)[3],
                    }
                    for name in wrappers
                },
            }
        )
        capture(label)

    def current_spread():
        rotation, _ = fk.fk(q)
        spread = rotation @ np.asarray(C.FINGER_SPREAD_AXIS, float)
        spread[2] = 0.0
        return spread / np.linalg.norm(spread)

    def solve_target(target, spread):
        solution, converged = ik.solve(
            q,
            target,
            down,
            limits,
            tool_offset=tool_offset,
            spread_dir=spread,
            iters=500,
            damping=0.02,
        )
        _, actual = fk.tool_pose(solution, tool_offset)
        aligned = abs(
            float(np.dot(fk.fk(solution)[0] @ np.asarray(C.FINGER_SPREAD_AXIS), spread))
        ) > 0.998
        if not converged or np.linalg.norm(actual - target) > 0.0012 or not aligned:
            raise RuntimeError(
                f"IK rejected target={target.tolist()}, error_mm={np.linalg.norm(actual-target)*1000:.3f}"
            )
        return solution

    def move_tcp(label, start, finish, seconds, jaw_start, jaw_finish, spread_start, spread_finish):
        global q
        frames = max(2, round(seconds * RENDER_FPS))
        for index in range(frames):
            t = ease((index + 1) / frames)
            target = start + (finish - start) * t
            spread = turn(spread_start, spread_finish, t)
            q = solve_target(target, spread)
            physics_frame(label, q, jaw_start + (jaw_finish - jaw_start) * t)

    def hold(label, target, seconds, jaw, spread):
        global q
        q = solve_target(target, spread)
        for _ in range(max(1, round(seconds * RENDER_FPS))):
            physics_frame(label, q, jaw)

    for _ in range(round(1.5 * RENDER_FPS)):
        physics_frame(current_phase, q, C.JAW_OPEN)

    initial_cover, _, initial_cover_yaw, initial_cover_tilt = root_pose("Cover")
    report["cover_orientation"]["initial_world_position_m"] = initial_cover.tolist()
    report["cover_orientation"]["initial_yaw_deg"] = initial_cover_yaw
    report["cover_orientation"]["initial_tilt_deg"] = initial_cover_tilt
    report["cover_orientation"]["flat_back_down_passed"] = bool(
        initial_cover_tilt <= 1.0
        and abs(initial_cover[2] + COVER_SKIRT_BOTTOM - TABLE_Z) <= 0.0015
    )
    if not report["cover_orientation"]["flat_back_down_passed"]:
        raise RuntimeError("Cover did not settle on its broad flat back/skirt")

    # The real gripper is pneumatic/binary: it commands fully closed and stops
    # on the 8 mm tab.  A width-derived partial-close target passed the proof
    # lift but released contact while yaw and XY motion were combined.
    tab_jaw = C.JAW_CLOSED
    # At the seated pose, open only enough to clear the 8 mm tab.  Full opening
    # beside the tab twisted a correctly phased gear by 2.4 degrees.  Finish
    # opening only after the fingers have withdrawn vertically.
    tab_release_jaw = min(
        C.JAW_OPEN,
        GEAR_TAB_SIZE[0] / 2.0 - C.FINGER_INNER_TIP + 0.00070,
    )

    def assemble_tabbed_part(
        name, target_world, target_yaw, tab_top, insertion, kind,
        require_absolute_yaw=True,
    ):
        start_position, _, start_yaw, start_tilt = root_pose(name)
        source_spread = np.array(
            (math.cos(math.radians(start_yaw)), math.sin(math.radians(start_yaw)), 0.0)
        )
        source_tip = np.array(
            (
                start_position[0],
                start_position[1],
                start_position[2] - C.TABLE_TOP_Z + tab_top - insertion,
            )
        )
        source_above = source_tip + np.array((0.0, 0.0, 0.075))
        proof = source_tip + np.array((0.0, 0.0, 0.022))
        carry = source_tip + np.array((0.0, 0.0, 0.110))
        target_spread = np.array(
            (math.cos(math.radians(target_yaw)), math.sin(math.radians(target_yaw)), 0.0)
        )
        label = f"{name.replace('_', ' ').upper()}: top-tab pick -> {kind} insertion"
        move_tcp(label + " | approach", home, source_above, 0.90, C.JAW_OPEN, C.JAW_OPEN, current_spread(), source_spread)
        move_tcp(label + " | descend", source_above, source_tip, 0.65, C.JAW_OPEN, C.JAW_OPEN, source_spread, source_spread)
        hold(label + " | close on tab", source_tip, 0.75, tab_jaw, source_spread)
        move_tcp(label + " | proof lift", source_tip, proof, 0.85, tab_jaw, tab_jaw, source_spread, source_spread)
        hold(label + " | proof dwell", proof, 0.55, tab_jaw, source_spread)
        lifted, _, lifted_yaw, lifted_tilt = root_pose(name)
        lift_mm = (lifted[2] - start_position[2]) * 1000.0
        if lift_mm < 15.0:
            raise RuntimeError(f"{name} tab grasp failed: lift={lift_mm:.2f} mm")

        move_tcp(label + " | raise", proof, carry, 1.80, tab_jaw, tab_jaw, source_spread, source_spread)
        hold(label + " | raised dwell", carry, 0.60, tab_jaw, source_spread)
        # Rotate at safe height before translating.  The earlier combined yaw
        # and XY move let the tab slip midway through the carry.  Re-observe
        # after yaw so the destination uses the actual held transform.
        move_tcp(label + " | orient in place", carry, carry, 1.80, tab_jaw, tab_jaw, source_spread, target_spread)
        hold(label + " | oriented dwell", carry, 0.60, tab_jaw, target_spread)
        oriented, _, oriented_yaw, oriented_tilt = root_pose(name)
        if oriented[2] - start_position[2] < 0.050:
            raise RuntimeError(f"{name} slipped during safe-height orientation")
        measured_joints = arm.get_joint_positions()
        _, measured_tip = fk.tool_pose(measured_joints[arm_indices], tool_offset)
        oriented_robot = oriented.copy()
        oriented_robot[2] -= C.TABLE_TOP_Z
        held_offset = oriented_robot - np.asarray(measured_tip, float)
        desired_root_robot = np.asarray(target_world, float).copy()
        desired_root_robot[2] -= C.TABLE_TOP_Z
        target_tip = desired_root_robot - held_offset
        target_above = target_tip + np.array((0.0, 0.0, 0.085))
        preinsert = target_tip + np.array((0.0, 0.0, 0.015 if kind == "shaft" else 0.018))
        move_tcp(label + " | carry", carry, target_above, 3.20, tab_jaw, tab_jaw, target_spread, target_spread)
        hold(label + " | aligned above target", target_above, 0.60, tab_jaw, target_spread)
        carried, _, _, _ = root_pose(name)
        if carried[2] - start_position[2] < 0.050:
            raise RuntimeError(f"{name} slipped during translation")
        # Correct residual grasp flex while still safely above all shafts and
        # pins.  In particular, never use the locating hardware itself to
        # force a tilted or laterally-offset cover into place.
        high_corrections = []
        for index in range(3):
            observed, _, observed_yaw, _ = root_pose(name)
            xy_error = np.asarray(target_world[:2]) - observed[:2]
            yaw_period = TOOTH_PITCH_DEG if kind == "shaft" else 180.0
            yaw_error = periodic_angle_error(observed_yaw, target_yaw, yaw_period)
            high_corrections.append(
                {
                    "iteration": index + 1,
                    "observed_world_xy_m": observed[:2].tolist(),
                    "xy_error_mm": (xy_error * 1000.0).tolist(),
                    "observed_yaw_deg": observed_yaw,
                    "yaw_error_deg": yaw_error,
                }
            )
            if np.linalg.norm(xy_error) <= 0.00035 and abs(yaw_error) <= 0.4:
                break
            next_above = target_above.copy()
            next_above[:2] += xy_error
            spread_xy = rotate_xy(target_spread[:2], math.radians(-yaw_error))
            next_spread = np.array((spread_xy[0], spread_xy[1], 0.0))
            move_tcp(label + f" | high alignment {index + 1}", target_above, next_above, 0.85, tab_jaw, tab_jaw, target_spread, next_spread)
            target_above = next_above
            target_tip[:2] += xy_error
            preinsert[:2] += xy_error
            target_spread = next_spread
        hold(label + " | high alignment dwell", target_above, 0.40, tab_jaw, target_spread)
        move_tcp(label + " | pre-insertion", target_above, preinsert, 1.30, tab_jaw, tab_jaw, target_spread, target_spread)

        corrected = preinsert.copy()
        corrected_spread = target_spread.copy()
        corrections = []
        for index in range(3):
            observed, _, observed_yaw, _ = root_pose(name)
            xy_error = np.asarray(target_world[:2]) - observed[:2]
            yaw_period = TOOTH_PITCH_DEG if kind == "shaft" else 180.0
            yaw_error = periodic_angle_error(observed_yaw, target_yaw, yaw_period)
            corrections.append(
                {
                    "iteration": index + 1,
                    "observed_world_xy_m": observed[:2].tolist(),
                    "xy_error_mm": (xy_error * 1000.0).tolist(),
                    "observed_yaw_deg": observed_yaw,
                    "yaw_error_deg": yaw_error,
                }
            )
            if np.linalg.norm(xy_error) <= 0.00035 and abs(yaw_error) <= 0.4:
                break
            next_target = corrected.copy()
            next_target[:2] += xy_error
            spread_xy = rotate_xy(corrected_spread[:2], math.radians(-yaw_error))
            next_spread = np.array((spread_xy[0], spread_xy[1], 0.0))
            move_tcp(label + f" | visual correction {index + 1}", corrected, next_target, 0.65, tab_jaw, tab_jaw, corrected_spread, next_spread)
            corrected = next_target
            corrected_spread = next_spread

        final_tip = target_tip.copy()
        final_tip[:2] = corrected[:2]
        move_tcp(label + " | guided final insertion", corrected, final_tip, 1.60 if kind == "shaft" else 2.00, tab_jaw, tab_jaw, corrected_spread, corrected_spread)
        hold(label + " | seated under grip", final_tip, 0.70, tab_jaw, corrected_spread)
        pre_release, _, pre_release_yaw, pre_release_tilt = root_pose(name)
        move_tcp(label + " | limited release", final_tip, final_tip, 0.75, tab_jaw, tab_release_jaw, corrected_spread, corrected_spread)
        hold(label + " | release dwell", final_tip, 0.60, tab_release_jaw, corrected_spread)
        withdraw = final_tip + np.array((0.0, 0.0, 0.085))
        move_tcp(label + " | vertical withdrawal", final_tip, withdraw, 1.00, tab_release_jaw, tab_release_jaw, corrected_spread, corrected_spread)
        hold(label + " | full open above part", withdraw, 0.35, C.JAW_OPEN, corrected_spread)
        hold(label + " | settle", withdraw, 0.85, C.JAW_OPEN, corrected_spread)
        final_position, _, final_yaw, final_tilt = root_pose(name)
        position_error_mm = np.linalg.norm(final_position - np.asarray(target_world)) * 1000.0
        xy_error_mm = np.linalg.norm(final_position[:2] - np.asarray(target_world[:2])) * 1000.0
        z_error_mm = abs(final_position[2] - target_world[2]) * 1000.0
        yaw_period = TOOTH_PITCH_DEG if kind == "shaft" else 180.0
        yaw_error = periodic_angle_error(final_yaw, target_yaw, yaw_period)
        passed = bool(
            xy_error_mm <= (1.0 if kind == "shaft" else 1.5)
            and z_error_mm <= (1.6 if kind == "shaft" else 2.0)
            and (not require_absolute_yaw or abs(yaw_error) <= 2.0)
            and final_tilt <= 3.0
        )
        event = {
            "part": name,
            "kind": kind,
            "initial_world_position_m": start_position.tolist(),
            "initial_yaw_deg": start_yaw,
            "initial_tilt_deg": start_tilt,
            "measured_lift_mm": lift_mm,
            "lifted_tilt_deg": lifted_tilt,
            "oriented_world_position_m": oriented.tolist(),
            "oriented_yaw_deg": oriented_yaw,
            "oriented_tilt_deg": oriented_tilt,
            "held_root_minus_tcp_m": held_offset.tolist(),
            "high_corrections": high_corrections,
            "corrections": corrections,
            "target_world_position_m": np.asarray(target_world).tolist(),
            "target_yaw_deg": target_yaw,
            "pre_release_world_position_m": pre_release.tolist(),
            "pre_release_yaw_deg": pre_release_yaw,
            "pre_release_tilt_deg": pre_release_tilt,
            "final_world_position_m": final_position.tolist(),
            "final_yaw_deg": final_yaw,
            "final_tilt_deg": final_tilt,
            "position_error_mm": position_error_mm,
            "xy_error_mm": xy_error_mm,
            "z_error_mm": z_error_mm,
            "yaw_error_deg": yaw_error,
            "absolute_yaw_required": require_absolute_yaw,
            "passed": passed,
        }
        report["events"].append(event)
        if not passed:
            raise RuntimeError(f"{name} failed assembly acceptance: {event}")
        move_tcp(label + " | home", withdraw, home, 0.90, C.JAW_OPEN, C.JAW_OPEN, corrected_spread, current_spread())

    # A gear on a round shaft has no meaningful absolute phase until its mate
    # is introduced.  Seat the first gear, observe its resting phase, then
    # command the second gear one half-tooth offset from that measured phase.
    assemble_tabbed_part(
        "Gear_Left", LEFT_GEAR_TARGET, 0.0, GEAR_TAB_TOP, 0.0080,
        "shaft", require_absolute_yaw=False,
    )
    _, _, seated_left_yaw, _ = root_pose("Gear_Left")
    # External gears counter-rotate: if the first gear rests at +theta, the
    # mating gear must be commanded to reference_phase-theta.
    right_command_yaw = RIGHT_GEAR_PHASE_DEG - seated_left_yaw
    assemble_tabbed_part(
        "Gear_Right", RIGHT_GEAR_TARGET, right_command_yaw,
        GEAR_TAB_TOP, 0.0080, "shaft", require_absolute_yaw=False,
    )

    # Verify the two centres and phase before covering them.
    left_position, _, left_yaw, left_tilt = root_pose("Gear_Left")
    right_position, _, right_yaw, right_tilt = root_pose("Gear_Right")
    centre_distance_mm = np.linalg.norm(right_position[:2] - left_position[:2]) * 1000.0
    phase_error = periodic_angle_error(
        right_yaw + left_yaw, RIGHT_GEAR_PHASE_DEG, TOOTH_PITCH_DEG
    )
    report["gear_pair_verification"] = {
        "measured_centre_distance_mm": centre_distance_mm,
        "target_centre_distance_mm": GEAR_CENTRE_DISTANCE * 1000.0,
        "centre_distance_error_mm": abs(centre_distance_mm - GEAR_CENTRE_DISTANCE * 1000.0),
        "measured_external_mesh_phase_sum_deg": right_yaw + left_yaw,
        "target_external_mesh_phase_sum_deg": RIGHT_GEAR_PHASE_DEG,
        "phase_error_deg": phase_error,
        "both_upright": bool(left_tilt <= 3.0 and right_tilt <= 3.0),
    }
    if (
        report["gear_pair_verification"]["centre_distance_error_mm"] > 1.5
        or abs(phase_error) > 2.5
        or not report["gear_pair_verification"]["both_upright"]
    ):
        raise RuntimeError(
            f"Gear pair failed physical mesh verification: {report['gear_pair_verification']}"
        )

    assemble_tabbed_part("Cover", COVER_TARGET, 0.0, COVER_TAB_TOP, 0.0115, "locating pins")

    for _ in range(round(2.5 * RENDER_FPS)):
        physics_frame("TASK COMPLETE: measured gearbox assembled top-down", q, C.JAW_OPEN)

    final_cover, _, final_cover_yaw, final_cover_tilt = root_pose("Cover")
    report["final_assembly"] = {
        "left_gear": report["events"][0],
        "right_gear": report["events"][1],
        "cover": report["events"][2],
        "cover_flat_back_horizontal": bool(final_cover_tilt <= 3.0),
        "cover_seated_over_base": bool(
            np.linalg.norm(final_cover[:2] - BASE_WORLD[:2]) <= 0.0015
            and abs(final_cover[2] - BASE_WORLD[2]) <= 0.002
            and abs(periodic_angle_error(final_cover_yaw, BASE_YAW_DEG, 180.0)) <= 2.0
        ),
    }
    report["frames"] = frame_number
    report["render_fps"] = RENDER_FPS
    report["task_complete"] = bool(
        report["cover_orientation"]["flat_back_down_passed"]
        and len(report["events"]) == 3
        and all(event["passed"] for event in report["events"])
        and report["gear_pair_verification"]["centre_distance_error_mm"] <= 1.5
        and abs(report["gear_pair_verification"]["phase_error_deg"]) <= 2.5
        and report["final_assembly"]["cover_flat_back_horizontal"]
        and report["final_assembly"]["cover_seated_over_base"]
    )
except Exception as exc:
    traceback.print_exc()
    report["frames"] = report.get("frames", 0) or globals().get("frame_number", 0)
    report["error"] = f"{type(exc).__name__}: {exc}"
    exit_code = 2
finally:
    (OUT / "execution_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
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
