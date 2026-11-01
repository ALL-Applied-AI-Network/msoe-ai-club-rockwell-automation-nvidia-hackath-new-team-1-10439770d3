"""Observation-only CAD registration candidates for method B.

This module is intentionally optional: Open3D is imported only when the policy
runs.  The policy starts from geometry-only RGB-D components, builds a fused
point cloud from the submitted cameras, and refines cube/bin poses with a real
FPFH + Fast Global Registration + point-to-plane ICP pipeline.  No simulator
object transform or scoring file is available to this code.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .advanced_methods import GeometryOBBScalePolicy, _replace_fixed_mounts
from .camera_geometry import canonical_square_yaw
from .policies import BIN_SIZE, CUBE_SIZE, BasePolicy, SceneObservation


def _box_surface(size_x: float, size_y: float, height: float, step: float) -> np.ndarray:
    xs = np.arange(-size_x / 2.0, size_x / 2.0 + step * 0.5, step)
    ys = np.arange(-size_y / 2.0, size_y / 2.0 + step * 0.5, step)
    zs = np.arange(0.0, height + step * 0.5, step)
    points = []
    for z in (0.0, height):
        xx, yy = np.meshgrid(xs, ys)
        points.append(np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z))))
    for x in (-size_x / 2.0, size_x / 2.0):
        yy, zz = np.meshgrid(ys, zs)
        points.append(np.column_stack((np.full(yy.size, x), yy.ravel(), zz.ravel())))
    for y in (-size_y / 2.0, size_y / 2.0):
        xx, zz = np.meshgrid(xs, zs)
        points.append(np.column_stack((xx.ravel(), np.full(xx.size, y), zz.ravel())))
    return np.unique(np.round(np.concatenate(points), 6), axis=0)


def _open_bin_surface(scale: float, step: float) -> np.ndarray:
    outer = BIN_SIZE * scale
    height = 0.054 * scale
    wall = 0.0075 * scale
    bottom = 0.029 * scale
    points = [_box_surface(outer, outer, bottom, step)]
    # Four hollow walls.  Their union reproduces the measured open-top bin
    # geometry much better than a solid 50 mm cuboid.
    long = outer
    thick = wall
    wall_height = height - bottom
    for x, y, sx, sy in (
        (0.0, outer / 2.0 - thick / 2.0, long, thick),
        (0.0, -outer / 2.0 + thick / 2.0, long, thick),
        (outer / 2.0 - thick / 2.0, 0.0, thick, long),
        (-outer / 2.0 + thick / 2.0, 0.0, thick, long),
    ):
        wall_points = _box_surface(sx, sy, wall_height, step)
        wall_points[:, 0] += x
        wall_points[:, 1] += y
        wall_points[:, 2] += bottom
        points.append(wall_points)
    return np.unique(np.round(np.concatenate(points), 6), axis=0)


def _fused_points(observation: SceneObservation, stride: int = 2) -> np.ndarray:
    groups = []
    for camera_obs in observation.cameras.values():
        if camera_obs.depth is None:
            continue
        points, _, _ = camera_obs.camera.depth_to_world(camera_obs.depth, stride=stride)
        use = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] > 0.12)
            & (points[:, 0] < 0.72)
            & (points[:, 1] > -0.41)
            & (points[:, 1] < 0.33)
            & (points[:, 2] > camera_obs.camera.table_z + 0.0015)
            & (points[:, 2] < camera_obs.camera.table_z + 0.13)
        )
        groups.append(points[use])
    return np.concatenate(groups) if groups else np.empty((0, 3), dtype=np.float64)


def _cloud(o3d, points: np.ndarray, voxel: float):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4.0, max_nn=40)
    )
    return cloud


def _features(o3d, cloud, voxel: float):
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 7.0, max_nn=80),
    )


def _tilt_degrees(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))


def _register_one(o3d, points: np.ndarray, item: dict, table_z: float) -> tuple[dict, dict]:
    category = item["category"]
    if category not in ("cube", "bin"):
        return dict(item), {"accepted": False, "reason": "truck_not_refined"}
    centre = np.asarray(item["xy"], dtype=np.float64)
    radius = 0.027 if category == "cube" else 0.046
    max_height = 0.025 if category == "cube" else 0.068
    crop = points[
        (np.linalg.norm(points[:, :2] - centre, axis=1) < radius)
        & (points[:, 2] < table_z + max_height)
    ]
    voxel = 0.0015 if category == "cube" else 0.0030
    if len(crop) < 35:
        return dict(item), {"accepted": False, "reason": "insufficient_points", "points": int(len(crop))}
    target = _cloud(o3d, crop, voxel)
    target_features = _features(o3d, target, voxel)
    best = None
    for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
        if category == "cube":
            template = _box_surface(CUBE_SIZE * scale, CUBE_SIZE * scale, CUBE_SIZE * scale, voxel)
        else:
            template = _open_bin_surface(scale, voxel)
        source = _cloud(o3d, template, voxel)
        source_features = _features(o3d, source, voxel)
        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel * 2.2,
            iteration_number=96,
        )
        try:
            global_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source, target, source_features, target_features, option
            )
            refined = o3d.pipelines.registration.registration_icp(
                source,
                target,
                voxel * 1.8,
                global_result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
            )
        except RuntimeError:
            continue
        transform = np.asarray(refined.transformation)
        xy = transform[:2, 3]
        tilt = _tilt_degrees(transform[:3, :3])
        displacement = float(np.linalg.norm(xy - centre))
        bottom_error = abs(float(transform[2, 3]) - table_z)
        quality = (
            float(refined.fitness)
            - 2.0 * float(refined.inlier_rmse)
            - 8.0 * displacement
            - 5.0 * bottom_error
            - 0.002 * tilt
        )
        candidate = (quality, scale, refined, displacement, bottom_error, tilt)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return dict(item), {"accepted": False, "reason": "registration_failed", "points": int(len(crop))}
    _, scale, refined, displacement, bottom_error, tilt = best
    transform = np.asarray(refined.transformation)
    accepted = bool(
        refined.fitness >= (0.12 if category == "cube" else 0.08)
        and displacement <= (0.010 if category == "cube" else 0.016)
        and bottom_error <= 0.012
        and tilt <= 25.0
    )
    output = dict(item)
    if accepted:
        yaw = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        output["xy"] = transform[:2, 3].tolist()
        output["yaw_deg"] = float(canonical_square_yaw(yaw))
        output["estimated_scale"] = float(scale)
        output["confidence"] = float(np.clip(0.35 + 0.60 * refined.fitness, 0.05, 0.995))
        output["source"] = "cad_fpfh_fgr_icp"
    diagnostic = {
        "accepted": accepted,
        "points": int(len(crop)),
        "fitness": float(refined.fitness),
        "inlier_rmse_mm": float(refined.inlier_rmse * 1000.0),
        "seed_displacement_mm": float(displacement * 1000.0),
        "bottom_error_mm": float(bottom_error * 1000.0),
        "tilt_deg": float(tilt),
        "scale": float(scale),
    }
    return output, diagnostic


class Open3DFGRPolicy(BasePolicy):
    """B2: measured-geometry FPFH/FGR/ICP with gravity plausibility gates."""

    name = "B2_cad_fpfh_fgr_icp"

    def __init__(self):
        self.seed = GeometryOBBScalePolicy()

    def predict(self, observation: SceneObservation) -> dict:
        import open3d as o3d

        started = time.perf_counter()
        corrected = _replace_fixed_mounts(observation)
        seed_result = self.seed.predict(corrected)
        points = _fused_points(corrected)
        table_z = corrected.cameras["table"].camera.table_z
        detections = []
        diagnostics = []
        for item in seed_result["detections"]:
            refined, diagnostic = _register_one(o3d, points, item, table_z)
            diagnostic.update({"category": item["category"], "colour": item.get("colour")})
            detections.append(refined)
            diagnostics.append(diagnostic)
        result = self._result(
            corrected,
            detections,
            started,
            ["actual Open3D FPFH + Fast Global Registration + point-to-plane ICP; gravity plausibility gate"],
        )
        result["registration_diagnostics"] = diagnostics
        return result

