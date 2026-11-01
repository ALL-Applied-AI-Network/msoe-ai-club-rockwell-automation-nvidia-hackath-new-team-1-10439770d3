"""TEASER++ CAD registration candidate for method B.

FPFH descriptors establish noisy CAD-to-observation correspondences; the
official TEASER++ Python binding estimates robust rotation, translation, and
scale.  A final ICP pass is allowed only inside gravity and seed-distance
plausibility gates.  Inputs remain serialized RGB-D observations only.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree

from .advanced_methods import GeometryOBBScalePolicy, _replace_fixed_mounts
from .cad_registration import (
    _box_surface,
    _cloud,
    _features,
    _fused_points,
    _open_bin_surface,
    _tilt_degrees,
)
from .camera_geometry import canonical_square_yaw
from .policies import BIN_SIZE, CUBE_SIZE, BasePolicy, SceneObservation


def _mutual_feature_correspondences(source_features, target_features):
    source = np.asarray(source_features.data, dtype=np.float64).T
    target = np.asarray(target_features.data, dtype=np.float64).T
    if not len(source) or not len(target):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    source_to_target_distance, source_to_target = cKDTree(target).query(source, k=1)
    _, target_to_source = cKDTree(source).query(target, k=1)
    source_indices = np.arange(len(source), dtype=int)
    mutual = target_to_source[source_to_target] == source_indices
    # Keep a robust distance tail.  FPFH magnitudes vary with visible surface
    # fraction, so an absolute descriptor threshold is not portable.
    limit = float(np.percentile(source_to_target_distance, 65.0))
    mutual &= source_to_target_distance <= max(limit, 1e-12)
    return source_indices[mutual], source_to_target[mutual]


def _register_one(o3d, teaserpp, points: np.ndarray, item: dict, table_z: float):
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
    template = (
        _box_surface(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, voxel)
        if category == "cube"
        else _open_bin_surface(1.0, voxel)
    )
    source = _cloud(o3d, template, voxel)
    target = _cloud(o3d, crop, voxel)
    source_features = _features(o3d, source, voxel)
    target_features = _features(o3d, target, voxel)
    source_index, target_index = _mutual_feature_correspondences(source_features, target_features)
    if len(source_index) < 8:
        return dict(item), {
            "accepted": False,
            "reason": "insufficient_feature_correspondences",
            "points": int(len(crop)),
            "correspondences": int(len(source_index)),
        }

    source_points = np.asarray(source.points)[source_index].T.copy(order="F")
    target_points = np.asarray(target.points)[target_index].T.copy(order="F")
    params = teaserpp.RobustRegistrationSolver.Params()
    params.cbar2 = 1.0
    params.noise_bound = voxel * 2.5
    params.estimate_scaling = True
    params.rotation_estimation_algorithm = (
        teaserpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-10
    solver = teaserpp.RobustRegistrationSolver(params)
    solver.solve(source_points, target_points)
    solution = solver.getSolution()
    scale = float(solution.scale)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(solution.rotation)
    transform[:3, 3] = np.asarray(solution.translation).reshape(3)

    scaled = source.scale(scale, center=(0.0, 0.0, 0.0))
    try:
        refined = o3d.pipelines.registration.registration_icp(
            scaled,
            target,
            voxel * 1.8,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
        )
        transform = np.asarray(refined.transformation)
        fitness = float(refined.fitness)
        rmse = float(refined.inlier_rmse)
    except RuntimeError:
        fitness, rmse = 0.0, math.inf
    xy = transform[:2, 3]
    tilt = _tilt_degrees(transform[:3, :3])
    displacement = float(np.linalg.norm(xy - centre))
    bottom_error = abs(float(transform[2, 3]) - table_z)
    accepted = bool(
        0.80 <= scale <= 1.20
        and fitness >= (0.12 if category == "cube" else 0.08)
        and displacement <= (0.010 if category == "cube" else 0.016)
        and bottom_error <= 0.012
        and tilt <= 25.0
    )
    output = dict(item)
    if accepted:
        yaw = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        output["xy"] = xy.tolist()
        output["yaw_deg"] = float(canonical_square_yaw(yaw))
        output["estimated_scale"] = scale
        output["confidence"] = float(np.clip(0.35 + 0.60 * fitness, 0.05, 0.995))
        output["source"] = "cad_fpfh_teaser_icp"
    return output, {
        "accepted": accepted,
        "points": int(len(crop)),
        "correspondences": int(len(source_index)),
        "fitness": fitness,
        "inlier_rmse_mm": rmse * 1000.0,
        "seed_displacement_mm": displacement * 1000.0,
        "bottom_error_mm": bottom_error * 1000.0,
        "tilt_deg": tilt,
        "scale": scale,
    }


class TeaserCADPolicy(BasePolicy):
    name = "B3_cad_fpfh_teaser_icp"

    def __init__(self):
        self.seed = GeometryOBBScalePolicy()

    def predict(self, observation: SceneObservation) -> dict:
        import open3d as o3d
        import teaserpp_python as teaserpp

        started = time.perf_counter()
        corrected = _replace_fixed_mounts(observation)
        seeds = self.seed.predict(corrected)["detections"]
        points = _fused_points(corrected)
        table_z = corrected.cameras["table"].camera.table_z
        detections, diagnostics = [], []
        for item in seeds:
            refined, diagnostic = _register_one(o3d, teaserpp, points, item, table_z)
            diagnostic.update({"category": item["category"], "colour": item.get("colour")})
            detections.append(refined)
            diagnostics.append(diagnostic)
        result = self._result(
            corrected,
            detections,
            started,
            ["official TEASER++ robust registration from FPFH correspondences; ICP and gravity plausibility gate"],
        )
        result["registration_diagnostics"] = diagnostics
        return result

