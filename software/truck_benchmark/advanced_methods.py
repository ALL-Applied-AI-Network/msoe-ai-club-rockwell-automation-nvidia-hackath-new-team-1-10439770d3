"""Advanced observation-only candidates for the A--K research campaign.

These policies deliberately remain independent of Isaac Sim.  They consume the
same serialized RGB, depth, calibration, joint state, and timestamp packets as
the original benchmark.  Heavy neural candidates live in a separate runner so
their optional dependencies cannot compromise these deterministic methods.
"""

from __future__ import annotations

from dataclasses import replace
import itertools
import math
import time

import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from .camera_geometry import CameraModel, canonical_axis_yaw, canonical_square_yaw
from .policies import (
    BIN_SIZE,
    CUBE_SIZE,
    TRUSTED_CAMERA_TO_WORLD,
    BasePolicy,
    CameraObservation,
    DepthGeometryPolicy,
    LockedCalibrationPolicy,
    SceneObservation,
    _deduplicate,
    _depth_detections,
    _fuse_groups,
    _pixel_world_maps,
    _task_inventory_filter,
)


def _replace_fixed_mounts(observation: SceneObservation) -> SceneObservation:
    cameras: dict[str, CameraObservation] = {}
    for name, item in observation.cameras.items():
        model = item.camera
        if name in TRUSTED_CAMERA_TO_WORLD:
            model = replace(model, camera_to_world=TRUSTED_CAMERA_TO_WORLD[name].copy())
        cameras[name] = CameraObservation(name, item.rgb_bgr, item.depth, model)
    return SceneObservation(
        scene_id=observation.scene_id,
        cameras=cameras,
        robot_state=dict(observation.robot_state),
        variation=dict(observation.variation),
    )


def _fit_table_plane(camera_obs: CameraObservation) -> tuple[np.ndarray, float, int]:
    """Robustly fit z=ax+by+c in the reported world frame."""

    if camera_obs.depth is None:
        return np.asarray((0.0, 0.0, 1.0)), camera_obs.camera.table_z, 0
    points, _, _ = camera_obs.camera.depth_to_world(camera_obs.depth, stride=4)
    use = (
        (points[:, 0] > 0.05)
        & (points[:, 0] < 0.80)
        & (points[:, 1] > -0.48)
        & (points[:, 1] < 0.38)
        & (np.abs(points[:, 2] - camera_obs.camera.table_z) < 0.035)
    )
    points = points[use]
    if len(points) < 100:
        return np.asarray((0.0, 0.0, 1.0)), camera_obs.camera.table_z, 0
    # Iteratively trim geometry above the dominant tabletop plane.
    selected = points
    coeff = np.asarray((0.0, 0.0, camera_obs.camera.table_z))
    for _ in range(4):
        design = np.column_stack((selected[:, 0], selected[:, 1], np.ones(len(selected))))
        coeff, *_ = np.linalg.lstsq(design, selected[:, 2], rcond=None)
        residual = points[:, 2] - (points[:, :2] @ coeff[:2] + coeff[2])
        scale = max(0.00035, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        selected = points[np.abs(residual) < 2.5 * scale]
        if len(selected) < 100:
            break
    normal = np.asarray((-coeff[0], -coeff[1], 1.0), dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    if normal[2] < 0:
        normal *= -1.0
    height = float(np.median(selected[:, 2])) if len(selected) else float(coeff[2])
    return normal, height, int(len(selected))


def _plane_correct_camera(camera_obs: CameraObservation) -> tuple[CameraObservation, dict]:
    normal, height, support = _fit_table_plane(camera_obs)
    # Column-vector rotation maps the observed normal onto +Z.  CameraModel
    # stores row-vector transforms, hence the transpose in the update.
    rotation, _ = Rotation.align_vectors(
        np.asarray(((0.0, 0.0, 1.0),)),
        np.asarray((normal,)),
    )
    correction = rotation.as_matrix()
    matrix = camera_obs.camera.camera_to_world.copy()
    matrix[:3, :3] = matrix[:3, :3] @ correction.T
    corrected_model = replace(camera_obs.camera, camera_to_world=matrix)
    probe = CameraObservation(camera_obs.name, camera_obs.rgb_bgr, camera_obs.depth, corrected_model)
    _, corrected_height, _ = _fit_table_plane(probe)
    matrix = matrix.copy()
    matrix[3, 2] += camera_obs.camera.table_z - corrected_height
    corrected_model = replace(camera_obs.camera, camera_to_world=matrix)
    tilt_deg = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    return (
        CameraObservation(camera_obs.name, camera_obs.rgb_bgr, camera_obs.depth, corrected_model),
        {
            "camera": camera_obs.name,
            "plane_support": support,
            "normal": normal.tolist(),
            "tilt_correction_deg": tilt_deg,
            "z_correction_mm": (camera_obs.camera.table_z - corrected_height) * 1000.0,
            "unobservable_dofs": ["x", "y", "yaw"],
        },
    )


class PlaneSelfCalibrationPolicy(BasePolicy):
    """A: live table-plane roll/pitch/Z correction before depth fusion."""

    name = "A_plane_selfcal_multiview_rawcube"

    def predict(self, observation: SceneObservation) -> dict:
        started = time.perf_counter()
        cameras: dict[str, CameraObservation] = {}
        diagnostics = []
        for name, item in observation.cameras.items():
            corrected, diagnostic = _plane_correct_camera(item)
            cameras[name] = corrected
            diagnostics.append(diagnostic)
        corrected_observation = SceneObservation(
            observation.scene_id,
            cameras,
            dict(observation.robot_state),
            dict(observation.variation),
        )
        groups = [
            _depth_detections(cameras[name], 0.006, 1, cube_pose_mode="raw", truck_pose_mode="achromatic")
            for name in ("table", "side", "wrist")
            if name in cameras
        ]
        detections = _fuse_groups(groups, match_radius=0.020)
        output = self._result(
            corrected_observation,
            _task_inventory_filter(detections),
            started,
            ["table-plane self-calibration corrects roll/pitch/Z; planar XY/yaw remain unobservable"],
        )
        output["calibration_diagnostics"] = diagnostics
        return output


class GeometryOBBScalePolicy(BasePolicy):
    """B: lighting-independent OBB geometry with observed scale estimates."""

    name = "B_geometry_obb_scale_cad"

    def predict(self, observation: SceneObservation) -> dict:
        started = time.perf_counter()
        corrected = _replace_fixed_mounts(observation)
        groups = [
            _depth_detections(
                corrected.cameras[name],
                0.006,
                1,
                cube_pose_mode="raw",
                truck_pose_mode="achromatic",
            )
            for name in ("table", "side", "wrist")
            if name in corrected.cameras
        ]
        detections = _fuse_groups(groups, match_radius=0.018)
        for item in detections:
            expected = CUBE_SIZE if item["category"] == "cube" else BIN_SIZE if item["category"] == "bin" else 0.18085018
            extents = np.asarray(item.get("raw_extents_m", item.get("extents_m", (expected, expected))), float)
            item["estimated_scale"] = float(np.clip(np.median(extents) / expected, 0.80, 1.20))
            item["source"] = "cad_scale_adaptive_obb"
        return self._result(
            corrected,
            _task_inventory_filter(detections),
            started,
            ["RANSAC-equivalent table rejection + Euclidean components + scale-adaptive planar OBB"],
        )


def _cuboid_edges(size_x: float, size_y: float, height: float, xy: np.ndarray, yaw_deg: float, z0: float):
    angle = math.radians(float(yaw_deg))
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    corners = []
    for z in (z0, z0 + height):
        for local in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            p = np.asarray(xy) + rotation @ (np.asarray(local) * np.asarray((size_x, size_y)))
            corners.append(np.asarray((p[0], p[1], z)))
    indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return corners, indices


def _edge_distance_maps(observation: SceneObservation) -> dict[str, np.ndarray]:
    maps = {}
    for name, item in observation.cameras.items():
        gray = cv2.cvtColor(item.rgb_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
        edge = cv2.Canny(gray, 45, 125)
        maps[name] = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return maps


def _wireframe_score(
    observation: SceneObservation,
    distance_maps: dict[str, np.ndarray],
    xy: np.ndarray,
    yaw: float,
    scale: float,
    size: tuple[float, float],
    height: float,
) -> float:
    costs = []
    for name in ("table", "side", "wrist"):
        item = observation.cameras.get(name)
        if item is None or name not in distance_maps:
            continue
        corners, indices = _cuboid_edges(
            size[0] * scale,
            size[1] * scale,
            height * scale,
            xy,
            yaw,
            item.camera.table_z,
        )
        values = []
        for a, b in indices:
            for t in np.linspace(0.0, 1.0, 13):
                uv = item.camera.project(corners[a] * (1.0 - t) + corners[b] * t)
                if uv is None:
                    continue
                u, v = int(round(uv[0])), int(round(uv[1]))
                if 0 <= u < item.camera.width and 0 <= v < item.camera.height:
                    values.append(min(float(distance_maps[name][v, u]), 20.0))
        if values:
            # Normalize in pixels but retain a robust tail penalty so one
            # coincident unrelated edge cannot dominate the result.
            costs.append(float(np.mean(values) + 0.25 * np.percentile(values, 75)))
    return float(np.mean(costs)) if costs else math.inf


def _edge_refine(
    observation: SceneObservation,
    distance_maps: dict[str, np.ndarray],
    detection: dict,
    multiview: bool = True,
) -> dict:
    category = detection["category"]
    if category not in ("cube", "bin"):
        return dict(detection)
    if not multiview:
        observation = SceneObservation(
            observation.scene_id,
            {"table": observation.cameras["table"]},
            observation.robot_state,
            observation.variation,
        )
    size = (CUBE_SIZE, CUBE_SIZE) if category == "cube" else (BIN_SIZE, BIN_SIZE)
    height = CUBE_SIZE if category == "cube" else 0.054
    centre = np.asarray(detection["xy"], float)
    yaw0 = float(detection["yaw_deg"])
    scale0 = float(detection.get("estimated_scale", 1.0))
    best = (math.inf, centre, yaw0, scale0)
    # Depth supplies only a bounded initialization.  RGB edges own the final
    # candidate selection.
    for dx, dy, dyaw, dscale in itertools.product(
        (-0.003, 0.0, 0.003),
        (-0.003, 0.0, 0.003),
        (-8.0, -4.0, 0.0, 4.0, 8.0),
        (-0.08, 0.0, 0.08),
    ):
        xy = centre + np.asarray((dx, dy))
        yaw = canonical_square_yaw(yaw0 + dyaw)
        scale = float(np.clip(scale0 + dscale, 0.85, 1.15))
        score = _wireframe_score(observation, distance_maps, xy, yaw, scale, size, height)
        if score < best[0]:
            best = (score, xy, yaw, scale)
    refined = dict(detection)
    refined["xy"] = best[1].tolist()
    refined["yaw_deg"] = float(best[2])
    refined["estimated_scale"] = float(best[3])
    refined["edge_cost_px"] = float(best[0])
    refined["confidence"] = float(np.clip(math.exp(-best[0] / 7.0), 0.05, 0.995))
    refined["source"] = "multiview_cad_edge_render_compare" if multiview else "table_wireframe_pnl"
    return refined


class MultiViewCADEdgePolicy(BasePolicy):
    """F: coarse-depth initialization followed by RGB-only CAD edge fitting."""

    name = "F_multiview_cad_edge_render_compare"

    def __init__(self):
        self.seed = GeometryOBBScalePolicy()

    def predict(self, observation: SceneObservation) -> dict:
        started = time.perf_counter()
        corrected = _replace_fixed_mounts(observation)
        seeds = self.seed.predict(corrected)["detections"]
        maps = _edge_distance_maps(corrected)
        detections = [_edge_refine(corrected, maps, item, multiview=True) for item in seeds]
        return self._result(
            corrected,
            _task_inventory_filter(_deduplicate(detections, 0.008)),
            started,
            ["rough depth centroid only; final cube/bin x/y/yaw/scale selected by three-view RGB CAD edges"],
        )


def _foreground_masks(observation: SceneObservation) -> dict[str, np.ndarray]:
    masks = {}
    for name, item in observation.cameras.items():
        hsv = cv2.cvtColor(item.rgb_bgr, cv2.COLOR_BGR2HSV)
        # White table is low-saturation and bright.  Keep both coloured parts
        # and dark/gray geometry; local seed bounds suppress the distant robot.
        mask = ((hsv[..., 1] > 28) | (hsv[..., 2] < 205)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        masks[name] = mask
    return masks


def _visual_hull_points(
    observation: SceneObservation,
    masks: dict[str, np.ndarray],
    detection: dict,
) -> np.ndarray:
    category = detection["category"]
    nominal = CUBE_SIZE if category == "cube" else BIN_SIZE
    height = CUBE_SIZE if category == "cube" else 0.054
    centre = np.asarray(detection["xy"], float)
    radius = nominal * (1.10 if category == "cube" else 0.75)
    step = 0.002 if category == "cube" else 0.003
    xs = np.arange(centre[0] - radius, centre[0] + radius + step / 2, step)
    ys = np.arange(centre[1] - radius, centre[1] + radius + step / 2, step)
    zs = np.arange(observation.cameras["table"].camera.table_z + step / 2,
                   observation.cameras["table"].camera.table_z + height * 1.15, step)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    voxels = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    keep = np.ones(len(voxels), dtype=bool)
    support_views = 0
    for name in ("table", "side", "wrist"):
        item = observation.cameras.get(name)
        if item is None:
            continue
        support_views += 1
        projected = []
        for point in voxels:
            uv = item.camera.project(point)
            projected.append((-1.0, -1.0) if uv is None else uv)
        uv = np.asarray(projected)
        u = np.rint(uv[:, 0]).astype(int)
        v = np.rint(uv[:, 1]).astype(int)
        inside = (u >= 0) & (u < item.camera.width) & (v >= 0) & (v < item.camera.height)
        view_keep = np.zeros(len(voxels), dtype=bool)
        view_keep[inside] = masks[name][v[inside], u[inside]] > 0
        keep &= view_keep
    return voxels[keep] if support_views >= 2 else np.empty((0, 3), dtype=float)


def _box_sdf(points_xy: np.ndarray, centre: np.ndarray, yaw: float, half_extent: float) -> np.ndarray:
    angle = -math.radians(float(yaw))
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    local = (np.asarray(points_xy) - np.asarray(centre)) @ rotation.T
    q = np.abs(local) - half_extent
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outside + inside


def _sdf_refine(detection: dict, hull: np.ndarray) -> dict:
    if len(hull) < 16 or detection["category"] not in ("cube", "bin"):
        output = dict(detection)
        output["visual_hull_voxels"] = int(len(hull))
        return output
    nominal = CUBE_SIZE if detection["category"] == "cube" else BIN_SIZE
    xy0 = np.asarray(detection["xy"], float)
    # Boundary of the carved XY support, approximated on a 1 mm occupancy map.
    lo = hull[:, :2].min(axis=0) - 0.001
    hi = hull[:, :2].max(axis=0) + 0.001
    shape = np.maximum(3, np.ceil((hi - lo) / 0.001).astype(int) + 1)
    occ = np.zeros(tuple(shape), dtype=bool)
    ij = np.clip(np.rint((hull[:, :2] - lo) / 0.001).astype(int), 0, shape - 1)
    occ[ij[:, 0], ij[:, 1]] = True
    edge = occ & ~binary_erosion(occ)
    edge_ij = np.argwhere(edge)
    surface = lo + edge_ij * 0.001
    best = (math.inf, xy0, float(detection["yaw_deg"]), 1.0)
    for dx, dy, yaw, scale in itertools.product(
        (-0.003, 0.0, 0.003),
        (-0.003, 0.0, 0.003),
        np.arange(-45.0, 45.0, 5.0),
        (0.90, 1.00, 1.10),
    ):
        centre = xy0 + np.asarray((dx, dy))
        cost = float(np.mean(np.abs(_box_sdf(surface, centre, yaw, nominal * scale / 2.0))))
        if cost < best[0]:
            best = (cost, centre, canonical_square_yaw(float(yaw)), float(scale))
    output = dict(detection)
    output["xy"] = best[1].tolist()
    output["yaw_deg"] = best[2]
    output["estimated_scale"] = best[3]
    output["sdf_surface_error_mm"] = best[0] * 1000.0
    output["visual_hull_voxels"] = int(len(hull))
    output["confidence"] = float(np.clip(math.exp(-best[0] / 0.004), 0.05, 0.995))
    output["source"] = "triview_visual_hull_sdf"
    return output


class VisualHullSDFPolicy(BasePolicy):
    """H: three-view RGB space carving followed by planar analytic SDF fit."""

    name = "H_triview_visual_hull_sdf"

    def __init__(self):
        self.seed = GeometryOBBScalePolicy()

    def predict(self, observation: SceneObservation) -> dict:
        started = time.perf_counter()
        corrected = _replace_fixed_mounts(observation)
        seeds = self.seed.predict(corrected)["detections"]
        masks = _foreground_masks(corrected)
        detections = []
        for item in seeds:
            hull = _visual_hull_points(corrected, masks, item) if item["category"] in ("cube", "bin") else np.empty((0, 3))
            detections.append(_sdf_refine(item, hull))
        return self._result(
            corrected,
            _task_inventory_filter(_deduplicate(detections, 0.008)),
            started,
            ["RGB foreground visual hull; coarse depth policy supplies only bounded object seeds"],
        )


class WireframePnLPolicy(BasePolicy):
    """I: fast single-view line/wireframe refinement."""

    name = "I_table_wireframe_pnl"

    def __init__(self):
        self.seed = GeometryOBBScalePolicy()

    def predict(self, observation: SceneObservation) -> dict:
        started = time.perf_counter()
        corrected = _replace_fixed_mounts(observation)
        seeds = self.seed.predict(corrected)["detections"]
        maps = _edge_distance_maps(corrected)
        detections = [_edge_refine(corrected, maps, item, multiview=False) for item in seeds]
        return self._result(
            corrected,
            _task_inventory_filter(_deduplicate(detections, 0.008)),
            started,
            ["table RGB line/corner alignment; planar PnL search around the coarse geometry seed"],
        )


class GravityPlanarPolicy(BasePolicy):
    """J: explicit gravity-constrained 4-parameter CAD state."""

    name = "J_gravity_planar_xy_yaw_scale"

    def __init__(self):
        self.geometry = GeometryOBBScalePolicy()

    def predict(self, observation: SceneObservation) -> dict:
        started = time.perf_counter()
        result = self.geometry.predict(observation)
        detections = []
        table_z = observation.cameras["table"].camera.table_z
        for item in result["detections"]:
            constrained = dict(item)
            constrained["roll_deg"] = 0.0
            constrained["pitch_deg"] = 0.0
            constrained["bottom_z_m"] = float(table_z)
            constrained["state_dof"] = ["x", "y", "yaw", "scale"]
            constrained["source"] = "gravity_constrained_cad"
            detections.append(constrained)
        return self._result(
            observation,
            detections,
            started,
            ["roll/pitch locked to the live table normal; bottom face clamped to the table"],
        )


def advanced_policy_registry() -> dict[str, BasePolicy]:
    policies: list[BasePolicy] = [
        PlaneSelfCalibrationPolicy(),
        GeometryOBBScalePolicy(),
        # D is the already implemented active-view method and is added by the
        # campaign runner from the original registry to keep one source of truth.
        MultiViewCADEdgePolicy(),
        VisualHullSDFPolicy(),
        WireframePnLPolicy(),
        GravityPlanarPolicy(),
    ]
    return {policy.name: policy for policy in policies}
