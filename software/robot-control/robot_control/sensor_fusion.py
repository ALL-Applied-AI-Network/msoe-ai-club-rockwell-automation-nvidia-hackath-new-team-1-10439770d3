"""Cross-check RGB geometry with calibrated table/wrist depth evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .depth_vision import DepthCluster


class SensorFusionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FusedBinPose:
    center_base_mm: tuple[float, float, float]
    wall_center_base_mm: tuple[float, float, float]
    gripper_yaw_deg: float
    rgb_depth_disagreement_mm: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _candidate_bin_clusters(clusters: list[DepthCluster]) -> list[DepthCluster]:
    return [
        item
        for item in clusters
        if item.category_hint == "bin"
        or (
            item.principal_extents_xy_m[0] >= 0.030
            and item.height_over_surface_m >= 0.020
            and item.confidence >= 0.40
        )
    ]


def fuse_bin_pose(rgb_detection, depth_clusters: list[DepthCluster]) -> FusedBinPose:
    """Fuse a calibrated RGB bin outline with a metric depth cluster.

    Metric table depth supplies centre and wall orientation. RGB supplies an
    independent identity/outline check. Disagreement beyond 25 mm fails
    closed rather than averaging incompatible detections.
    """

    rgb_center = np.asarray(rgb_detection.bin_center_base_mm, dtype=float)
    rgb_wall = np.asarray(rgb_detection.south_wall_center_base_mm, dtype=float)
    candidates = _candidate_bin_clusters(depth_clusters)
    if not candidates:
        raise SensorFusionError("depth did not contain a bin-scale cluster")
    depth = min(
        candidates,
        key=lambda item: float(
            np.linalg.norm(np.asarray(item.centroid_base_m[:2]) * 1000.0 - rgb_center[:2])
        ),
    )
    depth_center = np.asarray(depth.centroid_base_m, dtype=float) * 1000.0
    disagreement = float(np.linalg.norm(depth_center[:2] - rgb_center[:2]))
    if disagreement > 25.0:
        raise SensorFusionError(
            f"RGB/depth bin centres disagree by {disagreement:.1f} mm"
        )

    rgb_direction = rgb_wall[:2] - rgb_center[:2]
    norm = float(np.linalg.norm(rgb_direction))
    if not 15.0 <= norm <= 30.0:
        raise SensorFusionError("RGB wall normal has an implausible length")
    rgb_yaw = float(rgb_detection.gripper_yaw_deg)
    depth_yaw = float(depth.yaw_deg)
    yaw_disagreement = abs((depth_yaw - rgb_yaw + 45.0) % 90.0 - 45.0)
    if yaw_disagreement > 15.0:
        raise SensorFusionError(
            f"RGB/depth wall orientations disagree by {yaw_disagreement:.1f} degrees"
        )
    yaw_rad = math.radians(depth_yaw)
    direction = np.array((math.sin(yaw_rad), -math.cos(yaw_rad)), dtype=float)
    # Use the metric depth cloud for translation.  A top-down RGB silhouette of
    # an open bin is systematically biased toward its visible front wall even
    # when camera calibration is excellent; RGB therefore supplies independent
    # identity, cross-checking and wall orientation, not a second noisy centre.
    fused_xy = depth_center[:2]
    wall_xy = fused_xy + direction * 21.25
    confidence = float(
        min(
            float(rgb_detection.confidence),
            float(depth.confidence),
            max(0.0, 1.0 - disagreement / 25.0),
            max(0.0, 1.0 - yaw_disagreement / 15.0),
        )
    )
    if confidence < 0.45:
        raise SensorFusionError(f"fused bin confidence is too low: {confidence:.3f}")
    return FusedBinPose(
        center_base_mm=(float(fused_xy[0]), float(fused_xy[1]), float(depth_center[2])),
        wall_center_base_mm=(float(wall_xy[0]), float(wall_xy[1]), float(depth_center[2])),
        gripper_yaw_deg=depth_yaw,
        rgb_depth_disagreement_mm=disagreement,
        confidence=confidence,
    )


class LiftEvidenceTracker:
    """Verify micro-lift, full lift and retention from metric depth over time."""

    def __init__(self, center_xy_mm: tuple[float, float], baseline_z_mm: float):
        self.center_xy_mm = np.asarray(center_xy_mm, dtype=float)
        self.baseline_z_mm = float(baseline_z_mm)
        self.last_z_mm = float(baseline_z_mm)

    def observe(self, clusters: list[DepthCluster], maximum_xy_error_mm: float = 30.0) -> float:
        candidates = _candidate_bin_clusters(clusters)
        if not candidates:
            raise SensorFusionError("bin disappeared from both the expected scale and height gate")
        item = min(
            candidates,
            key=lambda cluster: float(
                np.linalg.norm(np.asarray(cluster.centroid_base_m[:2]) * 1000 - self.center_xy_mm)
            ),
        )
        xy = np.asarray(item.centroid_base_m[:2]) * 1000.0
        error = float(np.linalg.norm(xy - self.center_xy_mm))
        if error > maximum_xy_error_mm:
            raise SensorFusionError(f"tracked bin moved laterally by {error:.1f} mm")
        self.center_xy_mm = xy
        self.last_z_mm = float(item.centroid_base_m[2] * 1000.0)
        return self.last_z_mm - self.baseline_z_mm

    def gate(self, clusters: list[DepthCluster], minimum_lift_mm: float) -> bool:
        return self.observe(clusters) >= float(minimum_lift_mm)


def wrist_presence_crosscheck(
    clusters: list[DepthCluster],
    expected_center_xy_mm: tuple[float, float] | np.ndarray,
    *,
    maximum_xy_error_mm: float = 20.0,
    minimum_confidence: float = 0.35,
) -> tuple[bool, float | None]:
    """Use wrist depth as a presence vote, never as grasp translation.

    The fingers can occlude one side of the bin, so the wrist footprint may be
    incomplete. It is still useful as independent near-field evidence when a
    bin-scale cluster remains close to the table-camera track.
    """

    expected = np.asarray(expected_center_xy_mm, dtype=float)
    candidates = [
        item
        for item in _candidate_bin_clusters(clusters)
        if item.confidence >= float(minimum_confidence)
    ]
    if not candidates:
        return False, None
    distances = [
        float(
            np.linalg.norm(
                np.asarray(item.centroid_base_m[:2], dtype=float) * 1000.0 - expected
            )
        )
        for item in candidates
    ]
    distance = min(distances)
    return distance <= float(maximum_xy_error_mm), distance
