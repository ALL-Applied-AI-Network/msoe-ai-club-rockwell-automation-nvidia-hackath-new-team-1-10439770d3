"""Calibrated depth clustering for the hackathon table and wrist cameras.

The organizer supplies rectified depth plus camera-to-base calibration.  This
module deliberately keeps metric geometry deterministic: foundation models may
name an object or interpret an instruction, but they do not invent grasp
coordinates.  All returned positions are in the FANUC base frame, in metres.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


SURFACE_Z_M = -0.0051
WORKSPACE_X_M = (0.20, 0.65)
WORKSPACE_Y_M = (-0.45, 0.35)


class DepthVisionError(RuntimeError):
    """Raised when calibrated depth is missing or cannot support a decision."""


@dataclass(frozen=True)
class DepthCluster:
    centroid_base_m: tuple[float, float, float]
    minimum_base_m: tuple[float, float, float]
    maximum_base_m: tuple[float, float, float]
    principal_extents_xy_m: tuple[float, float]
    yaw_deg: float
    height_over_surface_m: float
    pixel_bbox: tuple[int, int, int, int]
    point_count: int
    category_hint: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _category(extents: np.ndarray, height: float) -> tuple[str, float]:
    long_side, short_side = sorted((float(extents[0]), float(extents[1])), reverse=True)
    if 0.035 <= long_side <= 0.080 and 0.025 <= short_side <= 0.080 and height >= 0.020:
        error = abs(long_side - 0.050) + abs(short_side - 0.050)
        return "bin", max(0.0, 1.0 - error / 0.060)
    if 0.006 <= long_side <= 0.024 and 0.005 <= short_side <= 0.024 and 0.004 <= height <= 0.030:
        squareness = short_side / max(long_side, 1e-9)
        return "cube", min(1.0, squareness / 0.70)
    if 0.018 <= long_side <= 0.045 and short_side <= 0.018 and height <= 0.018:
        elongation = long_side / max(short_side, 0.002)
        return "screw", min(1.0, elongation / 3.0)
    return "unknown", 0.25


def clusters_from_base_points(
    points_base_m: np.ndarray,
    pixels_uv: np.ndarray,
    image_shape: tuple[int, int],
    *,
    minimum_height_m: float = 0.002,
    maximum_height_m: float = 0.25,
    minimum_points: int = 20,
) -> list[DepthCluster]:
    """Cluster object points already transformed into the robot base frame.

    Separating this operation from camera projection makes it independently
    testable and permits the exact same logic for fixed table and moving wrist
    depth streams.
    """

    points = np.asarray(points_base_m, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels_uv, dtype=np.int32).reshape(-1, 2)
    if len(points) != len(pixels):
        raise DepthVisionError("point and pixel arrays have different lengths")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise DepthVisionError("invalid depth image shape")

    finite = np.all(np.isfinite(points), axis=1)
    workspace = (
        finite
        & (points[:, 0] >= WORKSPACE_X_M[0])
        & (points[:, 0] <= WORKSPACE_X_M[1])
        & (points[:, 1] >= WORKSPACE_Y_M[0])
        & (points[:, 1] <= WORKSPACE_Y_M[1])
        & (points[:, 2] >= SURFACE_Z_M + minimum_height_m)
        & (points[:, 2] <= SURFACE_Z_M + maximum_height_m)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    points, pixels = points[workspace], pixels[workspace]
    if len(points) < minimum_points:
        return []

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clusters: list[DepthCluster] = []
    for label in range(1, count):
        member = labels[pixels[:, 1], pixels[:, 0]] == label
        cloud = points[member]
        if len(cloud) < minimum_points:
            continue
        xy = cloud[:, :2]
        rectangle = cv2.minAreaRect(np.asarray(xy, dtype=np.float32))
        footprint_centre = np.asarray(rectangle[0], dtype=float)
        box = np.asarray(cv2.boxPoints(rectangle), dtype=float)
        edges = np.roll(box, -1, axis=0) - box
        edge_lengths = np.linalg.norm(edges, axis=1)
        edge_angles = [
            (float(np.degrees(np.arctan2(edge[1], edge[0]))) + 90.0) % 180.0
            - 90.0
            for edge in edges
        ]
        centre = np.array(
            (footprint_centre[0], footprint_centre[1], np.median(cloud[:, 2])),
            dtype=float,
        )
        extents = np.sort(np.asarray(rectangle[1], dtype=float))[::-1]
        minimum, maximum = np.min(cloud, axis=0), np.max(cloud, axis=0)
        object_height = float(maximum[2] - SURFACE_Z_M)
        category, shape_score = _category(extents, object_height)
        if category in {"bin", "cube"}:
            # A square repeats every 90 degrees.  Choose the edge closest to
            # robot X, which identifies the wall whose outward normal points
            # most toward robot -Y in our bounded +/-30 degree bin envelope.
            yaw = min(edge_angles, key=abs)
        else:
            yaw = edge_angles[int(np.argmax(edge_lengths))]
        point_score = min(1.0, len(cloud) / 250.0)
        confidence = float(min(shape_score, 0.35 + 0.65 * point_score))
        x, y, w, h, _ = [int(value) for value in stats[label]]
        clusters.append(
            DepthCluster(
                centroid_base_m=tuple(float(v) for v in centre),
                minimum_base_m=tuple(float(v) for v in minimum),
                maximum_base_m=tuple(float(v) for v in maximum),
                principal_extents_xy_m=tuple(float(v) for v in extents),
                yaw_deg=float(yaw),
                height_over_surface_m=object_height,
                pixel_bbox=(x, y, w, h),
                point_count=int(len(cloud)),
                category_hint=category,
                confidence=confidence,
            )
        )
    return sorted(clusters, key=lambda item: item.confidence, reverse=True)


def detect_depth_clusters(
    depth_mm: np.ndarray,
    stream: str,
    *,
    joints=None,
    stride: int = 2,
    minimum_points: int = 20,
    minimum_height_m: float = 0.0045,
) -> list[DepthCluster]:
    """Transform and cluster an organizer-provided table or wrist depth frame.

    ``minimum_height_m`` is deliberately explicit because the organizer's
    table-depth calibration reports a 2.31 mm plane residual.  The 4.5 mm
    default is a conservative general foreground gate; callers looking for a
    55 mm bin should use 6 mm so plane noise cannot merge the whole table into
    one component.
    """

    if stream not in {"table", "wrist"}:
        raise DepthVisionError("stream must be 'table' or 'wrist'")
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise DepthVisionError("depth must be a uint16 HxW image in millimetres")
    if stream == "wrist" and joints is None:
        raise DepthVisionError("wrist depth requires synchronized joints")
    from calibration.geometry import depth_to_base, load

    camera = load()[f"{stream}_depth"]
    points, pixels = depth_to_base(
        camera,
        depth,
        stride=stride,
        with_pixels=True,
        joints=joints,
    )
    return clusters_from_base_points(
        points,
        pixels,
        depth.shape,
        minimum_height_m=minimum_height_m,
        minimum_points=minimum_points,
    )
