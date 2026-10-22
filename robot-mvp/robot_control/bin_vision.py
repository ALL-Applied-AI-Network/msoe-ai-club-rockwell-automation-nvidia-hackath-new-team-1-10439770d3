"""Conservative colour-camera localization for the official green bin.

The detector accepts one saturated green region of the expected apparent size,
maps its centre through the organizer-provided table-camera calibration, and
returns the centreline of the wall facing robot -Y. It raises instead of
inventing a target when the evidence is ambiguous.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


BIN_TOP_Z_M = 0.0499
BIN_WALL_CENTER_OFFSET_M = 0.02125


class BinVisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class BinDetection:
    image_size: tuple[int, int]
    pixel_center: tuple[float, float]
    contour_area_px: float
    bbox_px: tuple[int, int, int, int]
    bin_center_base_mm: tuple[float, float, float]
    south_wall_center_base_mm: tuple[float, float, float]
    gripper_yaw_deg: float
    fitted_edge_lengths_mm: tuple[float, float, float, float]
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _modulo_gripper_yaw(angle_deg: float) -> float:
    """Parallel-jaw closing axes repeat every 180 degrees."""
    return (float(angle_deg) + 90.0) % 180.0 - 90.0


def locate_green_bin(rgb: np.ndarray) -> BinDetection:
    """Locate an upright green 50 mm bin in a table-camera RGB frame."""

    # The calibration package belongs to the official SIDC client. Import it
    # lazily so non-vision MVP tests can run before installation into that
    # client while physical execution still requires the official calibration.
    from calibration.geometry import load, pixel_to_base

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise BinVisionError("table image must be HxWx3 RGB uint8")
    height, width = image.shape[:2]
    if width < 320 or height < 180:
        raise BinVisionError(f"table image is too small: {width}x{height}")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array((35, 55, 35), dtype=np.uint8),
        np.array((95, 255, 255), dtype=np.uint8),
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [contour for contour in contours if cv2.contourArea(contour) >= 150.0]
    candidates.sort(key=cv2.contourArea, reverse=True)
    if not candidates:
        raise BinVisionError("no green-bin-sized region found")
    if len(candidates) > 1 and cv2.contourArea(candidates[1]) > 0.45 * cv2.contourArea(candidates[0]):
        raise BinVisionError("multiple similarly sized green regions found")

    contour = candidates[0]
    area = float(cv2.contourArea(contour))
    area_fraction = area / float(width * height)
    if not 0.0005 <= area_fraction <= 0.02:
        raise BinVisionError(f"green region has implausible area fraction {area_fraction:.5f}")
    x, y, w, h = [int(value) for value in cv2.boundingRect(contour)]
    if min(w, h) < 15 or max(w, h) > 100:
        raise BinVisionError(f"green region has implausible bounding box {w}x{h}")

    moments = cv2.moments(contour)
    if not moments["m00"]:
        raise BinVisionError("green region has zero contour moment")
    u = float(moments["m10"] / moments["m00"])
    v = float(moments["m01"] / moments["m00"])

    camera = load()["table"]
    calibration_width, calibration_height = camera["size"]

    def to_base(pixel) -> np.ndarray:
        scaled = (
            float(pixel[0]) * calibration_width / width,
            float(pixel[1]) * calibration_height / height,
        )
        return pixel_to_base(camera, scaled, z=BIN_TOP_Z_M)

    center = to_base((u, v))
    if not (0.20 <= center[0] <= 0.65 and -0.45 <= center[1] <= 0.35):
        raise BinVisionError(
            f"detected centre lies outside the approved workspace: {center.tolist()}"
        )

    rectangle_px = cv2.boxPoints(cv2.minAreaRect(contour))
    rectangle_base = np.asarray([to_base(pixel)[:2] for pixel in rectangle_px])
    edges = [
        rectangle_base[(index + 1) % 4] - rectangle_base[index]
        for index in range(4)
    ]
    lengths = np.asarray([np.linalg.norm(edge) for edge in edges])
    if np.any(lengths < 0.030) or np.any(lengths > 0.105):
        raise BinVisionError(
            "fitted green region does not have bin-scale edges: "
            + ", ".join(f"{length * 1000:.1f}" for length in lengths)
            + " mm"
        )

    # A square's yaw is ambiguous by 90 degrees; choose the fitted edge normal
    # that points most strongly toward robot -Y, then grip that wall.
    normals = []
    for edge, length in zip(edges, lengths):
        tangent = edge / length
        normal = np.array((-tangent[1], tangent[0]))
        if normal[1] > 0:
            normal = -normal
        normals.append(normal)
    south_normal = min(normals, key=lambda normal: float(normal[1]))
    wall_center = center.copy()
    wall_center[:2] += south_normal * BIN_WALL_CENTER_OFFSET_M
    yaw = _modulo_gripper_yaw(math.degrees(math.atan2(-south_normal[0], south_normal[1])))

    dimension_error = float(np.mean(np.abs(lengths - 0.050)))
    size_score = max(0.0, 1.0 - dimension_error / 0.050)
    area_score = min(1.0, area / 500.0)
    confidence = min(size_score, area_score)
    if confidence < 0.40:
        raise BinVisionError(f"green-bin confidence is too low: {confidence:.3f}")

    return BinDetection(
        image_size=(width, height),
        pixel_center=(u, v),
        contour_area_px=area,
        bbox_px=(x, y, w, h),
        bin_center_base_mm=tuple(float(value * 1000.0) for value in center),
        south_wall_center_base_mm=tuple(float(value * 1000.0) for value in wall_center),
        gripper_yaw_deg=float(yaw),
        fitted_edge_lengths_mm=tuple(float(value * 1000.0) for value in lengths),
        confidence=float(confidence),
    )
