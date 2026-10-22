"""Independent side-camera evidence for a green-bin lift.

Metric table/wrist depth is the primary measurement.  The fixed side RGB view
adds a deliberately different failure mode: a real lift must move the green
region upward in the image while retaining a plausible area and lateral
location.  This cue cannot choose a grasp coordinate and therefore cannot
silently steer the robot from uncalibrated pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


class SideVisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GreenRegion:
    center_px: tuple[float, float]
    bbox_px: tuple[int, int, int, int]
    area_px: float

    def to_dict(self) -> dict:
        return asdict(self)


def locate_green_region(rgb: np.ndarray) -> GreenRegion:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise SideVisionError("side image must be HxWx3 RGB uint8")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array((35, 55, 35), dtype=np.uint8),
        np.array((95, 255, 255), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [item for item in contours if cv2.contourArea(item) >= 100.0]
    if not contours:
        raise SideVisionError("side camera contains no green-bin-sized region")
    contours.sort(key=cv2.contourArea, reverse=True)
    if len(contours) > 1 and cv2.contourArea(contours[1]) > 0.50 * cv2.contourArea(contours[0]):
        raise SideVisionError("side camera contains multiple ambiguous green regions")
    contour = contours[0]
    moments = cv2.moments(contour)
    if not moments["m00"]:
        raise SideVisionError("side green region has zero contour moment")
    x, y, width, height = [int(value) for value in cv2.boundingRect(contour)]
    return GreenRegion(
        center_px=(
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ),
        bbox_px=(x, y, width, height),
        area_px=float(cv2.contourArea(contour)),
    )


class SideLiftTracker:
    def __init__(self, baseline: GreenRegion):
        self.baseline = baseline

    def observe(self, region: GreenRegion) -> float:
        area_ratio = region.area_px / max(self.baseline.area_px, 1.0)
        lateral_shift = abs(region.center_px[0] - self.baseline.center_px[0])
        if not 0.40 <= area_ratio <= 2.50:
            raise SideVisionError(f"side green-region area changed by {area_ratio:.2f}x")
        if lateral_shift > 80.0:
            raise SideVisionError(f"side green region moved laterally by {lateral_shift:.1f} px")
        # Image +v points down.  A physical upward movement therefore reduces
        # v and produces a positive lift signal.
        return float(self.baseline.center_px[1] - region.center_px[1])

