"""Camera geometry shared by observation-only policies.

This module is deliberately independent of Isaac Sim.  Policies receive a
serialized pinhole model and sensor arrays; they never import simulator APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    camera_to_world: np.ndarray
    table_z: float

    @classmethod
    def from_json(cls, value: dict) -> "CameraModel":
        width, height = value["resolution"]
        return cls(
            width=int(width),
            height=int(height),
            focal_length_mm=float(value["focal_length_mm"]),
            horizontal_aperture_mm=float(value["horizontal_aperture_mm"]),
            vertical_aperture_mm=float(value["vertical_aperture_mm"]),
            camera_to_world=np.asarray(value["camera_to_world"], dtype=np.float64),
            table_z=float(value["table_top_z"]),
        )

    @property
    def tan_h(self) -> float:
        return self.horizontal_aperture_mm / (2.0 * self.focal_length_mm)

    @property
    def tan_v(self) -> float:
        return self.vertical_aperture_mm / (2.0 * self.focal_length_mm)

    @property
    def rotation_rows(self) -> np.ndarray:
        return self.camera_to_world[:3, :3]

    @property
    def eye(self) -> np.ndarray:
        return self.camera_to_world[3, :3]

    def project(self, xyz: Iterable[float]) -> np.ndarray | None:
        point = np.asarray(tuple(xyz), dtype=np.float64) - self.eye
        camera = self.rotation_rows @ point
        if camera[2] >= -1e-9:
            return None
        x_ndc = camera[0] / (-camera[2]) / self.tan_h
        y_ndc = camera[1] / (-camera[2]) / self.tan_v
        return np.asarray(
            ((x_ndc + 1.0) * 0.5 * self.width, (1.0 - y_ndc) * 0.5 * self.height),
            dtype=np.float64,
        )

    def rays(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(u, dtype=np.float64) / self.width * 2.0 - 1.0
        y = 1.0 - np.asarray(v, dtype=np.float64) / self.height * 2.0
        camera = np.stack((x * self.tan_h, y * self.tan_v, -np.ones_like(x)), axis=-1)
        world = camera @ self.rotation_rows
        world /= np.maximum(np.linalg.norm(world, axis=-1, keepdims=True), 1e-12)
        return np.broadcast_to(self.eye, world.shape), world

    def pixels_to_plane(self, uv: np.ndarray, z: float) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        origin, direction = self.rays(uv[:, 0], uv[:, 1])
        scale = (float(z) - origin[:, 2]) / direction[:, 2]
        return origin + scale[:, None] * direction

    def depth_to_world(self, depth: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert distance-to-image-plane depth into world points.

        Returns `(points, u, v)` for finite positive samples.  Isaac's
        distance-to-image-plane value is `-camera_z`, not Euclidean range.
        """

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        vv, uu = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
        dd = depth[::stride, ::stride]
        valid = np.isfinite(dd) & (dd > 1e-5) & (dd < 10.0)
        uu = uu[valid].astype(np.float64)
        vv = vv[valid].astype(np.float64)
        dd = dd[valid].astype(np.float64)
        x_ndc = uu / self.width * 2.0 - 1.0
        y_ndc = 1.0 - vv / self.height * 2.0
        camera = np.column_stack((x_ndc * self.tan_h * dd, y_ndc * self.tan_v * dd, -dd))
        points = camera @ self.rotation_rows + self.eye
        return points, uu.astype(np.int32), vv.astype(np.int32)


def angle_mod_90_error(a_deg: float, b_deg: float) -> float:
    """Smallest absolute orientation error for a square object."""

    return abs((float(a_deg) - float(b_deg) + 45.0) % 90.0 - 45.0)


def angle_mod_180_error(a_deg: float, b_deg: float) -> float:
    """Smallest absolute orientation error for an unoriented long axis."""

    return abs((float(a_deg) - float(b_deg) + 90.0) % 180.0 - 90.0)


def canonical_square_yaw(yaw_deg: float) -> float:
    return (float(yaw_deg) + 45.0) % 90.0 - 45.0


def canonical_axis_yaw(yaw_deg: float) -> float:
    return (float(yaw_deg) + 90.0) % 180.0 - 90.0

