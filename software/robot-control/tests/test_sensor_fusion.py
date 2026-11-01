from types import SimpleNamespace
import unittest

from robot_control.depth_vision import DepthCluster
from robot_control.sensor_fusion import (
    LiftEvidenceTracker,
    SensorFusionError,
    fuse_bin_pose,
    wrist_presence_crosscheck,
)


def cluster(x_mm=452, y_mm=-12, z_mm=50, confidence=0.9, yaw_deg=0):
    return DepthCluster(
        centroid_base_m=(x_mm / 1000, y_mm / 1000, z_mm / 1000),
        minimum_base_m=((x_mm - 25) / 1000, (y_mm - 25) / 1000, -0.005),
        maximum_base_m=((x_mm + 25) / 1000, (y_mm + 25) / 1000, z_mm / 1000),
        principal_extents_xy_m=(0.05, 0.05),
        yaw_deg=yaw_deg,
        height_over_surface_m=0.055,
        pixel_bbox=(10, 10, 50, 50),
        point_count=400,
        category_hint="bin",
        confidence=confidence,
    )


class SensorFusionTests(unittest.TestCase):
    def setUp(self):
        self.rgb = SimpleNamespace(
            bin_center_base_mm=(450.0, -10.0, 49.9),
            south_wall_center_base_mm=(450.0, -31.25, 49.9),
            gripper_yaw_deg=0.0,
            confidence=0.9,
        )

    def test_fuses_independent_centres_and_preserves_rgb_wall_normal(self):
        pose = fuse_bin_pose(self.rgb, [cluster()])
        self.assertAlmostEqual(pose.center_base_mm[0], 452.0, places=3)
        self.assertAlmostEqual(pose.wall_center_base_mm[1], -33.25, places=2)
        self.assertGreater(pose.confidence, 0.8)

    def test_large_cross_sensor_disagreement_fails_closed(self):
        with self.assertRaises(SensorFusionError):
            fuse_bin_pose(self.rgb, [cluster(x_mm=500, y_mm=50)])

    def test_metric_depth_orientation_selects_wall(self):
        rgb = SimpleNamespace(
            bin_center_base_mm=(450.0, 0.0, 49.9),
            south_wall_center_base_mm=(453.7, -20.9, 49.9),
            gripper_yaw_deg=10.0,
            confidence=0.9,
        )
        pose = fuse_bin_pose(
            rgb, [cluster(x_mm=450, y_mm=0, yaw_deg=12.0)]
        )
        self.assertAlmostEqual(pose.gripper_yaw_deg, 12.0, places=3)
        self.assertGreater(pose.wall_center_base_mm[0], 454.0)
        self.assertLess(pose.wall_center_base_mm[1], -20.0)

    def test_lift_tracker_requires_metric_vertical_motion(self):
        tracker = LiftEvidenceTracker((452, -12), 50)
        self.assertFalse(tracker.gate([cluster(z_mm=51.5)], 2.0))
        self.assertTrue(tracker.gate([cluster(z_mm=58.0)], 7.0))

    def test_wrist_is_presence_crosscheck_not_pose_override(self):
        present, distance = wrist_presence_crosscheck(
            [cluster(x_mm=456, y_mm=-8)], (450.0, -10.0)
        )
        self.assertTrue(present)
        self.assertAlmostEqual(distance, 40.0**0.5, places=4)

        present, distance = wrist_presence_crosscheck(
            [cluster(x_mm=480, y_mm=20)], (450.0, -10.0)
        )
        self.assertFalse(present)
        self.assertGreater(distance, 20.0)


if __name__ == "__main__":
    unittest.main()
