import unittest

import numpy as np

from robot_control.depth_vision import DepthVisionError, clusters_from_base_points


def rectangular_cloud(cx, cy, sx, sy, top_z, u0, v0, pixels_x, pixels_y):
    xs = np.linspace(cx - sx / 2, cx + sx / 2, pixels_x)
    ys = np.linspace(cy - sy / 2, cy + sy / 2, pixels_y)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, top_z)
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    uu, vv = np.meshgrid(np.arange(u0, u0 + pixels_x), np.arange(v0, v0 + pixels_y))
    pixels = np.column_stack((uu.ravel(), vv.ravel()))
    return points, pixels


class DepthVisionTests(unittest.TestCase):
    def test_metric_clusters_separate_bin_and_cube(self):
        bin_points, bin_pixels = rectangular_cloud(
            0.45, -0.10, 0.050, 0.050, 0.050, 20, 20, 20, 20
        )
        cube_points, cube_pixels = rectangular_cloud(
            0.36, 0.08, 0.010, 0.010, 0.010, 80, 80, 10, 10
        )
        clusters = clusters_from_base_points(
            np.vstack((bin_points, cube_points)),
            np.vstack((bin_pixels, cube_pixels)),
            (120, 120),
            minimum_points=20,
        )
        self.assertEqual({item.category_hint for item in clusters}, {"bin", "cube"})
        # The small synthetic cube has only 100 samples, so its evidence score
        # should be usable but must not pretend to be a dense wrist close-up.
        self.assertGreater(min(item.confidence for item in clusters), 0.6)

    def test_table_plane_and_outside_workspace_are_rejected(self):
        plane, pixels = rectangular_cloud(
            0.45, 0.0, 0.050, 0.050, -0.0051, 20, 20, 20, 20
        )
        outside = plane.copy()
        outside[:, 0] = 0.80
        clusters = clusters_from_base_points(
            np.vstack((plane, outside)),
            np.vstack((pixels, pixels + [40, 40])),
            (120, 120),
        )
        self.assertEqual(clusters, [])

    def test_footprint_centre_is_not_biased_by_uneven_surface_density(self):
        full, full_pixels = rectangular_cloud(
            0.45, 0.0, 0.050, 0.050, 0.050, 20, 20, 20, 20
        )
        # An open bin exposes more points on the wall facing the camera.  The
        # grasp centre must come from the fitted footprint, not the raw median.
        biased = np.repeat(full[full[:, 1] < -0.015], 5, axis=0)
        biased_pixels = np.repeat(full_pixels[full[:, 1] < -0.015], 5, axis=0)
        clusters = clusters_from_base_points(
            np.vstack((full, biased)),
            np.vstack((full_pixels, biased_pixels)),
            (120, 120),
            minimum_points=20,
        )
        self.assertEqual(len(clusters), 1)
        self.assertAlmostEqual(clusters[0].centroid_base_m[0], 0.45, places=4)
        self.assertAlmostEqual(clusters[0].centroid_base_m[1], 0.0, places=4)

    def test_rotated_bin_uses_metric_footprint_yaw(self):
        points, pixels = rectangular_cloud(
            0.45, 0.0, 0.050, 0.050, 0.050, 20, 20, 24, 24
        )
        angle = np.radians(12.0)
        rotation = np.array(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )
        points[:, :2] = (points[:, :2] - [0.45, 0.0]) @ rotation.T + [0.45, 0.0]
        cluster = clusters_from_base_points(
            points, pixels, (120, 120), minimum_points=20
        )[0]
        self.assertAlmostEqual(cluster.yaw_deg, 12.0, delta=0.2)

    def test_mismatched_arrays_fail_closed(self):
        with self.assertRaises(DepthVisionError):
            clusters_from_base_points(np.zeros((2, 3)), np.zeros((1, 2)), (20, 20))


if __name__ == "__main__":
    unittest.main()
