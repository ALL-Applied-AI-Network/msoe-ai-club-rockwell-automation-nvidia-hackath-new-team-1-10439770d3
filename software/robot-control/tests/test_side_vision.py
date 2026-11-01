import unittest

import cv2
import numpy as np

from robot_control.side_vision import (
    GreenRegion,
    SideLiftTracker,
    SideVisionError,
    locate_green_region,
)


class SideVisionTests(unittest.TestCase):
    def test_locates_green_region(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (120, 100), (180, 170), (20, 180, 45), -1)
        region = locate_green_region(image)
        self.assertAlmostEqual(region.center_px[0], 150, places=1)
        self.assertAlmostEqual(region.center_px[1], 135, places=1)

    def test_upward_motion_is_positive(self):
        baseline = GreenRegion((150, 135), (120, 100, 60, 70), 4000)
        tracker = SideLiftTracker(baseline)
        moved = GreenRegion((151, 122), (121, 87, 60, 70), 3900)
        self.assertEqual(tracker.observe(moved), 13)

    def test_implausible_area_change_fails_closed(self):
        baseline = GreenRegion((150, 135), (120, 100, 60, 70), 4000)
        tracker = SideLiftTracker(baseline)
        with self.assertRaises(SideVisionError):
            tracker.observe(GreenRegion((150, 120), (140, 110, 10, 10), 100))


if __name__ == "__main__":
    unittest.main()
