import importlib.util
import pathlib
import sys
import unittest


MODULE = pathlib.Path(__file__).parents[1] / "simulation" / "one_wall_bin_model.py"
SPEC = importlib.util.spec_from_file_location("one_wall_bin_model", MODULE)
one_wall = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = one_wall
SPEC.loader.exec_module(one_wall)


class OneWallBinModelTests(unittest.TestCase):
    def test_official_dimensions_produce_7_5_mm_walls(self):
        spec = one_wall.BinSpec()
        self.assertAlmostEqual(spec.wall_x * 1000, 7.5)
        self.assertAlmostEqual(spec.wall_y * 1000, 7.5)
        self.assertAlmostEqual(spec.floor_height * 1000, 30.0)

    def test_compound_volume_and_com_match_cad_scale(self):
        spec = one_wall.BinSpec()
        self.assertAlmostEqual(spec.volume_m3 * 1e6, 106.875, places=3)
        self.assertAlmostEqual(spec.center_of_mass_z * 1000, 23.2, delta=0.2)

    def test_grip_stays_above_pocket_floor(self):
        spec = one_wall.BinSpec()
        plan = one_wall.GripPlan()
        self.assertAlmostEqual(plan.wall_center_xy(spec)[1] * 1000, -21.25)
        self.assertAlmostEqual(plan.grip_tip(spec)[2] * 1000, 35.9)
        self.assertAlmostEqual(
            plan.minimum_tip_clearance_over_floor(spec) * 1000, 11.0
        )

    def test_acceptance_gate(self):
        good = one_wall.assess_trial(
            commanded_lift_mm=10,
            measured_lift_mm=9.6,
            max_downward_slip_mm=0.4,
            max_tilt_deg=2.0,
            jaw_gap_mm=7.5,
        )
        self.assertTrue(good["passed"])
        slipped = one_wall.assess_trial(
            commanded_lift_mm=10,
            measured_lift_mm=7.0,
            max_downward_slip_mm=1.8,
            max_tilt_deg=7.0,
            jaw_gap_mm=7.5,
        )
        self.assertFalse(slipped["passed"])
        self.assertFalse(slipped["checks"]["slip"])


if __name__ == "__main__":
    unittest.main()
