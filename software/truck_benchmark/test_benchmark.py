from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from truck_benchmark.camera_geometry import angle_mod_180_error, angle_mod_90_error
from truck_benchmark.contract import audit_policy_source
from truck_benchmark.policies import policy_registry


class ContractTests(unittest.TestCase):
    def test_policy_module_has_no_simulator_api(self):
        path = Path(__file__).with_name("policies.py")
        self.assertEqual(audit_policy_source(path), [])

    def test_registry_has_baselines_variants_and_hybrids(self):
        names = set(policy_registry())
        self.assertIn("fixed_layout_baseline", names)
        self.assertTrue(any(name.startswith("rgb_fixed") for name in names))
        self.assertTrue(any(name.startswith("depth_table") for name in names))
        self.assertTrue(any("multiview" in name for name in names))
        self.assertIn("ensemble_confidence_gated", names)
        self.assertGreaterEqual(len(names), 15)

    def test_physics_gate_reads_object_pose_only_after_evaluator_boundary(self):
        """Prevent accidental feedback from PhysX truth into robot commands."""

        path = Path(__file__).with_name("execute_physics_finalist.py")
        text = path.read_text(encoding="utf-8")
        marker = "# ---------------- evaluator boundary: no values below feed the controller ----------------"
        self.assertIn(marker, text)
        controller, evaluator = text.split(marker, maxsplit=1)
        self.assertNotIn("get_world_pose", controller)
        self.assertNotIn("evaluator_pose(", controller)
        self.assertIn("get_world_pose", evaluator)
        self.assertLess(controller.index('write_text(plan_text'), controller.index('command(label + " approach"'))


class GeometryTests(unittest.TestCase):
    def test_square_symmetry(self):
        self.assertAlmostEqual(angle_mod_90_error(44.0, -44.0), 2.0)
        self.assertAlmostEqual(angle_mod_90_error(5.0, 95.0), 0.0)

    def test_axis_symmetry(self):
        self.assertAlmostEqual(angle_mod_180_error(-179.0, 1.0), 0.0)
        self.assertAlmostEqual(angle_mod_180_error(89.0, -89.0), 2.0)


if __name__ == "__main__":
    unittest.main()
