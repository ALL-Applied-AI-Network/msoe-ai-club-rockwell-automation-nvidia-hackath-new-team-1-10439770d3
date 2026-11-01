import unittest

import numpy as np

from robot_control.adaptive_bin_policy import AdaptiveBinLiftPolicy


class AdaptivePolicyContractTests(unittest.TestCase):
    def test_requests_all_rgb_views_and_both_depth_streams(self):
        self.assertEqual(AdaptiveBinLiftPolicy.CAMERAS, ["table", "side", "wrist"])
        self.assertEqual(AdaptiveBinLiftPolicy.DEPTH, ["table", "wrist"])
        self.assertEqual(AdaptiveBinLiftPolicy.ACTION, "tcp")
        self.assertFalse(AdaptiveBinLiftPolicy.PREFETCH)

    def test_reset_clears_all_rollout_state(self):
        policy = AdaptiveBinLiftPolicy()
        policy.audit.append({"event": "test"})
        policy.evidence_samples["proof_lift"] = [1.0]
        policy.reset()
        self.assertEqual(policy.audit, [])
        self.assertEqual(policy.evidence_samples, {})
        self.assertEqual(policy.wrist_presence_samples, {})
        self.assertIsNone(policy.controller)

    def test_invalid_initial_image_holds_observed_pose(self):
        policy = AdaptiveBinLiftPolicy()
        observation = {
            "images": {
                "table": np.zeros((480, 848, 3), dtype=np.uint8),
                "side": np.zeros((480, 848, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 848, 3), dtype=np.uint8),
            },
            "depth": {
                "table": np.zeros((400, 640), dtype=np.uint16),
                "wrist": np.zeros((480, 848), dtype=np.uint16),
            },
            "tcp_pose": [450.0, 0.0, -200.0, -180.0, 0.0, 0.0],
            "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        self.assertEqual(
            policy.act(observation),
            [450.0, 0.0, -200.0, -180.0, 0.0, 0.0, 1.0],
        )
        self.assertTrue(
            "BinVisionError" in policy.initialization_error
            or "ModuleNotFoundError" in policy.initialization_error
        )


if __name__ == "__main__":
    unittest.main()
