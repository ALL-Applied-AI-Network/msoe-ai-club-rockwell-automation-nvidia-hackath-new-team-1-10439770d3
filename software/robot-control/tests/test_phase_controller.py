import unittest

from robot_control.phase_controller import (
    MotionPhase,
    PhaseController,
    PhaseExecutionError,
    bin_lift_phases,
    tcp_chunk,
)


class PhaseControllerTests(unittest.TestCase):
    def test_chunk_has_exact_endpoint_and_binary_gripper_switch(self):
        start = [300, 0, -90, -180, 0, 0, 0]
        target = (450, 0, -235, -180, 0, 10, 1)
        chunk = tcp_chunk(start, target, 2.0)
        self.assertEqual(chunk[-1], list(target))
        self.assertTrue(all(row[6] == 0 for row in chunk[:-1]))
        self.assertEqual(chunk[-1][6], 1)

    def test_lag_only_corrects_same_target_and_never_reverses_phase(self):
        phases = [
            MotionPhase("lift", (450, 0, -285, -180, 0, 0, 1), 1.0),
            MotionPhase("hold", (450, 0, -285, -180, 0, 0, 1), 0.5),
        ]
        controller = PhaseController(phases, settled_observations=1)
        lagging = {"tcp_pose": [450, 0, -295, -180, 0, 0], "joints": [0] * 6 + [1]}
        first = controller.act(lagging)
        second = controller.act(lagging)
        self.assertEqual(controller.phase.name, "lift")
        self.assertGreater(first[-1][2], lagging["tcp_pose"][2])
        self.assertGreater(second[-1][2], lagging["tcp_pose"][2])
        reached = {"tcp_pose": [450, 0, -285, -180, 0, 0], "joints": [0] * 6 + [1]}
        controller.act(reached)
        self.assertEqual(controller.phase.name, "hold")

    def test_verification_gate_holds_then_fails_closed(self):
        phase = MotionPhase(
            "close", (450, 0, -298, -180, 0, 0, 1), 0.5, verification="grasp"
        )
        controller = PhaseController([phase], settled_observations=1)
        obs = {"tcp_pose": [450, 0, -298, -180, 0, 0], "joints": [0] * 6 + [1]}
        controller.act(obs)
        controller.act(obs, {"grasp": None})
        self.assertFalse(controller.complete)
        with self.assertRaises(PhaseExecutionError):
            controller.act(obs, {"grasp": False})

    def test_bin_proof_ends_at_hold_not_lower(self):
        phases = bin_lift_phases(450, -21, 0)
        self.assertEqual([phase.name for phase in phases][-2:], ["lift", "hold"])
        self.assertIn("proof_lift", [phase.name for phase in phases])
        self.assertNotIn("lower", [phase.name for phase in phases])


if __name__ == "__main__":
    unittest.main()
