"""Drop-in policy for the official SIDC Lab Physical AI client.

The policy only consumes a plan that a local human approved through the A-Team
CLI. It defaults to mock mode. Physical mode remains locked unless the operator
sets ATEAM_EXECUTION_MODE=physical during an approved test slot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MVP_ROOT = Path(os.environ.get("ATEAM_MVP_ROOT", Path(__file__).resolve().parent))
if str(MVP_ROOT) not in sys.path:
    sys.path.insert(0, str(MVP_ROOT))

from robot_control.safety import as_tcp_action  # noqa: E402
from robot_control.store import PlanStore  # noqa: E402


class Policy:
    CAMERAS = []
    IMAGE_SIZE = (320, 180)
    IMAGE_FORMAT = "jpeg"
    ACTION = "tcp"
    PREFETCH = False
    DEPTH = []

    def __init__(self, checkpoint: str | None = None):
        self.mode = os.environ.get("ATEAM_EXECUTION_MODE", "mock").strip().lower()
        if self.mode not in {"mock", "physical"}:
            raise RuntimeError("ATEAM_EXECUTION_MODE must be mock or physical")
        if self.mode == "physical" and os.environ.get("ATEAM_ALLOW_PHYSICAL") != "YES":
            raise RuntimeError("physical execution is locked; an on-site operator must arm it")
        self.store = PlanStore()
        self.plan = None
        self.actions: list[list[float]] = []
        self.sent = False

    def reset(self) -> None:
        self.plan = self.store.claim_next(self.mode)
        if self.plan is None:
            raise RuntimeError(
                f"no human-approved {self.mode} plan is queued; draft and approve one before starting the run"
            )
        self.actions = [as_tcp_action(item) for item in self.plan["waypoints"]]
        self.sent = False

    def act(self, obs: dict):
        if not self.sent:
            self.sent = True
            return self.actions
        target = self.actions[-1]
        observed = obs.get("tcp_pose")
        if observed is not None:
            position_error = sum((float(observed[i]) - target[i]) ** 2 for i in range(3)) ** 0.5
            if position_error <= 2.0 and self.plan is not None and self.plan["status"] == "executing":
                self.plan = self.store.complete(self.plan["id"], f"reported TCP error {position_error:.2f} mm")
        return target

