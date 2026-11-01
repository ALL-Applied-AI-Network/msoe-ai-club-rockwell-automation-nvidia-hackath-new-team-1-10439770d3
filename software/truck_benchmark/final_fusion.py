"""Online observation-only fusion used by the final truck-task benchmark.

This module turns the previously offline ``L_consensus_inventory_fusion``
experiment into a real policy.  Every component consumes only the current
RGB-D camera packet, reported calibration, robot state, and timestamp.  No
Isaac Sim or USD API is imported here.
"""

from __future__ import annotations

import os

from .advanced_methods import GeometryOBBScalePolicy, PlaneSelfCalibrationPolicy
from .campaign_fusion import fuse_document
from .neural_methods import GroundingDinoProvider, UltralyticsPromptSegmenter, run_scene
from .policies import BasePolicy, LockedCalibrationPolicy, SceneObservation, TaskHybridRecoveryPolicy


class OnlineConsensusInventoryFusionPolicy(BasePolicy):
    """Task hybrid + methods A/B + GroundingDINO/MobileSAM consensus.

    Model construction is lazy so importing the deterministic policy registry
    does not allocate GPU memory.  One instance is reused for every camera
    retry in an episode, so neural weights are loaded only once.
    """

    name = "L_online_consensus_inventory_fusion"

    def __init__(self) -> None:
        self.primary = LockedCalibrationPolicy(TaskHybridRecoveryPolicy())
        self.plane = PlaneSelfCalibrationPolicy()
        self.geometry = GeometryOBBScalePolicy()
        self._provider = None
        self._segmenter = None

    def _neural_components(self):
        if self._provider is None:
            model_id = os.environ.get(
                "A2K_DINO_MODEL_ID",
                "IDEA-Research/grounding-dino-tiny",
            )
            mobile_sam = os.environ.get("A2K_MOBILE_SAM_WEIGHTS", "mobile_sam.pt")
            self._provider = GroundingDinoProvider(model_id, confidence=0.12)
            self._segmenter = UltralyticsPromptSegmenter("mobilesam", mobile_sam)
        return self._provider, self._segmenter

    def predict(self, observation: SceneObservation) -> dict:
        provider, segmenter = self._neural_components()
        primary = self.primary.predict(observation)
        primary["policy"] = self.primary.name
        plane = self.plane.predict(observation)
        plane["policy"] = self.plane.name
        geometry = self.geometry.predict(observation)
        geometry["policy"] = self.geometry.name
        neural = run_scene(
            provider,
            observation,
            segmenter,
            camera_names=(
                "table",
                "side",
                "wrist",
                "wrist_scan_near",
                "wrist_scan_far",
                "wrist_scan_oblique_west",
            ),
        )
        neural["policy"] = "C_grounding_dino_mobilesam_mask_depth"
        fused = fuse_document(
            primary,
            {"A": plane, "B": geometry, "dino": neural},
            cluster_radius_m=0.004,
            dedupe_radius_m=0.005,
            min_sources=1,
            require_rgb_source=False,
            refine_pose=True,
        )
        fused["policy"] = self.name
        fused.setdefault("notes", []).extend(
            [
                "online fusion; all component predictions came from the current observation packet",
                "configuration frozen from the prior 48-scene L fusion sweep",
            ]
        )
        return fused


def final_policy_registry() -> dict[str, BasePolicy]:
    policy = OnlineConsensusInventoryFusionPolicy()
    return {policy.name: policy}
