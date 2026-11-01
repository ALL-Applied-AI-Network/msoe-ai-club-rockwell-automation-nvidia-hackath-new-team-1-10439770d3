"""Run the optional Open3D CAD registration policy over serialized observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cad_registration import Open3DFGRPolicy
from .contract import assert_policy_source_clean, validate_observation_manifest
from .policies import load_scene_observation


parser = argparse.ArgumentParser()
parser.add_argument("--observations", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()


def main() -> int:
    package = Path(__file__).resolve().parent
    assert_policy_source_clean(package / "cad_registration.py")
    observations = Path(args.observations)
    scenes = sorted(path.parent for path in observations.glob("scene_*/observation.json"))
    if not scenes:
        raise SystemExit("No observations found")
    policy = Open3DFGRPolicy()
    root = Path(args.out)
    destination = root / "predictions" / policy.name
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"policy": policy.name, "completed": 0, "failed": 0, "errors": []}
    for scene in scenes:
        try:
            validate_observation_manifest(scene / "observation.json")
            prediction = policy.predict(load_scene_observation(scene))
            (destination / f"{scene.name}.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
            summary["completed"] += 1
            print(f"[cad-fgr] {scene.name}", flush=True)
        except Exception as exc:
            summary["failed"] += 1
            summary["errors"].append({"scene": scene.name, "error": repr(exc)})
    (root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
