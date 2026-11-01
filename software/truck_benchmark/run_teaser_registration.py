"""Run the optional TEASER++ CAD policy over serialized observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import assert_policy_source_clean, validate_observation_manifest
from .policies import load_scene_observation
from .teaser_registration import TeaserCADPolicy


parser = argparse.ArgumentParser()
parser.add_argument("--observations", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--max-scenes", type=int, default=0)
args = parser.parse_args()


def main() -> int:
    package = Path(__file__).resolve().parent
    assert_policy_source_clean(package / "teaser_registration.py")
    scenes = sorted(path.parent for path in Path(args.observations).glob("scene_*/observation.json"))
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    policy = TeaserCADPolicy()
    root = Path(args.out)
    destination = root / "predictions" / policy.name
    destination.mkdir(parents=True, exist_ok=True)
    errors = []
    for index, scene in enumerate(scenes):
        try:
            validate_observation_manifest(scene / "observation.json")
            prediction = policy.predict(load_scene_observation(scene))
            (destination / f"{scene.name}.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
            print(f"[teaser] {index + 1}/{len(scenes)} {scene.name}", flush=True)
        except Exception as exc:
            errors.append({"scene": scene.name, "error": repr(exc)})
    summary = {"policy": policy.name, "scenes": len(scenes), "errors": errors}
    (root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
