"""Run the deterministic A/B/D/F/H/I/J campaign candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .advanced_methods import advanced_policy_registry
from .contract import assert_policy_source_clean, validate_observation_manifest
from .policies import load_scene_observation, policy_registry


parser = argparse.ArgumentParser()
parser.add_argument("--observations", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--policies", nargs="*", default=[])
args = parser.parse_args()


def main() -> int:
    package = Path(__file__).resolve().parent
    assert_policy_source_clean(package / "advanced_methods.py")
    assert_policy_source_clean(package / "policies.py")
    observations = Path(args.observations)
    scene_dirs = sorted(path.parent for path in observations.glob("scene_*/observation.json"))
    if not scene_dirs:
        raise SystemExit(f"No observation scenes found under {observations}")

    candidates = advanced_policy_registry()
    originals = policy_registry()
    # D and the best prior fixed perception policy are explicit controls.
    for name in (
        "locked_calibration__active_wrist_selective_rgbd_robust_occlusion_oblique_recovery_rawcube",
        "locked_calibration__task_hybrid_oblique_cube_recovery_achromatic_truck",
        "locked_calibration__depth_table_h6mm_s1_rawcube",
    ):
        if name in originals:
            candidates[f"D_{name}" if "active_wrist" in name else f"CONTROL_{name}"] = originals[name]

    selected = args.policies or sorted(candidates)
    missing = [name for name in selected if name not in candidates]
    if missing:
        raise SystemExit(f"Unknown campaign policies: {missing}")

    out = Path(args.out)
    prediction_root = out / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    summary = {"scene_count": len(scene_dirs), "policies": {}, "errors": []}
    for campaign_name in selected:
        policy = candidates[campaign_name]
        policy_dir = prediction_root / campaign_name
        policy_dir.mkdir(parents=True, exist_ok=True)
        completed = 0
        for scene_dir in scene_dirs:
            try:
                validate_observation_manifest(scene_dir / "observation.json")
                prediction = policy.predict(load_scene_observation(scene_dir))
                prediction["policy"] = campaign_name
                (policy_dir / f"{scene_dir.name}.json").write_text(
                    json.dumps(prediction, indent=2), encoding="utf-8"
                )
                completed += 1
            except Exception as exc:  # campaign must preserve per-method failures
                summary["errors"].append(
                    {"policy": campaign_name, "scene": scene_dir.name, "error": repr(exc)}
                )
        summary["policies"][campaign_name] = {
            "completed": completed,
            "failed": len(scene_dirs) - completed,
        }
        print(f"[advanced] {campaign_name}: {completed}/{len(scene_dirs)}", flush=True)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
