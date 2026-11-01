"""Run every observation-only policy against captured scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truck_benchmark.contract import assert_policy_source_clean, validate_observation_manifest
from truck_benchmark.policies import load_scene_observation, policy_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policies", nargs="*", default=[])
    args = parser.parse_args()

    root = Path(args.observations).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert_policy_source_clean(Path(__file__).with_name("policies.py"))
    registry = policy_registry()
    selected = args.policies or sorted(registry)
    missing = sorted(set(selected) - set(registry))
    if missing:
        raise SystemExit(f"Unknown policies: {missing}")

    scene_dirs = sorted(path.parent for path in root.glob("scene_*/observation.json"))
    if not scene_dirs:
        raise SystemExit(f"No observation scenes found under {root}")
    summary = {"scene_count": len(scene_dirs), "policies": {}, "errors": []}
    for name in selected:
        policy_dir = out / name
        policy_dir.mkdir(parents=True, exist_ok=True)
        completed = 0
        for scene_dir in scene_dirs:
            validate_observation_manifest(scene_dir / "observation.json")
            try:
                prediction = registry[name].predict(load_scene_observation(scene_dir))
                (policy_dir / f"{scene_dir.name}.json").write_text(
                    json.dumps(prediction, indent=2), encoding="utf-8"
                )
                completed += 1
            except Exception as exc:  # preserve failures for ranking
                summary["errors"].append(
                    {
                        "policy": name,
                        "scene": scene_dir.name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        summary["policies"][name] = {"completed": completed, "failed": len(scene_dirs) - completed}
        print(f"[policy] {name}: {completed}/{len(scene_dirs)}", flush=True)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

