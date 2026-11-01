"""Validate the derived package without modifying source media."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name", "-of", "json", str(path)
    ])
    return json.loads(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--package-dir", type=Path, required=True)
    args = ap.parse_args()
    sources = sorted(args.source_dir.glob("*.mp4"))
    rendered = sorted((args.package_dir / "grounded_annotated").glob("PXL_*_grounded.mp4"))
    annotations = json.loads((Path(__file__).parent / "verified_annotations.json").read_text(encoding="utf-8"))
    task = json.loads((args.package_dir / "compiled_task" / "robot_task.json").read_text(encoding="utf-8"))
    checks = {
        "six_sources": len(sources) == 6,
        "six_grounded_videos": len(rendered) == 6,
        "four_placements_each": len(annotations) == 6 and all(len(s["placements"]) == 4 for s in annotations.values()),
        "task_has_safety_gates": len(task.get("safety_gates", [])) >= 5,
        "task_marks_unobserved_metric_3d": any("robot-frame" in x or "depth" in x for x in task["evidence_scope"]["not_observed"]),
        "behavior_tree_present": (args.package_dir / "compiled_task" / "behavior_tree.json").is_file(),
        "contact_graph_present": (args.package_dir / "compiled_task" / "contact_graph.json").is_file(),
    }
    media = []
    source_by_stem = {p.stem: p for p in sources}
    for video in rendered:
        stem = video.stem.removesuffix("_grounded")
        info = probe(video)
        streams = {(s["codec_type"], s["codec_name"]) for s in info["streams"]}
        source_duration = float(probe(source_by_stem[stem])["format"]["duration"])
        output_duration = float(info["format"]["duration"])
        media.append({
            "video": video.name,
            "h264_video": ("video", "h264") in streams,
            "aac_audio": ("audio", "aac") in streams,
            "duration_delta_s": round(output_duration - source_duration, 4),
        })
    checks["all_videos_h264_aac"] = all(m["h264_video"] and m["aac_audio"] for m in media)
    checks["durations_preserved"] = all(abs(m["duration_delta_s"]) <= 0.12 for m in media)
    result = {"passed": all(checks.values()), "checks": checks, "media": media}
    out = args.package_dir / "validation_report.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

