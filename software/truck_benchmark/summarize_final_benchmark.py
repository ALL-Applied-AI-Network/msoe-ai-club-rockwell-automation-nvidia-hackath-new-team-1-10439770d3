"""Summarize end-to-end final truck benchmark execution reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def mean(values):
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("episodes/*/execution_report.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        evaluation = doc.get("evaluation") or {}
        episode = path.parent.name
        rows.append({
            "episode": episode,
            "policy": doc.get("policy", "unknown"),
            "controller": doc.get("controller", "unknown"),
            "variation": doc.get("variation", "unknown"),
            "seed": doc.get("seed"),
            "task_complete": bool(doc.get("task_complete")),
            "full_task_success": bool(evaluation.get("full_task_success")),
            "cubes_in_matching_bins": int(evaluation.get("cubes_in_matching_bins", 0)),
            "bins_on_deck": int(evaluation.get("bins_on_deck", 0)),
            "perception_latency_ms": doc.get("perception_latency_ms"),
            "motion_wall_time_s": doc.get("motion_wall_time_s"),
            "action_count": doc.get("action_count_including_active_scans", doc.get("action_count")),
            "tcp_path_length_m": doc.get("tcp_path_length_m"),
            "frames": doc.get("frames", 0),
            "error": None if doc.get("error") is None else doc["error"].get("message"),
        })

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    ranking = []
    for policy, subset in grouped.items():
        successes = sum(row["full_task_success"] for row in subset)
        ranking.append({
            "policy": policy,
            "episodes": len(subset),
            "successes": successes,
            "success_rate": successes / len(subset),
            "inventory_gate_rate": sum(row["task_complete"] for row in subset) / len(subset),
            "mean_cubes": mean([row["cubes_in_matching_bins"] for row in subset]),
            "mean_bins": mean([row["bins_on_deck"] for row in subset]),
            "mean_perception_ms": mean([row["perception_latency_ms"] for row in subset if row["perception_latency_ms"] is not None]),
            "mean_motion_s": mean([row["motion_wall_time_s"] for row in subset if row["motion_wall_time_s"] is not None]),
            "mean_actions": mean([row["action_count"] for row in subset if row["action_count"] is not None]),
            "mean_tcp_path_m": mean([row["tcp_path_length_m"] for row in subset if row["tcp_path_length_m"] is not None]),
        })
    ranking.sort(key=lambda row: (row["success_rate"], row["mean_cubes"], row["mean_bins"]), reverse=True)

    if rows:
        with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps({"ranking": ranking, "episodes": rows}, indent=2), encoding="utf-8")

    lines = [
        "# Final observation-only truck benchmark",
        "",
        "No policy received simulator object transforms. Object state was read only after all motion for scoring.",
        "",
        "| Rank | Policy | Success | Inventory/execution gate | Cubes | Bins | Perception | Motion | Actions | TCP XY |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranking, 1):
        def fmt(value, suffix=""):
            return "n/a" if value is None else f"{value:.2f}{suffix}"
        lines.append(
            f"| {index} | `{row['policy']}` | {row['successes']}/{row['episodes']} ({100*row['success_rate']:.1f}%) "
            f"| {100*row['inventory_gate_rate']:.1f}% | {row['mean_cubes']:.2f}/12 | {row['mean_bins']:.2f}/3 "
            f"| {fmt(row['mean_perception_ms'], ' ms')} | {fmt(row['mean_motion_s'], ' s')} "
            f"| {fmt(row['mean_actions'])} | {fmt(row['mean_tcp_path_m'], ' m')} |"
        )
    lines.extend(["", "## Episodes", ""])
    for row in rows:
        status = "PASS" if row["full_task_success"] else "FAIL"
        lines.append(
            f"- **{status}** `{row['episode']}` — cubes {row['cubes_in_matching_bins']}/12, "
            f"bins {row['bins_on_deck']}/3" + (f"; {row['error']}" if row["error"] else "")
        )
    (root / "FINAL_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(ranking, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

