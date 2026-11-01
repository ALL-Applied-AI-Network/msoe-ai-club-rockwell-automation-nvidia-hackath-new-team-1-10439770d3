"""Aggregate honest open-loop PhysX episodes and combine them with perception scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def mean(values):
    return statistics.fmean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--perception-report")
    args = parser.parse_args()

    root = Path(args.episodes).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    perception = {}
    if args.perception_report:
        doc = json.loads(Path(args.perception_report).read_text(encoding="utf-8"))
        perception = {item["policy"]: item for item in doc.get("ranking", [])}

    groups = defaultdict(list)
    episodes = []
    for path in sorted(root.rglob("execution_report.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["report_path"] = str(path)
        episodes.append(item)
        groups[(item.get("policy", "unknown"), item.get("controller", "unknown"))].append(item)

    ranking = []
    for (policy, controller), rows in groups.items():
        complete = [row for row in rows if row.get("task_complete")]
        full = [bool(row.get("evaluation", {}).get("full_task_success")) for row in rows]
        cubes = [float(row.get("evaluation", {}).get("cubes_in_matching_bins", 0)) for row in rows]
        bins = [float(row.get("evaluation", {}).get("bins_on_deck", 0)) for row in rows]
        latencies = [float(row.get("perception_latency_ms", math.inf)) for row in complete]
        path_lengths = [float(row.get("tcp_path_length_m", 0.0)) for row in complete]
        actions = [float(row.get("action_count", 0.0)) for row in complete]
        wall_times = [float(row.get("motion_wall_time_s", 0.0)) for row in complete]
        perception_score = float(perception.get(policy, {}).get("ranking_score", 0.0))
        physics_score = (
            0.62 * mean(full)
            + 0.23 * (mean(cubes) / 12.0)
            + 0.15 * (mean(bins) / 3.0)
        )
        # Efficiency never outweighs reliability, but it breaks ties using
        # exactly the latency/length/work quantities requested by the task.
        efficiency_score = (
            0.25 * math.exp(-mean(latencies) / 2000.0)
            + 0.30 * math.exp(-mean(path_lengths) / 12.0)
            + 0.20 * math.exp(-mean(actions) / 180.0)
            + 0.25 * math.exp(-mean(wall_times) / 30.0)
        ) if complete else 0.0
        by_variation = {}
        for variation in sorted({str(row.get("variation", "unknown")) for row in rows}):
            subset = [row for row in rows if str(row.get("variation", "unknown")) == variation]
            by_variation[variation] = {
                "episodes": len(subset),
                "full_success_rate": mean([
                    bool(row.get("evaluation", {}).get("full_task_success")) for row in subset
                ]),
                "mean_cubes": mean([
                    float(row.get("evaluation", {}).get("cubes_in_matching_bins", 0)) for row in subset
                ]),
                "mean_bins": mean([
                    float(row.get("evaluation", {}).get("bins_on_deck", 0)) for row in subset
                ]),
            }
        ranking.append({
            "policy": policy,
            "controller": controller,
            "episodes": len(rows),
            "completed": len(complete),
            "failed_to_complete": len(rows) - len(complete),
            "full_successes": sum(full),
            "full_success_rate": mean(full),
            "mean_cubes_in_matching_bins": mean(cubes),
            "mean_bins_on_deck": mean(bins),
            "perception_latency_ms_mean": mean(latencies),
            "action_count_mean": mean(actions),
            "tcp_path_length_m_mean": mean(path_lengths),
            "motion_wall_time_s_mean": mean(wall_times),
            "perception_ranking_score": perception_score,
            "physics_score": physics_score,
            "combined_score": 0.72 * physics_score + 0.28 * perception_score,
            "efficiency_score": efficiency_score,
            "balanced_score": 0.75 * physics_score + 0.15 * perception_score + 0.10 * efficiency_score,
            "by_variation": by_variation,
        })
    ranking.sort(key=lambda item: (-item["balanced_score"], item["failed_to_complete"], item["perception_latency_ms_mean"], item["policy"], item["controller"]))

    document = {
        "ranking": ranking,
        "episodes": episodes,
        "interpretation": (
            "The combined score is 72% end-to-end PhysX and 28% held-out perception. "
            "A failed/incomplete episode scores zero; simulator truth is evaluator-only."
        ),
    }
    (out / "physics_results.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    fields = [
        "policy", "controller", "episodes", "completed", "failed_to_complete", "full_successes",
        "full_success_rate", "mean_cubes_in_matching_bins", "mean_bins_on_deck",
        "perception_latency_ms_mean", "action_count_mean", "tcp_path_length_m_mean",
        "motion_wall_time_s_mean", "perception_ranking_score", "physics_score", "efficiency_score",
        "combined_score", "balanced_score",
    ]
    with (out / "physics_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranking)

    lines = [
        "# Truck end-to-end PhysX ranking",
        "",
        "Every run derived commands only from RGB-D observations. Open-loop controllers froze the complete plan before motion; `vision_replan` refreshed bin/truck poses before transport; `vision_recovery` additionally retried camera-observed cubes still outside their matching bin. Object transforms were unavailable to every controller and read only after the final retreat.",
        "",
        "| Rank | Policy | Controller | Full success | Failures | Cubes | Bins | Perception | Actions | TCP path | Motion time | Efficiency | Balanced |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(ranking, start=1):
        lines.append(
            f"| {index} | `{item['policy']}` | `{item['controller']}` | "
            f"{item['full_successes']}/{item['episodes']} ({item['full_success_rate']:.1%}) | "
            f"{item['episodes'] - item['full_successes']} | "
            f"{item['mean_cubes_in_matching_bins']:.2f}/12 | {item['mean_bins_on_deck']:.2f}/3 | "
            f"{item['perception_latency_ms_mean']:.1f} ms | {item['action_count_mean']:.1f} | "
            f"{item['tcp_path_length_m_mean']:.2f} m | {item['motion_wall_time_s_mean']:.1f} s | "
            f"{item['efficiency_score']:.4f} | {item['balanced_score']:.4f} |"
        )
    lines.extend([
        "",
        "A passing Isaac run reduces risk but cannot guarantee physical success. Real camera/tool calibration, finger geometry, compliance, friction, controller timing, and unmodelled obstacles remain transfer variables.",
    ])
    (out / "PHYSICS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(ranking, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
