"""Score policy predictions against evaluator-only simulator labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera_geometry import angle_mod_180_error, angle_mod_90_error


THRESHOLDS = {
    "cube_xy_mm": 3.0,
    "cube_yaw_deg": 10.0,
    "bin_xy_mm": 4.0,
    "bin_yaw_deg": 8.0,
    "truck_xy_mm": 7.0,
    "truck_yaw_deg": 5.0,
}


def _identity(item: dict) -> tuple[str, str]:
    return str(item.get("category")), str(item.get("colour"))


def _match(truth: list[dict], predicted: list[dict]) -> tuple[list[tuple[dict, dict, float]], list[dict], list[dict]]:
    matches = []
    unmatched_truth = []
    unmatched_pred = list(predicted)
    for identity in sorted({_identity(item) for item in truth} | {_identity(item) for item in predicted}):
        gt_group = [item for item in truth if _identity(item) == identity]
        pred_group = [item for item in predicted if _identity(item) == identity]
        if not gt_group:
            continue
        if not pred_group:
            unmatched_truth.extend(gt_group)
            continue
        cost = np.asarray(
            [
                [np.linalg.norm(np.asarray(gt["xy"], float) - np.asarray(pred["xy"], float)) for pred in pred_group]
                for gt in gt_group
            ],
            dtype=float,
        )
        rows, columns = linear_sum_assignment(cost)
        used_gt, used_pred = set(), set()
        for row, column in zip(rows, columns):
            gt, pred = gt_group[row], pred_group[column]
            matches.append((gt, pred, float(cost[row, column])))
            used_gt.add(row)
            used_pred.add(column)
            if pred in unmatched_pred:
                unmatched_pred.remove(pred)
        unmatched_truth.extend(gt for index, gt in enumerate(gt_group) if index not in used_gt)
    return matches, unmatched_truth, unmatched_pred


def _plan_metrics(detections: list[dict]) -> dict:
    home = np.asarray((0.30, -0.02), dtype=float)
    cursor = home.copy()
    length = 0.0
    actions = 0
    retries = 0
    by_identity = defaultdict(list)
    for item in detections:
        by_identity[_identity(item)].append(item)
    truck = by_identity.get(("truck", "truck"), [])
    truck_centre = np.asarray(truck[0]["xy"], float) if truck else None
    for colour, offset in zip(("red", "orange", "blue"), (-0.057, 0.0, 0.057)):
        bins = by_identity.get(("bin", colour), [])
        cubes = by_identity.get(("cube", colour), [])
        if not bins:
            continue
        bin_xy = np.asarray(bins[0]["xy"], float)
        remaining = [np.asarray(item["xy"], float) for item in cubes]
        while remaining:
            index = int(np.argmin([np.linalg.norm(point - cursor) for point in remaining]))
            cube = remaining.pop(index)
            length += float(np.linalg.norm(cube - cursor) + np.linalg.norm(bin_xy - cube))
            cursor = bin_xy.copy()
            actions += 1
        if truck_centre is not None:
            yaw = math.radians(float(truck[0].get("yaw_deg", -41.292)))
            target = truck_centre + offset * np.asarray((math.cos(yaw), math.sin(yaw)))
            length += float(np.linalg.norm(bin_xy - cursor) + np.linalg.norm(target - bin_xy))
            cursor = target
            actions += 1
        retries += sum(float(item.get("confidence", 0.0)) < 0.55 for item in cubes)
    return {
        "action_count": actions,
        "estimated_tcp_xy_path_m": length,
        "low_confidence_retry_budget": int(retries),
    }


def score_scene(truth_doc: dict, prediction: dict) -> dict:
    truth = truth_doc["objects"]
    predicted = prediction.get("detections", [])
    matches, missed, extra = _match(truth, predicted)
    object_rows = []
    for gt, pred, distance in matches:
        category = gt["category"]
        symmetry_error = (
            angle_mod_180_error(pred.get("yaw_deg", 0.0), gt.get("yaw_deg", 0.0))
            if category == "truck"
            else angle_mod_90_error(pred.get("yaw_deg", 0.0), gt.get("yaw_deg", 0.0))
        )
        xy_mm = distance * 1000.0
        if category == "cube":
            passed = xy_mm <= THRESHOLDS["cube_xy_mm"] and symmetry_error <= THRESHOLDS["cube_yaw_deg"]
        elif category == "bin":
            passed = xy_mm <= THRESHOLDS["bin_xy_mm"] and symmetry_error <= THRESHOLDS["bin_yaw_deg"]
        else:
            passed = xy_mm <= THRESHOLDS["truck_xy_mm"] and symmetry_error <= THRESHOLDS["truck_yaw_deg"]
        object_rows.append(
            {
                "name": gt["name"],
                "category": category,
                "colour": gt["colour"],
                "xy_error_mm": xy_mm,
                "yaw_error_deg": symmetry_error,
                "passed": bool(passed),
                "confidence": float(pred.get("confidence", 0.0)),
            }
        )
    expected = {"cube": 12, "bin": 3, "truck": 1}
    counts = {category: sum(item.get("category") == category for item in predicted) for category in expected}
    category_pass = {
        category: counts[category] == expected[category]
        and sum(row["category"] == category and row["passed"] for row in object_rows) == expected[category]
        for category in expected
    }
    inventory_pass = all(counts[category] == expected[category] for category in expected)
    task_pass = bool(inventory_pass and all(category_pass.values()) and not missed and not extra)
    plan_metrics = _plan_metrics(predicted)
    active = prediction.get("active_perception", {})
    plan_metrics["action_count"] += int(active.get("estimated_motion_actions", 0))
    plan_metrics["estimated_tcp_xy_path_m"] += float(active.get("estimated_tcp_xy_path_m", 0.0))
    return {
        "scene_id": truth_doc["scene_id"],
        "variation": truth_doc["variation"],
        "policy": prediction.get("policy", "unknown"),
        "task_pass": task_pass,
        "inventory_pass": inventory_pass,
        "category_pass": category_pass,
        "counts": counts,
        "missed": [{"name": item.get("name"), "category": item.get("category"), "colour": item.get("colour")} for item in missed],
        "false_positives": [{"category": item.get("category"), "colour": item.get("colour"), "xy": item.get("xy")} for item in extra],
        "objects": object_rows,
        "latency_ms": float(prediction.get("latency_ms", math.inf)) + float(active.get("estimated_acquisition_ms", 0.0)),
        "active_perception": active,
        **plan_metrics,
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.inf


def aggregate(rows: list[dict]) -> dict:
    object_rows = [item for row in rows for item in row["objects"]]

    def category_rate(category: str) -> float:
        values = [item["passed"] for item in object_rows if item["category"] == category]
        return float(np.mean(values)) if values else 0.0

    result = {
        "scenes": len(rows),
        "task_successes": sum(row["task_pass"] for row in rows),
        "task_failures": sum(not row["task_pass"] for row in rows),
        "task_success_rate": float(np.mean([row["task_pass"] for row in rows])) if rows else 0.0,
        "inventory_success_rate": float(np.mean([row["inventory_pass"] for row in rows])) if rows else 0.0,
        "cube_pose_pass_rate": category_rate("cube"),
        "bin_pose_pass_rate": category_rate("bin"),
        "truck_pose_pass_rate": category_rate("truck"),
        "latency_ms_mean": statistics.fmean(row["latency_ms"] for row in rows) if rows else math.inf,
        "latency_ms_p95": percentile([row["latency_ms"] for row in rows], 95),
        "action_count_mean": statistics.fmean(row["action_count"] for row in rows) if rows else 0.0,
        "estimated_tcp_xy_path_m_mean": statistics.fmean(row["estimated_tcp_xy_path_m"] for row in rows) if rows else 0.0,
        "retry_budget_mean": statistics.fmean(row["low_confidence_retry_budget"] for row in rows) if rows else 0.0,
    }
    by_variation = {}
    for variation in sorted({row["variation"] for row in rows}):
        subset = [row for row in rows if row["variation"] == variation]
        by_variation[variation] = {
            "scenes": len(subset),
            "successes": sum(row["task_pass"] for row in subset),
            "success_rate": float(np.mean([row["task_pass"] for row in subset])),
        }
    result["by_variation"] = by_variation
    # Ranking score weights whole-scene success most heavily and uses object
    # rates only to distinguish policies that fail the strict all-object gate.
    result["ranking_score"] = (
        0.55 * result["task_success_rate"]
        + 0.20 * result["cube_pose_pass_rate"]
        + 0.12 * result["bin_pose_pass_rate"]
        + 0.08 * result["truck_pose_pass_rate"]
        + 0.05 * result["inventory_success_rate"]
        - min(result["latency_ms_mean"] / 100000.0, 0.02)
    )
    return result


def markdown_report(ranking: list[dict], thresholds: dict) -> str:
    lines = [
        "# Truck observation-only benchmark ranking",
        "",
        "Policies received only RGB, depth, reported camera calibration, joint/gripper state, and timestamps. Simulator object transforms were evaluator-only.",
        "",
        "## Acceptance thresholds",
        "",
        "```json",
        json.dumps(thresholds, indent=2),
        "```",
        "",
        "## Ranking",
        "",
        "| Rank | Policy | Full scenes | Failures | Inventory | Cube pose | Bin pose | Truck pose | Mean latency | Actions | XY path | Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | `{item['policy']}` | {item['task_successes']}/{item['scenes']} "
            f"({item['task_success_rate']:.1%}) | {item['task_failures']} | {item['inventory_success_rate']:.1%} | "
            f"{item['cube_pose_pass_rate']:.1%} | {item['bin_pose_pass_rate']:.1%} | {item['truck_pose_pass_rate']:.1%} | "
            f"{item['latency_ms_mean']:.1f} ms | {item['action_count_mean']:.1f} | "
            f"{item['estimated_tcp_xy_path_m_mean']:.3f} m | {item['ranking_score']:.4f} |"
        )
    lines.extend(["", "## Robustness by variation", ""])
    variations = sorted({name for item in ranking for name in item["by_variation"]})
    lines.append("| Policy | " + " | ".join(variations) + " |")
    lines.append("|---|" + "|".join("---:" for _ in variations) + "|")
    for item in ranking:
        values = []
        for variation in variations:
            cell = item["by_variation"].get(variation, {"success_rate": 0.0, "scenes": 0})
            values.append(f"{cell['success_rate']:.0%} ({cell['scenes']})")
        lines.append(f"| `{item['policy']}` | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A full-scene success requires all twelve cubes, all three bins, and the truck deck to be detected with actionable pose error. Object rates explain near misses; they do not override a failed complete task.",
            "",
            "This ranking is the perception/planning gate. Full physics results are reported separately because a correct pose estimate can still fail through contact, slip, collision, or release dynamics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prediction_root = Path(args.predictions).resolve()
    evaluator_root = Path(args.evaluator).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    policies = sorted(path for path in prediction_root.iterdir() if path.is_dir())
    all_scene_scores = []
    ranking = []
    for policy_dir in policies:
        rows = []
        for prediction_path in sorted(policy_dir.glob("scene_*.json")):
            truth_path = evaluator_root / prediction_path.name
            if not truth_path.exists():
                continue
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            row = score_scene(truth, prediction)
            rows.append(row)
            all_scene_scores.append(row)
        item = aggregate(rows)
        item["policy"] = policy_dir.name
        ranking.append(item)
    ranking.sort(key=lambda item: (-item["ranking_score"], item["latency_ms_mean"], item["policy"]))

    report = {"thresholds": THRESHOLDS, "ranking": ranking, "scene_scores": all_scene_scores}
    (out / "benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "BENCHMARK_REPORT.md").write_text(markdown_report(ranking, THRESHOLDS), encoding="utf-8")
    if ranking:
        fieldnames = [
            "policy", "scenes", "task_successes", "task_failures", "task_success_rate",
            "inventory_success_rate", "cube_pose_pass_rate", "bin_pose_pass_rate",
            "truck_pose_pass_rate", "latency_ms_mean", "latency_ms_p95",
            "action_count_mean", "estimated_tcp_xy_path_m_mean", "retry_budget_mean", "ranking_score",
        ]
        with (out / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ranking)
    print(json.dumps(ranking[:5], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
