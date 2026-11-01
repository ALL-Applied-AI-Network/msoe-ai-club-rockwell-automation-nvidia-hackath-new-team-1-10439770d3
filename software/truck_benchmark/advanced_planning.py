"""Evaluate E (task ordering/yaw symmetry) and K (CAD grasp filtering).

Plans are generated from policy detections and observation packets only.  The
separate label directory is opened after planning solely to score collision
proxies, mirroring the perception benchmark boundary.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

from .policies import TRUSTED_CAMERA_TO_WORLD, load_scene_observation


HOME = np.asarray((0.30, -0.02), dtype=float)
COLOURS = ("red", "orange", "blue")
ZONE_OFFSET = {"red": -0.057, "orange": 0.0, "blue": 0.057}


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _equivalent_nearest(yaw: float, current: float, period: float) -> tuple[float, float]:
    candidates = [float(yaw) + period * k for k in range(-4, 5)]
    selected = min(candidates, key=lambda value: abs(value - current))
    return selected, abs(selected - current)


def _objects(detections: list[dict]):
    bins = {
        colour: max(
            (item for item in detections if item.get("category") == "bin" and item.get("colour") == colour),
            key=lambda item: item.get("confidence", 0.0),
            default=None,
        )
        for colour in COLOURS
    }
    cubes = [item for item in detections if item.get("category") == "cube" and item.get("colour") in COLOURS]
    trucks = [item for item in detections if item.get("category") == "truck"]
    truck = max(trucks, key=lambda item: item.get("confidence", 0.0), default=None)
    return bins, cubes, truck


def _truck_zones(truck: dict | None) -> dict[str, np.ndarray]:
    if truck is None:
        return {}
    centre = np.asarray(truck.get("deck_centre_xy", truck["xy"]), float)
    yaw = math.radians(float(truck.get("deck_yaw_deg", truck.get("yaw_deg", 0.0))))
    axis = np.asarray((math.cos(yaw), math.sin(yaw)))
    return {colour: centre + ZONE_OFFSET[colour] * axis for colour in COLOURS}


def baseline_plan(detections: list[dict]) -> dict:
    bins, cubes, truck = _objects(detections)
    cursor = HOME.copy()
    path = 0.0
    actions = []
    wrist = 0.0
    wrist_rotation = 0.0
    for colour in COLOURS:
        if bins[colour] is None:
            continue
        remaining = [item for item in cubes if item.get("colour") == colour]
        target = np.asarray(bins[colour]["xy"], float)
        while remaining:
            item = min(remaining, key=lambda value: _dist(cursor, value["xy"]))
            remaining.remove(item)
            source = np.asarray(item["xy"], float)
            path += _dist(cursor, source) + _dist(source, target)
            raw = float(item.get("yaw_deg", 0.0))
            wrist_rotation += abs(raw - wrist)
            wrist = raw
            actions.append({"kind": "cube", "colour": colour, "source": source.tolist(), "target": target.tolist(), "yaw": raw})
            cursor = target
    zones = _truck_zones(truck)
    for colour in COLOURS:
        if bins[colour] is None or colour not in zones:
            continue
        source = np.asarray(bins[colour]["xy"], float)
        target = zones[colour]
        path += _dist(cursor, source) + _dist(source, target)
        raw = float(bins[colour].get("yaw_deg", 0.0))
        wrist_rotation += abs(raw - wrist)
        wrist = raw
        actions.append({"kind": "bin", "colour": colour, "source": source.tolist(), "target": target.tolist(), "yaw": raw})
        cursor = target
    return {"actions": actions, "xy_path_m": path, "wrist_rotation_deg": wrist_rotation}


def optimized_plan(detections: list[dict]) -> dict:
    bins, cubes, truck = _objects(detections)
    tasks = [
        {
            "cube": item,
            "source": np.asarray(item["xy"], float),
            "target": np.asarray(bins[item["colour"]]["xy"], float),
        }
        for item in cubes
        if bins.get(item.get("colour")) is not None
    ]
    n = len(tasks)

    @lru_cache(maxsize=None)
    def solve(mask: int, last: int) -> tuple[float, tuple[int, ...]]:
        if mask == (1 << n) - 1:
            return 0.0, ()
        cursor = HOME if last < 0 else tasks[last]["target"]
        best = (math.inf, ())
        for index, task in enumerate(tasks):
            if mask & (1 << index):
                continue
            step = _dist(cursor, task["source"]) + _dist(task["source"], task["target"])
            tail, order = solve(mask | (1 << index), index)
            if step + tail < best[0]:
                best = (step + tail, (index,) + order)
        return best

    cube_cost, order = solve(0, -1)
    cursor = HOME if not order else tasks[order[-1]]["target"]
    zones = _truck_zones(truck)
    available = [colour for colour in COLOURS if bins[colour] is not None and colour in zones]
    best_bins = (math.inf, ())
    for permutation in itertools.permutations(available):
        here = cursor.copy()
        cost = 0.0
        for colour in permutation:
            source = np.asarray(bins[colour]["xy"], float)
            cost += _dist(here, source) + _dist(source, zones[colour])
            here = zones[colour]
        if cost < best_bins[0]:
            best_bins = (cost, permutation)

    actions = []
    wrist = 0.0
    wrist_rotation = 0.0
    for index in order:
        task = tasks[index]
        item = task["cube"]
        yaw, rotation = _equivalent_nearest(float(item.get("yaw_deg", 0.0)), wrist, 90.0)
        wrist_rotation += rotation
        wrist = yaw
        actions.append({
            "kind": "cube",
            "colour": item["colour"],
            "source": task["source"].tolist(),
            "target": task["target"].tolist(),
            "yaw": yaw,
        })
    for colour in best_bins[1]:
        item = bins[colour]
        yaw, rotation = _equivalent_nearest(float(item.get("yaw_deg", 0.0)), wrist, 180.0)
        wrist_rotation += rotation
        wrist = yaw
        actions.append({
            "kind": "bin",
            "colour": colour,
            "source": list(item["xy"]),
            "target": zones[colour].tolist(),
            "yaw": yaw,
        })
    return {
        "actions": actions,
        "xy_path_m": cube_cost + best_bins[0],
        "wrist_rotation_deg": wrist_rotation,
    }


def _residual_obstacle_points(scene_dir: Path, detections: list[dict]) -> np.ndarray:
    observation = load_scene_observation(scene_dir)
    table = observation.cameras["table"]
    model = table.camera
    if "table" in TRUSTED_CAMERA_TO_WORLD:
        model = type(model)(
            width=model.width,
            height=model.height,
            focal_length_mm=model.focal_length_mm,
            horizontal_aperture_mm=model.horizontal_aperture_mm,
            vertical_aperture_mm=model.vertical_aperture_mm,
            camera_to_world=TRUSTED_CAMERA_TO_WORLD["table"].copy(),
            table_z=model.table_z,
        )
    points, _, _ = model.depth_to_world(table.depth, stride=3)
    use = (
        (points[:, 2] > model.table_z + 0.004)
        & (points[:, 2] < model.table_z + 0.12)
        & (points[:, 0] > 0.16)
        & (points[:, 0] < 0.69)
        & (points[:, 1] > -0.38)
        & (points[:, 1] < 0.30)
    )
    points = points[use]
    keep = np.ones(len(points), dtype=bool)
    for item in detections:
        category = item.get("category")
        radius = 0.013 if category == "cube" else 0.045 if category == "bin" else 0.20
        keep &= np.linalg.norm(points[:, :2] - np.asarray(item["xy"], float), axis=1) > radius
    return points[keep, :2]


def _corridor_hits(points: np.ndarray, centre: np.ndarray, yaw_deg: float) -> int:
    if len(points) == 0:
        return 0
    angle = -math.radians(float(yaw_deg))
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    local = (points - centre) @ rotation.T
    # Installed pair: 18 mm open tip gap + two 6 mm fingers, with 2 mm
    # clearance; 12 mm finger width plus the same clearance.
    return int(np.sum((np.abs(local[:, 0]) < 0.017) & (np.abs(local[:, 1]) < 0.008)))


def collision_filtered_plan(scene_dir: Path, detections: list[dict], ordered: dict) -> dict:
    obstacles = _residual_obstacle_points(scene_dir, detections)
    actions = []
    current_yaw = 0.0
    for action in ordered["actions"]:
        period = 90.0 if action["kind"] == "cube" else 180.0
        candidates = [float(action["yaw"]) + period * k for k in range(-2, 3)]
        centre = np.asarray(action["source"], float)
        scored = [
            (_corridor_hits(obstacles, centre, yaw), abs(yaw - current_yaw), yaw)
            for yaw in candidates
        ]
        hits, rotation, selected = min(scored)
        item = dict(action)
        item["yaw"] = selected
        item["predicted_obstacle_points_in_corridor"] = hits
        item["gripper_outer_span_mm"] = 34.0
        item["gripper_width_mm"] = 16.0
        actions.append(item)
        current_yaw = selected
    return {
        **ordered,
        "actions": actions,
        "predicted_corridor_hits": sum(item["predicted_obstacle_points_in_corridor"] for item in actions),
        "collision_free_actions": sum(item["predicted_obstacle_points_in_corridor"] == 0 for item in actions),
    }


def _label_collision_proxy(plan: dict, label_doc: dict) -> int:
    clutter = [np.asarray(item["xy"], float) for item in label_doc.get("clutter", []) if "xy" in item]
    if not clutter:
        return 0
    points = np.asarray(clutter)
    return sum(
        _corridor_hits(points, np.asarray(action["source"], float), float(action["yaw"])) > 0
        for action in plan["actions"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prediction_dir = Path(args.predictions)
    observation_root = Path(args.observations)
    label_root = Path(args.labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for prediction_path in sorted(prediction_dir.glob("scene_*.json")):
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        label_doc = json.loads((label_root / prediction_path.name).read_text(encoding="utf-8"))
        baseline = baseline_plan(prediction.get("detections", []))
        optimized = optimized_plan(prediction.get("detections", []))
        filtered = collision_filtered_plan(
            observation_root / prediction_path.stem,
            prediction.get("detections", []),
            optimized,
        )
        baseline_hits = _label_collision_proxy(baseline, label_doc)
        filtered_hits = _label_collision_proxy(filtered, label_doc)
        row = {
            "scene": prediction_path.stem,
            "variation": label_doc["variation"],
            "baseline_xy_path_m": baseline["xy_path_m"],
            "optimized_xy_path_m": optimized["xy_path_m"],
            "path_reduction_percent": 100.0 * (baseline["xy_path_m"] - optimized["xy_path_m"]) / max(baseline["xy_path_m"], 1e-9),
            "baseline_wrist_rotation_deg": baseline["wrist_rotation_deg"],
            "optimized_wrist_rotation_deg": optimized["wrist_rotation_deg"],
            "rotation_reduction_percent": 100.0 * (baseline["wrist_rotation_deg"] - optimized["wrist_rotation_deg"]) / max(baseline["wrist_rotation_deg"], 1e-9),
            "action_count": len(optimized["actions"]),
            "predicted_corridor_hits": filtered["predicted_corridor_hits"],
            "label_collision_proxy_baseline": baseline_hits,
            "label_collision_proxy_filtered": filtered_hits,
            "plan": filtered,
        }
        rows.append(row)
        (out / prediction_path.name).write_text(json.dumps(row, indent=2), encoding="utf-8")
    aggregate = {
        "scenes": len(rows),
        "mean_path_reduction_percent": statistics.fmean(row["path_reduction_percent"] for row in rows),
        "mean_rotation_reduction_percent": statistics.fmean(row["rotation_reduction_percent"] for row in rows),
        "mean_action_count": statistics.fmean(row["action_count"] for row in rows),
        "baseline_label_collision_proxy": sum(row["label_collision_proxy_baseline"] for row in rows),
        "filtered_label_collision_proxy": sum(row["label_collision_proxy_filtered"] for row in rows),
        "rows": rows,
    }
    (out / "planning_results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
