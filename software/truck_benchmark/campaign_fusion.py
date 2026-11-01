"""Observation-only late fusion for the truck benchmark.

The primary detector remains authoritative.  Secondary predictions may only
fill a missing inventory slot when spatially distinct detections form a
cross-policy consensus cluster.  Evaluator labels are deliberately absent
from this module.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


EXPECTED = {
    ("cube", "red"): 4,
    ("cube", "orange"): 4,
    ("cube", "blue"): 4,
    ("bin", "red"): 1,
    ("bin", "orange"): 1,
    ("bin", "blue"): 1,
    ("truck", "truck"): 1,
}


def _identity(item: dict) -> tuple[str, str]:
    return str(item.get("category")), str(item.get("colour"))


def _distance(left: dict, right: dict) -> float:
    return float(np.linalg.norm(np.asarray(left["xy"], float) - np.asarray(right["xy"], float)))


def _confidence(item: dict) -> float:
    return float(item.get("confidence", 0.0))


def _symmetry_error(left: float, right: float, period: float = 90.0) -> float:
    return abs((left - right + period / 2.0) % period - period / 2.0)


def _consensus_yaw(items: list[dict], period: float = 90.0) -> tuple[float, float]:
    yaws = [float(item.get("yaw_deg", 0.0)) for item in items]
    selected = min(yaws, key=lambda yaw: sum(_symmetry_error(yaw, other, period) for other in yaws))
    spread = max(_symmetry_error(selected, other, period) for other in yaws)
    return selected, spread


def _refine_primary_pose(item: dict, secondaries: dict[str, dict]) -> dict:
    """Apply conservative, source-specific pose repairs before inventory fill."""
    identity = _identity(item)
    refined = dict(item)
    if identity[0] == "bin" and item.get("source") == "rgb_bin_model_fit" and "dino" in secondaries:
        candidates = [entry for entry in secondaries["dino"].get("detections", []) if _identity(entry) == identity]
        if candidates:
            candidate = min(candidates, key=lambda entry: _distance(item, entry))
            if _distance(item, candidate) <= 0.012:
                refined["xy"] = list(candidate["xy"])
                refined["yaw_deg"] = float(candidate.get("yaw_deg", refined.get("yaw_deg", 0.0)))
                refined["source"] = "rgb_bin_model_fit+dino_pose_repair"
                refined["pose_repair_delta_mm"] = 1000.0 * _distance(item, candidate)
        return refined
    if identity[0] == "cube" and item.get("source") == "table_depth_geometry":
        alternatives = []
        for source_name in ("A", "B", "fgr", "teaser"):
            prediction = secondaries.get(source_name)
            if prediction is None:
                continue
            candidates = [entry for entry in prediction.get("detections", []) if _identity(entry) == identity]
            if not candidates:
                continue
            candidate = min(candidates, key=lambda entry: _distance(item, entry))
            if _distance(item, candidate) <= 0.0035:
                tagged = dict(candidate)
                tagged["_fusion_source"] = source_name
                alternatives.append(tagged)
        has_wrist_geometry = any(
            entry["_fusion_source"] == "A" and entry.get("source") == "wrist_depth_geometry"
            for entry in alternatives
        )
        if len(alternatives) >= 2 and has_wrist_geometry:
            yaw, spread = _consensus_yaw(alternatives)
            if spread <= 5.0 and _symmetry_error(float(item.get("yaw_deg", 0.0)), yaw) >= 6.0:
                refined["yaw_deg"] = yaw
                refined["source"] = "table_depth_geometry+multimethod_yaw_repair"
                refined["yaw_repair_support"] = sorted(entry["_fusion_source"] for entry in alternatives)
                refined["yaw_repair_spread_deg"] = spread
    return refined


def _cluster_candidates(candidates: list[dict], radius_m: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for candidate in sorted(candidates, key=_confidence, reverse=True):
        point = np.asarray(candidate["xy"], float)
        best = None
        best_distance = math.inf
        for cluster in clusters:
            centre = np.median(np.asarray([item["xy"] for item in cluster], float), axis=0)
            distance = float(np.linalg.norm(point - centre))
            if distance <= radius_m and distance < best_distance:
                best, best_distance = cluster, distance
        if best is None:
            clusters.append([candidate])
        elif candidate["_fusion_source"] not in {item["_fusion_source"] for item in best}:
            best.append(candidate)
    return clusters


def _representative(cluster: list[dict]) -> dict:
    # Median translation is robust to one bad secondary.  Keep the yaw and
    # metadata of the most confident member because cube yaw is modulo 90 deg.
    exemplar = max(cluster, key=_confidence)
    item = {key: value for key, value in exemplar.items() if not key.startswith("_fusion_")}
    item["xy"] = np.median(np.asarray([entry["xy"] for entry in cluster], float), axis=0).tolist()
    item["confidence"] = min(0.89, statistics.fmean(_confidence(entry) for entry in cluster))
    item["source"] = "consensus_fill:" + "+".join(sorted(entry["_fusion_source"] for entry in cluster))
    item["fusion_support"] = len(cluster)
    item["fusion_spread_mm"] = 1000.0 * max(
        float(np.linalg.norm(np.asarray(entry["xy"], float) - np.asarray(item["xy"], float)))
        for entry in cluster
    )
    return item


def fuse_document(
    primary: dict,
    secondaries: dict[str, dict],
    *,
    cluster_radius_m: float = 0.008,
    dedupe_radius_m: float = 0.006,
    min_sources: int = 2,
    require_rgb_source: bool = False,
    rgb_source_names: frozenset[str] = frozenset({"dino"}),
    refine_pose: bool = False,
) -> dict:
    started = time.perf_counter()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in primary.get("detections", []):
        if _identity(item) in EXPECTED:
            grouped[_identity(item)].append(
                _refine_primary_pose(item, secondaries) if refine_pose else dict(item)
            )

    # Cap over-detections per semantic identity using primary confidence only.
    for identity, expected in EXPECTED.items():
        grouped[identity] = sorted(grouped[identity], key=_confidence, reverse=True)[:expected]

    additions: list[dict] = []
    for identity, expected in EXPECTED.items():
        missing = expected - len(grouped[identity])
        if missing <= 0:
            continue
        candidates = []
        for source_name, prediction in secondaries.items():
            for raw in prediction.get("detections", []):
                if _identity(raw) != identity:
                    continue
                item = dict(raw)
                item["_fusion_source"] = source_name
                if any(_distance(item, present) <= dedupe_radius_m for present in grouped[identity]):
                    continue
                candidates.append(item)
        clusters = _cluster_candidates(candidates, cluster_radius_m)
        eligible = []
        for cluster in clusters:
            sources = {entry["_fusion_source"] for entry in cluster}
            if len(sources) < min_sources:
                continue
            if require_rgb_source and not (sources & rgb_source_names):
                continue
            centre = np.median(np.asarray([entry["xy"] for entry in cluster], float), axis=0)
            spread = max(float(np.linalg.norm(np.asarray(entry["xy"], float) - centre)) for entry in cluster)
            eligible.append((len(sources), -spread, statistics.fmean(_confidence(entry) for entry in cluster), cluster))
        for _, _, _, cluster in sorted(eligible, key=lambda row: row[:3], reverse=True)[:missing]:
            item = _representative(cluster)
            grouped[identity].append(item)
            additions.append(item)

    detections = [item for identity in EXPECTED for item in grouped[identity]]
    component_latency = float(primary.get("latency_ms", 0.0)) + sum(
        float(prediction.get("latency_ms", 0.0)) for prediction in secondaries.values()
    )
    return {
        **{key: value for key, value in primary.items() if key not in {"detections", "latency_ms", "policy"}},
        "policy": "L_consensus_inventory_fusion",
        "detections": detections,
        "latency_ms": component_latency + 1000.0 * (time.perf_counter() - started),
        "fusion": {
            "primary_policy": primary.get("policy"),
            "secondary_policies": sorted(secondaries),
            "cluster_radius_mm": 1000.0 * cluster_radius_m,
            "dedupe_radius_mm": 1000.0 * dedupe_radius_m,
            "min_sources": min_sources,
            "require_rgb_source": require_rgb_source,
            "refine_pose": refine_pose,
            "additions": additions,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-mm", type=float, default=8.0)
    parser.add_argument("--dedupe-mm", type=float, default=6.0)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--require-rgb", action="store_true")
    parser.add_argument("--refine-pose", action="store_true")
    args = parser.parse_args()
    primary_dir = Path(args.primary)
    secondary_dirs = dict(item.split("=", 1) for item in args.secondary)
    secondary_dirs = {name: Path(path) for name, path in secondary_dirs.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for primary_path in sorted(primary_dir.glob("scene_*.json")):
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        secondaries = {
            name: json.loads((path / primary_path.name).read_text(encoding="utf-8"))
            for name, path in secondary_dirs.items()
            if (path / primary_path.name).exists()
        }
        fused = fuse_document(
            primary,
            secondaries,
            cluster_radius_m=args.cluster_mm / 1000.0,
            dedupe_radius_m=args.dedupe_mm / 1000.0,
            min_sources=args.min_sources,
            require_rgb_source=args.require_rgb,
            refine_pose=args.refine_pose,
        )
        (out / primary_path.name).write_text(json.dumps(fused, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
