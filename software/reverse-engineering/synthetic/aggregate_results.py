"""Aggregate per-part evaluator reports into a two-part method ranking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", required=True)
    parser.add_argument("--closed", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reports = {
        "gearboxopen": json.loads(Path(args.open).read_text(encoding="utf-8")),
        "gearboxclosed": json.loads(Path(args.closed).read_text(encoding="utf-8")),
    }
    by_method: dict[str, list[tuple[str, dict]]] = {}
    failures = []
    for part, report in reports.items():
        for row in report.get("ranking", []):
            by_method.setdefault(row["method"], []).append((part, row))
        for row in report.get("failures", []):
            failures.append({"part": part, **row})

    rows = []
    for method, values in by_method.items():
        metrics = [value for _, value in values]
        row = {
            "method": method,
            "parts_scored": len(values),
            "mean_ranking_score": mean(v["ranking_score"] for v in metrics),
            "mean_fscore_0p5mm": mean(v["fscore_0p5mm"] for v in metrics),
            "mean_fscore_1p0mm": mean(v["fscore_1p0mm"] for v in metrics),
            "mean_fscore_2p0mm": mean(v["fscore_2p0mm"] for v in metrics),
            "mean_fscore_5p0mm": mean(v["fscore_5p0mm"] for v in metrics),
            "mean_chamfer_mm": mean(v["symmetric_chamfer_mean_mm"] for v in metrics),
            "mean_normal_consistency": mean(v["normal_consistency"] for v in metrics),
            "total_runtime_s": sum((v.get("runtime_s") or 0.0) for v in metrics),
            "both_parts": len(values) == 2,
            "part_results": {part: value for part, value in values},
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["both_parts"], row["mean_ranking_score"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"ranking": rows, "failures": failures}
    (out / "aggregate_ranking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "rank", "method", "parts_scored", "both_parts", "mean_ranking_score",
        "mean_fscore_0p5mm", "mean_fscore_1p0mm", "mean_fscore_2p0mm",
        "mean_fscore_5p0mm", "mean_chamfer_mm", "mean_normal_consistency", "total_runtime_s",
    ]
    with (out / "aggregate_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"ranking": [{key: row[key] for key in fields} for row in rows], "failures": failures}, indent=2))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
