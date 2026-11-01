"""Summarize perception-driven layout trials without copying their large timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for report_path in sorted(args.matrix_dir.glob("*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        perception = report.get("perception", {})
        measurements = report.get("measurements", {})
        configuration = report.get("configuration", {})
        rows.append(
            {
                "trial": report_path.parent.name,
                "corrected_depth_orientation": report_path.parent.name.endswith("_retry")
                or report_path.parent.name == "right_front",
                "x_mm": configuration.get("bin_x_mm"),
                "y_mm": configuration.get("bin_y_mm"),
                "yaw_deg": configuration.get("bin_yaw_deg"),
                "perception_passed": perception.get("passed", False),
                "acceptance_passed": report.get("acceptance", {}).get("passed", False),
                "wall_error_mm": perception.get("wall_error_mm"),
                "measured_lift_mm": measurements.get("measured_lift_mm"),
                "slip_mm": measurements.get("max_hold_downward_slip_mm"),
                "tilt_deg": measurements.get("max_lift_and_hold_tilt_deg"),
                "error": report.get("error"),
            }
        )

    corrected = [row for row in rows if row["corrected_depth_orientation"]]
    output = {
        "evidence_boundary": "Isaac Sim 6.0 only; no physical reliability claim.",
        "trials": rows,
        "corrected_summary": {
            "completed": len(corrected),
            "perception_passed": sum(bool(row["perception_passed"]) for row in corrected),
            "physics_acceptance_passed": sum(bool(row["acceptance_passed"]) for row in corrected),
            "maximum_wall_error_mm": max(
                float(row["wall_error_mm"]) for row in corrected if row["wall_error_mm"] is not None
            ),
        },
    }
    (args.matrix_dir / "summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    lines = [
        "# Perception-driven layout matrix",
        "",
        "Isaac Sim 6.0 only. The perceived wall pose drove the robot; ground truth was used only for evaluation.",
        "",
        "| Trial | Corrected estimator | Layout x/y/yaw | Perception | Physics | Wall error mm | Lift mm | Slip mm | Tilt deg |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def number(key: str) -> str:
            value = row[key]
            return "—" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {row['trial']} | {row['corrected_depth_orientation']} | "
            f"{row['x_mm']}/{row['y_mm']}/{row['yaw_deg']} | "
            f"{row['perception_passed']} | {row['acceptance_passed']} | "
            f"{number('wall_error_mm')} | {number('measured_lift_mm')} | "
            f"{number('slip_mm')} | {number('tilt_deg')} |"
        )
    summary = output["corrected_summary"]
    lines.extend(
        [
            "",
            f"Corrected estimator: {summary['perception_passed']}/{summary['completed']} perception passes and {summary['physics_acceptance_passed']}/{summary['completed']} physics passes; maximum wall error {summary['maximum_wall_error_mm']:.3f} mm.",
            "",
            "The retained pre-fix failures are an ablation: RGB orientation biased the wall target by more than the 2 mm gate. The corrected estimator uses the metric depth footprint for orientation and RGB as an identity/consistency check.",
            "",
            "The `center` directory contains an infrastructure startup timeout and no report; it is not counted as a policy result.",
            "",
        ]
    )
    (args.matrix_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["corrected_summary"], indent=2))


if __name__ == "__main__":
    main()
