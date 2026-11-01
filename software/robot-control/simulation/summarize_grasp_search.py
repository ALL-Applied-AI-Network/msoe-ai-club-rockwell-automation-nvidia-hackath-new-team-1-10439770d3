"""Summarize Isaac reports without counting infrastructure timeouts as trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        cfg = report.get("configuration", {})
        metrics = report.get("measurements", {})
        rows.append(
            {
                "trial": path.stem,
                "passed": bool(report.get("task_complete")),
                "binary_close": float(cfg.get("jaw_target_mm_per_jaw", -1)) == 0.0,
                "mass_kg": cfg.get("mass_kg"),
                "friction": cfg.get("friction"),
                "insertion_mm": cfg.get("insertion_mm"),
                "force_n_per_finger": cfg.get("jaw_max_force_n_per_finger"),
                "x_error_mm": cfg.get("grip_x_error_mm", 0.0),
                "y_error_mm": cfg.get("grip_y_error_mm", 0.0),
                "yaw_error_deg": cfg.get("grip_yaw_error_deg", 0.0),
                "lift_mm": metrics.get("measured_lift_mm"),
                "slip_mm": metrics.get("max_hold_downward_slip_mm"),
                "tilt_deg": metrics.get("max_lift_and_hold_tilt_deg"),
                "gap_mm": metrics.get("contact_jaw_gap_mm"),
                "source": path.name,
            }
        )

    binary = [row for row in rows if row["binary_close"]]
    deployable = [
        row
        for row in binary
        if row["force_n_per_finger"] is not None
        and float(row["force_n_per_finger"]) <= 34.0
    ]
    summary = {
        "scope": "Isaac Sim 6.0 physics; binary gripper command compatibility",
        "completed_reports": len(rows),
        "binary_close_reports": len(binary),
        "deployable_force_envelope_reports": len(deployable),
        "deployable_force_envelope_passes": sum(row["passed"] for row in deployable),
        "all_deployable_force_envelope_passed": bool(deployable)
        and all(row["passed"] for row in deployable),
        "important_boundary": (
            "Partial-jaw trials are exploratory only because the physical API is binary. "
            "Only binary-close trials at <=34 N/finger support a deployable claim."
        ),
        "trials": rows,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Isaac grasp-search summary",
        "",
        summary["scope"],
        "",
        f"Completed reports: {len(rows)}. Binary-close reports at or below the "
        f"SMC 34 N/finger catalog condition: {len(deployable)}; passes: "
        f"{sum(row['passed'] for row in deployable)}.",
        "",
        summary["important_boundary"],
        "",
        "| Trial | Pass | Binary | Mass kg | Friction | Force N | XYZ/yaw error | Lift mm | Slip mm | Tilt deg | Gap mm |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {trial} | {passed} | {binary_close} | {mass_kg} | {friction} | "
            "{force_n_per_finger} | {x_error_mm}/{y_error_mm}/{yaw_error_deg} | "
            "{lift_mm:.3f} | {slip_mm:.3f} | {tilt_deg:.3f} | {gap_mm:.3f} |".format(
                **row
            )
        )
    lines += [
        "",
        "Infrastructure timeouts produced no JSON report and are deliberately absent; they are neither passes nor manipulation failures.",
    ]
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
