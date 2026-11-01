"""Extract policy-relevant telemetry from organizer-provided LeRobot archives.

This deliberately analyzes behavior and timing rather than exporting or replaying
another team's trajectories.  An operator-ended episode is not labelled a
success: the archive does not contain the operator's success score.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _parquet_members(archive: tarfile.TarFile, marker: str) -> list[str]:
    return sorted(
        name
        for name in archive.getnames()
        if marker in name and name.endswith(".parquet")
    )


def _read_table(archive: tarfile.TarFile, names: list[str]) -> pa.Table:
    tables = []
    for name in names:
        member = archive.extractfile(name)
        if member is None:
            continue
        tables.append(pq.read_table(io.BytesIO(member.read())))
    if not tables:
        raise ValueError("archive does not contain the expected parquet data")
    return pa.concat_tables(tables, promote_options="default")


def _arrays(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _transitions(values: np.ndarray) -> int:
    binary = values >= 0.5
    return int(np.count_nonzero(binary[1:] != binary[:-1]))


def _best_tracking_lag(
    commands: np.ndarray, observations: np.ndarray, max_lag_samples: int = 20
) -> dict[str, float | int]:
    """Return the lag whose robust command/observation error is smallest."""
    best: dict[str, float | int] | None = None
    for lag in range(max_lag_samples + 1):
        if lag:
            command = commands[:-lag]
            observed = observations[lag:]
        else:
            command = commands
            observed = observations
        if len(command) < 5:
            continue
        errors = np.linalg.norm(command - observed, axis=1)
        candidate = {
            "samples": lag,
            "median_error": float(np.median(errors)),
            "p95_error": float(np.quantile(errors, 0.95)),
        }
        if best is None or candidate["median_error"] < best["median_error"]:
            best = candidate
    assert best is not None
    return best


def _range(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": np.min(values, axis=0).round(4).tolist(),
        "max": np.max(values, axis=0).round(4).tolist(),
    }


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def analyze_archive(path: Path) -> dict[str, Any]:
    with tarfile.open(path) as archive:
        names = archive.getnames()
        manifest_name = next(name for name in names if name.endswith("manipbench.json"))
        manifest_file = archive.extractfile(manifest_name)
        if manifest_file is None:
            raise ValueError(f"cannot read {manifest_name}")
        manifest = {
            int(item["episode_index"]): item
            for item in json.loads(manifest_file.read().decode("utf-8"))
        }
        data = _read_table(archive, _parquet_members(archive, "/data/"))
        episodes = _read_table(archive, _parquet_members(archive, "/meta/episodes/"))

    episode_meta = {
        int(row["episode_index"]): row
        for row in episodes.select(["episode_index", "tasks", "length"]).to_pylist()
    }
    rows = data.select(
        [
            "observation.state",
            "observation.tcp_pose",
            "action",
            "timestamp",
            "episode_index",
        ]
    ).to_pylist()

    result: dict[str, Any] = {
        "archive": path.name,
        "evidence_boundary": (
            "Telemetry only. Operator-ended episodes are not assumed successful; "
            "leaderboard scores are not stored in this archive."
        ),
        "episodes": [],
    }
    for episode_index in sorted(manifest):
        episode_rows = [row for row in rows if int(row["episode_index"]) == episode_index]
        if not episode_rows:
            continue
        state = _arrays([row["observation.state"] for row in episode_rows])
        tcp = _arrays([row["observation.tcp_pose"] for row in episode_rows])
        action = _arrays([row["action"] for row in episode_rows])
        timestamps = _arrays([row["timestamp"] for row in episode_rows])
        duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        rate = (len(timestamps) - 1) / duration if duration > 0 else 0.0
        action_kind = str(manifest[episode_index]["action_kind"])
        ignored_prefix_frames = 0
        active = np.ones(len(action), dtype=bool)
        if action_kind == "tcp":
            # The recordings begin with controller/home rows whose action field
            # contains joint-shaped values even though the submitted policy is
            # in TCP mode.  They are not TCP jumps.  Start analysis at the first
            # pose-shaped row and retain the full episode duration separately.
            pose_rows = (action[:, 0] > 100.0) & (np.abs(action[:, 3]) > 90.0)
            indices = np.flatnonzero(pose_rows)
            if len(indices):
                ignored_prefix_frames = int(indices[0])
                active[: indices[0]] = False
        active_action = action[active]
        active_tcp = tcp[active]
        active_state = state[active]
        command_space = active_action[:, :6]
        observed_space = active_tcp if action_kind == "tcp" else active_state[:, :6]
        lag = _best_tracking_lag(command_space, observed_space)
        lag["milliseconds"] = round(float(lag["samples"]) * 1000.0 / rate, 1) if rate else 0.0

        if action_kind == "tcp":
            command_step = np.linalg.norm(np.diff(command_space[:, :3], axis=0), axis=1)
            units = "mm"
        else:
            command_step = np.linalg.norm(np.diff(command_space, axis=0), axis=1)
            units = "degrees"

        meta = episode_meta.get(episode_index, {})
        episode_result: dict[str, Any] = {
            "episode_index": episode_index,
            "trial_id": manifest[episode_index].get("trial_id"),
            "task": (meta.get("tasks") or [""])[0],
            "action_kind": action_kind,
            "end_reason": manifest[episode_index].get("end_reason"),
            "frames": len(episode_rows),
            "duration_s": round(duration, 3),
            "sample_rate_hz": round(rate, 3),
            "ignored_controller_prefix_frames": ignored_prefix_frames,
            "observed_tcp_xyz_mm": _range(tcp[:, :3]),
            "observed_tcp_radial_max_mm": round(
                float(np.linalg.norm(tcp[:, :2], axis=1).max()), 3
            ),
            "observed_tcp_path_mm": round(_path_length(tcp[:, :3]), 3),
            "observed_gripper_transitions": _transitions(state[:, 6]),
            "commanded_gripper_transitions": _transitions(active_action[:, 6]),
            "commanded_gripper_closed_fraction": round(float(np.mean(active_action[:, 6] >= 0.5)), 4),
            "tracking_best_lag": lag,
            "command_step_units": units,
            "command_step_median": round(float(np.median(command_step)), 4),
            "command_step_p95": round(float(np.quantile(command_step, 0.95)), 4),
            "command_step_max": round(float(np.max(command_step)), 4),
        }
        if action_kind == "tcp":
            episode_result["commanded_tcp_xyz_mm"] = _range(active_action[:, :3])
            episode_result["commanded_tcp_radial_max_mm"] = round(
                float(np.linalg.norm(active_action[:, :2], axis=1).max()), 3
            )
            episode_result["commanded_tcp_path_mm"] = round(
                _path_length(active_action[:, :3]), 3
            )
        result["episodes"].append(episode_result)

    result["summary"] = {
        "episode_count": len(result["episodes"]),
        "end_reasons": dict(Counter(row["end_reason"] for row in result["episodes"])),
        "action_kinds": dict(Counter(row["action_kind"] for row in result["episodes"])),
        "total_frames": sum(row["frames"] for row in result["episodes"]),
        "median_sample_rate_hz": round(
            float(np.median([row["sample_rate_hz"] for row in result["episodes"]])), 3
        ),
    }
    return result


def _flat_rows(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for analysis in analyses:
        for episode in analysis["episodes"]:
            xyz = episode["observed_tcp_xyz_mm"]
            lag = episode["tracking_best_lag"]
            rows.append(
                {
                    "archive": analysis["archive"],
                    "episode": episode["episode_index"],
                    "trial_id": episode["trial_id"],
                    "task": episode["task"],
                    "action_kind": episode["action_kind"],
                    "end_reason": episode["end_reason"],
                    "frames": episode["frames"],
                    "duration_s": episode["duration_s"],
                    "sample_rate_hz": episode["sample_rate_hz"],
                    "ignored_prefix_frames": episode["ignored_controller_prefix_frames"],
                    "min_tcp_z_mm": xyz["min"][2],
                    "max_tcp_radius_mm": episode["observed_tcp_radial_max_mm"],
                    "gripper_transitions": episode["commanded_gripper_transitions"],
                    "best_lag_ms": lag["milliseconds"],
                    "lag_median_error": round(float(lag["median_error"]), 4),
                    "max_command_step": episode["command_step_max"],
                    "command_step_units": episode["command_step_units"],
                }
            )
    return rows


def _markdown(analyses: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    faults = [row for row in rows if row["end_reason"] == "controller fault"]
    nonfaults = [row for row in rows if row["end_reason"] != "controller fault"]

    def med(items: list[dict[str, Any]], key: str) -> str:
        if not items:
            return "n/a"
        return f"{float(np.median([item[key] for item in items])):.2f}"

    lines = [
        "# Organizer Reference Rollout Analysis",
        "",
        "This report extracts timing, command, state, and termination evidence from the three organizer-provided LeRobot archives. It does **not** copy trajectories into our policy and does **not** infer success from an operator-ended recording. The archives contain termination reasons, not leaderboard success labels.",
        "",
        "## Corpus",
        "",
        "| Archive | Episodes | Frames | Action mode | End reasons |",
        "|---|---:|---:|---|---|",
    ]
    for analysis in analyses:
        summary = analysis["summary"]
        reasons = ", ".join(f"{key}: {value}" for key, value in summary["end_reasons"].items())
        modes = ", ".join(summary["action_kinds"])
        lines.append(
            f"| `{analysis['archive']}` | {summary['episode_count']} | {summary['total_frames']} | {modes} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "All recordings are approximately 10 Hz, matching the official action-step contract. The physical arm's documented tracking delay means our policy must gate phase changes on observed TCP/joint state, not merely on elapsed action rows.",
            "",
            "## Episode telemetry",
            "",
            "| Archive / ep. | End | Mode | Duration (s) | Min Z (mm) | Max radius (mm) | Grip transitions | Best lag (ms) | Max command step |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['archive']}` / {row['episode']} | {row['end_reason']} | {row['action_kind']} | {row['duration_s']:.1f} | {row['min_tcp_z_mm']:.1f} | {row['max_tcp_radius_mm']:.1f} | {row['gripper_transitions']} | {row['best_lag_ms']:.0f} | {row['max_command_step']:.2f} {row['command_step_units']} |"
        )
    lines.extend(
        [
            "",
            "## Findings we can safely use",
            "",
            f"- Controller faults are present in {len(faults)} of {len(rows)} recorded episodes. Fault and non-fault recordings overlap in workspace and duration (median minimum Z: {med(faults, 'min_tcp_z_mm')} vs {med(nonfaults, 'min_tcp_z_mm')} mm; median maximum radius: {med(faults, 'max_tcp_radius_mm')} vs {med(nonfaults, 'max_tcp_radius_mm')} mm). A single global Z or radius threshold cannot explain these faults.",
            "- Gripper commands are binary, and transitions are sparse. Our candidate therefore uses fully open/fully closed commands and verifies the resulting grasp through motion and vision.",
            "- Recorded commands and observations are not synchronous enough to treat a command as achieved immediately. Our controller advances only after observed TCP/joints meet a tolerance for consecutive observations. Controller/home prefix rows were excluded from TCP command statistics because those rows contain joint-shaped values in the action column.",
            "- The three synchronized RGB cameras are valuable for independent checks. The current candidate uses table RGB/depth for localization, wrist depth for near-field presence, and side RGB for lift confirmation rather than averaging all sensors into one potentially biased pose.",
            "- Layouts change between rollouts, so the recorded coordinates are evidence about interfaces and dynamics—not reusable waypoints.",
            "",
            "## Evidence boundary",
            "",
            "The archives do not include the operator's success/failure score in `manipbench.json`. `end_reason: operator` means only that the operator ended the episode. Visual outcome scoring would require the result page or a reviewed video annotation. The lag statistic is a telemetry correlation estimate, not a controller specification.",
            "",
            "Machine-readable details are in `reference_rollout_analysis.json` and `reference_rollout_episodes.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    analyses = [analyze_archive(path) for path in args.archives]
    rows = _flat_rows(analyses)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reference_rollout_analysis.json").write_text(
        json.dumps({"archives": analyses}, indent=2), encoding="utf-8"
    )
    with (args.out_dir / "reference_rollout_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "REFERENCE_ROLLOUT_ANALYSIS.md").write_text(
        _markdown(analyses, rows), encoding="utf-8"
    )
    print(f"wrote {len(rows)} episode summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
