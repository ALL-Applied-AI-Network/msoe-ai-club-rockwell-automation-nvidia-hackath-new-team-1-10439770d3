"""Attach recorded FANUC state to the prepared clean-orbit frame manifest.

The clean 85-second clip begins at frame 3350 of LeRobot episode 2.  That offset
was established by exhaustive 80x45 luminance-frame matching; the best match
had MSE 0.1603 and the next unrelated candidate was 0.7725.  Telemetry remains
an optional, separately reported pose-prior track—it is never mixed into the
pure image-only reconstruction ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def to_list(value: object) -> list[float]:
    return [float(item) for item in value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean-to-original-offset", type=int, default=3350)
    args = parser.parse_args()

    data = pd.read_parquet(args.parquet)
    with args.selected.open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    records: list[dict] = []
    for row in selected:
        clean_frame = int(row["frame"])
        source_frame = args.clean_to_original_offset + clean_frame
        if source_frame < 0 or source_frame >= len(data):
            raise IndexError(f"Frame {source_frame} outside telemetry table")
        telemetry = data.iloc[source_frame]
        records.append(
            {
                "set": row["set"],
                "output_index": int(row["output_index"]),
                "stem": row["stem"],
                "clean_frame": clean_frame,
                "original_episode_frame": source_frame,
                "clean_time_s": float(row["time_s"]),
                "episode_time_s": float(telemetry["timestamp"]),
                "joint_deg_and_gripper": to_list(telemetry["observation.state"]),
                "tcp_xyz_mm_wpr_deg": to_list(telemetry["observation.tcp_pose"]),
                "commanded_action": to_list(telemetry["action"]),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_parquet": str(args.parquet.resolve()),
                "clean_to_original_frame_offset": args.clean_to_original_offset,
                "offset_evidence": {
                    "method": "exhaustive 80x45 luminance MSE match of clean frame 0 against episode 2",
                    "best_original_frame": 3350,
                    "best_mse": 0.16027778387069702,
                    "next_unrelated_mse": 0.7724999785423279,
                },
                "camera_hand_eye_known": False,
                "note": "TCP poses are trajectory priors, not calibrated camera extrinsics.",
                "records": records,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {len(records)} telemetry records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
