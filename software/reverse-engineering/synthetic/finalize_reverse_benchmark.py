"""Register official outputs, run the final evaluator, and render winners.

This finisher is deliberately observation-only.  External meshes are clipped
to the RGB-D TSDF envelope, never aligned to the withheld source geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PARTS = ("gearboxopen", "gearboxclosed")


def read_number(path: Path, default: float = 0.0) -> float:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return default


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, check=check)


def write_failure(path: Path, method: str, error: str, runtime_s: float | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "method": method,
                "status": "failed",
                "runtime_s": runtime_s,
                "source_mesh_access": False,
                "error": error,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def external_runtime(root: Path, family: str, part: str, pause_seconds: float) -> float:
    result = root / "official_results" / family / part
    if family == "neuralangelo":
        train = read_number(result / "train_wall_seconds.txt")
        # Only the open-part training spanned the documented Docker pause.
        if part == "gearboxopen":
            train = max(0.0, train - pause_seconds)
        mesh_extra = 0.0
        train_stamp = result / "train_wall_seconds.txt"
        mesh_stamp = result / "mesh.exit"
        if train_stamp.exists() and mesh_stamp.exists():
            mesh_extra = max(0.0, mesh_stamp.stat().st_mtime - train_stamp.stat().st_mtime)
        return train + mesh_extra
    return read_number(result / "wall_seconds.txt")


def register_external(
    root: Path,
    tools: Path,
    part: str,
    method: str,
    raw: Path,
    runtime_s: float,
) -> None:
    recon = root / "results" / part / "reconstructions"
    report = recon / f"{method}.json"
    if raw.exists():
        result = run(
            sys.executable,
            str(tools / "crop_external_to_observation.py"),
            "--input",
            str(raw),
            "--reference",
            str(recon / "tsdf_fk.ply"),
            "--out",
            str(recon),
            "--method",
            method,
            "--runtime-s",
            str(runtime_s),
            "--padding-mm",
            "5",
            check=False,
        )
        if result.returncode not in (0, 2):
            write_failure(report, method, f"registration process exited {result.returncode}", runtime_s)
    else:
        write_failure(report, method, f"expected output is missing: {raw}", runtime_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/work/isaac_run")
    parser.add_argument("--tools", default="/work/isaac_run/input")
    parser.add_argument("--pause-seconds", type=float, default=22957.0)
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--turntable-points", type=int, default=160_000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tools = Path(args.tools).resolve()

    for part in PARTS:
        neural_runtime = external_runtime(root, "neuralangelo", part, args.pause_seconds)
        register_external(
            root,
            tools,
            part,
            "neuralangelo",
            root / "official_results" / "neuralangelo" / part / "neuralangelo.ply",
            neural_runtime,
        )
        colmap_runtime = external_runtime(root, "colmap_mvs", part, args.pause_seconds)
        register_external(
            root,
            tools,
            part,
            "colmap_mvs",
            root / "official_results" / "colmap_mvs" / part / "poisson.ply",
            colmap_runtime,
        )

    for part in PARTS:
        run(
            sys.executable,
            str(tools / "evaluate_reconstructions.py"),
            "--dataset",
            str(root / "datasets" / part),
            "--reconstructions",
            str(root / "results" / part / "reconstructions"),
            "--out",
            str(root / "results" / part / "evaluation"),
            "--samples",
            str(args.samples),
        )

    aggregate = root / "results" / "aggregate"
    run(
        sys.executable,
        str(tools / "aggregate_results.py"),
        "--open",
        str(root / "results" / "gearboxopen" / "evaluation" / "evaluation.json"),
        "--closed",
        str(root / "results" / "gearboxclosed" / "evaluation" / "evaluation.json"),
        "--out",
        str(aggregate),
    )

    ranking = json.loads((aggregate / "aggregate_ranking.json").read_text(encoding="utf-8"))["ranking"]
    winners = [row for row in ranking if row.get("both_parts")][:3]
    videos = aggregate / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    for winner in winners:
        method = winner["method"]
        for part in PARTS:
            mesh = root / "results" / part / "reconstructions" / f"{method}.ply"
            if not mesh.exists():
                continue
            run(
                sys.executable,
                str(tools / "make_mesh_turntable.py"),
                "--mesh",
                str(mesh),
                "--out",
                str(videos / f"{part}_{method}.mp4"),
                "--label",
                f"{part} | {method}",
                "--points",
                str(args.turntable_points),
            )

    (aggregate / "finalization.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_mesh_used_by_reconstruction": False,
                "evaluation_samples_per_surface": args.samples,
                "pause_seconds_removed_from_active_runtime": args.pause_seconds,
                "winner_methods": [row["method"] for row in winners],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("FINALIZATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
