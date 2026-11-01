"""Write a human-readable report from the final reverse-engineering results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--reachability", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.archive).resolve()
    aggregate = json.loads((root / "results/aggregate/aggregate_ranking.json").read_text(encoding="utf-8"))
    part_reports = {
        part: json.loads((root / f"results/{part}/evaluation/evaluation.json").read_text(encoding="utf-8"))
        for part in ("gearboxopen", "gearboxclosed")
    }

    reach_rows = []
    for path in sorted(Path(args.reachability).glob("reach_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        reach_rows.append(
            (
                path.name,
                value.get("radius_m"),
                value.get("camera_offset_m"),
                value.get("ik_successes"),
                value.get("poses"),
                value.get("ik_success_fraction"),
            )
        )

    lines = [
        "# FANUC orbit reverse-engineering benchmark — final report",
        "",
        "## Outcome",
        "",
    ]
    valid = [row for row in aggregate["ranking"] if row.get("both_parts")]
    if valid:
        winner = valid[0]
        lines.append(
            f"The highest-scoring two-part method was **{winner['method']}**, with mean score "
            f"{fmt(winner['mean_ranking_score'])}, mean F-score@1 mm "
            f"{fmt(winner['mean_fscore_1p0mm'])}, and mean symmetric Chamfer distance "
            f"{fmt(winner['mean_chamfer_mm'])} mm."
        )
    lines += [
        "",
        "The strongest practical result is therefore geometric: fuse the calibrated upright orbit, "
        "flip or re-grasp the part, then fuse a second pass. This closes bottom-face occlusion and "
        "outperformed every single-pass, photometric, splatting, and neural-implicit candidate in this test.",
        "",
        "## Experiment contract",
        "",
        "- Parts: `gearboxopen.stl` and `gearboxclosed.stl`.",
        "- Capture: 90 upright plus 90 flipped RGB-D views per part; 960×720; 18 azimuths × five elevations.",
        "- Pose input: reported FANUC-FK/hand-eye poses with 0.8 mm / 0.2° perturbation.",
        "- Reconstruction inputs: masked RGB, noisy quantized depth, reported poses, declared flip transform, and method-specific simulated sensor observations.",
        "- Withheld from all reconstruction methods: clean depth, true camera poses, and source triangles.",
        "- Final evaluation: 120,000 sampled points per surface using bidirectional distance, F-scores, normal agreement, dimension error, bottom recall, runtime, and a one-point watertightness tie-breaker.",
        "- The supplied source STLs are non-watertight triangle assemblies, so watertightness is not treated as ground truth.",
        "",
        "## Aggregate ranking",
        "",
        "| Rank | Method | Both parts | Score | F@0.5 mm | F@1 mm | F@2 mm | Chamfer mm | Normal | Total runtime s |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["ranking"]:
        lines.append(
            f"| {row['rank']} | {row['method']} | {'yes' if row['both_parts'] else 'no'} | "
            f"{fmt(row['mean_ranking_score'])} | {fmt(row['mean_fscore_0p5mm'])} | "
            f"{fmt(row['mean_fscore_1p0mm'])} | {fmt(row['mean_fscore_2p0mm'])} | "
            f"{fmt(row['mean_chamfer_mm'])} | {fmt(row['mean_normal_consistency'])} | "
            f"{fmt(row['total_runtime_s'], 1)} |"
        )

    lines += ["", "## Per-part leaders", ""]
    for part, report in part_reports.items():
        lines += [
            f"### {part}",
            "",
            "| Rank | Method | Score | F@1 mm | F@2 mm | Chamfer mm | Runtime s |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in report["ranking"][:8]:
            lines.append(
                f"| {row['rank']} | {row['method']} | {fmt(row['ranking_score'])} | "
                f"{fmt(row['fscore_1p0mm'])} | {fmt(row['fscore_2p0mm'])} | "
                f"{fmt(row['symmetric_chamfer_mean_mm'])} | {fmt(row.get('runtime_s'), 1)} |"
            )
        lines.append("")

    lines += [
        "## Method prognosis",
        "",
        "1. **Dual-pass TSDF — best accuracy/completeness.** Use this when the robot may flip/re-grasp the part or when a human may reorient it once. It directly solves the unseen-bottom problem.",
        "2. **NBV TSDF — best fast single-pass choice.** It uses a small, coverage-oriented subset of views and is the recommended first real-cell orbit when cycle time and reachability matter.",
        "3. **FK TSDF / depth Poisson — robust deterministic fallbacks.** They are straightforward and stable; bounded ICP gave only a modest quality gain for a much larger runtime in this implementation.",
        "4. **2DGS — inconsistent for metric CAD.** It was usable on one state but collapsed on the other. Photorealistic view synthesis did not imply dimensional mesh accuracy.",
        "5. **Neural implicit methods — not reliable here.** The compact neural SDF scored below geometric fusion, NeuS2 failed observation-volume registration, and any official Neuralangelo failure is retained below rather than hidden.",
        "6. **Visual hull — a fallback when depth is absent.** It preserved coarse silhouettes but could not recover concavities or bottom geometry with millimetric accuracy.",
        "",
        "## ER-4iA reachability gate",
        "",
        "These numbers are inverse-kinematics feasibility only; they do not certify self-collision, table collision, singularity margin, cable routing, or camera visibility.",
        "",
        "| File | Orbit radius m | Camera offset m | IK success | Fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, radius, offset, successes, poses, fraction in reach_rows:
        lines.append(
            f"| {name} | {fmt(radius, 2)} | {fmt(offset, 2)} | {successes}/{poses} | {fmt(fraction)} |"
        )

    failures = aggregate.get("failures", [])
    lines += ["", "## Preserved failures", ""]
    if failures:
        lines += ["| Part | Method | Status | Error |", "|---|---|---|---|"]
        for row in failures:
            error = str(row.get("error", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row.get('part', '—')} | {row.get('method', '—')} | {row.get('status', '—')} | {error} |")
    else:
        lines.append("No evaluator failures were recorded.")

    lines += [
        "",
        "## Real-cell recommendation",
        "",
        "Start with a collision-checked reachable subset of the **NBV TSDF** orbit, not the ideal 180-view trajectory. "
        "Use the measured camera mount transform, validate hand-eye calibration against a fixed target, average depth at each stop, "
        "reject grazing-angle/flying-edge pixels, and inspect live coverage. For a printable watertight deliverable, add the second "
        "flip/re-grasp pass and fuse it as **dual-pass TSDF**. If flipping is impossible, explicitly cap only a verified planar bottom "
        "and label that face as inferred rather than scanned.",
        "",
        "Do not claim the synthetic ranking guarantees real-cell dimensional accuracy. The next validation must scan a measured artifact, "
        "compare it with caliper/CMM dimensions, and repeat from at least three randomized placements.",
        "",
        "## Artifact map",
        "",
        "- `datasets/`: observations, calibration, manifests, and evaluator-only truth kept structurally separate.",
        "- `results/<part>/reconstructions/`: normalized `.ply`, `.obj`, `.stl`, and per-method metadata.",
        "- `results/<part>/evaluation/`: final JSON and CSV metrics.",
        "- `results/aggregate/`: aggregate JSON/CSV, finalization record, and winner videos.",
        "- `official_results/`: raw official-method logs/checkpoints/meshes.",
        "- `input/`: exact scripts used to capture, reconstruct, evaluate, and render.",
    ]

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
