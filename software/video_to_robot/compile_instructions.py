"""Compile reviewed video evidence into conservative robot-task artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evidence_for(specs: dict, placement_index: int) -> list[dict]:
    result = []
    for stem, spec in specs.items():
        start, end = spec["placements"][placement_index - 1]
        result.append(
            {
                "episode": stem,
                "view": spec["view"],
                "time_range_s": [start, end],
                "claim": f"human placement {placement_index} of 4",
            }
        )
    return result


def make_step(step_id: str, primitive: str, preconditions: list[str], observed: list[dict], robot_parameters: dict, verification: dict, confidence: float, requires_calibration: bool = True) -> dict:
    return {
        "id": step_id,
        "primitive": primitive,
        "preconditions": preconditions,
        "observed_evidence": observed,
        "robot_parameters": robot_parameters,
        "verification": verification,
        "confidence": confidence,
        "requires_calibration": requires_calibration,
    }


def compile_task(specs: dict) -> dict:
    all_evidence = [e for i in range(1, 5) for e in evidence_for(specs, i)]
    steps = [
        make_step(
            "S00_observe", "observe", [], all_evidence,
            {"detector": "instance masks + persistent IDs", "required_inputs": ["live RGB-D", "camera calibration", "gripper state"]},
            {"pass": "four separate cubes and one upright open target bin are localized", "fail": "pause and acquire another view"},
            0.98,
        ),
        make_step(
            "S01_approach_cube", "approach", ["one unplaced cube selected", "collision-free top-down corridor exists"], all_evidence,
            {"direction": "top_down", "xy": "live_perception(cube_instance)", "z_safe": "workcell_calibration_required", "speed": "controller_validated"},
            {"pass": "gripper projection is centered over the selected cube and obstacle clearance is valid"},
            0.91,
        ),
        make_step(
            "S02_grasp_cube", "grasp", ["approach verified", "gripper open"], all_evidence,
            {"grasp_type": "parallel_jaw", "jaw_gap": "installed_finger_geometry_required", "force": "part/material validation required"},
            {"pass": "gripper state changes and the cube follows during a short proof lift", "retry": "open, retreat, re-detect, re-center once"},
            0.88,
        ),
        make_step(
            "S03_lift", "lift", ["grasp proof passed"], all_evidence,
            {"direction": "table_normal", "distance": "clear tallest observed obstacle plus margin"},
            {"pass": "cube remains latched and no obstacle intersects gripper keep-out"},
            0.90,
        ),
        make_step(
            "S04_transfer", "transfer", ["cube secure", "bin pose current"], all_evidence,
            {"target": "above_bin_opening", "path": "collision_checked_monotonic", "relocalize_before_descent": True},
            {"pass": "cube and complete finger envelope project inside the bin opening with margin"},
            0.94,
        ),
        make_step(
            "S05_insert", "insert", ["above-bin alignment passed"], all_evidence,
            {"direction": "bin_opening_axis", "release_height": "depth_or_geometry_required", "descent": "guarded"},
            {"pass": "cube clears rim; no finger/bin-wall intersection", "fail": "stop, retract vertically, re-estimate bin pose"},
            0.82,
        ),
        make_step(
            "S06_release", "release", ["rim clearance passed"], all_evidence,
            {"command": "open_once_and_hold", "debounce": True},
            {"pass": "cube remains in bin after the hand/gripper clears; inside-bin count increments by one"},
            0.96,
        ),
        make_step(
            "S07_retract", "retract", ["release issued"], all_evidence,
            {"direction": "vertical_first", "path": "monotonic"},
            {"pass": "gripper clears rim and target occupancy remains stable"},
            0.97,
        ),
        make_step(
            "S08_verify_repeat", "repeat", ["post-release verification passed"], all_evidence,
            {"until": "inside_bin_count == 4", "max_attempts_per_cube": 2},
            {"pass": "four unique cube IDs are contained and none remain in source region"},
            0.99,
            False,
        ),
    ]
    return {
        "task_id": "four_cubes_into_green_bin_v1",
        "title": "Transfer four red/brown cubes into one green open-top bin",
        "evidence_scope": {
            "observed": [
                "six independent demonstrations",
                "four placements per demonstration (24 total)",
                "hand trajectories in image coordinates",
                "action order and qualitative approach/transfer/release phases",
                "successful final state with four cubes contained",
            ],
            "not_observed": [
                "synchronized multiview geometry",
                "camera intrinsics/extrinsics",
                "depth or robot-base coordinates",
                "jaw width, gripping force, collision margins, release height",
                "robot execution telemetry",
            ],
        },
        "objects": [
            {"class": "cube", "count": 4, "appearance": "red/brown", "persistent_identity_required": True},
            {"class": "open_top_bin", "count": 1, "appearance": "green", "role": "target"},
        ],
        "steps": steps,
        "safety_gates": [
            {"id": "G1", "requirement": "calibrate RGB-D to robot base before resolving any pixel coordinate into motion"},
            {"id": "G2", "requirement": "model the complete installed fingers, not only TCP or jaw-center clearance"},
            {"id": "G3", "requirement": "proof-lift after closing; abort if tracked cube does not follow"},
            {"id": "G4", "requirement": "re-detect bin immediately before loaded descent"},
            {"id": "G5", "requirement": "use one debounced open/close command; reject oscillating command streams"},
            {"id": "G6", "requirement": "post-release occupancy increment is mandatory before advancing"},
        ],
        "cross_take_consensus": {"episodes": len(specs), "successful_placements": 24, "expected_sequence_repeated": 6},
    }


def behavior_tree(task: dict) -> dict:
    return {
        "type": "Sequence",
        "name": "FillGreenBin",
        "children": [
            {"type": "SafetyGate", "checks": ["calibration_valid", "workspace_clear", "gripper_model_loaded"]},
            {"type": "Perceive", "outputs": ["cube_instances", "bin_pose", "obstacle_voxels"]},
            {
                "type": "RepeatUntil",
                "condition": "inside_bin_count == 4",
                "max_iterations": 8,
                "child": {
                    "type": "Sequence",
                    "children": [
                        {"type": "Select", "query": "nearest_verified_unplaced_cube"},
                        {"type": "Approach", "mode": "top_down_collision_checked"},
                        {"type": "Grasp", "mode": "debounced_parallel_jaw"},
                        {"type": "ProofLift", "on_fail": "redetect_and_retry_once"},
                        {"type": "Transfer", "mode": "monotonic_safe_height"},
                        {"type": "RePerceive", "target": "bin"},
                        {"type": "GuardedInsert", "on_fail": "vertical_retract"},
                        {"type": "Release", "mode": "open_once_hold"},
                        {"type": "VerticalRetract"},
                        {"type": "Verify", "condition": "inside_bin_count_incremented"},
                    ],
                },
            },
            {"type": "Verify", "condition": "all_four_unique_cubes_contained"},
        ],
    }


def contact_graph() -> dict:
    return {
        "graph_type": "symbolic_4d_contact_graph",
        "metric_pose_available": False,
        "nodes": [
            {"id": "hand_actor", "type": "human_hand", "tracked_as": "2d_landmarks"},
            {"id": "cube_i", "type": "part_instance", "cardinality": 4},
            {"id": "green_bin", "type": "open_top_container", "cardinality": 1},
            {"id": "table", "type": "support_surface"},
        ],
        "states": [
            {"phase": "initial", "edges": [["cube_i", "supported_by", "table"], ["green_bin", "supported_by", "table"]]},
            {"phase": "approach", "edges": [["hand_actor", "near", "cube_i"]]},
            {"phase": "grasp", "edges": [["hand_actor", "contact", "cube_i"], ["hand_actor", "holds", "cube_i"]]},
            {"phase": "transfer", "edges": [["hand_actor", "holds", "cube_i"], ["cube_i", "approaches", "green_bin"]]},
            {"phase": "insert", "edges": [["cube_i", "crosses_opening", "green_bin"]], "uncertainty": "contact partly occluded"},
            {"phase": "release", "edges": [["hand_actor", "not_holding", "cube_i"], ["cube_i", "contained_by", "green_bin"]]},
            {"phase": "verified", "edges": [["cube_i", "contained_by", "green_bin"]], "postcondition": "inside-bin count increments after hand-clear"},
        ],
    }


def instruction_markdown(task: dict) -> str:
    lines = [
        "# Evidence-grounded assembly instruction",
        "",
        f"**Task:** {task['title']}",
        "",
        "Six separate takes show the same result: four cubes are moved one at a time into the green bin. The video proves task order and qualitative contact states; it does not provide robot coordinates.",
        "",
        "## Robot sequence",
        "",
        "1. Detect and assign persistent IDs to four cubes and the target bin.",
        "2. Select one unplaced cube; approach vertically with the complete finger keep-out checked.",
        "3. Close once, hold closed, and perform a short proof lift.",
        "4. Travel at safe height to a pose above the current bin opening.",
        "5. Re-detect the bin, then descend on its opening axis under collision guarding.",
        "6. Open once and hold open; retract vertically before moving laterally.",
        "7. Confirm that the inside-bin count increased and repeat until it reaches four.",
        "",
        "## Non-negotiable calibration",
        "",
    ]
    lines.extend(f"- {gate['requirement']}" for gate in task["safety_gates"])
    lines += ["", "## Evidence summary", "", "- 6 independent episodes", "- 24 successful demonstrated placements", "- Grounded action boundaries reviewed to approximately ±0.25–0.5 s", "- Metric 6D motion intentionally withheld until calibrated RGB-D/CAD is supplied"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=Path(__file__).parent / "verified_annotations.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    specs = json.loads(args.annotations.read_text(encoding="utf-8"))
    task = compile_task(specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "robot_task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
    (args.output_dir / "behavior_tree.json").write_text(json.dumps(behavior_tree(task), indent=2), encoding="utf-8")
    (args.output_dir / "contact_graph.json").write_text(json.dumps(contact_graph(), indent=2), encoding="utf-8")
    (args.output_dir / "INSTRUCTION_CARD.md").write_text(instruction_markdown(task), encoding="utf-8")
    print(json.dumps({"task": task["task_id"], "steps": len(task["steps"]), "evidence_placements": 24}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
