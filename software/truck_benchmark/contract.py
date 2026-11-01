"""Static and runtime checks for the no-simulator-truth policy boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path


FORBIDDEN_IMPORT_PREFIXES = (
    "omni",
    "pxr",
    "isaacsim",
    "carb",
    "warp",
)

FORBIDDEN_SYMBOLS = {
    "get_world_pose",
    "get_world_poses",
    "get_local_pose",
    "get_local_poses",
    "get_prim_at_path",
    "GetPrimAtPath",
    "SingleXFormPrim",
    "RigidPrim",
    "XFormPrim",
    "ground_truth",
    "scene_state",
    "evaluator",
}


def audit_policy_source(path: str | Path) -> list[str]:
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    failures.append(f"forbidden import {alias.name} at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                failures.append(f"forbidden import {module} at line {node.lineno}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_SYMBOLS:
            failures.append(f"forbidden symbol {node.id} at line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_SYMBOLS:
            failures.append(f"forbidden attribute {node.attr} at line {node.lineno}")
    return sorted(set(failures))


def assert_policy_source_clean(path: str | Path) -> None:
    failures = audit_policy_source(path)
    if failures:
        raise RuntimeError("Policy boundary violation:\n" + "\n".join(failures))


def validate_observation_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(value).lower()
    for forbidden in ("ground_truth", "scene_state", "prim_path", "world_pose"):
        if forbidden in text:
            raise RuntimeError(f"Observation manifest leaks forbidden field: {forbidden}")
    required = {"scene_id", "cameras", "robot_state", "variation"}
    missing = required - set(value)
    if missing:
        raise RuntimeError(f"Observation manifest missing keys: {sorted(missing)}")
    return value

