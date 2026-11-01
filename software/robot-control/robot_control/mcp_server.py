"""Read/plan-only MCP surface for the A-Team robot MVP.

No MCP tool approves or physically executes motion. A nearby human must use the
separate CLI approval command after reviewing the waypoints.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .safety import ENVELOPE, SafetyError
from .store import PlanStore


mcp = FastMCP("A-Team Robot Planner")


def _summary(plan: dict) -> dict:
    return {
        "id": plan["id"],
        "label": plan["label"],
        "status": plan["status"],
        "execution_mode": plan["execution_mode"],
        "waypoints": plan["waypoints"],
        "approved_by": plan["approved_by"],
        "created_at": plan["created_at"],
        "note": plan["note"],
    }


@mcp.tool()
def robot_status() -> dict:
    """Show the conservative envelope and recent plans. This never moves the robot."""
    return {
        "physical_execution_enabled": False,
        "safety_envelope": {
            "x_mm": ENVELOPE.x_mm,
            "y_mm": ENVELOPE.y_mm,
            "z_mm": ENVELOPE.z_mm,
            "yaw_deg": ENVELOPE.yaw_deg,
            "max_segment_mm": ENVELOPE.max_segment_mm,
        },
        "recent_plans": [_summary(plan) for plan in PlanStore().list(10)],
        "next_step": "A nearby human reviews and approves a pending plan with the local ateam-robot CLI.",
    }


@mcp.tool()
def draft_point_to_point_move(
    start_x_mm: float,
    start_y_mm: float,
    start_z_mm: float,
    end_x_mm: float,
    end_y_mm: float,
    end_z_mm: float,
    yaw_deg: float = 0.0,
    label: str = "MCP point-to-point move",
) -> dict:
    """Validate and save a two-waypoint TCP plan for human review; do not execute it."""
    try:
        plan = PlanStore().create(
            label,
            [
                {"x_mm": start_x_mm, "y_mm": start_y_mm, "z_mm": start_z_mm, "yaw_deg": yaw_deg, "gripper": 0},
                {"x_mm": end_x_mm, "y_mm": end_y_mm, "z_mm": end_z_mm, "yaw_deg": yaw_deg, "gripper": 0},
            ],
        )
    except SafetyError as exc:
        return {"accepted": False, "error": str(exc)}
    return {
        "accepted": True,
        "plan": _summary(plan),
        "approval_required": True,
        "approval_command": f"ateam-robot approve {plan['id']} --mode mock --operator YOUR_NAME",
    }


@mcp.tool()
def draft_safe_demo() -> dict:
    """Save the standard open-gripper left-to-right demo for human review; do not execute it."""
    plan = PlanStore().create(
        "A-Team safe point-to-point demo",
        [
            {"x_mm": 450, "y_mm": -70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
            {"x_mm": 450, "y_mm": 70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
        ],
    )
    return {
        "accepted": True,
        "plan": _summary(plan),
        "approval_required": True,
        "approval_command": f"ateam-robot approve {plan['id']} --mode mock --operator YOUR_NAME",
    }


@mcp.tool()
def cancel_unexecuted_plan(plan_id: str) -> dict:
    """Cancel a pending or approved plan. Executing/completed plans cannot be changed."""
    try:
        return {"cancelled": True, "plan": _summary(PlanStore().cancel(plan_id))}
    except (KeyError, ValueError) as exc:
        return {"cancelled": False, "error": str(exc)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

