"""Human-side approval and inspection CLI for the A-Team robot MVP."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .store import PlanStore


PHYSICAL_CONFIRMATION = "I AM AT THE CELL WITH THE OPERATOR AND E-STOP READY"


def one_wall_bin_waypoints() -> list[dict]:
    """Nominal human-review preview; live x/y/yaw come from table vision."""
    x, y = 450.0, -21.25
    approach_z, grip_z, lifted_z = -235.0, -294.0, -284.0
    return [
        {"x_mm": x, "y_mm": y, "z_mm": approach_z, "yaw_deg": 0, "gripper": 0},
        {"x_mm": x, "y_mm": y, "z_mm": grip_z, "yaw_deg": 0, "gripper": 0},
        {"x_mm": x, "y_mm": y, "z_mm": grip_z, "yaw_deg": 0, "gripper": 1},
        {"x_mm": x, "y_mm": y, "z_mm": lifted_z, "yaw_deg": 0, "gripper": 1},
        {"x_mm": x, "y_mm": y, "z_mm": grip_z, "yaw_deg": 0, "gripper": 1},
        {"x_mm": x, "y_mm": y, "z_mm": grip_z, "yaw_deg": 0, "gripper": 0},
        {"x_mm": x, "y_mm": y, "z_mm": approach_z, "yaw_deg": 0, "gripper": 0},
    ]


def print_json(value) -> None:
    print(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="list recent plans")
    demo = sub.add_parser("draft-demo", help="create the standard two-point demo")
    demo.add_argument("--label", default="A-Team safe point-to-point demo")
    bin_grip = sub.add_parser(
        "draft-one-wall-bin",
        help="create camera-guided empty-bin one-wall micro-lift plan previews",
    )
    bin_grip.add_argument("--count", type=int, default=3)
    bin_grip.add_argument("--label", default="A-Team vision one-wall bin micro-lift")

    approve = sub.add_parser("approve", help="human approval gate")
    approve.add_argument("plan_id")
    approve.add_argument("--mode", choices=("mock", "physical"), required=True)
    approve.add_argument("--operator", required=True)
    approve.add_argument("--confirm", help="required exact phrase for physical mode")

    cancel = sub.add_parser("cancel", help="cancel a pending/approved plan")
    cancel.add_argument("plan_id")

    interrupt = sub.add_parser("interrupt", help="close an executing plan without marking it completed")
    interrupt.add_argument("plan_id")
    interrupt.add_argument("--note", default="official rollout ended before local plan completion")

    args = parser.parse_args()
    store = PlanStore()
    if args.command == "status":
        print_json(store.list())
        return
    if args.command == "draft-demo":
        print_json(
            store.create(
                args.label,
                [
                    {"x_mm": 450, "y_mm": -70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
                    {"x_mm": 450, "y_mm": 70, "z_mm": -230, "yaw_deg": 0, "gripper": 0},
                ],
            )
        )
        return
    if args.command == "draft-one-wall-bin":
        if not 1 <= args.count <= 10:
            sys.exit("--count must be between 1 and 10")
        print_json(
            [
                store.create(f"{args.label} rollout {index}", one_wall_bin_waypoints())
                for index in range(1, args.count + 1)
            ]
        )
        return
    if args.command == "cancel":
        print_json(store.cancel(args.plan_id))
        return
    if args.command == "interrupt":
        print_json(store.interrupt(args.plan_id, args.note))
        return
    if args.command == "approve":
        if args.mode == "physical":
            if os.environ.get("ATEAM_ALLOW_PHYSICAL") != "YES":
                sys.exit("physical mode is locked; set ATEAM_ALLOW_PHYSICAL=YES only during an approved lab slot")
            if args.confirm != PHYSICAL_CONFIRMATION:
                sys.exit(f'physical approval requires --confirm "{PHYSICAL_CONFIRMATION}"')
        print_json(store.approve(args.plan_id, args.mode, args.operator))


if __name__ == "__main__":
    main()
