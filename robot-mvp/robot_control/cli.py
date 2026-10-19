"""Human-side approval and inspection CLI for the A-Team robot MVP."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .store import PlanStore


PHYSICAL_CONFIRMATION = "I AM AT THE CELL WITH THE OPERATOR AND E-STOP READY"


def print_json(value) -> None:
    print(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="list recent plans")
    demo = sub.add_parser("draft-demo", help="create the standard two-point demo")
    demo.add_argument("--label", default="A-Team safe point-to-point demo")

    approve = sub.add_parser("approve", help="human approval gate")
    approve.add_argument("plan_id")
    approve.add_argument("--mode", choices=("mock", "physical"), required=True)
    approve.add_argument("--operator", required=True)
    approve.add_argument("--confirm", help="required exact phrase for physical mode")

    cancel = sub.add_parser("cancel", help="cancel a pending/approved plan")
    cancel.add_argument("plan_id")

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
    if args.command == "cancel":
        print_json(store.cancel(args.plan_id))
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

