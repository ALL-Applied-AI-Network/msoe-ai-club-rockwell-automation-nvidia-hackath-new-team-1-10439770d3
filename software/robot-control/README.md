# Robot control package

This package is the human-gated control layer used by the A-Team prototype. It combines RGB/RGB-D perception, phase-aware motion, safety envelopes, plan storage, MCP drafting tools, and adapters for the approved SIDC client.

## Safety model

1. An AI or user drafts a bounded plan.
2. The plan is saved as pending.
3. A nearby operator reviews and separately approves it.
4. The policy adapter consumes only that approved plan through the official interface.
5. The workcell controller, limits, and operator remain authoritative.

There is intentionally no MCP tool that approves or executes physical motion. Mock mode is the default.

## Package layout

- `robot_control/` — perception, phase controller, safety, plan store, CLI, and MCP server
- `policy.py` — bounded point-to-point official-client adapter
- `policy_adaptive_candidate.py` — adaptive perception/controller candidate
- `simulation/` — point-to-point and wall-grasp simulation helpers
- `tests/` — 39 hardware-free regression tests
- `evidence/` — compact commissioning and point-to-point reports
- `scripts/` — simulation, mock-client, and DGX setup helpers

## Install and test

From the repository root with Python 3.12+:

```bash
python -m pip install -e software/robot-control pytest
python -m pytest software/robot-control/tests -q
```

## MCP drafting surface

- `robot_status` — current envelope and recent plans
- `draft_point_to_point_move` — validate and save a two-point plan
- `draft_safe_demo` — save the standard bounded demo
- `cancel_unexecuted_plan` — cancel a pending or approved plan

## Verified milestones

- The original host suite and the package's 39 tests passed.
- Isaac Sim reproduced the two-waypoint controller-frame conversion with sub-millimeter TCP error in the recorded run.
- The first supervised physical rollout moved the FANUC from home to waypoint A and exposed a too-strict waypoint tolerance, which was corrected and covered by regression testing.
- Later supervised trials informed the RGB-D, proof-lift, command-debouncing, packing, and recovery logic documented in [Perception and control](../../docs/perception-and-control.md).

Compact reports:

- [`point_to_point_report_2026-10-19.json`](evidence/point_to_point_report_2026-10-19.json)
- [`physical_commissioning_2026-10-20.json`](evidence/physical_commissioning_2026-10-20.json)
- [First physical run report](../../docs/experiments/first-physical-run.md)

## Physical-operation warning

Do not copy recorded targets to another installation. Physical execution requires an approved session, an on-site operator with E-stop access, verified TCP/work/tool frames, current obstacle review, and the site-specific confirmation process. Never provide a private session token to an AI tool or connect directly to an undocumented controller endpoint.

The public release intentionally omits the facility request client and physical-launch scripts. The retained policy source has no controller transport of its own; recorded targets remain only as nonportable experiment evidence and test fixtures.

