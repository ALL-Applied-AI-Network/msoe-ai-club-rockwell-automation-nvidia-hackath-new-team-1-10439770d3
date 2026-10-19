# A-Team Robot MVP

This directory provides the smallest defensible path from an AI request to a
FANUC ER-4iA point-to-point action:

1. Codex drafts a bounded TCP plan through the local `ateam-robot` MCP server.
2. The plan is stored as **pending**. MCP has no approval or execution tool.
3. A nearby person reviews the coordinates and approves them with the local CLI.
4. The drop-in `policy.py` consumes the approved plan through the official SIDC
   client. The cell still enforces its own limits and an operator supervises the
   physical rollout.

The default execution mode is `mock`. Physical mode is deliberately locked.

## What is installed locally

Run from PowerShell:

```powershell
cd "E:\Hackathon data\Hackathon main event\official-team-repository\robot-mvp"
.\scripts\install_local.ps1
.\scripts\run_mock_demo.ps1
```

The installer creates a Python 3.12 environment, runs the tests, copies the
drop-in policy into the downloaded official client, and registers a local
stdio MCP server named `ateam-robot` with Codex.

The mock demo sends two valid open-gripper TCP targets:

- A: `[450, -70, -230, -180, 0, 0, 0]`
- B: `[450,  70, -230, -180, 0, 0, 0]`

Units are millimetres/degrees in controller UF0. This only proves interface and
limit compatibility; the mock contains no kinematics or collision physics.

## MCP tools

- `robot_status`: envelope and recent plans.
- `draft_point_to_point_move`: validate and save a two-point plan.
- `draft_safe_demo`: save the standard A-to-B plan.
- `cancel_unexecuted_plan`: cancel a pending/approved plan.

There is intentionally no `execute` or `approve` MCP tool.

## Human approval

Approve a plan for the mock only:

```powershell
.\.venv\Scripts\ateam-robot.exe approve PLAN_ID --mode mock --operator "Your Name"
```

Physical execution additionally requires all of the following:

- an approved SIDC Lab submission and scheduled operator slot;
- an on-site person at the cell with the emergency stop accessible;
- coordinates rechecked against the current tool frame and obstacles;
- `ATEAM_ALLOW_PHYSICAL=YES` and `ATEAM_EXECUTION_MODE=physical` set locally;
- the exact physical confirmation phrase required by `ateam-robot --help`.

Never pass a private run token to an AI tool, commit `.request.json`, or connect
directly to an undocumented robot-controller endpoint.

## Isaac Sim 6.0

Dataset 3 was validated on Linux, Isaac Sim 6.0, and an RTX A6000. The local RTX
3060 Laptop GPU has 6 GB VRAM, below NVIDIA's current 16 GB minimum; the E: drive
also has less than the 50 GB minimum free space. Use the provisioned DGX Spark or
Brev GPU instead.

After the captain creates the YAAD team and supplies the displayed SSH login:

```powershell
scp -r "E:\Hackathon data\Datasets\fanuc_er4ia_hackathon_starter_v05\fanuc_er4ia_sim" YOUR_SSH_LOGIN:/workspace/
scp ".\scripts\setup_dgx_spark.sh" YOUR_SSH_LOGIN:/workspace/
ssh YOUR_SSH_LOGIN "bash /workspace/setup_dgx_spark.sh"
```

The script pulls Isaac Sim, runs a headless application smoke test, and—when the
starter is present—builds Dataset 3's USD stage at
`/workspace/fanuc_er4ia_sim/runs/point_to_point_scene.usd`. Do not interpret a
sim target as a physical target until the organizer verifies the tool frame,
camera calibration, work surface, and installed fingers on the real cell.
