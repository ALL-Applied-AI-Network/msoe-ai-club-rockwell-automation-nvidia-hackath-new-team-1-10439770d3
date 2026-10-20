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

Verified on October 19, 2026:

- the official SIDC client completed one mock rollout with 29 actions and zero
  rejected actions, ending with `Ready for the real robot`;
- the Dataset 3 host suite passed all 33 tests;
- Dataset 3's calibrated ER-4iA inverse kinematics converged for both demo
  endpoints after applying the documented 330 mm UF0/base-frame offset;
- the MVP's five safety, state-machine, and live stdio-MCP tests passed.

Updated for the physical interface opened on October 20, 2026: the policy now
interpolates from the observed physical TCP at 10 Hz, limits each row to 2 mm of
translation and 1 degree of rotation, and requires one separately approved
local plan per rollout. Seven local tests cover the resulting control path.

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

After the captain creates the YAAD team and configures an SSH alias such as
`yaad-ateam`, upload a clean Dataset 3 source archive and the two deployment
files. The SSH account's home directory is the persistent YAAD workspace:

```powershell
tar.exe -czf fanuc_er4ia_sim_source.tar.gz `
  --exclude=fanuc_er4ia_sim/.venv `
  --exclude=fanuc_er4ia_sim/runs `
  -C "E:\Hackathon data\Datasets\fanuc_er4ia_hackathon_starter_v05" `
  fanuc_er4ia_sim
scp fanuc_er4ia_sim_source.tar.gz yaad-ateam:
scp .\scripts\setup_dgx_spark.sh yaad-ateam:
ssh yaad-ateam "tar -xzf fanuc_er4ia_sim_source.tar.gz"
scp .\simulation\isaac_point_to_point.py yaad-ateam:fanuc_er4ia_sim/
ssh yaad-ateam "chmod +x setup_dgx_spark.sh && bash setup_dgx_spark.sh"
```

The script installs host Python when needed, runs Dataset 3's 33 host tests,
pulls Isaac Sim 6.0, runs a headless application smoke test, builds the
calibrated USD stage, executes the two-waypoint simulation and rejects the run
unless the measured report passes. Persistent artifacts are under
`~/fanuc_er4ia_sim/runs/`. Do not interpret a simulator target as a physical
target until the organizer verifies the tool frame, camera calibration, work
surface, installed fingers and collision envelope on the real cell.

### Verified DGX Spark result

On October 19, 2026, YAAD job `9e4fd737-6a10-40a6-8613-0fd046cdc394`
successfully launched `nvcr.io/nvidia/isaac-sim:6.0.0` on the team's NVIDIA GB10.
Isaac Sim reached `app ready`, advanced a frame, printed
`ATEAM_ISAAC_SIM_6_SMOKE_OK`, and exited with code 0. The runtime reported an
ARM64 host and 120 GiB of unified GPU memory.

The same day, the interactive A-Team workspace reproduced the setup end to end:

- all 33 Dataset 3 host tests passed on the ARM64 Spark;
- the scene exported to `runs/point_to_point_scene.usd` (7.3 MiB; SHA-256
  `99a76b7dfa62c82846657b74b5d9cdfeec857c2a15562952815e5efbdb2cb470`);
- the simulated open-gripper TCP moved from controller target
  `[450, -70, -230]` to `[450, 70, -230]` mm after the documented UF0/base
  conversion;
- measured TCP errors were 0.231 mm and 0.220 mm; maximum joint tracking error
  was 0.0212 degrees; both waypoints were verified;
- [`evidence/point_to_point_report_2026-10-19.json`](evidence/point_to_point_report_2026-10-19.json)
  preserves the exact measured values (repository-file SHA-256
  `7944fb6fa6d5b1cc223566b047084085b04bbdad84cb850199500c3de86ea9fe`);
  the raw Spark artifact SHA-256 is
  `0ac4cd8eaf6aeceb0df333a1a4c293e7ddc3fde4be99e53f4fb8cc45e7151588`.

This proves the bounded motion in the supplied digital twin. It does not prove
collision-free transfer to the physical cell or authorize unattended motion.

## First physical rollout

The authorized next step is a supervised run through the official SIDC client,
not a direct robot-controller connection. Follow
[`../docs/PHYSICAL_ROLLOUT_RUNBOOK.md`](../docs/PHYSICAL_ROLLOUT_RUNBOOK.md).
The included preparation script installs a three-rollout commissioning request
without submitting it; the physical-slot script stays locked until the SIDC
request is approved and a nearby human confirms the operator and E-stop are
ready.

The reusable job definition is [`jobs/isaac_sim_6_smoke.json`](jobs/isaac_sim_6_smoke.json).
[`scripts/yaad_call.py`](scripts/yaad_call.py) provides a token-safe MCP client
for setup sessions: it reads `YAAD_TOKEN` from the environment and never prints
the credential.
