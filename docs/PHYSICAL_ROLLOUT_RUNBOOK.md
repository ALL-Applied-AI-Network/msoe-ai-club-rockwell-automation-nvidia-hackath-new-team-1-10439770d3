# The A-Team — First Physical Robot Rollout

## Outcome

The software path is ready for a first supervised commissioning motion. The
remaining gate is operational: an approved SIDC submission, a 30-minute testbed
slot, and a nearby cell operator who verifies the real tool frame, installed
fingers, scene clearance, and emergency-stop access.

Do **not** connect directly to a FANUC controller address. The official SIDC
client is the authorized control path:

```text
A-Team policy on laptop
        ↓ act(obs)
Official run.py / cell_client.py
        ↓ approved authenticated run
SIDC cell server and safety checks
        ↓ operator starts each rollout
FANUC ER-4iA
```

The policy runs on the team's laptop. For every step, the cell sends camera and
robot observations; the policy returns a TCP target. The cell validates and
executes it. The local `ateam-robot` database adds a separate human approval
gate before the policy can claim a physical plan.

## What is already complete

- Official client installed with its Python environment.
- A-Team `policy.py` copied into the client.
- Official mock passed: 1 rollout, 29 accepted actions, 0 rejected.
- Two endpoints solved and executed in the Dataset 3 Isaac Sim twin.
- Simulated TCP error: 0.231 mm at A and 0.220 mm at B.
- Physical mode defaults to locked.

## The first motion

The commissioning path keeps the current fingers open and moves laterally
between these controller-UF0 targets:

```text
A = [450, -70, -230, -180, 0, 0, 0]
B = [450,  70, -230, -180, 0, 0, 0]
```

The internal project envelope is smaller than the public cell limits. The path
is 140 mm long and remains approximately 105 mm above the documented work
surface. That does not replace a physical clearance check: the operator must
confirm the active frame, actual TCP, installed fingers, bin, screws, and route
from the home pose.

## Step 1 — prepare locally

From the repository's `robot-mvp` directory:

```powershell
.\scripts\prepare_first_motion.ps1 `
  -ContactName "Abdallah Benelmadjat" `
  -ContactEmail "YOUR_UWM_EMAIL"
```

This performs no submission and contacts no robot. It:

- backs up the official client's current submission files;
- installs the first-motion submission and placement templates; and
- refreshes the official client's `policy.py` and its small local
  `robot_control` dependency.

Review the generated YAML with the mentor/operator. It requests three
commissioning rollouts, no custom hardware, and the documented default scene.

## Step 2 — request approval

Only after the organizer confirms that this is the desired first test:

```powershell
cd "E:\Hackathon data\Hackathon main event\site_crawl\sidclab_2026-10-19\PhysicalAI-Hackathon-2026"
uv run run.py --submit
```

The command prints a request ID and writes `.request.json`. The file contains a
request secret, is already ignored by the official client's Git configuration,
and must not be copied into this repository or an AI prompt.

Open the SIDC website's **Calendar** tab, enter the printed request ID, and pick
an available 30-minute Central Time slot. The organizers approve the request for
that slot. If you also send a coordination note, use:

> The A-Team requests a supervised 30-minute commissioning slot for request
> `<REQUEST_ID>`. This first run uses no custom hardware and keeps the gripper
> open while moving between TCP points A `[450,-70,-230,-180,0,0,0]` and B
> `[450,70,-230,-180,0,0,0]` in controller UF0. Please confirm the active tool
> frame, installed-finger clearance, route from home, and clearance above the
> default screw/bin scene before approval. We will be present with the assigned
> operator and E-stop accessible.

Check the request without exposing its private file:

```powershell
uv run run.py --status
```

## Step 3 — run only during the approved slot

Be physically present with the assigned operator. The operator must home the
arm, verify the current setup, and retain the stop controls. Then run:

```powershell
cd "E:\Hackathon data\Hackathon main event\official-team-repository\robot-mvp"
.\scripts\run_physical_slot.ps1 -Operator "Actual Nearby Operator Name"
```

The script refuses to arm unless:

- the official private request file exists;
- `run.py --status` reports `approved`;
- a human types the exact physical confirmation phrase; and
- the local plan passes the conservative envelope checks.

The operator argument is part of the audit trail. The launcher rejects the
literal placeholder `OPERATOR NAME`; enter the actual nearby operator's name.

The launcher creates and separately approves three identical local plans, one
for each declared rollout. It converts the two high-level waypoints into 10 Hz
TCP rows limited to 2 mm of translation and 1 degree of rotation per row
(nominally 20 mm/s and 10 degrees/s) from the observed physical pose. It refuses
physical motion if the cell does not provide `tcp_pose`.

It enables physical mode only for the lifetime of the official client and
removes the physical-mode environment variables afterward, including after an
error. Any plan claimed by a rollout that ends before the final waypoint is
recorded as `interrupted`, while unused plans are cancelled.

The October 20 commissioning run showed that the real cell can settle roughly
2.54 mm from a requested point. The commissioning policy therefore uses a 5 mm
waypoint-completion gate while retaining the 2 mm per-row motion limit. That
5 mm gate is for this open-gripper, non-contact path—not a general grasping
accuracy specification.

For each rollout, the SIDC operator—not the script—homes the arm, prepares the
scene, and starts the motion. The operator can stop the rollout or handle a
controller fault at any point.

## Step 4 — preserve the evidence

After the run:

```powershell
cd "E:\Hackathon data\Hackathon main event\site_crawl\sidclab_2026-10-19\PhysicalAI-Hackathon-2026"
uv run run.py --status
uv run run.py --download
```

Record:

- request and run IDs, without committing their secrets;
- policy Git commit and local plan ID;
- installed tool/finger configuration;
- physical frame/TCP confirmation;
- success, failure, fault, or intervention for each rollout;
- camera videos and robot telemetry;
- measured final TCP error and cycle time; and
- operator observations.

## What follows after first motion

Once this connection and frame are verified, replace the fixed motion with one
bounded perception task:

1. use table/wrist observations;
2. identify one approved object;
3. compute a pre-grasp above it;
4. re-observe before descent;
5. grasp, lift, and verify; and
6. retry once or request help.

Only after that baseline is repeatable should the team add generated fingers or
demonstration-to-task-graph reasoning.
