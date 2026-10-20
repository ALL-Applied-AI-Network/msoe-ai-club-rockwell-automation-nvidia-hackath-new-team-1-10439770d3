# First Supervised Physical FANUC Run

## Outcome

On October 20, 2026, The A-Team completed its first approved, operator-supervised
physical commissioning run through the official SIDC Lab interface. The real
FANUC ER-4iA accepted the A-Team policy's TCP commands and moved from its home
pose toward waypoint A using conservative 10 Hz interpolation.

This is successful end-to-end physical-control commissioning, but it is not yet
a complete A→B demonstration. In both recorded episodes, the arm came within
approximately 2.54 mm of A. The policy required 2.00 mm or less before moving to
B, so it continued holding A until the operator ended the rollout.

The official results record scored both recorded rollouts as successful: 2/2,
100% observed success, zero faults and zero interventions. The overall run is
still marked incomplete because one of three planned trials was cancelled. The
official success labels and the narrower A→B engineering acceptance criterion
are both recorded; one must not be substituted for the other.

## Configuration

| Item | Value |
|---|---|
| Robot | FANUC ER-4iA |
| Interface | Official SIDC participant client and supervised cell operator |
| Action type | TCP pose in controller UF0 |
| Planned A | `[450, -70, -230, -180, 0, 0, 0]` |
| Planned B | `[450, 70, -230, -180, 0, 0, 0]` |
| Gripper | Open for the entire path |
| Command rate | 10 Hz |
| Translation limit | 2 mm per action row, or 20 mm/s nominal |
| Rotation limit | 1° per action row, or 10°/s nominal |
| Scene | Official default screws and yellow bin scene |
| Supervision | Nearby operator, homing workflow and cell stop controls |

The private results URL, view key, request secret and execution token are not in
this repository.

The local plan rows recorded the literal placeholder `OPERATOR NAME` in the
`approved_by` field. That field is therefore not valid evidence of the person's
identity. The official supervised cell workflow and user confirmation establish
that an operator was present, but the next run must record the operator's actual
name; the launcher now rejects this placeholder.

## Official and recorded run structure

The approved run exposed three official trial identifiers. Trial `t00` was
cancelled; `t01` and `t02` finished and were scored successful by the operator. The
downloaded LeRobotDataset therefore contains two episodes.

| Official metric | Result |
|---|---:|
| Successes | 2 / 2 recorded rollouts |
| Observed success rate | 100% |
| Faults | 0 |
| Interventions | 0 |
| Mean cycle time | 33.445 s |
| Mean policy latency | 84.15 ms |
| Run completion | Incomplete: 2 / 3 planned rollouts recorded |

| Episode | Frames | Duration | Closest to A | First within 5 mm | Closest to B | Final TCP XYZ |
|---:|---:|---:|---:|---:|---:|---|
| 0 / `t01` | 517 | 51.6 s | 2.531 mm | 12.0 s | 131.863 mm | `[450.0, -70.0, -234.58]` |
| 1 / `t02` | 153 | 15.2 s | 2.540 mm | 12.0 s | 131.863 mm | `[450.0, -70.0, -234.20]` |

Neither episode entered the 2 mm completion radius. Both entered a 3 mm radius
at about 12.1–12.2 seconds. The gripper-closed channel remained zero in every
recorded frame.

## Additional interface intelligence from the cell conversation

The machine transcript captured during the visit is imperfect and speaker
labels are not independently verified, so these points are paraphrased rather
than quoted:

- organizers described the platform as supporting repeated remote policy tests,
  with in-person testing especially useful while debugging networking and
  policy behavior;
- the three RGB views are recorded and downloadable for each rollout;
- real-time physical-AI policy control is supported through the official
  observation/action loop;
- data-collection sequences and scheduled policy-evaluation slots are related
  but distinct workflows;
- calibrated geometry is represented in the supplied simulation environment;
  and
- table or wrist depth can be requested through the policy API by adding the
  appropriate depth field, rather than requiring a separate controller path.

These points agree with the published participant FAQ. Exact quotations or
speaker attribution should be checked against the original recording before
use in a presentation.

## What the evidence establishes

- The laptop client, approved run token, cell observation loop and TCP-action
  path function together on the real robot.
- The policy's interpolated commands produced a controlled home→A motion rather
  than a single large target jump.
- Table, side and wrist recordings show the tool moving above the table without
  contacting the supplied objects.
- The official recorder captured 670 total frames at 10 Hz, synchronized with
  J1–J6, gripper state, TCP pose and commanded action.
- The two episodes reproduced the same approximately 2.54 mm closest approach,
  giving us a concrete physical tolerance/calibration observation rather than a
  simulation assumption.

## What it does not establish

- The A→B leg did not execute.
- No object was perceived, grasped, sorted, stacked or assembled.
- No custom fingers or automatic tool-change hardware were installed.
- The run does not authorize direct controller access, unattended motion or an
  AI bypass around the official operator workflow.
- Official success was judged at the rollout level. It does not prove that the
  intended two-waypoint engineering acceptance criterion was met; telemetry
  shows B was not reached.

## Interpreting the warnings

The console printed `POST /actions: no trial is running (robot homing)` after
each operator-ended episode. The dataset marks both episode end reasons as
`operator`, and the public run endpoint marks the trials `done`, not controller
fault. The warning is therefore consistent with a final policy action arriving
while the cell had already transitioned into homing. It should not be presented
as proof of a robot fault.

## Corrections made after the run

1. The non-contact commissioning completion tolerance was raised from 2 mm to
   5 mm. The per-action motion limits remain unchanged.
2. A regression test reproduces the measured 2.54 mm residual and verifies that
   the policy advances to B under the corrected threshold.
3. The physical launcher now redacts the private results link in console logs.
4. The launcher refuses a placeholder operator name.
5. If an official rollout ends before a local plan reaches its final waypoint,
   the plan is recorded as `interrupted`, not left indefinitely `executing` and
   not mislabeled `completed`.

## Next physical experiment

Repeat the same supervised non-contact path with the corrected policy before
adding perception or grasping. The acceptance criteria should be:

- both A and B are reached during every recorded episode;
- each waypoint is reached within the declared 5 mm commissioning tolerance;
- the gripper remains open;
- no intervention or controller fault occurs;
- local plan state ends as `completed`; and
- the run records all three camera streams and telemetry.

Only after that repeat should the team proceed to a bounded perception-guided
pick or a custom-finger experiment.

## Related evidence

- [`../robot-mvp/evidence/physical_commissioning_2026-10-20.json`](../robot-mvp/evidence/physical_commissioning_2026-10-20.json)
- [`PHYSICAL_ROLLOUT_RUNBOOK.md`](PHYSICAL_ROLLOUT_RUNBOOK.md)
- [`../robot-mvp/policy.py`](../robot-mvp/policy.py)

The raw run dataset remains in the machine-local LeRobot cache. It is excluded
from Git because it contains large recordings and team-private evaluation data.
