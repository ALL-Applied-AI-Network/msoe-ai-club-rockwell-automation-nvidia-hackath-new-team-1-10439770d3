# The A-Team — Master Project Reference

> **Purpose:** one durable, security-safe source of truth for everything The
> A-Team has researched, decided, built, and verified for the Rockwell
> Automation + NVIDIA Physical AI Hackathon.
>
> **Current through:** October 19, 2026
>
> **Team:** The A-Team
>
> **Working project:** **The Adaptive Workcell: Show the Job, Generate the Tool**

This file is the orientation document. It summarizes the work and links to the
evidence; it does not replace the underlying transcripts, reports, source code,
or measured result files.

## 1. Executive status

The team has moved from broad event discovery to a concrete Dataset 3 robotics
direction. We now have:

- a searchable kickoff knowledge base built from the event recordings and all
  photographed slides;
- a local archive and evidence-led audit of the official hackathon resources;
- an analysis of all three supplied datasets and their credible winning paths;
- a focused project concept combining demonstration-based task teaching with
  adaptive, generated robot tooling;
- reconstructed and validated gripper CAD assets, with the important mounting
  uncertainties explicitly identified;
- a human-gated robot-control MVP that prevents an AI agent from approving or
  directly executing a physical motion;
- a working A-Team DGX Spark environment with Dataset 3 deployed;
- Isaac Sim 6.0 running on the Spark; and
- measured proof that the supplied FANUC ER-4iA digital twin can execute our
  bounded two-waypoint motion to sub-millimetre TCP error.

The key boundary is equally important: **we have not moved the physical FANUC
robot yet**. The present result proves the software path and simulated motion,
not physical-cell safety, calibration, collision clearance, or authorization.

## 2. Project thesis

High-mix manufacturing still requires considerable engineering work whenever a
new part, tool, or assembly task enters a robotic cell. The A-Team's proposed
adaptive workcell reduces that changeover burden through two connected
capabilities.

### Workstream A — Toolsmith

The workcell observes a new object from calibrated views and reconstructs only
the geometry needed for manipulation. It then:

1. estimates dimensions, pose, graspable surfaces, and exclusion regions;
2. determines whether an existing gripper/finger set is suitable;
3. selects an existing tool when possible;
4. otherwise asks for missing facts rather than inventing hidden geometry;
5. uses CADVector/Forge to generate bounded finger candidates;
6. checks the mounting interface, actuator opening, collisions, approach
   clearance, and grasp suitability in the twin;
7. exports a reviewed candidate for 3D printing; and
8. compares the physical grasp against the original fingers.

The winning claim is not “the robot can grab anything.” A defensible claim is:

> The workcell recognizes when current tooling is inadequate, generates a
> compatible candidate for a bounded new part, and demonstrates a measurable
> grasp improvement after human-reviewed manufacture and installation.

### Workstream B — Show & Tell

A person demonstrates a bounded assembly, kitting, stacking, grouping, or
pick-and-place task while describing the intent. The system converts the video
and speech into a structured task graph instead of copying the person's exact
joint path. It then:

1. identifies objects and their relationships;
2. extracts goals, ordering constraints, and completion conditions;
3. maps them to a small approved robot-action vocabulary;
4. displays its interpretation for human confirmation;
5. executes the task in a changed tabletop layout;
6. verifies each step from camera evidence; and
7. retries safely, replans, or requests help when evidence is insufficient.

The first credible vocabulary is `observe`, `pick`, `place`, `orient`, `stack`,
`group`, `verify`, `retry`, and `request_help`. Arbitrary gearbox assembly,
threaded fastening, snap fits, cable routing, and unknown disassembly forces are
future work—not MVP promises.

### Combined demonstration story

The preferred final story is:

1. A person demonstrates and narrates a small task once.
2. The robot interprets the goal and performs it after the objects are moved.
3. A new object is introduced.
4. The system explains why the installed tooling is or is not suitable.
5. For one carefully selected incompatible object, it generates and validates a
   printable finger design.
6. After human review, printing, and supervised installation, the robot
   completes the same class of task with the new object.

This is one coherent manufacturing story: **teach the job, check capability,
generate the missing tool, and verify the result**.

## 3. Why this direction fits the kickoff

The recordings repeatedly emphasized several principles that shaped the
project:

| Kickoff signal | Project consequence |
|---|---|
| Start with a real existing problem | Frame the work around changeover engineering and validated operator pain, not a technology demo alone. |
| Go beyond classification into reasoning | Object recognition is a component; capability checking, explanation, planning, recovery, and verification are the contribution. |
| Learn from examples without fine-tuning | Teach a new bounded task or constraint through demonstrations/context while model weights remain fixed. |
| Show impact, creativity, scale, innovation, and contextual learning | Prepare distinct measured evidence for each claim rather than relying on one accuracy number. |
| Design for legacy equipment and deployment | Use the official policy interface and cameras without modifying safety controls or requiring a modernized production line. |
| Deploy and iterate on equipment | Treat supervised physical trials as required evidence, not an optional closing slide. |
| Preserve experienced-worker knowledge | Translate demonstrations and spoken intent into a reusable, reviewable task representation. |

The timestamped source is
[`knowledge-base-kickoff/ALL_TRANSCRIPTS.md`](knowledge-base-kickoff/ALL_TRANSCRIPTS.md).
It was generated with Whisper small and is a working transcription. Verify exact
quotations and speaker attribution against the recordings before using them in a
public presentation.

## 4. The three datasets and our decision

| Dataset | Opportunity | Decision |
|---|---|---|
| **1 — Final Vial Inspection** | Detect missing, crooked, damaged, or absent vials/caps at known fixture positions. | Good, feasible baseline; not selected as the primary project because classification alone is too modest for our chosen thesis. |
| **2 — Fluid Mixing Station 3** | Combine current imagery with recipe history and changeover context to diagnose fill/color anomalies and possible carryover. | Strongest non-robotics alternative and still the best fallback for contextual reasoning with lower hardware risk. |
| **3 — Grasping and Sorting** | Use the FANUC twin, multiple cameras, planning, verification, and recovery to adapt manipulation under variation. | **Selected primary dataset.** It supports both Toolsmith and Show & Tell and offers the strongest physical demonstration. |

The full evidence, leakage risks, baseline ladders, validation rules, and metrics
are in
[`knowledge-base-kickoff/THREE_EXPERIMENTS_DATASET_STRATEGY.md`](knowledge-base-kickoff/THREE_EXPERIMENTS_DATASET_STRATEGY.md).

### Dataset 3 source and verified baseline

Local supplied source:

```text
E:\Hackathon data\Datasets\fanuc_er4ia_hackathon_starter_v05\fanuc_er4ia_sim
```

The starter already contains a Cosmos-assisted six-cube stacking pipeline,
calibrated geometry, hybrid perception, planning, execution checks, and
confidence gates. Merely changing the cube order would not be a substantial
contribution. Our work must add rapid task teaching, tooling adaptation,
failure recovery, or a carefully measured combination.

Verified on both the local host and the DGX Spark: **all 33 Dataset 3 host tests
pass**.

## 5. Hardware and CAD knowledge

### Confirmed supplied hardware model

- **Robot:** FANUC ER-4iA, six-axis industrial robot.
- **Actuator:** SMC MHZL2-16D2 long-stroke, double-acting parallel air gripper.
- **Actuator finger option:** through-hole moving jaws.
- **Custom end fingers/blades:** approximately 76 × 12 × 6 mm in the supplied
  reconstruction; actual physical mounting details still require measurement.
- **Observations:** table, side, and wrist camera streams are supported by the
  evaluation system; some policy configurations also expose depth.

### What the organizer supplied versus what we reconstructed

The SMC actuator and moving jaw came from an official manufacturer 3D model.
They were reverse-engineered into editable SolidWorks parts with named features
and checked against the source solids:

| Part | Validation result |
|---|---|
| SMC actuator body | Exact bidirectional solid subtraction: zero volume difference; matching volume, area, and bounds. |
| SMC moving jaw | Exact; the same part rotated 180° matches the opposite jaw. |
| Custom 76 mm blade | Rebuilt from the kit's photo-based mesh; approximately 0.008% volume difference and 0.03 mm average surface difference. |
| Assembly | Body, two jaws, and two blades placed as represented by the starter assets. |

The supplied blade reconstruction is not yet a manufacturing drawing. In the
current assembly it geometrically intersects the jaw, its nominal 13 mm hole
spacing does not match the SMC jaw's 7 mm hole spacing, and the hole axes do not
produce a credible fastening path. The relationship between the manufacturer's
26.9 mm finger position and the starter's 0–12 mm gripping gap is also not
established.

Before printing a custom finger, measure the physical cell:

- jaw mounting face and its 5 × 8 mm interface;
- fastener diameter, hole axes, spacing, depth, and thread;
- installed blade seating/orientation;
- fully open and fully closed jaw/finger gap;
- usable stroke, allowed contact regions, and collision clearance; and
- one or more approved test objects: dimensions, mass, material, and required
  orientation.

Only the custom finger/inserts or a deliberately designed quick-change adapter
would normally be 3D printed. The FANUC arm and SMC actuator are supplied
hardware and are not reprinted.

### CADVector/Forge advantage

The team's existing parametric CAD system is documented locally at:

```text
E:\PROJECTOPENSCAD\forge\README.md
```

It already supports clarification questions, parameterized generation,
variants, compilation checks, visual checks, and printable exports. For this
project it still needs robot-specific mechanical constraints: verified mounting
geometry, actuator limits, grasp/collision tests, material/strength assumptions,
and human approval before fabrication.

## 6. Research and knowledge base completed

### Kickoff media extraction

- Transcribed all eight large event recordings with Whisper small.
- Preserved timestamped combined transcripts.
- Analyzed and contextualized all 27 photographed slides/images.
- Combined all photo context into one Markdown file.
- Produced a concise kickoff report for teammates who missed the event.

Primary artifacts:

- [`knowledge-base-kickoff/ALL_TRANSCRIPTS.md`](knowledge-base-kickoff/ALL_TRANSCRIPTS.md)
- [`knowledge-base-kickoff/ALL_IMAGE_CONTEXT.md`](knowledge-base-kickoff/ALL_IMAGE_CONTEXT.md)
- [`knowledge-base-kickoff/THE_A_TEAM_KICKOFF_REPORT.md`](knowledge-base-kickoff/THE_A_TEAM_KICKOFF_REPORT.md)

### Official event resource crawl

The official hackathon article and relevant linked public resources were
archived and analyzed with provenance, hashes, extracted evidence, access gaps,
and a follow-up queue. This includes the public guide, dataset documentation,
compute/tooling guidance, schedule material, and linked technical resources.

Primary artifacts:

- [`research/official-resource-audit/README.md`](research/official-resource-audit/README.md)
- [`research/official-resource-audit/COMPREHENSIVE_RESOURCE_EVIDENCE.md`](research/official-resource-audit/COMPREHENSIVE_RESOURCE_EVIDENCE.md)
- [`research/official-resource-audit/STRATEGIC_DONT_MISS_BRIEF.md`](research/official-resource-audit/STRATEGIC_DONT_MISS_BRIEF.md)
- [`research/official-resource-audit/ACCESS_GAPS_AND_FOLLOW_UP.md`](research/official-resource-audit/ACCESS_GAPS_AND_FOLLOW_UP.md)

The crawl is evidence-rich, but authenticated pages, Slack history, dynamic
content, and resources that were not publicly reachable remain explicit gaps.
“Public-resource audit” should not be described as proof that every private or
future event resource was captured.

### Physical-AI evaluation platform audit

We studied the SIDC Lab submission/evaluation platform, benchmark inventory,
policy categories, action/observation interfaces, repeated-rollout scoring,
operator workflow, calibration clues, downloadable telemetry, and public CAD
and media.

Important conclusions:

- policies are evaluated through standardized physical rollouts;
- layouts can change while the object set remains fixed;
- table, side, and wrist imagery and six-joint/gripper telemetry are logged;
- local operators retain homing, stopping, calibration, rerun, and fault
  controls;
- the online coordinator and locally supervised robot service are separate
  operational layers; and
- use the documented submission/client path rather than connecting an AI agent
  directly to an undocumented controller endpoint.

Primary artifacts:

- [`research/physical-ai-platform/SIDCLAB_PLATFORM_INTELLIGENCE_REPORT.md`](research/physical-ai-platform/SIDCLAB_PLATFORM_INTELLIGENCE_REPORT.md)
- [`research/physical-ai-platform/THE_A_TEAM_IMPLEMENTATION_BRIEF.md`](research/physical-ai-platform/THE_A_TEAM_IMPLEMENTATION_BRIEF.md)
- [`research/physical-ai-platform/RESOURCE_MANIFEST.csv`](research/physical-ai-platform/RESOURCE_MANIFEST.csv)

## 7. Team plan preserved

The working team-plan title is:

> **The Adaptive Workcell: Show the Job, Generate the Tool**

The plan names Dataset 3 and describes the two integrated workstreams above.
Its proposed evaluation includes:

- original versus generated fingers: pickup success, slips, placement accuracy,
  engineering/changeover time, human edits, print time, and installation time;
- demonstration learning: task-graph accuracy, step-verification reliability,
  completion under new layouts, clarification quality, recovery success, and
  correct refusal;
- system qualities: latency, compute cost, example budget, and operation with
  fixed foundation-model weights; and
- evidence discipline: deterministic geometry/motion checks and human approval
  around generative AI suggestions.

Compute planning originally included Brev GPU time and an MSOE GB10. The team
subsequently received both NVIDIA Brev credit and a private YAAD DGX Spark
workspace at UWM/CSI. DGX Spark is now the verified Isaac Sim environment.

The plan text was saved as a draft in the event workspace. Confirm the current
workspace submission state and any mentor/compute checkboxes directly in the
event portal; this reference does not treat a saved draft as proof of final
submission.

## 8. Access and compute established

### YAAD / DGX Spark

- Team workspace: **A-TEAM**.
- Persistent working directory on the interactive Spark:
  `/home/yt-ateamdd25/workspace`.
- Home Spark label: `spark-a318`.
- Platform: Ubuntu 24.04 on ARM64, NVIDIA GB10, approximately 119 GiB unified
  memory, with roughly 3.4 TiB free storage at verification time.
- Safe local SSH alias: `yaad-ateam`.
- Dataset 3 remote source:
  `/home/yt-ateamdd25/workspace/fanuc_er4ia_sim`.

The website's batch-job convention uses `/workspace`, while the persistent
interactive account uses the path above. The deployment script detects the
usable workspace instead of assuming these are identical.

### Brev

NVIDIA provisioned the captain with $500 in Brev credits and advised using
start/stop-capable RTX PRO 6000 or 4500 instances where appropriate. No Brev
instance should be left running when unused. Brev remains useful for parallel
experiments, but it is no longer a prerequisite for the verified robot MVP.

### Network authorization and boundaries

The organizer explicitly authorized responsible use of the provided connection
details by the team and its AI agents, with these constraints:

- do not alter, block, or interfere with shared network configuration;
- do not disrupt other users or prevent access;
- remain under 50 requests per second; and
- use the official interfaces and supervision model.

The CSI VPN and passwordless YAAD-only SSH key were repaired and verified on
October 19. Exact IPs, passwords, team codes, agent tokens, private keys, and
private run tokens are intentionally not recorded in this repository.

The security-safe service and credential-location inventory is maintained in
[`docs/ACCESS_REFERENCE.md`](docs/ACCESS_REFERENCE.md). A more convenient
machine-specific card is stored locally as the Git-ignored
`PRIVATE_ACCESS_REFERENCE.local.md`.

## 9. Robot-control MVP completed

[`robot-mvp/`](robot-mvp/) implements the smallest defensible path from an AI
request to a candidate FANUC motion:

1. Codex drafts a bounded TCP plan through the local `ateam-robot` MCP server.
2. The plan remains pending; MCP exposes neither approval nor execution.
3. A nearby human reviews and approves through a separate local CLI.
4. The drop-in policy uses the official SIDC client interface.
5. The cell's own limits and an on-site operator remain authoritative.

Available AI-facing tools are limited to status, drafting a two-point move,
drafting the standard safe demo, and cancelling an unexecuted plan. Physical
mode is locked by default and requires explicit local environmental gates plus
the CLI's physical confirmation phrase.

Verified locally:

- 5/5 MVP safety, state-machine, and live stdio-MCP tests pass;
- the official SIDC mock completed one rollout;
- 29 actions were accepted with zero rejections; and
- the mock ended with `Ready for the real robot`.

The mock validates interface and limit compatibility. It does not implement
collision physics or prove a safe physical target.

## 10. Isaac Sim and point-to-point proof

### Environment

- Image: `nvcr.io/nvidia/isaac-sim:6.0.0`.
- Required container entry point: `/isaac-sim/python.sh`.
- Root-container compatibility: `OMNI_KIT_ALLOW_ROOT=1`.
- Required Isaac terms variables are handled by the setup workflow.
- Headless startup reached application ready, advanced a frame, and exited
  successfully.

The local RTX 3060 Laptop GPU has only 6 GB VRAM and was not used for this
proof; the DGX Spark is the correct current target.

### Bounded simulated motion

Controller-space TCP targets, in UF0 millimetres/degrees with the gripper open:

```text
A = [450, -70, -230, -180, 0, 0, 0]
B = [450,  70, -230, -180, 0, 0, 0]
```

After applying the starter's documented +0.330 m base/UF0 Z conversion, the
simulator targets are `(0.45, -0.07, 0.10)` m and `(0.45, 0.07, 0.10)` m.

Measured result:

| Check | Result |
|---|---:|
| Task complete | `true` |
| Waypoint A TCP error | 0.230605 mm |
| Waypoint B TCP error | 0.219793 mm |
| Maximum joint tracking error | approximately 0.02112° |
| Dataset 3 tests on Spark | 33/33 passed |

Evidence:

- [`robot-mvp/simulation/isaac_point_to_point.py`](robot-mvp/simulation/isaac_point_to_point.py)
- [`robot-mvp/evidence/point_to_point_report_2026-10-19.json`](robot-mvp/evidence/point_to_point_report_2026-10-19.json)
- [`robot-mvp/scripts/setup_dgx_spark.sh`](robot-mvp/scripts/setup_dgx_spark.sh)

Generated remote artifacts:

```text
fanuc_er4ia_sim/runs/point_to_point_scene.usd
fanuc_er4ia_sim/runs/point_to_point_report.json
```

Integrity values:

- scene USD: 7.3 MiB, SHA-256
  `99a76b7dfa62c82846657b74b5d9cdfeec857c2a15562952815e5efbdb2cb470`;
- raw Spark report SHA-256:
  `0ac4cd8eaf6aeceb0df333a1a4c293e7ddc3fde4be99e53f4fb8cc45e7151588`;
- formatted repository evidence SHA-256:
  `7944fb6fa6d5b1cc223566b047084085b04bbdad84cb850199500c3de86ea9fe`.

This proves bounded motion in the supplied digital twin. It does **not** prove:

- collision-free transfer to the real workcell;
- agreement with the current physical tool frame;
- installed-finger clearance;
- camera/world calibration on the day of the run;
- physical repeatability; or
- permission for unattended or direct robot control.

## 11. Evidence ledger

| Claim | State | Evidence |
|---|---|---|
| Kickoff speech is searchable | Verified | Combined timestamped transcripts |
| All photographed kickoff slides have combined context | Verified | `ALL_IMAGE_CONTEXT.md` |
| All three supplied datasets were analyzed | Verified | Dataset strategy and local source review |
| Dataset 3 exists locally and remotely | Verified | Local dataset path and Spark deployment |
| FANUC starter host suite works | Verified | 33/33 tests locally and on Spark |
| Official SIDC client accepts our bounded policy | Verified in mock | One rollout, 29 actions, zero rejections |
| Human-gated MCP planner works | Verified locally | 5/5 tests and stdio protocol test |
| Isaac Sim 6.0 runs on DGX Spark | Verified | Headless smoke and job evidence |
| FANUC twin executes A→B accurately | Verified in simulation | JSON report and USD stage |
| Existing custom blade fits the real SMC jaw | **Not established; current reconstruction indicates mismatch** | CAD comparison and missing measurements |
| CADVector can generate a mechanically valid FANUC finger | Proposed | Integration and constraints not yet implemented |
| Video + speech can produce a correct robot task graph | Proposed | Pipeline not yet implemented |
| Generated tooling improves physical grasps | Not yet tested | Requires manufacture and repeated trials |
| Physical FANUC executes our policy safely | Not yet tested | Requires official supervised evaluation |

## 12. Next milestones, in order

### Gate 1 — Confirm the physical interface

- Schedule supervised access to the FANUC cell.
- Measure the actual SMC jaw, installed blades, holes, stroke, open/closed gap,
  tool frame, and relevant clearances.
- Photograph the interface with a scale and record measurement uncertainty.
- Update the editable CAD and re-run solid/interference checks.

**Exit criterion:** a reviewed manufacturing interface drawing whose dimensions
agree with the physical cell.

### Gate 2 — First supervised robot motion

- Use the official SIDC submission/run workflow.
- Confirm installed tool, active frame, table height, homing, obstacles, and
  E-stop access with the operator.
- Begin with one conservative motion or approved benchmark—not autonomous tool
  change.
- Record run ID, policy commit, frames, telemetry, outcome, and any fault.

**Exit criterion:** one operator-approved physical rollout with complete
telemetry and no safety-boundary bypass.

The exact preparation, submission, on-site execution, and evidence workflow is
documented in
[`docs/PHYSICAL_ROLLOUT_RUNBOOK.md`](docs/PHYSICAL_ROLLOUT_RUNBOOK.md).

**October 20 operational update:** the physical testing interface and Calendar
opened. A-Team request `q50c904` is submitted and pending calendar booking and
organizer approval. The installed client policy now converts the two reviewed
waypoints into 10 Hz rows limited to 2 mm translation and 1 degree rotation per
row from the observed physical TCP; it refuses unobserved physical motion and
requires a separately human-approved local plan for each of the three rollouts.

### Gate 3 — Bounded Toolsmith experiment

- Select 3–4 approved test objects: current-tool success, geometry-specific
  improvement, controlled orientation, and correct refusal.
- Integrate verified interface parameters into CADVector.
- Generate a small candidate family, test fit/collision/grasp in simulation,
  review, print, and install one candidate.
- Compare original and generated tooling across repeated randomized trials.

**Exit criterion:** measured benefit or an honestly documented negative result.

### Gate 4 — Bounded Show & Tell experiment

- Record short demonstrations with synchronized spoken instructions.
- Implement the restricted task-graph schema and human confirmation screen.
- Execute in changed layouts with perception and motion planning separated from
  semantic inference.
- Test ambiguity, missing objects, failed pickups, and correct requests for
  help.

**Exit criterion:** an unseen layout completed from a demonstration/context
example without foundation-model weight updates.

### Gate 5 — Final integration and pitch

- Join only the stable portions of Toolsmith and Show & Tell.
- Prepare one clean live story plus a recorded fallback.
- Present baselines, repetitions, failures, time/cost, and evidence for each
  award theme.
- Never imply that a simulation result is a physical result.

## 13. Award evidence plan

| Award theme | Evidence to build |
|---|---|
| **Most impactful** | Reduced engineering/changeover effort, higher grasp/task success, faster recovery, and a validated operator pain point. |
| **Most innovative** | The workcell reasons about capability and generates needed tooling rather than only classifying objects. |
| **Best contextual learning** | A new task or constraint is handled from a small disclosed example/context budget with fixed model weights and held-out layouts. |
| **Best presentation** | A visible new-part or disturbed-layout challenge, clear explanation, safe action, verification, and concise before/after metrics. |
| **Most scalable** | Fixed mounting schema, reusable action/task representation, onboarding time, inference cost, latency, and a legacy-equipment deployment path. |

No project can guarantee an award. Our strongest strategy is to make every
claim visible, reproducible, and grounded in the judges' stated manufacturing
priorities.

## 14. Security and credential hygiene

Secrets were shared during setup and must never be copied into Git, reports,
screenshots, issues, prompts, or team chat. Required cleanup:

- revoke and recreate the exposed YAAD personal access token;
- change the CSI VPN password that was previously shared;
- retire the old encrypted SSH key anywhere it was authorized because its
  private-key body was exposed, even though its passphrase was not recovered;
- preserve the new YAAD-only private key only on the local machine and back it
  up securely if desired;
- share only `.pub` public keys; and
- keep physical evaluation run tokens outside AI prompts and repository files.

Credential changes do not delete project source, Git history, Spark workspace
data, Isaac artifacts, or simulation results. Replacing an SSH public key only
changes which local private key can authenticate.

## 15. Repository and local workspace map

Official Git repository:

```text
https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-1-10439770d3
```

Local official clone:

```text
E:\Hackathon data\Hackathon main event\official-team-repository
```

High-value local supporting directories:

```text
E:\Hackathon data\Datasets
E:\Hackathon data\Hackathon main event\transcripts_large_v3
E:\Hackathon data\Hackathon main event\image_context
E:\Hackathon data\Hackathon main event\site_crawl
E:\Hackathon data\Set 3 robotic arm cad hand
E:\PROJECTOPENSCAD\forge
```

Repository guide:

| Location | Purpose |
|---|---|
| [`README.md`](README.md) | Public-facing project narrative and navigation |
| [`PROJECT_REFERENCE.md`](PROJECT_REFERENCE.md) | This master state/history document |
| [`docs/ACCESS_REFERENCE.md`](docs/ACCESS_REFERENCE.md) | SSH/VPN/YAAD/Git access map, public-key fingerprint, rotation, and recovery steps |
| [`docs/PHYSICAL_ROLLOUT_RUNBOOK.md`](docs/PHYSICAL_ROLLOUT_RUNBOOK.md) | Exact official-client workflow for the first supervised physical motion |
| [`knowledge-base-kickoff/`](knowledge-base-kickoff/) | Transcripts, photo context, kickoff report, dataset strategy |
| [`research/official-resource-audit/`](research/official-resource-audit/) | Official public-resource archive and strategic analysis |
| [`research/physical-ai-platform/`](research/physical-ai-platform/) | SIDC platform, hardware, policy, calibration, and implementation research |
| [`robot-mvp/`](robot-mvp/) | Human-gated MCP planner, official-client policy, tests, DGX/Isaac deployment, evidence |

## 16. Git history checkpoints

These commits preserve the major milestones through this reference date:

| Commit | Milestone |
|---|---|
| `42fa213` | High-fidelity project README and three-dataset strategy |
| `2b744d7` | Complete official resource audit (legacy history) |
| `679b714` | SIDC Physical AI platform intelligence |
| `14f27ce` | Human-gated FANUC point-to-point MCP MVP |
| `767bd55` | Robot MVP verification evidence |
| `7c5f7e4` | Verified YAAD Isaac Sim deployment workflow |
| `3c6be3c` | FANUC point-to-point motion verified on DGX Spark |

## 17. Working rules from here

1. Use this file to orient the team, then follow the linked primary evidence.
2. Label results as physical, simulated, mock, inferred, or proposed.
3. Record the hypothesis, baseline, metric, data split, software commit, and
   artifact for each experiment.
4. Keep training/context examples separate from held-out test examples.
5. Prefer a correct refusal or clarification over a confident invented fact.
6. Keep AI outside safety PLCs and undocumented robot-controller interfaces.
7. Require a nearby operator and official workflow for every physical motion.
8. Update this reference whenever a milestone changes status.

---

**Current bottom line:** The A-Team has a strong concept, a substantial evidence
base, a working human-gated control path, and a reproducible simulated FANUC
motion on NVIDIA hardware. The next decisive evidence is physical: measure the
real gripper, perform a supervised rollout, and prove one bounded adaptive
tooling or demonstration-learning improvement through repeated trials.
