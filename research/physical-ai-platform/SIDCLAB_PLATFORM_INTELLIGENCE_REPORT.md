# SIDC Lab Physical-AI Platform Intelligence Report

**Prepared for:** The A-Team

**Snapshot date:** October 19, 2026 (America/Chicago)

**Primary target:** [hackathon.sidclab.org](https://hackathon.sidclab.org/)

**Public client:** [SIDCLab/PhysicalAI-Hackathon-2026](https://github.com/SIDCLab/PhysicalAI-Hackathon-2026)

**Track:** Track 3 — Act on it: teach the robot

## Executive findings

The site is not merely a leaderboard. It is the participant-facing layer of a physical-robot evaluation service. A participant's policy runs on the participant's computer; the cell streams observations to it; the policy returns joint-space or tool-center-point targets; the cell applies its own validation and safety constraints; a local operator prepares and supervises every rollout; and the platform records scores, videos, timing, and robot state.

The strongest new facts for The A-Team are:

1. The evaluation robot is explicitly identified as a **FANUC ER-4iA**.
2. The gripper is an **SMC MHZL2-16D2** with **custom aluminium fingers**, not stock fingers.
3. Teams are explicitly allowed to propose their own fingers, objects, or bins through the `custom_hardware` field, subject to organizer review.
4. Teams may define **any manipulation task on the cell**, rather than being limited to the example leaderboard tasks.
5. The interface directly supports language-conditioned execution through `obs["trial"]["instruction"]`.
6. The real arm follows commanded targets about **0.8 seconds behind**; that behavior is absent from the twin.
7. Each rollout is scored as a **single binary success or failure**, so one missed step can invalidate an otherwise impressive assembly sequence.
8. Depth is available from the table and wrist cameras, but table depth cannot see the screws, and depth is **not recorded** in the returned dataset.
9. The supplied starter contains a consequential vial-scale mismatch: the real vial is **Ø27.18 × 203 mm**, while starter v0.5 uses **Ø17 × 140 mm**.
10. The public policy client and local mock expose the exact action limits, timing, calibration, and submission schema needed to start implementation now.

These findings strengthen both proposed A-Team directions, but they also narrow what should be demonstrated. Toolsmith is supported as approved custom hardware; Show & Tell is supported as an instruction-conditioned task. An autonomous printer or live magnetic tool-ejection mechanism is not part of the standard interface and would require explicit safety review. The most credible entry is therefore a closed-loop hybrid policy that interprets a short demonstrated task, executes a bounded action vocabulary, verifies every stage, and uses one pre-approved custom finger design when it measurably improves a grasp.

## Scope, ethics, and evidence boundary

This audit used only public, read-only material:

- the public participant dashboard and its self-contained HTML/JavaScript;
- all directly linked same-origin public CAD and media files;
- the public GitHub client repository and its history;
- the public local mock supplied by the organizers; and
- the screenshot-derived briefing provided by the team.

No organizer login was attempted. No private result key, request secret, run token, contact detail, or network identifier was guessed, enumerated, used, or reproduced. The site's public leaderboard is currently in private mode, so team-specific policies and results visible in screenshots are treated as screenshot evidence rather than independently verified public records.

Confidence labels used below:

- **Confirmed:** stated in the site or public client, or reproduced with the local mock.
- **Observed:** visible in public media or team-provided screenshots.
- **Inferred:** a reasoned architectural or strategic conclusion, clearly labeled as such.

## 1. What the platform is

The site describes itself as an online platform for standardized submission, remote execution, and reproducible evaluation of robot policies on physical robots. The public workflow confirms this architecture:

```text
simulation / demonstrations / training
                  |
                  v
        policy on team's machine
                  |
          obs <--> act(obs)
                  |
                  v
       SIDC cell execution server
                  |
                  v
        FANUC ER-4iA + cameras
                  |
                  v
 score + videos + state + dataset
```

The important separation is between policy inference and physical execution. The policy is not uploaded as a container to an opaque evaluator; it stays on the team's machine. The team's client holds the approved run token, waits for the local operator, receives observations, and returns actions. The cell remains responsible for action validation and physical safety limits.

The operator workflow shown in the team's screenshots—homing, calibration controls, run selection, trial readiness, stop/rerun, and scoring—is consistent with the public client. This is operator-supervised remote execution, not unattended autonomy.

## 2. Exact policy contract

The team's policy is a Python `Policy` class in `policy.py`. The constructor loads a model or checkpoint, `reset()` runs at the start of every rollout, and `act(obs)` returns either one target or an action chunk.

### Observation fields

| Field | Confirmed contents | Engineering consequence |
|---|---|---|
| `obs["images"]` | Selected RGB streams: `table`, `side`, and/or `wrist`; whole frames resized, not cropped; default 640 × 360; PNG or JPEG | Crop or resize consistently with simulation; do not assume a cropped camera model |
| `obs["joints"]` | J1–J6 in degrees plus gripper state | Required for joint policies and moving wrist-depth geometry |
| `obs["tcp_pose"]` | Fingertip `[x, y, z, w, p, r]` in millimetres/degrees | Lets a task planner work in Cartesian coordinates without client-side inverse kinematics |
| `obs["trial"]` | Current rollout metadata, including the natural-language instruction | Direct hook for Show & Tell or language-conditioned task graphs |
| `obs["depth"]` | Optional table and/or wrist `uint16` depth in millimetres; 0 means no depth | Useful for geometric verification, but must be logged locally if needed later |

The latest frame from each requested camera is included; the public documentation does not promise exact cross-camera exposure synchronization. A policy that depends on fine temporal alignment should timestamp and test that assumption rather than silently relying on it.

### Action formats

The policy selects one action space:

| Setting | Target | Best fit |
|---|---|---|
| `ACTION = "joints"` | `[J1, J2, J3, J4, J5, J6, gripper]` in degrees | ACT, diffusion policies, and other learned visuomotor policies |
| `ACTION = "tcp"` | `[x, y, z, w, p, r, gripper]` in mm/degrees | Task-and-motion planning, VLM-driven behavior selection, and deterministic skills |

For the A-Team's proposed architecture, TCP actions are the clearest first choice: foundation models can select and parameterize a small skill library, while deterministic geometry and the cell controller handle precise execution. Joint actions remain appropriate if the team later trains a visuomotor behavior model.

### Timing and chunking

- Default target duration: **0.1 s**, matching the starter's 10 Hz data.
- Allowed target duration in the mock: **0.1–1.0 s**.
- Maximum action chunk: **100 targets**.
- Maximum queued targets: **100**.
- `act(obs)` has no strict time limit; the arm holds still while it computes.
- `PREFETCH = True` starts planning the next chunk while the current one plays when the chunk is long enough to cover inference and network delay.
- The leaderboard's latency field represents median time from serving an observation to receiving the next action chunk. It is not the robot cycle time.

The real robot's approximately 0.8-second target-following lag is the central sim-to-real timing risk. The team should avoid long open-loop chunks near contact. Use coarse motion chunks in free space and short, re-observed steps near the object, fixture, or stack.

### Validated safety envelope in the public mock

The mock states that it shares the real cell's API, limits, and timing. It validates:

- TCP x: **150 to 650 mm** in UF0;
- TCP y: **−450 to 450 mm**;
- TCP z: **−334 to 300 mm**;
- W/P/R: each within **±360°**;
- J1: **−170° to 170°**;
- J2: **−110° to 120°**;
- J3: **−69° to 205°**;
- J4: **−190° to 190°**;
- J5: **−120° to 120°**;
- J6: **−360° to 360°**;
- finite numeric values only; and
- gripper values clipped to `[0,1]`, with values at or above 0.5 treated as closed.

The site separately warns that targets beyond about **0.55 m from the base axis** can cause a controller fault and terminate the rollout. The rectangular mock envelope is therefore necessary but not sufficient; radial reach, collision clearance, and the real fixture geometry still matter.

## 3. Coordinate systems and calibration

The client includes a unusually complete calibration package.

### Frames

- **Robot base:** x forward, y left, z up, in metres. Work surface at z = **−0.0051 m**.
- **UF0:** x and y equal the base frame; `z_UF0 = z_base − 330 mm`. Work surface is therefore approximately **−335 mm**.
- **UTOOL1/TCP:** fingertip, 223 mm beyond the flange face and 293 mm from the J6 origin.
- **Downward finger orientation:** `w,p,r = −180,0,R`, with R controlling yaw.
- **Twin joint convention:** simulated J3 is controller J3 + J2.

### Camera package

| Camera | Native/served geometry | Calibration quality disclosed | Notes |
|---|---:|---:|---|
| Table RGB | 1920 × 1080 native | 0.465 px RMS | Fixed overhead/front view |
| Side RGB | 1920 × 1080 native | 3.986 px RMS | Re-aimed and re-solved after testbed rebuild |
| Wrist RGB | 1280 × 720 model; captured 848 × 480 at 15 fps over USB 2.0 | 0.904 px RMS | Moves with flange |
| Table depth | 640 × 400 | 2.31 mm plane RMS | Fixed; cannot see screws reliably |
| Wrist depth | 848 × 480 | 3.0 mm plane RMS | Requires the joints from the same observation |

The supplied `geometry.py` provides pixel projection, table-plane back-projection, forward kinematics, and depth-to-base conversion. This enables a geometry-first perception baseline before any large model is added.

The table plan in the client reports a physical work-surface boundary of approximately x **−371 to 711 mm** and y **−724 to 744 mm**, but the dashed reach circle is about 550 mm. The default scene places eight screws and a yellow bin in the reachable front half of the table. Placement coordinates use the same x/y robot frame as `obs["tcp_pose"]`.

## 4. Cell hardware and public CAD assets

### Robot and gripper

- Robot: **FANUC ER-4iA**, six-axis industrial manipulator.
- Actuator: **SMC MHZL2-16D2** long-stroke parallel air gripper.
- Installed fingers: custom aluminium, **76 × 12 × 6 mm**, with a hooked notch.
- Jaw travel: **6 mm per jaw**.
- Straight-face gap when open: **12 mm**.

This official clarification supersedes the older repository wording that referred to a SCHUNK gripper.

The public SMC STEP downloaded from the platform is a SolidWorks 2022 AP214 export with separate product entries for the body and prong/finger components. Its SHA-256 differs from the earlier CADENAS export already in the team's modeling kit. Different hashes do not prove different geometry: the files use different exporters, entity ordering, and metadata. Continue using geometric comparison—body subtraction, bounding boxes, hole axes, and mounting planes—rather than byte identity.

### Objects

| Object | Confirmed geometry/details |
|---|---|
| Countersunk screw | ISO 10642 M6 × 30, 90° head, 4 mm hex; threads omitted from public models |
| Bin | 50 × 50 mm outside, 55 mm tall; 35 × 35 mm pocket, 25 mm deep; corner radius approximate |
| Cubes | 10 mm and 16 mm; 16 mm cube has 1 mm rounded edges; 10 mm radius approximate; origin at bottom-face center |
| Vial | Clear plastic, Ø27.18 × 203 mm, part 0890-01 |
| Cap | MOCAP TWF.820/.930 tapered plug, Ø31.75 mm flange, 15.7 mm tall |

### Critical starter correction

Starter v0.5 scales the vial to Ø17 × 140 mm. The site instructs teams to set:

```python
VIAL_RADIUS = 0.02718 / 2
VIAL_LENGTH = 0.203
```

The cap is already at the correct size. Any vial-gripper design or synthetic data generated before this correction should be considered geometrically suspect and regenerated.

### Public asset bundle

The archive contains:

- SMC gripper STEP;
- screw STEP and STL;
- bin STEP and STL;
- 10 mm and 16 mm cube STEP/STL pairs;
- vial USDA and STL;
- cap USDA and STL;
- real tabletop texture PNG; and
- synchronized-looking table and side demonstration clips with poster images.

The two public clips are both 640 × 360 H.264 at 10 fps, 109.4 seconds long, with 1,094 frames. They show the FANUC cell, hooked gripper fingers, eight scattered screws, three colored cubes, and yellow/blue/clear bins. The footage appears to demonstrate repeated screw collection into the yellow bin while the cubes and other bins act as scene clutter. That task interpretation is observational, not a published ground-truth annotation.

## 5. Submission and physical-evaluation workflow

The exact participant path is:

1. Clone the public client and run `uv sync`.
2. Implement `Policy.act(obs)` and select cameras, image format, action format, depth, and prefetch settings.
3. Run `uv run run.py --mock` to validate shapes, limits, timing, and connectivity.
4. Complete `submission/submission.yaml`.
5. Complete `submission/placement.yaml` unless using the unchanged default eight-screw scene.
6. Submit with `uv run run.py --submit`.
7. Book a 30-minute test slot in Slack channel `#data-and-testbed-requests`, including the printed request ID.
8. After approval, run `uv run run.py` and leave the client connected while the operator works through the rollouts.
9. Retrieve the private results link with `uv run run.py --status`.
10. Download the accumulated LeRobotDataset with `uv run run.py --download`.

The request file written by submission contains a secret and must remain private. The results key is likewise team-private. Neither belongs in Git.

The client is documented for Linux and macOS, including Apple Silicon. On this Windows PowerShell host, the mock completed its action checks but initially failed only when printing the Unicode check mark under the default CP1252 console encoding. Setting `PYTHONUTF8=1` made the mock finish successfully. For the real slot, use the supported Linux environment or WSL and test the entire client path in advance.

## 6. Task, placement, and custom-hardware schema

`submission.yaml` records:

- team and contact;
- policy name;
- model type (`VA`, `VLA`, `WAM`, `VLM + BM`, `TAMP + FMs`, or `Hybrid`);
- task ID and one-sentence task;
- whether objects and layouts remain the same;
- rollout count, ten by default; and
- custom hardware.

`placement.yaml` records per-rollout object positions and optional instructions. If either objects or layout is `different`, the client requires one placement entry per rollout. Object coordinates are in robot-frame millimetres with optional yaw. Anything that changes between rollouts must be found from observations rather than assumed.

This has two consequences for the A-Team:

1. **Show & Tell fits the interface.** A concise task graph or instruction can be supplied per rollout, while object positions change. Generalization can be demonstrated by executing the same relational task on unseen layouts.
2. **Toolsmith fits the review process, not the live control loop.** A printed finger can be declared as custom hardware and installed for an approved run. Dynamic printing or automatic tool ejection is not part of the supplied action interface and should not be assumed permissible.

## 7. Scoring, results, and datasets

- A standard run requests **10 rollouts** by default.
- Each booking slot is **30 minutes**.
- The operator prepares each scene, homes the robot, starts the rollout, and records the result.
- The rollout is scored **success or failure as a whole**, not per picked object or assembly step.
- Controller faults and operator interventions are separately noted.
- Successful rollouts record start-to-success cycle time.
- A policy cannot declare the rollout finished. Once done, it must hold the current pose while the operator scores it.
- Every rollout records three camera streams and robot state at **10 Hz** as a LeRobotDataset episode.
- The public viewer code supports camera clips and a trajectory trace containing time, six joints, and gripper state.
- Depth is available live but explicitly **not recorded**.
- Results are private to the team until the organizers choose to reveal the leaderboard; teams can also request that runs remain off it.

The site's leaderboard schema can compare success rate, confidence interval, successful/attempted rollouts, successful cycle time, layouts, model type, action type, latency, cameras, date, and—when supplied—code, weights, and data links. Its policy comparison view is layout-by-layout, reinforcing that generalization across placements is a core evaluation concept.

## 8. Interpreting the screenshot-derived benchmark intelligence

The team's screenshots add policy names, benchmark tabs, run counts, provisional results, and local operator details that the current public page does not expose. They suggest tasks such as screw collection, two-cube stacking, and a six-cube tower, with policies labeled VLM + BM, VLA, VA, and Hybrid.

Those screenshots are valuable competitive intelligence, but the following rules prevent overclaiming:

- A displayed success percentage during an unfinished run is provisional.
- `CYCLE S` is successful rollout duration, not model latency.
- A fault is not automatically equivalent to a policy error; the platform tracks interventions and controller faults separately.
- Counts on tabs may be result counts rather than total submissions.
- Policy/model labels describe the submitted architecture category, not independently audited internals.
- Private contact and network fields should never be copied into team documents.

The public platform now confirms the screenshots' broad architecture: repeated physical rollouts, operator supervision, local execution control, multiple policy types, multi-camera observations, and downloadable trajectory datasets.

## 9. Strategic consequences for The A-Team

### Toolsmith: what is now defensible

The winning version is not “the robot reverse-engineers anything and prints a hand in real time.” The defensible version is:

> Given a bounded family of unfamiliar parts, the system determines whether the approved finger can grasp the part. When it cannot, it generates a constrained insert or replacement finger, verifies fit and clearance in CAD/simulation, and demonstrates a statistically meaningful improvement with organizer-approved custom hardware.

Required proof:

- fixed mounting interface and exact real-gripper measurements;
- at least one part where stock/current fingers already work, proving the system can decline unnecessary redesign;
- at least one part where a generated design improves repeated physical pickup success;
- one part correctly rejected as outside reach, stroke, force, or safe-contact capability;
- collision, jaw-travel, fastener, strength, and approach-clearance checks;
- transparent separation between generated proposal and deterministic engineering validation.

The tool-change carousel, magnetic coupler, or self-ejection mechanism should remain a stretch concept until organizers approve it. Pneumatic routing, retained payloads, collision behavior, fail-safe attachment, TCP recalibration, and safety review make it a separate mechatronics project.

### Show & Tell: what is now defensible

The strongest version is:

> A human demonstrates and narrates a short bounded task. The system extracts a relational task graph, asks for clarification when needed, and executes it on a new layout using verified skills and closed-loop stage checks.

Start with four or five primitives:

- pick;
- place in region/bin;
- stack;
- group/sort; and
- verify/re-observe/retry.

Do not start with arbitrary gearbox assembly. Tight insertions, thread engagement, hidden forces, compliance, backlash, occlusion, and tool use exceed the evidence available from ordinary RGB video. A gearbox can remain the long-term story while the scored physical task uses blocks, bins, or a generous-clearance two-part assembly.

Required proof:

- demonstration-to-task-graph accuracy;
- execution on layouts not used in the demonstration;
- one spoken correction that changes behavior without retraining;
- explicit confirmation before physical execution;
- step-level verification and bounded recovery;
- comparison against coordinate replay or a fixed scripted baseline.

### Recommended combined entry

**The Adaptive Workcell: Show the Job, Generate the Tool** remains coherent if scope is disciplined:

1. Interpret one demonstrated kitting/stacking task into a task graph.
2. Execute it with TCP skills on randomized placements.
3. Detect that one approved object is incompatible with the current finger.
4. Present a CADVector-generated candidate and deterministic validation report.
5. Run a separately approved evaluation with the printed finger installed.
6. Compare repeated pickup and full-task success against the baseline finger.

This yields one narrative—rapidly commissioning a workcell for a new product—without depending on an unapproved live tool-changing mechanism.

## 10. Recommended technical architecture

```text
video + narration
       |
       v
multimodal task interpreter
       |
       v
typed task graph + object relations
       |
       +-----------------------------+
       |                             |
       v                             v
scene grounding               capability/tool check
RGB + wrist depth             stroke/contact/clearance
       |                             |
       v                             v
deterministic TCP skill library   CADVector proposal
       |                         + CAD/sim validation
       v                             |
execute -> re-observe -> verify <----+
       |
       v
success / retry / safe refusal
```

Foundation models should handle semantics, ambiguity, and candidate generation. Deterministic components should enforce coordinate transforms, allowed actions, reach, collision constraints, grasp geometry, tool compatibility, and stop/refusal conditions.

## 11. Experiment plan aligned to the platform

### Experiment A — task understanding

Compare:

1. coordinate replay;
2. language instruction only;
3. video only; and
4. video plus narration and one confirmation step.

Measure task-graph exact match, critical-step recall, object-relation accuracy, clarification quality, and success on randomized layouts.

### Experiment B — closed-loop execution

Compare:

1. open-loop full sequence;
2. re-observation between high-level steps; and
3. re-observation plus bounded retry.

Measure full-rollout success, controller faults, unnecessary retries, cycle time, and recovery success. Because scoring is binary, full-rollout success is the primary metric; step-level measures explain why it changed.

### Experiment C — custom finger value

Compare the installed finger with one CADVector-generated, reviewed design on the same object family and placement distribution. Measure pickup success with confidence intervals, slip/drop rate, placement accuracy, setup time, design time, human corrections, and correct refusal rate.

### Experimental discipline

- Freeze the test placements before final tuning.
- Keep the task demonstration separate from evaluation layouts.
- Use at least ten repeated physical attempts per condition when access allows.
- Record local depth if it influences a decision because the platform dataset will not preserve it.
- Report faults and interventions separately from ordinary task failures.
- Preserve failed runs; they are essential evidence for recovery and limitation claims.

## 12. Immediate operational checklist

- [ ] Clone and pin public client commit `5c458b605012a72c6f6ab433573c2350e20c9c41`.
- [ ] Develop in Linux/WSL; run the local mock after every action-interface change.
- [ ] Patch the starter's vial dimensions before generating vial data or tooling.
- [ ] Import the supplied camera calibrations and validate one known table point.
- [ ] Start with `ACTION = "tcp"`, JPEG, and short near-contact chunks.
- [ ] Implement a hard workspace/reach gate before sending targets.
- [ ] Implement `hold current pose` after task completion.
- [ ] Save live depth locally with observation timestamps when used.
- [ ] Draft `submission.yaml` with model type `TAMP + FMs` or `Hybrid`.
- [ ] Define ten legal placement entries that actually test layout generalization.
- [ ] Describe any printed finger precisely under `custom_hardware`.
- [ ] Request organizer approval before designing an automatic coupler or tool carousel into the live demo.
- [ ] Book `#data-and-testbed-requests` early and include the request ID.
- [ ] Budget setup and reset time inside the 30-minute slot.
- [ ] Pre-register the scoring rule for the full task and each failure category.

## 13. Questions that still require organizer confirmation

1. Which custom-finger materials, fasteners, protrusion limits, and load cases are acceptable?
2. Is a manually installed printed finger sufficient for custom-hardware evaluation, and how much installation time is available?
3. Would any automatic coupler or magnetic retention system be permitted, and what safety documentation would it require?
4. May the team save live depth locally during official rollouts?
5. How are operator interventions and controller faults treated in award judging, beyond being annotated?
6. What is the exact success condition for the team's custom task, and who approves it before the slot?
7. Can a single approved request include different natural-language instructions across its ten rollouts?
8. Are team-defined objects allowed to vary in mass/material if their geometry is declared?
9. Is the wrist RGB stream served at a stable effective frame rate when all RGB and both depth streams are requested?
10. Will there be opportunities for calibration-only or hardware-fit checks before scored runs?

## 14. Reproducibility record

Local archive:

`E:\Hackathon data\Hackathon main event\site_crawl\sidclab_2026-10-19`

Key artifacts:

- `index.html` — public dashboard snapshot, 107,588 bytes, SHA-256 `B64B31E765E58C068D00DBE77C7688E53688058D7F816F1E5B766F0F0C6AAF28`.
- `PhysicalAI-Hackathon-2026/` — full public Git history, pinned at commit `5c458b6` for this audit.
- `public_assets/` — all 18 directly linked public CAD/media assets plus generated contact sheets.
- `RESOURCE_MANIFEST.csv` — URL, size, hash, and purpose for the archived assets.

The public dashboard was built at `2026-10-19 21:19` according to its embedded data. It was in private-results mode with no public task or leaderboard rows. `robots.txt` and `sitemap.xml` returned 403 to GET during the audit, while the main page and directly linked assets were publicly retrievable.

## Bottom line

The platform is unusually favorable to The A-Team's thesis. It explicitly supports custom fingers, natural-language rollout instructions, multi-camera and depth observations, team-defined manipulation tasks, local foundation-model inference, and downloadable physical-trial data.

It also makes the winning standard clearer: the entry will be judged by reliable whole-task execution on real hardware, not by an impressive generative demo alone. The A-Team should use AI for task understanding and tool ideation, but win the physical evaluation through constrained skills, geometry, calibration, re-observation, measurable baselines, and safe refusal.
