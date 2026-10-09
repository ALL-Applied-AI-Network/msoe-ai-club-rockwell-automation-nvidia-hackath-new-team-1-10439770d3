# The A-Team — Hackathon Kickoff Brief

**MSOE AI Club + UWM CSI · Rockwell Automation + NVIDIA Hackathon**  
**Kickoff:** Thursday, October 8, 2026, 4:00–6:00 PM CDT · Rockwell Automation HQ, Milwaukee  
**Purpose:** A concise handoff for teammates who missed the kickoff

## Executive summary

This is a four-week **Physical AI** challenge built around real manufacturing systems at UWM's Connected Systems Institute (CSI). The core expectation is not merely to produce a model that works on a laptop: teams must solve a specific manufacturing pain point, measure improvement against a baseline, explain operational and business value, acknowledge failure modes, and demonstrate the solution on the real CSI line or the separate FANUC robot setup.

The three project paths are final-vial inspection, fluid-mixing anomaly detection, and robotic grasping/sorting. Digital twins and synthetic data are central because real defects can be rare and useful training data often becomes available late. Teams also have access to sponsor mentorship and NVIDIA compute/tooling.

**Immediate priorities for The A-Team:** rename the workspace team, accept the pending GitHub invitation, choose one dataset, verify the data, join Slack, define a measurable baseline, request only the compute/data/mentor help we can justify, and submit the team plan by **Wednesday, October 14 at 11:59 PM CT**.

## What the kickoff emphasized

- **Physical AI closes the loop.** The system must perceive a real environment, reason about spatial/physical relationships, and support or execute an action under real-time and safety constraints.
- **A factory result matters more than a screen-only demo.** The project should survive contact with real equipment, changing recipes/products, imperfect views, and operational limits.
- **Start with a pain point and a baseline.** Tie the work to scrap, rework, downtime, labor, changeover effort, or operator trust; then report a metric that shows improvement.
- **Simulation can move development earlier.** Digital twins support model/operator training, synthetic-data generation, virtual commissioning, risk reduction, and testing before touching the physical system.
- **Keep the scope narrow and complete.** The official guidance favors one strong solution around a single idea or dataset over several incomplete ideas.

### What the speakers added

The recordings sharpened the written guidance in several useful ways:

- **Do not invent a problem.** Learn the actual workflow at CSI, ask operators and mentors what fails today, and solve something that already costs time, quality, or effort.
- **Go beyond classification.** A label is useful only if it drives the right operational response. The speakers explicitly contrasted “most accurate” with “most competitive” and stressed that manufacturing error costs are asymmetric: missing a shipped defect may matter far more than sending a good part to reinspection.
- **Design for the installed base.** Industrial equipment can remain in service for decades; consider whether the approach can work across new and legacy systems, and state where it cannot.
- **Use the access opportunity.** Visit CSI—or send a teammate—observe the process, seek feedback early, and arrange a supervised run. The judges want to see the system operate, not only hear a proposal.
- **Experiment safely while it is cheap to fail.** The hosts encouraged curiosity and trying new ideas during development, before failures can stop production where downtime can be extremely expensive.

## The three challenge options

| Dataset | Manufacturing problem | Provided context | Strong first baseline | Key risks |
|---|---|---|---|---|
| **1. Final Vial Inspection** | Detect missing/crooked caps, damaged vials, empty fixtures, or unexpected vials at final overhead inspection. | Overhead view of final pick-and-place plus a digital twin for synthetic data. | Fixture localization followed by object/defect detection; report per-class recall, false rejects, and latency. | Rare defects, class imbalance, glare/occlusion, sim-to-real gap. |
| **2. Fluid Mixing Station 3** | Detect incorrect fill level, cross-contamination, recipe inconsistency, and color drift during changeover. | Synchronized, timestamped tank/filling/final-inspection views plus a filling-station twin. | Align views by time/recipe and flag deviations in fill and color; compare with a simple threshold baseline. | Expected transition vs. fault, lighting/color calibration, temporal alignment, leakage across runs. |
| **3. Grasping and Sorting** | Find, grasp, and relocate objects that appear in unexpected positions. | Standalone FANUC 4iA arm, pneumatic two-finger gripper, multi-view RGB cameras, and a digital twin. | Detect/segment objects, estimate pose or grasp point, and measure pick success across controlled position changes. | Collision safety, calibration, occlusion, grasp failures, sim-to-real transfer and closed-loop latency. |

**Important:** Dataset 3's FANUC setup is separate from the vial production line.

## Resources available

- **Compute:** DGX Spark systems (10 at MSOE and 10 at UWM were shown), NVIDIA Brev GPU credits by request, build.nvidia.com model APIs, MSOE Dell GB10 access described in the event guide, and an in-person CSI GPU workstation for Emulate3D.
- **Simulation and AI:** Isaac Sim/digital twins, synthetic-data tooling, NVIDIA Cosmos, vision-language/in-context approaches, NuRec real-to-sim, and NeMo orchestration were presented as options.
- **People:** NVIDIA and Rockwell Automation mentors, plus MSOE/UWM/CSI organizers and dataset experts.
- **Coordination:** Slack channels for announcements, each dataset, testbed/data requests, and compute questions.

Treat this as a menu. Every requested tool or credit should map to a concrete experiment, workload estimate, or blocker.

## Testbed guardrails

Teams may visit CSI, observe the plant and robot arms, collect data with an approved setup, request more data, ask questions, and seek feedback. Teams may **not** move existing cameras/hardware, change PLC or control programming, interfere with safety/operational systems, or introduce components that could damage or disrupt the plant. Plan supervised, additive tests with a fail-safe path.

## Judging and deliverables

The final submission is due **Sunday, November 1 at 11:59 PM CST** and consists of:

- an approximately five-minute video;
- a one-page project summary;
- the team repository; and
- references/credits for data, models, code, and tools.

Finals are **Thursday, November 5, 5:00–8:00 PM** at MSOE Diercks Hall. Judges represent Rockwell Automation, NVIDIA, MSOE, and UWM. The official page emphasizes a working hardware demonstration, a specific problem, applicability to other manufacturers, measured results, and clear limitations. Kickoff slides also highlighted impact, innovation, in-context learning, presentation quality, and scalability; detailed weights were still being finalized.

The advertised prize pool is **$5,000 cash**, plus RTX 5080/5070 GPUs and sponsor swag; the exact distribution was not yet final.

## The A-Team

**Requested team name:** The A-Team  
**Current workspace name:** New Team 1 (rename needed)

| Member | Workspace role |
|---|---|
| Abdallah Benelmadjat | Lead / participant |
| Pablo Camarillo | Participant |
| Benjamin Hess | Participant |
| Tar Meh | Participant |
| Nathan Baker | Participant |

**Repository:** `ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-1-10439770d3`  
**Current status:** invitation pending for `@abdallah-benelmadjat`; accept it while signed into that GitHub account.

## Recommended 48-hour action plan

1. Rename **New Team 1** to **The A-Team** and accept/check repository access.
2. Join Slack and monitor `#hackathon-announcements`; join the three dataset channels long enough to compare data quality and mentor responsiveness.
3. Open every candidate dataset and record: sample count, labels, class balance, camera geometry, timestamps, train/evaluation restrictions, and missing documentation.
4. Choose one dataset using three filters: shortest route to a reliable baseline, access to a real-hardware test, and a clear manufacturing value metric.
5. Define the first experiment and success metric before choosing advanced tooling.
6. Submit the team plan with the chosen problem, needed data, justified compute/agent resources, requested mentorship, and open questions.

## Dates and discrepancy to watch

| Item | Authoritative working date |
|---|---|
| Team plan | **Oct 14, 11:59 PM CT** |
| Final submission | **Nov 1, 11:59 PM CST** |
| Finals | **Nov 5, 5:00–8:00 PM** |

One photographed kickoff slide says the project plan was due **Oct 13**. The current event workspace and saved event page say **Oct 14 at 11:59 PM CT**; follow the workspace and recheck `#hackathon-announcements` for changes.

## Source package

- [`Context about hackathon.txt`](Context%20about%20hackathon.txt) — saved official event article/context.
- [`image_context/README.md`](image_context/README.md) — index to 27 detailed, image-specific Markdown notes.
- [`transcripts_large_v3/`](transcripts_large_v3/) — higher-capacity Whisper Large v3 timestamped transcripts and structured segment JSON for all eight recordings.
- [`transcripts/`](transcripts/) — an alternate small-model, beam-searched pass; useful for comparison where the distant-room audio is unclear.

### Accuracy note

This brief distinguishes visible/official material from contextual interpretation. Video transcripts are machine-generated and should be checked against the recordings before quoting a speaker verbatim. Dates, access status, and resource availability can change; the event workspace and announcement channel take precedence.
