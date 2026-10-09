# The A-Team — Hackathon “Don’t Miss Anything” Brief

**Event:** MSOE AI Club + UWM CSI + Rockwell Automation + NVIDIA — Fall 2026 Physical AI Innovation Lab  
**Prepared:** October 9, 2026 (America/Chicago)  
**Source of truth:** [Official event overview](https://msoe-ai-club.github.io/maic-content/competitions/Innovation%20Labs/Rockwell-NVIDIA-Fall-2026/Rockwell-NVIDIA-Fall-2026.md)

## Executive takeaway

The strongest competitive path is not “use the most AI.” It is to deliver one narrow, repeatable system that works on the real CSI line or FANUC arm, beats a stated baseline, reports failures honestly, and connects the result to manufacturing cost, deployment effort, and operator trust.

The official site explicitly recommends one strong solution around a single dataset. The final package must include an approximately five-minute video, one-page summary, repository, and references. Judges expect a real-hardware demonstration, a specific problem, a transferable business case, and measured results.

## What the crawl covered

- Indexed the complete public MAIC content manifest: **812 assets**.
- Downloaded and searched **all 217 public Markdown articles**, with **0 failures**.
- Followed the event’s same-site article graph to the requested maximum of 10 levels.
- Found **8 connected hackathon pages**: the event hub, three dataset pages, and four operational/resource guides.
- The deepest meaningful internal level is **1**, because the event hub links directly to every related page and those pages link back to the hub/guides. There is no hidden level 2–10 branch.
- Relevance-scored all unlinked articles and found **no additional omitted hackathon page**.
- Inventoried **104 external-link references** and **17 embedded event images/media references**.

Exact crawl data is in `CRAWL_INDEX.json`, `EXTERNAL_LINKS.csv`, and `pages/` beside this file.

## Dates that cannot slip

| Date | Deliverable |
|---|---|
| **Wednesday, October 14, 11:59 PM CT** | Team plan due through the ALL AI Network form |
| **Sunday, November 1, 11:59 PM CT** | Final submission: ~5-minute video, one-page summary, repository, references/credits |
| **Thursday, November 5, 5:00–8:00 PM** | Finals at MSOE Diercks Hall |

The plan is a resource request, not a binding technical contract, and can be updated before the deadline. It should name the dataset, problem, extra data, compute and estimated credits, coding-agent needs, mentor support, and open questions.

## Immediate A-Team checklist

1. **Join Slack now:** [Rockwell/NVIDIA Hackathon Slack invitation](https://join.slack.com/t/rockwellnvidi-wq82211/shared_invite/zt-4bp05nsn9-UW3afFtM9wjzWtZiQoJBHg).
2. Open `#hackathon-announcements`, the chosen dataset channel, `#data-and-testbed-requests`, and `#compute-questions`.
3. Confirm every member can access the [Innovation Lab data folder](https://drive.google.com/drive/folders/1GWuHV2WUNLztq_Fga6xBH7qFzhxDaac8?usp=sharing).
4. Confirm the team repository invitation and ALL AI Network workspace access.
5. Pick one primary dataset and write a one-sentence problem plus one primary metric.
6. Establish a simple baseline before selecting advanced tools.
7. Put specific compute, coding-agent, data, CSI-visit, and live-test requests into the team plan.
8. Reserve physical testing early; the final submission is expected to show real equipment.

## Three dataset decisions

### Dataset 1 — Final Vial Inspection

**Task:** Inspect two 30-position fixtures from the final overhead camera for `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`, and `no_vial` at the target-position level.

**Critical nuance:** A trailing empty aperture may simply not have been filled yet. Use the robot’s fixed fill order, placement count, or order-complete signal so the system does not confuse progress with a missed placement. The judging priority is defect escape prevention: a missed defect matters more than a false alarm.

**Good baseline:** Position crop + golden-reference comparison or a small classifier. Report false alarms at a threshold that catches every known defect, per-class false negatives, and complete-fixture results—not only crop accuracy.

**High-value extension:** Use the digital twin to target rare defects and domain variations (pose, glare, light, cap seating), then gate every new model against a fixed test set.

### Dataset 2 — Fluid Mixing Station 3

**Task:** Use synchronized `vial_fill`, `tank_level`, and `cap_seal` views plus recipe context to detect color/fill faults and explain their likely cause.

**Critical nuance:** Residual fluid in the common mixing path can discolor the first vial after a recipe change. The previous recipe is causal context, not merely metadata. T-300 was unavailable; do not use it. Do not independently change recipes or plant controls.

**Good baseline:** Timestamp alignment plus classical color/fill measurements. Test whether previous-recipe features improve prediction before adding a large vision-language model.

**Evaluation:** Split by run/time rather than neighboring frames. With 30 labeled vials across 10 recipes, use leave-one-recipe-out evaluation. Grade root-cause explanations separately from anomaly detection.

### Dataset 3 — Grasping and Sorting

**Task:** Use two overhead cameras, a wrist camera, the FANUC ER-4iA digital twin, and a SCHUNK two-finger gripper to produce repeatable pick-and-sort behavior.

**Critical current blocker:** The public page says the robotic digital twin is to be released and that camera samples, calibration files, finger CAD, and Rosie access details are still being finalized. Verify this in Slack/Drive immediately rather than planning around unavailable assets.

**Good baseline:** Make the supplied object/fingers work in simulation, repeat across varied starting positions, then introduce one variable at a time. Measure pick success, placement accuracy, cycle time, and failure recovery.

**Safety:** Never command the physical arm, change its controller, or install printed fingers without plant-manager/mentor approval and clearance review.

## The rule that changes the technical plan

For Datasets 1 and 2, the organizers say to inspect the real line data but **avoid training on it**. The intended workflow is synthetic training data from the digital twin, then evaluation on real or held-out data. This is central to the high-mix, low-volume manufacturing story and should be explicit in the pitch.

Keep raw data immutable. Store cleaned data, labels, and generated data separately. Do not commit large datasets, checkpoints, credentials, or API keys. Commit a small representative sample and instructions for authorized data access.

## Compute and tool requests worth putting in the team plan

- **Brev:** Request a workload, GPU, estimated hours/credits, and date. Prefer RTX PRO 6000 when available for training/Isaac Sim; L40S is the lower-cost fallback. Stop instances immediately after use and record GPU, runtime, credits, model/version, and result.
- **Campus systems:** Request access to UWM DGX Spark, MSOE Dell GB10/Rosie, or the CSI workstation only when the workload justifies it.
- **Free NVIDIA APIs:** Cosmos 3 Nano Generator and Reasoner on build.nvidia.com may provide a fast multimodal baseline.
- **Coding agents:** Free Gemini via Antigravity is the default. The organizers allow requests for Claude or Codex plans, but the team must specify plan/count/users/use case and why Gemini is insufficient.
- **Isaac Sim:** The Brev launchable is headless. Use CLI/file copy, render outputs, preserve important work elsewhere, and stop the instance after each session.

Choose tools only after naming the baseline failure they address, their I/O, access/compute needs, and how they will be compared against the baseline.

## Data collection and live-test protocol

Use `#data-and-testbed-requests` and tag **Shamar Webster** for extra data, CSI visits, or physical testing. A useful request includes:

- team and dataset;
- question being answered;
- exact recording/examples/counterexamples needed;
- camera, framing, resolution, frame rate, recipe/conditions;
- sample count or duration;
- labels and metadata required;
- requested date and success criterion.

For the live test, freeze a runnable version in advance. Document inputs, outputs, start/stop procedure, expected metrics, failure behavior, and every attempt’s software/model/input/prediction/expected result/latency/operator intervention.

Never move cameras; alter PLC, controller, FactoryTalk, safety, recipe, or robot programs; connect to the industrial network; or create collision risk without explicit approval.

## Slack map and contacts

| Need | Channel | Contact |
|---|---|---|
| Announcements/deadlines | `#hackathon-announcements` | Monitor continuously |
| Dataset/line questions | Dataset-specific channel | Brett Storoe, Dr. Derek Riley, or Tanner Cellio |
| Extra data, CSI visit, live test | `#data-and-testbed-requests` | Shamar Webster |
| Brev/GPU/compute | `#compute-questions` | Dan Schneider |
| Website, signup, team, Slack | `#hackathon-general` | Brett Storoe |

For a blocking issue, start the message with **critical-path blocker**, then include what was tried, the exact error, and the date by which it must be resolved.

## How The A-Team can differentiate

1. **Tell one coherent manufacturing story:** problem → baseline → failure analysis → synthetic intervention → fixed real-data evaluation → physical demonstration → adoption economics.
2. **Measure the ugly cases:** false negatives, recipe transitions, domain shift, clutter, recovery, latency—not just average accuracy.
3. **Make reproducibility visible:** one-command setup/evaluation where possible, frozen configs, experiment ledger, data lineage, model cards, and credited sources.
4. **Demonstrate operator trust:** confidence/abstention, understandable cause, review workflow, and safe failure mode.
5. **Quantify deployment:** hardware, cycle time, setup/changeover effort, compute/credit cost, and likely reduction in scrap/rework/downtime/labor.
6. **Show the real system:** schedule and rehearse the supervised hardware run early enough to recover from failure.

## Public-site facts that are still unresolved

- Detailed judging weights and extra award categories are not yet published.
- Cash split among first/second/third and additional prize details are pending; the page lists a **$5,000 cash pool**, RTX 5080, RTX 5070, and sponsor swag.
- Dataset 3 asset release/access details are pending on the public page.
- DGX Spark/GB10 and CSI workstation access/reservations have incomplete public instructions.
- A held-out real test set for Datasets 1/2 is mentioned as possible, not guaranteed.

These should be watched in Slack rather than inferred.

## Authenticated blind spots

The public crawl cannot see content behind login or invitation walls. A complete operational audit still requires a signed-in review of:

- Slack announcements, pins, files, and dataset-channel updates;
- ALL AI Network team status, plan draft, organizer messages, and final-submission form;
- Google Drive folder contents, timestamps, and recently added assets;
- GitHub invitation/team repository status;
- Brev organization, credits, and launchable availability;
- campus/CSI booking systems and released Dataset 3 files.

These are the most likely places for new information that the public MAIC pages do not yet contain.

## Official source pages

- [Event overview and requirements](https://msoe-ai-club.github.io/maic-content/competitions/Innovation%20Labs/Rockwell-NVIDIA-Fall-2026/Rockwell-NVIDIA-Fall-2026.md)
- [Dataset 1: Final Vial Inspection](https://msoe-ai-club.github.io/maic-content/unlisted/Rockwell-NVIDIA-Fall-2026-Vial-Inspection/Rockwell-NVIDIA-Fall-2026-Vial-Inspection.md)
- [Dataset 2: Fluid Mixing Station](https://msoe-ai-club.github.io/maic-content/unlisted/Rockwell-NVIDIA-Fall-2026-Fluid-Mixing/Rockwell-NVIDIA-Fall-2026-Fluid-Mixing.md)
- [Dataset 3: Grasping and Sorting](https://msoe-ai-club.github.io/maic-content/unlisted/Rockwell-NVIDIA-Fall-2026-Grasping-Sorting/Rockwell-NVIDIA-Fall-2026-Grasping-Sorting.md)
- [Compute guide](https://msoe-ai-club.github.io/maic-content/unlisted/Rockwell-NVIDIA-Fall-2026-Compute/Rockwell-NVIDIA-Fall-2026-Compute.md)
- [Data collection and physical testing](https://msoe-ai-club.github.io/maic-content/unlisted/Rockwell-NVIDIA-Fall-2026-Data-Collection/Rockwell-NVIDIA-Fall-2026-Data-Collection.md)
- [Coding agents](https://msoe-ai-club.github.io/maic-content/unlisted/Rockwell-NVIDIA-Fall-2026-Coding-Agents/Rockwell-NVIDIA-Fall-2026-Coding-Agents.md)
- [Tools and resources](https://msoe-ai-club.github.io/maic-content/unlisted/Rockwell-NVIDIA-Fall-2026-Resources/Rockwell-NVIDIA-Fall-2026-Resources.md)
