# Experiment 3 | Robot Trial Analysis

**Rockwell Automation × NVIDIA Hackathon · The A-Team**<br>
Initial evidence review and plan for measuring robot reliability

> **Status: Draft for team review**<br>
> Evidence reviewed: October 22, 2026 · Repository reference: `27e027a`<br>
> Analysis: descriptive statistics and structured video review

## Overview

This package establishes a documented baseline for evaluating the FANUC robot's recorded behavior. It connects observed actions to episode records, distinguishes robot-performed outcomes from human-assisted outcomes, and defines the information needed for repeatable follow-up tests.

The recordings show examples of cube stacking and screw transfers, alongside interrupted attempts, human interventions and camera gaps. These observations guide the next experiments; the available mix of tasks and conditions does not yet establish overall robot reliability.

## Review the deliverables

| Deliverable | Contents | Use |
| :--- | :--- | :--- |
| **[Trial analysis report](Robot_Data_and_Video_Review_1.docx)** | Recorded behavior, findings, evidence limitations, proposed tests and website sample provenance | Understand what the evidence supports |
| **[Attempt analysis workbook](Robot_Attempt_Analysis_1.xlsx)** | Episode log, grasp events, findings, separate commissioning evidence and blank future-run records | Inspect individual records and capture the next trials |

Download the files to view the complete Word report and Excel workbook. The workbook includes six tabs: **Findings and next tests**, **Episode log**, **Grasp events**, **Next runs**, **Guide and sources**, and **Repository evidence**.

## Evidence at a glance

| Evidence reviewed | Coverage |
| :--- | ---: |
| Recorded robot episodes | **17** |
| Telemetry rows | **14,442** |
| Camera streams | **51** |
| Decoded camera frames | **43,326** |
| Phone clips of setup or human demonstrations | **7** |

The phone demonstrations are separate from the recorded robot episodes. Multiple camera views of an episode are evidence of the same attempt, not additional trials.

## Principal findings

| Finding | Supporting observation | Implication for the next run |
| :--- | :--- | :--- |
| **Robot-performed stacking is visible** | Blue-on-red stacks remain standing after retreat in `d41abe` episode 2 and `ecd8ec` episode 3 | Repeat a fixed stacking task to measure consistency |
| **Screw transfers show both completion and recovery** | In `d41abe` episode 0, later cycles deposit both initial screws after earlier unsuccessful cycles | Record first-attempt success separately from completion after retries |
| **Human assistance changes the interpretation** | Human handling contributes to outcomes in `ecd8ec` episode 1 and `d41abe` episode 4 | Record intervention timestamps and report assisted outcomes separately |
| **Some outcomes remain unverified** | The green-bin lift is unproven; the reported yellow-cube pickup/drop was not identified | Capture clear final-state footage and confirm the relevant source files |
| **Recording quality affects what can be concluded** | Some camera streams are black, and two recordings are stationary | Check all camera views and telemetry before each test |

Short run identifiers in this table refer to the full run IDs recorded in the report and workbook. “Robot-performed” describes visible behavior; it does not by itself establish autonomous policy control.

## Source boundaries and traceability

- **Archive provenance:** The supplied run IDs are dated October 19–20. Exact session ownership and executed code/settings remain unverified for some episodes. Sharing a file with the team does not establish who executed the run.
- **Commissioning evidence:** Run `r20261020-5fea53` is separately reported in the repository. Its two recorded trials received official success labels, but the A-to-B leg did not execute. Its raw files were absent from the supplied archives, so its metrics remain source-reported context.
- **Website sample:** `ourwebsite/public/assets/testbed-demo.mp4` is a shortened, re-encoded copy of the first 30 seconds of `r20261019-d41abe`, episode 2, trial `t02`, side-camera `file-002.mp4`. It is already counted in the analysis.
- **Version scope:** The report includes website changes through `27e027a`; the workbook retains its earlier repository audit reference. Website prototype outputs are not additional measured robot trial results.

## What kind of analysis this is

We use **descriptive statistical analysis** to summarize counts, durations, event timing and recording coverage, supported by structured review of the videos. This describes the evidence available and identifies questions for the next experiments.

A single success percentage across these different tasks, settings and assistance conditions would be misleading. The current package does not estimate the probability of success on future trials. The reliability plan below supplements the selected report.

## Plan for measuring reliability

### 1. Define one repeatable task

Agree on the starting layout, time limit and success criteria before testing. For example, a cube-placement task could require the cube to remain fully inside the bin for three seconds after release and retreat, without human repositioning. This is a proposed criterion for team agreement.

### 2. Capture every scheduled trial

| Record | Why it matters |
| :--- | :--- |
| Run, episode, trial and target-object IDs | Connect each observation to its source |
| Actual robot-computer commit, local edits, policy and effective settings | Identify exactly what was tested |
| Starting layout and relevant operating conditions | Make comparisons fair and reproducible |
| Attempts, retries, outcome and elapsed time | Separate first-attempt success from eventual completion |
| Human assistance, faults, timeouts and abort reasons | Preserve incomplete and unsuccessful results |
| Synchronized camera footage, telemetry, alarms and final state | Allow independent checking of outcomes |

Keep all scheduled trials in the inventory and explain exclusions. Use the workbook's **Next runs** tab to capture the records and **Guide and sources** for field explanations.

### 3. Repeat under consistent conditions

Begin with a fixed task and layout, then evaluate other layouts and sessions deliberately. Five trials can serve as a pilot to check the procedure; they do not establish reliability. Agree on the required reliability and precision before choosing a larger test size.

Compare code versions under matched conditions and interleave them when practical. Repeated attempts within one trial are linked observations and should not be treated as independent trials.

### 4. Report results with their limits

For each task and code/settings group, report:

- Successes out of eligible trials, with an uncertainty interval.
- First-attempt success and eventual completion after retries.
- Retry, assistance and fault counts.
- Median completion time and its spread, with timeouts reported separately.
- Sample size, test conditions and the reasons for any exclusions.

Reliability conclusions apply to the conditions tested. More varied tasks require additional evidence.

## Requested review

Before merging, please review:

- [ ] Source ownership and session mapping.
- [ ] Episode outcomes and human-assistance labels.
- [ ] Available mappings to executed code versions and settings.
- [ ] Next-test success criteria, time limits and recording requirements.

**Review objective:** establish a shared baseline and an agreed data-collection process for the next robot session.
