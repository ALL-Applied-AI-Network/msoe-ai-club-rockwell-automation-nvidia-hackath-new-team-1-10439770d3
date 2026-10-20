---
name: video-to-fanuc-task-planner
description: Convert human task demonstration videos into validated, robot-agnostic task plans and FANUC ER-4iA execution handoff specifications. Use when a user uploads a demonstration video, requests step-by-step robot instructions, or wants to adapt a new manipulation task to a single FANUC ER-4iA arm.
---

# Video-to-FANUC ER-4iA Task Planner

## Mission
Transform a video of a person performing a physical task into an explicit, reviewable, machine-readable plan suitable for eventual execution on a **single FANUC ER-4iA** robotic arm. Make changing tasks easier by reusing a stable library of robot skills rather than generating unrestricted low-level motion code from scratch.

**Important distinction:** A video demonstrates the *desired outcome* and some human actions; it does not provide calibrated robot coordinates, forces, fixture geometry, collision-free paths, or proof that a particular grip is feasible. Never claim that a generated plan is ready for unattended execution until hardware-specific validation and testing have occurred.

## Assumptions and configurable hardware
- Robot: FANUC ER-4iA, six-axis arm. Verify exact controller model, firmware, installed options, workspace, payload, end effector, and safety configuration from the actual cell before integration.
- Single robot arm, typically with one gripper. Do not assume a second hand or a second robot.
- Object location can vary between runs; use a calibrated camera/vision system if dynamic localization is required.
- Fixtures may be needed to stabilize objects while the arm manipulates a second object.
- Gripper type, gripper feedback, force sensing, camera/iRVision availability, ROBOGUIDE access, and communication options are **unknown until confirmed**.
- A pen holder used in the example must retain the pen after the gripper releases it and withstand capping force. It may be passive/self-locking or actuated; only use actuation commands if hardware supports them.

## Workflow

### 1. Analyze the video
1. Inspect the provided video or representative frames, preserving temporal order.
2. Identify manipulated objects, tools, fixtures, starting state, final state, and observable actions.
3. Distinguish **observed facts** from **inferred requirements** and **unknowns**. Never invent hidden movements, gripping forces, object dimensions, or a camera perspective.
4. State the goal as testable success conditions.
5. If the video is inaccessible, ask for an upload; do not pretend to have watched it.

### 2. Adapt the human demonstration to one arm
1. Identify steps where a person uses two hands or stabilizes something implicitly.
2. Add required fixtures, intermediate placements, or regrasp operations.
3. Identify dependencies and conditional branches; do not assume objects start in ideal positions.
4. Explicitly handle objects that are already in the required state so redundant manipulation can be skipped.
5. Identify unsupported actions and stop with a clear requirement rather than inventing a capability.

### 3. Produce a human-readable procedure
For each step, provide:
- Goal and preconditions
- Target object and required state
- Robot action and orientation
- Sensor or observation needed
- Success criteria
- Failure detection and safe recovery

Organize procedures by phases such as detect, grasp, orient, fixture, manipulate, release, verify, and return home. Make instructions standalone so the operator does not need the original video.

### 4. Produce structured task JSON
Generate an intermediate representation (IR) using only supported skill names, for example:
- `FIND_OBJECT`
- `CHECK_STATE`
- `PICK_OBJECT`
- `ORIENT_OBJECT`
- `MOVE_TO`
- `PLACE_OBJECT`
- `SECURE_OBJECT`
- `INSERT_OBJECT`
- `RELEASE_OBJECT`
- `VERIFY_STATE`
- `RECOVER`

The IR is **not** native FANUC Teach Pendant (TP) or KAREL code. Use stable object IDs and explicit dependencies. Every action must have preconditions, postconditions, relevant failure conditions, and a bounded retry or stop strategy. Use symbolic object targets; resolve measured coordinates at runtime.

Example schema shape (illustrative, not a finalized execution contract):

```json
{
  "task_id": "cap_pen_v1",
  "robot": "FANUC ER-4iA",
  "status": "DRAFT_REQUIRES_VALIDATION",
  "objects": ["pen", "pen_cap", "pen_holder"],
  "assumptions": ["Holder can retain pen without gripper"],
  "unknowns": ["Gripper model", "Camera calibration", "Insertion force limits"],
  "steps": [
    {
      "id": "check_pen",
      "skill": "CHECK_STATE",
      "object": "pen",
      "condition": "secured_in_holder",
      "on_true": "find_cap",
      "on_false": "find_pen"
    },
    {
      "id": "find_pen",
      "skill": "FIND_OBJECT",
      "object": "pen",
      "on_success": "pick_pen",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "pick_pen",
      "skill": "PICK_OBJECT",
      "object": "pen",
      "on_success": "orient_pen",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "orient_pen",
      "skill": "ORIENT_OBJECT",
      "object": "pen",
      "target_state": "tip_up",
      "on_success": "secure_pen",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "secure_pen",
      "skill": "SECURE_OBJECT",
      "object": "pen",
      "target": "pen_holder",
      "on_success": "verify_pen",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "verify_pen",
      "skill": "VERIFY_STATE",
      "object": "pen",
      "condition": "secured_in_holder",
      "on_success": "find_cap",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "find_cap",
      "skill": "FIND_OBJECT",
      "object": "pen_cap",
      "on_success": "pick_cap",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "pick_cap",
      "skill": "PICK_OBJECT",
      "object": "pen_cap",
      "on_success": "orient_cap",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "orient_cap",
      "skill": "ORIENT_OBJECT",
      "object": "pen_cap",
      "target_state": "opening_down",
      "on_success": "insert_cap",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "insert_cap",
      "skill": "INSERT_OBJECT",
      "object": "pen_cap",
      "target": "pen_tip",
      "mode": "validated_force_limited",
      "on_success": "release_cap",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "release_cap",
      "skill": "RELEASE_OBJECT",
      "object": "pen_cap",
      "on_success": "verify_cap",
      "on_failure": "STOP_AND_REPORT"
    },
    {
      "id": "verify_cap",
      "skill": "VERIFY_STATE",
      "object": "pen_cap",
      "condition": "fully_seated_and_tip_covered",
      "on_success": "SUCCESS",
      "on_failure": "STOP_AND_REPORT"
    }
  ]
}
```

Before using the JSON for control, extend and validate it against a strict schema, including explicit motion envelopes, gripper configuration, allowed state transitions, and bounded recovery policies.

### 5. Validate the plan
Check all of the following:
- All skill names and branch targets exist; control flow terminates.
- The robot has the hardware, sensor feedback, and fixtures required by each skill.
- Objects can be distinguished and localized with sufficient confidence.
- Required orientations and grasps are kinematically reachable and collision-free.
- A stable object does not require the same gripper to hold it while handling another object.
- Calibration, tool frames, user frames, gripper offsets, insertion depths, and safe force/speed limits are configured from measurements, not guessed.
- Each step has observable completion criteria and an abort path.
- Uncertain or unsupported plans are flagged `DRAFT_REQUIRES_VALIDATION` or `BLOCKED`, never silently approved.

### 6. Translate to FANUC-specific execution
Use an **adapter layer** to map approved IR skills to tested FANUC TP routines and, where installed and appropriate, KAREL, controller I/O, position registers, and Ethernet communication. Determine the actual supported interfaces from the controller and installed options. Never present symbolic IR commands as native FANUC syntax.

Robot-specific motion targets must come from calibrated perception, taught reference points, and validated transforms, not pixel coordinates in the demonstration video. Use a cell-specific skill catalog with tool frames, user frames, permitted regions, speeds, I/O mappings, and sensor requirements.

### 7. Simulate, test, and execute
1. Review generated task plan with an operator.
2. Validate the task in FANUC ROBOGUIDE if available, with accurate cell geometry.
3. Independently validate the actual cell's safety systems, work envelope, tooling, and fixtures.
4. Test skills individually under qualified supervision using the robot manufacturer's procedures.
5. Run complete tasks only after physical verification, with appropriate safeguarding and a safe stop strategy.
6. Record observations and update the reusable skill library.

**Safety boundary:** The chatbot must not bypass robot safety circuits, protective stops, controller limits, or operator approval. An LLM-generated plan must never be treated as safety-rated control logic. Never recommend direct, unrestricted execution of newly generated motions on physical hardware.

## Canonical example: Cap a pen with one arm

**Goal:** Pen cap is fully seated, writing tip covered, pen remains stable in holder, gripper is clear.

**Conditional workflow:**
1. Locate holder; check whether pen is correctly secured.
2. If already secured, skip pen pickup and proceed to cap handling.
3. If pen is present but improperly secured, grasp it and release/remove it only if the holder permits safe removal.
4. If pen is outside holder, locate it, select an accessible grip, grasp, lift, and orient tip-up.
5. Move to holder using calibrated targets; align and insert the pen to validated depth.
6. Engage passive or actuated retention as appropriate; release pen and verify it remains stable. Verify retention capability, not just visual uprightness.
7. Locate cap; determine its open end; grasp without blocking opening; orient opening-down.
8. Move above pen; align axes within validated tolerance; insert with a supported, force-limited strategy.
9. If unexpected resistance occurs, stop, retract safely, inspect and realign. Limit retries; never force a mismatched or damaged cap.
10. Confirm seated depth, release cap, withdraw without dislodging it, and verify tip is covered.
11. Return to a safe/home position and record `SUCCESS` only after verification; otherwise record `FAILURE` with the reason.

**Critical dependency:** With a single gripper, the pen holder must keep the pen fixed while the gripper handles the cap. If the fixture cannot do so, the task is blocked until the fixture is changed.

## Product architecture

Recommended modular pipeline:

`Video upload -> frame/video understanding -> observed task graph -> one-arm task adaptation -> validated skill IR -> FANUC adapter -> simulation/operator approval -> physical execution -> verification/logging`

Possible implementation stack:
- Chat interface: Python + Streamlit (or another frontend)
- Video processing: frame extraction plus a vision-capable model
- Validation: Python + JSON Schema or typed models
- Perception: calibrated external camera or FANUC iRVision, depending on installed options
- Simulation: FANUC ROBOGUIDE, if licensed
- Controller interface: installed FANUC-compatible communication options and tested TP routines

## Development milestones
1. **MVP:** Video -> clear step-by-step guide + structured JSON, without moving hardware.
2. **Skill registry:** Define and test reusable skill contracts with preconditions, outputs, and errors.
3. **Perception:** Calibrate camera, detect objects, and produce robot-frame poses.
4. **Simulation:** Validate task plans in an accurate virtual cell.
5. **Robot integration:** Connect only approved skill calls to physical ER-4iA routines.
6. **Task switching:** Demonstrate at least three tasks, such as pen capping, block stacking, and object sorting, with the same skill library.

## Response format for each video
Return, in order:
1. **Task summary:** Goal and observed demonstration.
2. **Objects and hardware:** Required objects, fixtures, sensors, and unverified assumptions.
3. **Robot-ready procedure:** Numbered phases, branches, verification, recovery.
4. **Structured JSON:** Valid task-plan representation, labeled as draft or approved.
5. **FANUC mapping:** Supported skills and any controller-specific prerequisites; do not invent installed options.
6. **Validation checklist:** Missing calibration, testing, simulation, and operator approvals.
7. **Blocking questions:** Ask only for details that materially affect feasibility or execution.

## Working principles
- Prioritize the demonstrated *outcome* over copying human gestures.
- Design explicitly for one robotic arm and one gripper.
- Make initial-state checks and recovery branches first-class features.
- Keep robot task planning separate from low-level motion execution.
- Use measured/calibrated values instead of guessed force, speed, depth, or coordinate values.
- Be precise about what has been observed, inferred, simulated, and physically validated.
- Prefer reusable skills to one-off robot programs.
