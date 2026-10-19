# The A-Team Implementation Brief

## Recommended entry

**The Adaptive Workcell: Show the Job, Generate the Tool**

Demonstrate a short, bounded manipulation task once with video and narration. Convert it into a typed task graph, execute it on an unseen arrangement, verify every stage, and show that the workcell can identify when its current fingers are unsuitable. For one approved object, generate and validate a printable finger design with CADVector, install it between approved runs, and prove that it improves repeated physical grasp success.

## What to build first

1. A geometry-first TCP policy using table RGB, wrist RGB/depth, and the supplied calibration.
2. Four deterministic skills: observe, pick, place, and verify.
3. A task-graph representation that refers to object relationships rather than replayed coordinates.
4. A bounded recovery policy: re-observe, retry once, choose another eligible object, or stop safely.
5. A capability check that compares object geometry with current jaw stroke, contact regions, and approach clearance.
6. One CADVector finger design constrained to the real SMC mounting interface.

## Do not put on the critical path

- arbitrary gearbox assembly;
- reconstructing exact CAD for every unseen object from one camera;
- online 3D printing during a rollout;
- an automatic magnetic/pneumatic tool changer without organizer approval;
- unrestricted natural-language action generation; or
- long open-loop motion sequences near contact.

## Physical evaluation

Use ten randomized layouts per condition and compare:

| Condition | Purpose |
|---|---|
| Scripted coordinate replay | Weak baseline |
| Interpreted task graph, no stage verification | Tests semantic generalization |
| Task graph with re-observation | Tests closed-loop value |
| Task graph with re-observation and bounded retry | Tests recovery value |
| Existing finger vs generated finger | Tests Toolsmith's physical value |

Primary metric: complete-rollout success. Secondary metrics: pickup success, slip/drop rate, placement error, recovery success, controller faults, interventions, cycle time, inference latency, and clarification/refusal correctness.

## Platform settings to start with

```python
CAMERAS = ["table", "wrist"]
IMAGE_SIZE = (640, 360)
IMAGE_FORMAT = "jpeg"
ACTION = "tcp"
DEPTH = ["wrist"]
PREFETCH = False
```

Use the side camera only when its viewpoint materially improves verification. Keep prefetch off until action chunks are long enough and safe enough to benefit from it. Save depth locally because the official recorded dataset omits it.

## Organizer request wording

> The A-Team requests a supervised fit-and-clearance check for one 3D-printed finger pair designed for the SMC MHZL2-16D2 and a later 30-minute evaluated run. The fingers will use the measured mounting interface and declared fasteners; we will provide CAD, material, dimensions, mass, maximum protrusion, and collision-clearance evidence before installation. The evaluated task will use organizer-approved objects and ten declared placements. No automatic tool-changing mechanism will be used unless separately reviewed and approved.

## Definition of a compelling demo

The audience sees a human demonstrate and narrate a simple task. The system shows the inferred task graph. Objects are rearranged. The robot executes the intended relationships, notices one failed or ambiguous stage, recovers or asks for help, and verifies completion. A second segment shows that the original fingers fail repeatedly on a deliberately chosen object while the generated, approved fingers succeed measurably more often.

That story demonstrates contextual learning, physical AI, engineering discipline, and manufacturing changeover value without claiming arbitrary manipulation.
