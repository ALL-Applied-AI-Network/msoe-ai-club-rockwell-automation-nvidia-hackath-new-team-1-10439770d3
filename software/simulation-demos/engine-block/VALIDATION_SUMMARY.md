# Inline-Four Engine — Isaac Sim Validation Summary

Status: **PASSED — MEASURED ISAAC SIM RUN**  
Evidence scope: Isaac Sim assembly evidence only; this document does not authorize or claim physical execution.

## Objective

Validate a vertical, robot-oriented assembly of the supplied inline-four engine geometry with every movable item represented as a separate part. The four connecting rods and four pistons are independently staged, grasped, transported, and placed.

## Exact geometry and provenance

- Source: the supplied 3MF files now retained under [`cad/engine`](../../../cad/engine/).
- Coordinate frame: shared final-assembly frame reconstructed from the supplied assembled and exploded exports.
- Engine envelope: **132.379996 × 58.0 × 76.99 mm**.
- Twelve simulation assets are exported as individual watertight binary STLs:
  - `block.stl`
  - `crankshaft.stl`
  - `rod_1.stl` through `rod_4.stl`
  - `cover.stl`
  - `head.stl`
  - `piston_1.stl` through `piston_4.stl`
- Exact hashes, colors, bounds, volumes, source transforms, and cylinder mappings are recorded in `assets/asset_manifest.json`.
- The cover export excludes four zero-volume two-triangle artifacts from the source while retaining its complete watertight body.

## Validated assembly order

1. Engine block
2. Crankshaft
3. Connecting rod 1
4. Connecting rod 2
5. Connecting rod 3
6. Connecting rod 4
7. Drop-on cover
8. Cylinder head
9. Piston 1
10. Piston 2
11. Piston 3
12. Piston 4

This order installs the rods before the head so the complete gripper fingers do not need to enter deeply through the head after it is present. The cover and head subsequently drop over the installed rods, followed by the four independent pistons.

## Validation model and claim boundary

- Robot: FANUC ER-4iA with actual inverse-kinematics solutions and simulated articulation motion.
- Motion: top-down approach, vertical engagement, proof lift, guarded transport, vertical placement, release, and vertical retreat.
- Part carriage: kinematic attachment through the selected grasp transform while the jaws are commanded closed.
- This can validate:
  - reachability and IK continuity;
  - individual-part sequencing;
  - monotonic approach and retreat geometry;
  - vertical finger corridors;
  - transport trajectories;
  - final part transforms and assembly ordering;
  - rendered visual evidence from three camera views.
- This does **not** validate:
  - physical gripping force or friction;
  - FDM tolerance variation;
  - compliance, deformation, or insertion force;
  - real-camera perception;
  - a collision-free physical run without a final hardware dry-run.

## Grasp-interface caveat

The crankshaft, rods, and pistons contain usable robot-oriented handling geometry. The supplied block, cover, and cylinder-head meshes do **not** contain the advertised standardized 16 × 8 × 14 mm grasp fins. Their present simulation plan therefore uses documented provisional edge grasps:

- block: end-wall straddle;
- cover: external edge grasp;
- cylinder head: external edge/web grasp.

Those three placements may demonstrate simulated reach and geometric transport, but they must not be presented as grip-force-validated. Before physical execution, add dedicated standardized fins or use purpose-built fixtures and repeat clearance/contact validation.

## Acceptance criteria

The run is accepted only if all of the following are measured and pass:

| Check | Acceptance threshold | Measured result |
|---|---:|---:|
| Distinct transported parts | 12 of 12 | **12 of 12 — PASS** |
| IK failures | 0 | **0 — PASS** |
| Completed proof lifts | 12 of 12 | **12 of 12 — PASS** |
| Minimum proof-lift height | ≥ 18 mm | **19.9658 mm — PASS** |
| Final translation error, each part | ≤ 0.10 mm | **max 0.0000146 mm — PASS** |
| Final rotation error, each part | ≤ 0.10° | **max 0.000° — PASS** |
| Unsafe/unbounded motion | none | **none observed — PASS** |
| Rendered camera streams | 3 complete views | **3 × 786 frames — PASS** |
| Fatal runtime errors | 0 | **0 — PASS** |

## Measured run results

- Run identifier: **20261028T211929Z-render**
- Isaac Sim version: **6.0.1 application stack / Isaac Sim 6 runtime**
- Runtime mode: **headless physics + synchronized RGB rendering**
- Task complete: **true**
- Acceptance passed: **true**
- Total duration: **78.6 s**
- Total frames: **786 per camera, 2,358 rendered frames total**
- IK failures: **0**
- Minimum/maximum proof lift: **19.9658 / 19.9890 mm**
- Maximum final translation error: **0.0000146 mm**
- Maximum final rotation error: **0.000°**
- Runtime warnings affecting validity: **none; the missing optional mat texture warning did not affect geometry, motion, or rendered evidence**
- Report path: **`execution_report.json`**
- Log path: **`run.log`**

### Per-part evidence

| Order | Part | Cylinder | IK | Proof lift | Translation error | Rotation error | Result |
|---:|---|:---:|:---:|---:|---:|---:|:---:|
| 1 | block | — | pass | 19.9845 mm | 0.0000146 mm | 0.000° | **PASS*** |
| 2 | crankshaft | — | pass | 19.9886 mm | 0.0000146 mm | 0.000° | **PASS** |
| 3 | rod_1 | 1 | pass | 19.9746 mm | 0.0000146 mm | 0.000° | **PASS** |
| 4 | rod_2 | 2 | pass | 19.9808 mm | 0.0000146 mm | 0.000° | **PASS** |
| 5 | rod_3 | 3 | pass | 19.9848 mm | 0.0000146 mm | 0.000° | **PASS** |
| 6 | rod_4 | 4 | pass | 19.9873 mm | 0.0000146 mm | 0.000° | **PASS** |
| 7 | cover | — | pass | 19.9718 mm | 0.0000146 mm | 0.000° | **PASS*** |
| 8 | head | — | pass | 19.9658 mm | 0.0000146 mm | 0.000° | **PASS*** |
| 9 | piston_1 | 1 | pass | 19.9767 mm | 0.0000146 mm | 0.000° | **PASS** |
| 10 | piston_2 | 2 | pass | 19.9825 mm | 0.0000146 mm | 0.000° | **PASS** |
| 11 | piston_3 | 3 | pass | 19.9863 mm | 0.0000146 mm | 0.000° | **PASS** |
| 12 | piston_4 | 4 | pass | 19.9890 mm | 0.0000146 mm | 0.000° | **PASS** |

\* Simulation reach/trajectory/final-pose pass using a provisional edge or wall grasp; not a physical grip-force certification.

## Expected deliverables

- Physics-only Isaac run report (`report.json`) and full runtime log.
- Three rendered frame sequences:
  - hero view;
  - assembly close-up;
  - opposite oblique view.
- Three individual H.264 videos and one synchronized three-view composite.
- Exact twelve-part asset manifest and individual STL exports.
- Contact sheets or representative frames used for visual QA.
- This validation summary updated with measured values, with caveats retained.

## Physical-execution gate

Do not transfer this trajectory directly to hardware solely because the Isaac acceptance checks pass. Before a physical run:

1. Add standardized grasp fins or validated fixtures to the block, cover, and head.
2. Measure the printed interfaces and update clearances from the actual parts.
3. Register the real table and assembly fixture to the robot base frame.
4. Dry-run each approach at reduced speed with the gripper open and a conservative Z offset.
5. Validate jaw engagement, proof lifts, insertion clearance, and stop/recovery behavior one part at a time.

