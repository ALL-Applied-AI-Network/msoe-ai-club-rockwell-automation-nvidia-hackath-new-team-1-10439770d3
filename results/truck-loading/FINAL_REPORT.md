# Truck loading benchmark — final vision-recovery controller

## Outcome

`vision_recovery_v3` completed all six final Isaac Sim physics episodes. Every episode ended with all 12 cubes in their matching bins and all three loaded bins on the truck deck.

| Metric | Previous `vision_recovery` | Final `vision_recovery_v3` | Change |
|---|---:|---:|---:|
| Full-task success | 3/6 (50.0%) | 6/6 (100.0%) | +50 percentage points |
| Matching cubes | 11.50/12 mean | 12.00/12 mean | +0.50 |
| Bins on deck | 3.00/3 mean | 3.00/3 mean | unchanged |
| Balanced score | 0.5630 | 0.7921 | +0.2291 |
| Mean perception latency | 274.8 ms | 1066.2 ms | +791.4 ms |
| Mean action count | 157.3 | 174.0 | +16.7 |
| Mean TCP path | 12.48 m | 12.81 m | +0.34 m |
| Mean simulated motion wall time | 22.1 s | 28.9 s | +6.8 s |

The reliability improvement is deliberately purchased with additional observation, proof-lift, settling, and recovery actions. Efficiency fell to 0.4213, but full-task reliability and the balanced score increased materially.

## Final test matrix

The final clean matrix used two independently randomized seeds for each of three scene variations:

The values `20261117` and `20261118` are deterministic numeric RNG seeds, not execution dates.

| Variation | Seed | Matching cubes | Bins on deck | Full success |
|---|---:|---:|---:|---:|
| combined | 20261117 | 12/12 | 3/3 | yes |
| combined | 20261118 | 12/12 | 3/3 | yes |
| depth noise | 20261117 | 12/12 | 3/3 | yes |
| depth noise | 20261118 | 12/12 | 3/3 | yes |
| nominal | 20261117 | 12/12 | 3/3 | yes |
| nominal | 20261118 | 12/12 | 3/3 | yes |

This is a clean rerun of the exact final source. The earlier targeted runs used to tune the controller are not counted in the 6/6 result.

## Experimental boundary

The controller received RGB, depth, reported camera calibration, joint/gripper state, and timestamps. It did not receive simulator object transforms. Object transforms were read only by the evaluator after the final retreat to compute the score. Initial randomized object transforms were also evaluator-only.

Raw observations, inferred plans, action sequences, jaw-gap checks, timestamps, recovery diagnostics, and final evaluator measurements are preserved in the full evidence archive. The `boundary` object in each `execution_report.json` records the observation-only contract.

## What was tested before the final controller

The perception benchmark compared fixed-calibration RGB-D, depth-anchor and hue/model hybrids, a task-specific oblique hybrid, and RGB-only model fitting across 24 randomized observation scenes. These covered nominal appearance, photometric change, depth noise, calibration drift, clutter/occlusion, and combined disturbances. The best perception-only hybrid reached 12/24 full scenes and a 0.6780 ranking score; the selected fast RGB-D policy reached 11/24 and 0.6617. This showed that no single detector was reliable enough to own the complete task.

The physics benchmark then compared direct frozen plans, camera replanning, and progressively stronger recovery controllers. The former leader, `vision_recovery`, completed 3/6 physics episodes. Its three exact failures were replayed while changes were developed:

- one red cube remained outside its bin under depth noise;
- one orange cube remained outside its bin in a combined scene;
- one blue cube spilled from a tightly loaded bin during transport.

The first two were corrected by bounded residual search and a second grasp orientation. Tighter packing initially made the third failure worse: four cubes increased payload/contact sensitivity, and a delayed recovery into a bottom-layer slot could push an existing cube out. The final design therefore combines perception recovery with contact-aware transport and recovery-aware packing.

## Final control method

### 1. Multi-detector residual search

The primary calibrated RGB-D detector establishes metric poses. A raw RGB hue/model-fit detector is used only as a bounded residual proposal source. RGB proposals are admitted only when they satisfy task inventory, color, workspace, and duplicate-distance gates. This keeps RGB useful when depth fragments or misses a cube without allowing an unconstrained contour to overwrite a sound metric estimate.

### 2. Camera re-observation and two-pass cube recovery

After the first cube-placement pass, the arm retreats to clear the cameras and captures a new observation. Detected cubes still outside the matching bin are eligible for one bounded recovery. If contact is not proven, the controller captures again and retries once with the grasp axis rotated 90 degrees. The final run records each proposal, jaw target, measured contact gap, destination slot, and outcome.

### 3. Recovery-aware packing

The nominal four-cube pattern was tightened to ±6.25 mm in the bin frame. A late recovered cube is never driven into an already occupied bottom slot; it is redirected to the top layer. This change fixed the nominal scene where bottom-slot insertion destabilized an otherwise successful stack.

### 4. Payload settling

After cube recovery, the controller holds a two-second payload-settle interval before attempting bin transport. This reduced cube motion caused by immediately transitioning from recovery to a wall grasp.

### 5. Loaded-bin transport guard

Loaded bins use a deeper 20 mm wall insertion, slow close, 10 mm micro-lift, proof dwell, staged raise, and jaw-gap checks after close, proof, and raise. A contact loss aborts transport rather than carrying an ungripped or slipping bin. One camera re-centering retry is available if the proof lift does not preserve contact.

### 6. Centre-preserving wall pivot

The TCP follows the contacted wall's circular locus while the bin rotates about its centre. Rotating the wrist in place would sweep the bin centre and can drag or tip a loaded bin. The centre-preserving arc made the successful one-wall grasp usable for truck-aligned placement.

### 7. Bounded final spill recovery

The final camera observation can trigger a single spill recovery only when the same color already has camera-confirmed recovery history and the observed bin is close to its commanded truck zone. RGB confirms that a residual cube exists; the calibrated commanded deck zone and yaw define the insertion frame. This distinction fixed an earlier attempt in which an approximately 20 mm RGB contour-center bias placed a cube on the bin rim.

## Why RGB-only and SAM 3 were not selected as the primary controller

RGB-only fitting was useful as a recovery proposal generator but was less reliable as a metric pose owner across color, lighting, occlusion, and calibration variations. It has no direct metric depth and therefore still depends on camera calibration, plane/size assumptions, or another ranging source.

SAM 3 is a segmentation/foundation model, not a complete robot pose-and-control system. It could improve instance masks in a future detector, especially for unfamiliar parts, but masks still need color/identity association, metric 3D pose, grasp-axis estimation, calibration, temporal gating, and contact verification. The official checkpoint is access-gated, so it was not silently substituted into this reproducible benchmark. The final hybrid instead uses the available calibrated RGB-D evidence for metric pose and RGB only as a tightly gated fallback.

## Reproducibility artifacts

- `report/physics_results.json`: aggregate and per-episode results.
- `report/physics_ranking.csv`: machine-readable final ranking.
- `report/PHYSICS_REPORT.md`: generated summary.
- `episodes/*/execution_report.json`: complete action, recovery, timing, grip-check, and evaluator record.
- `episodes/*/observations/`: raw RGB, depth, calibration, and observation metadata for each camera refresh (full archive only).
- `truck_benchmark/execute_physics_finalist.py`: exact final controller source in the source archive/workspace.

## Limits and transfer requirements

Six of six randomized simulation episodes is evidence of improvement, not a physical guarantee. The real cell still requires verified camera extrinsics, TCP/tool geometry, the measured tapered finger profile, jaw calibration, physical bin/cube dimensions, friction/compliance validation, speed/force limits, collision review, and supervised low-height proof lifts. The simulator also cannot fully reproduce pneumatic response, printed-surface variation, camera exposure, structural flex, or every contact mode.

The appropriate next physical step is therefore a supervised calibration and proof-lift protocol, not an unsupervised full truck-loading run.
