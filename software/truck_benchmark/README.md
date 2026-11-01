# Truck observation-only autonomy benchmark

This suite measures autonomous perception and planning for the measured truck task without simulator-pose shortcuts.

## Policy boundary

Policies receive only:

- table, side, and wrist RGB images;
- table, side, and wrist distance-to-image-plane depth;
- reported camera intrinsics/extrinsics;
- joint positions, gripper state, and timestamps;
- the task specification.

They do **not** receive USD prims, semantic/instance annotations, simulator object transforms, scene-state files, or evaluator labels. `contract.py` rejects simulator imports and pose/label symbols in policy code. Isaac writes observations and labels to separate directories; `run_policies.py` is launched against only the observation directory.

Ground truth is allowed in exactly two places: randomized scene construction and offline scoring.

## Candidates

The registry compares 48 candidates:

- fixed-layout open-loop baseline;
- three fixed-HSV RGB contour variants;
- three adaptive-chromaticity RGB variants;
- three table RGB-D height thresholds;
- multi-view table+side RGB-D;
- full three-camera table+side+wrist depth and depth-anchored fusion;
- active wrist-scan selective recovery that fills only missing cube slots and
  never overwrites a trusted fixed-camera pose;
- robust two-scan and high-west oblique three-scan recovery, with cross-view,
  RGB/depth, known-size silhouette, and bin-registration gates;
- an achromatic RGB-D truck estimator that separates a coloured bin touching
  the gray trailer before estimating the deck pose;
- a task-routed hybrid using active recovery only for cubes and fixed RGB-D
  for bins and the truck;
- versioned fixed-mount calibration wrappers, tested against corrupted runtime extrinsics;
- fixed-calibration RGB-only model-fit variants for a direct RGB versus RGB-D comparison;
- raw-silhouette cube-pose variants that remove the oblique top-face centre bias;
- small/medium/large morphology variants for partially occluded colour silhouettes;
- RGB/RGB-D open fusion;
- adaptive RGB/RGB-D table fusion;
- adaptive multi-view fusion;
- strict multi-view consensus gating;
- a confidence-gated ensemble spanning thresholds and modalities.

## Randomization

The held-out scenes vary object position/yaw, cube/bin manufacturing scale, truck position/yaw, per-object colour, exposure, white balance, gamma, blur, pixel noise, depth noise/dropout/quantization, reported calibration drift, and unlabelled clutter/partial occlusion. Variations cycle through nominal, photometric, depth-noise, calibration-drift, clutter/occlusion, and combined stress.

## Metrics

The report ranks full-scene success, failures, inventory correctness, cube/bin/truck actionable-pose success, inference latency, planned action count, estimated TCP XY path length, and a low-confidence retry budget. Results are also sliced by variation.

Perception winners must still pass end-to-end PhysX execution. A simulator result reduces risk but cannot guarantee physical success; camera/tool calibration, material compliance, friction, controller behavior, and unmodelled geometry remain transfer risks.

The physics gate compares direct nominal motion, a geometry-corrected one-wall
lever, a slower guarded controller, a camera-only mid-task replan, and a
camera-only residual-cube recovery controller. Every mid-task observation
forces fresh Isaac render steps before reading RGB-D; physics-only stepping
does not update camera render products.
The cube loading pattern deliberately reserves the inner-finger corridor at the
south wall before a filled bin is lifted. Cube orientation at the final state is
not scored: a cube lying on its side inside the matching bin is correctly sorted.

## Run

From the Isaac project root:

```bash
chmod +x software/truck_benchmark/run_suite.sh
SCENES=48 software/truck_benchmark/run_suite.sh runs/truck_observation_benchmark
```

The top-level outputs are `report/BENCHMARK_REPORT.md`, `report/ranking.csv`, and `report/benchmark_results.json`.

## Deep second phase

`run_deep_phase2.sh` re-scores fixed-calibration RGB-only model fitting,
RGB-D, and selective active-wrist recovery on the original and held-out data,
then captures a new randomized active-view set. Active sensing is not treated
as free: each scan adds a conservative 1.25 s acquisition, 0.25 m TCP-path,
and one motion-action charge, plus the return-to-task action.

`run_deep_phase3.sh` tests fixed sensing, two overhead wrist scans, and a
kinematically verified high-west oblique scan on identical unseen scenes.
`run_final_confirmation.sh` repeats the finalists with a new random seed. The
task-routed hybrid never replaces a fixed-camera cube pose: active sensing may
only fill a declared missing inventory slot after geometric corroboration.

`run_replan_matrix.sh` compares the guarded open-loop controller with a
camera-only mid-task replan on unseen seeds. The replan clears the cameras,
re-observes after cube loading, and updates bin/truck targets without reading
USD or PhysX object transforms.

The `vision_recovery` candidate adds two safety gates. Refreshed bin and
truck poses are accepted only when their displacement is physically plausible,
and a retry is permitted only for a refreshed same-colour cube that remains
outside its matching bin and can be associated within 60 mm of one original
source attempt. It then uses that attempt's reserved packing slot. On six
paired hard PhysX episodes (nominal, depth noise, and combined stress), it
achieved 3/6 complete tasks, 11.50/12 mean cubes, and 3.00/3 mean bins versus
1/6, 10.67/12, and 2.67/3 for the guarded controller. This is simulation
evidence, not a guarantee of physical transfer.

The subsequent frozen `vision_recovery_v3` controller completed the final
clean 6/6 physics matrix with 12/12 cubes and 3/3 bins in every episode. Its
additional proof, settling, packing, and bounded recovery steps increased
latency and path length. See
[`../../results/truck-loading/FINAL_REPORT.md`](../../results/truck-loading/FINAL_REPORT.md)
for the exact comparison and episode boundary.
