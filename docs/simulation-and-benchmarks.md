# Simulation and benchmarks

**Status:** reproducible source and compact result package  
**Primary suite:** [`software/truck_benchmark`](../software/truck_benchmark/)  
**Final results:** [`results/truck-loading`](../results/truck-loading/)

## Observation-only contract

The benchmark was designed to prevent simulator shortcuts. Policies received only:

- table, side, and wrist RGB;
- table, side, and wrist distance-to-image-plane depth;
- reported intrinsics and extrinsics;
- joint positions, gripper state, and timestamps; and
- the task specification.

They did not receive USD prims, semantic labels, instance labels, simulator object transforms, scene-state files, or evaluator annotations. Source auditing rejects simulator imports and prohibited pose/label symbols in policy modules. Ground truth is used only to create scenes and score the final state.

## Randomization

Scenes vary object position and yaw, cube/bin manufacturing scale, truck position/yaw, object color, exposure, white balance, gamma, blur, pixel noise, depth noise, dropout, quantization, reported calibration drift, clutter, and partial occlusion. Conditions are grouped as nominal, photometric, depth-noise, calibration-drift, clutter/occlusion, and combined stress.

## Candidate families

The suite compares 48 policy candidates, including:

- fixed-layout and color-threshold baselines;
- adaptive chromaticity;
- table RGB-D height segmentation;
- multi-view RGB-D;
- calibration wrappers;
- RGB-only model fitting;
- active wrist recovery;
- robust oblique scans;
- consensus and ensemble fusion; and
- task-routed hybrids.

Every perception finalist must pass end-to-end PhysX execution. Perception-only accuracy is not treated as task completion.

## Final physics result

| Metric | Previous recovery controller | Final controller |
|---|---:|---:|
| Full truck task | 3/6 | **6/6** |
| Mean matching cubes | 11.50/12 | **12.00/12** |
| Mean bins on deck | 3.00/3 | **3.00/3** |
| Balanced score | 0.5630 | **0.7921** |
| Mean perception latency | 274.8 ms | 1066.2 ms |
| Mean actions | 157.3 | 174.0 |
| Mean TCP XY path | 12.48 m | 12.81 m |
| Mean simulated motion time | 22.1 s | 28.9 s |

The final clean matrix used two new seeds for each of nominal, depth-noise, and combined conditions. Earlier tuning episodes are not counted in the 6/6 result.

## What changed

The final controller combined metric RGB-D poses with tightly gated RGB residual proposals, camera re-observation, a two-pass cube recovery, recovery-aware packing, payload settling, guarded loaded-bin transport, and one bounded spill recovery. Reliability improved at the cost of time and actions.

## Other simulation demonstrators

- [Two-gear gearbox](../software/simulation-demos/gearbox/)
- [Inline-four engine](../software/simulation-demos/engine-block/)
- [Five-part Brewers badge](../software/simulation-demos/brewers-badge/)
- [Legacy measured truck and gearbox runs](../results/legacy/)

## Reproduce

Hardware-free contract and geometry tests run with Pytest. Full scene capture and physics execution require Isaac Sim. Exact commands and expected outputs are in [Reproduction](reproduction.md).

## Interpretation

Six successful simulator episodes demonstrate improvement under the tested randomizations. They do not establish a physical failure rate. Real transfer still depends on camera/TCP calibration, tool geometry, printed tolerances, friction, pneumatic response, speed limits, and operator procedures.

[Back to documentation](README.md)

