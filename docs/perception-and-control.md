# Perception and control

**Status:** tested in simulation and used during supervised physical trials  
**Source:** [`software/robot-control`](../software/robot-control/)  
**Detailed field notes:** [real robot findings](../software/robot-control/README.md) and [physical experiments](physical-experiments.md)

## Available observations

The CSI interface supplied table, side, and wrist RGB images; table and wrist depth where available; reported camera intrinsics/extrinsics; joint positions; TCP state; gripper state; and timestamps. The control code treats camera data as measurements with different strengths rather than assuming every stream is equally accurate.

| View | Primary use | Common failure mode |
|---|---|---|
| Table | global inventory and XY position | occlusion by the arm or bins; depth referenced to the wrong plane |
| Side | height, wall clearance, and approach sanity | weak top-surface information |
| Wrist | local re-centering and final descent | camera-to-TCP offset and narrow field of view |

## Table-relative depth

Depth becomes most useful after estimating the table plane and expressing object height relative to it. A floor-relative or uncorrected depth display compresses the useful range and can make small parts hard to separate from the table. The tested process was:

1. estimate or load the table plane;
2. convert depth to table-relative height;
3. reject invalid, edge, and grazing-angle pixels;
4. cluster above-table regions;
5. combine color/shape identity with metric height; and
6. confirm local geometry from the wrist before a tight descent.

## Control structure

The phase controller executes monotonic, complete trajectories rather than repeatedly issuing tiny competing corrections:

```text
observe → high approach → local re-center → descend → close
        → proof lift → transfer → guarded lower → release → retreat
```

The implementation includes:

- workspace and Z-floor clamps;
- action and phase state;
- debounced gripper commands;
- proof-lift and jaw-gap validation;
- fresh observation after the arm clears fixed cameras;
- bounded same-object recovery;
- packing slots that preserve finger corridors; and
- abort paths when grip state or displacement is implausible.

## Loaded-bin transport

A center grasp was unreliable for the one-wall printed bins and could collide with the contents. The successful strategy used a measured wall approach, deeper controlled wall engagement, slow close, 10 mm micro-lift, proof dwell, staged raise, and a center-preserving wrist pivot. The final simulator controller also waited for payload settling and checked jaw contact after close, proof, and raise.

## Small-motion oscillation

The observed back-and-forth behavior during tiny corrections was consistent with a command/observation timing problem: several incremental targets were queued while the arm was still converging, noisy estimates crossed a threshold, and open/close commands could be resent without a latched phase. The mitigation is to send one monotonic segment, wait on a completion condition with hysteresis, rate-limit re-plans, and update only at explicit checkpoints. Large high-clearance moves improved first because their signal was much larger than the measurement and control deadband.

## Physical execution boundary

The repository can generate and validate motion plans, but it cannot make physical motion safe by itself. Verify TCP, tool, work frame, table plane, camera transforms, part scale, gripper stroke, and clearance on the actual cell. Use the official client and a qualified operator.

[Back to documentation](README.md)

