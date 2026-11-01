# FANUC/Isaac reverse-engineering benchmark

This benchmark keeps the source STL evaluator-only after synthetic sensor
capture.  Reconstruction programs consume only the exported observations:
RGB, noisy depth, reported camera poses, optional multi-light frames, and the
declared object pose for a simulated in-hand second pass.

The initial experiment contains two measured assemblies:

- `gearboxclosed.stl`
- `gearboxopen.stl`

Both source STL files are multi-component and non-watertight.  Evaluation is
therefore based primarily on bidirectional surface distance and normal
agreement, not volume alone.

## Stages

1. `capture_isaac_orbit.py` renders a 360-degree upper-hemisphere RGB-D orbit.
2. `reconstruct_classical.py` runs depth fusion, pose refinement, Poisson,
   laser-stripe, visual-hull, next-best-view, and dual-pass variants.
3. `evaluate_reconstructions.py` scores meshes against the evaluator-only STL.
4. Neural SDF, 2DGS, and external photogrammetry methods are run through
   separate dependency gates so an unavailable project is never reported as
   an experimental failure.

The camera-pose convention follows Isaac/USD: row vectors, local +X right,
+Y up, and -Z forward. Depth is distance to the image plane in metres.

