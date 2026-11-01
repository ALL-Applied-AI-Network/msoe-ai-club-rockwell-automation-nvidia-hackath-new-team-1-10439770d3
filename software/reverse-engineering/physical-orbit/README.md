# Physical orbit RGB reconstruction benchmark

This benchmark reconstructs the real gearbox-base orbit captured in run
`r20261027-af2bec`.  Reconstruction methods receive only the physical RGB orbit
and, in a separately labeled telemetry-assisted track, robot state/kinematics.
They never receive the measured CAD, simulator object transforms, or depth.

The measured STL files are held back until a candidate output is frozen.  They
are then used only for similarity-aligned geometric evaluation.  Because the
orbit sees the upper hemisphere only, every closed underside must be labeled as
inferred or table-plane-capped rather than observed.

Prepare the input sets:

```powershell
python software/reverse-engineering/physical-orbit/prepare_orbit_inputs.py \
  --video /path/to/orbit_wrist.mp4 \
  --out runs/physical_orbit/inputs
```

The generated `orbit_endpoints_18` set follows the actual 18 stopped physical
viewpoints; `keyframes_18`, `keyframes_36`, and `keyframes_60` are blur-filtered
dense variants confined to the 12.4–54.9 s closed revolution.  Every set
includes raw full-frame RGB, white-background masked RGB, RGBA, silhouettes,
centered crops, and a manifest with exact source frame/time and quality metrics.
