# Measured Gearbox Assembly — Isaac Sim Validation

Date: 2026-10-23/24 (America/Chicago)  
Compute: NVIDIA L40S, Isaac Sim 6.0.1  
Result: **PASS** (`task_complete: true`)

## Demonstrated sequence

1. Pick the left gear by its integrated top tab and insert its 8.6 mm bore over the 8.0 mm shaft.
2. Measure the first gear's resting phase.
3. Pick the right gear, counter-rotate it to the derived mating phase, and insert it.
4. Verify the physical 28-tooth mesh.
5. Pick the fused cover tab with the cover's broad back horizontal.
6. Align above the base, lower over the four locating pins, release, and withdraw vertically.

The exact supplied STL files are used for visible geometry. Compound colliders preserve the bores, shafts, locating holes, pins, cover skirt, tabs, and all 28 teeth on each gear.

## Rendered-run acceptance evidence

- Left gear XY error: **0.288 mm**
- Right gear XY error: **0.055 mm**
- Gear center-distance error: **0.027 mm**
- External mesh phase error: **0.992°**
- Cover XY error: **0.436 mm**
- Cover Z error: **0.000024 mm**
- Cover tilt: **0.0000085°**
- Cover staged and carried broad-back horizontal: **pass**
- All three parts: **pass**

## Important corrections established during validation

- The exported cover STL is in a vertical print pose. The verified inverse transform is:
  - `x_assembly = x_export`
  - `y_assembly = z_export - 45.0 mm`
  - `z_assembly = 21.9 mm - y_export`
- The measured cover STL gives COM approximately `(0, -0.798, 15.708) mm`; the earlier guessed COM produced false swing and pin collisions.
- Carrying while rotating caused tab slip. The validated motion rotates at safe height, re-measures the held transform, then translates without simultaneous yaw.
- A seated gear's absolute phase is not meaningful until its mate is present. For equal external gears, the second command is `reference_phase - measured_first_phase`, and mesh verification uses the sum of the two angles.
- Full jaw opening beside a seated tab can disturb the part. The validated release opens only enough to clear the tab, withdraws vertically, then fully opens above it.

## Artifacts

- `gearbox_assembly.mp4`: 71.5 s, H.264, 1280×720, 12 fps, 858 frames
- `execution_report.json`: full phase, pose, correction, and telemetry record
- `run.log`: Isaac Sim runtime log
- `final_frame.png`: final assembled state
- `contact_sheet.png`: six representative frames
- `isaac_gearbox_assembly.py`: validated simulation source
- `run_gearbox_assembly.sh`: reproducible runner

This is Isaac Sim evidence, not authorization for unattended physical execution. Physical deployment still requires the real cell frame/tool calibration, collision review, and operator-controlled first run.
