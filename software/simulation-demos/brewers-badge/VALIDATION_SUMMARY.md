# Brewers Badge — Isaac Sim Assembly Validation

## Outcome

**PASS.** The FANUC ER-4iA completed the five-part, top-down sequence in NVIDIA Isaac Sim 6 on a Brev L40S:

1. `S` display stand
2. `1` navy badge base
3. `2` ball-in-glove emblem
4. `3` rim and lettering carrier
5. `P` EST. 1970 nameplate

All five exact 3MF-derived parts were seated at their measured assembled transforms. Native 3MF navy, gold, and white triangle materials were preserved in the rendered meshes.

## Measured results

| Part | Proof lift | Final position error | Final yaw error | Result |
|---|---:|---:|---:|---|
| Stand | 19.96 mm | 0.00 mm | 0.00° | PASS |
| Base | 19.97 mm | 0.00 mm | 0.00° | PASS |
| Emblem | 19.94 mm | 0.00 mm | 0.00° | PASS |
| Detail | 19.96 mm | 0.00 mm | 0.00° | PASS |
| Nameplate | 19.97 mm | 0.00 mm | 0.00° | PASS |

- Simulation duration: 37.0 seconds
- Render cadence: 10 fps
- Frames per camera: 370
- Cameras: hero, overhead, opposite
- Full-finger approach corridor checked: 34 × 20 mm
- Installed-finger model: 76 × 12 × 6 mm per finger
- Grasp fin: 16 × 8 × 14 mm with vertical top-down engagement

## Validation boundary

The FANUC arm uses real inverse kinematics and articulated joint motion. The thin printed carriers are transported with a deterministic grasp-fin-to-TCP attachment after the close command. This validates reachability, motion order, vertical approach corridors, orientation, and final assembly transforms. It does **not** claim contact-force or friction robustness for a physical print.

## Deliverables

- `brewers_assembly_hero.mp4`
- `brewers_assembly_overhead.mp4`
- `brewers_assembly_opposite.mp4`
- `brewers_assembly_three_view.mp4`
- `physics_execution_report.json`
- `brewers_render_result/execution_report.json`
- `assets/asset_manifest.json`
- `isaac_brewers_assembly.py`
- `run_brewers_assembly.sh`

The combined three-view video places the hero view at 1280 × 720 and the overhead/opposite views in a synchronized 640 × 360 stack, producing a 1920 × 720 H.264 deliverable.
