# Demonstrations and status

This page distinguishes what was physically executed, what was validated in Isaac Sim, and what remains a prototype or design artifact.

| System | Physical status | Simulation/offline status | Key evidence |
|---|---|---|---|
| Cube sorting | Supervised cube-to-bin transfers, including four consecutive placements in one reviewed window | Included in randomized end-to-end truck task | [physical experiments](physical-experiments.md), [truck results](../results/truck-loading/) |
| Loaded-bin transport | Supervised one-wall grasp, proof lift, and transport trials; failures and corrections retained | All three loaded bins placed on truck in 6/6 final episodes | [physical bin image](../media/images/physical-bin-lift.jpg), [final report](../results/truck-loading/FINAL_REPORT.md) |
| Two-gear gearbox | Printed parts assembled on the FANUC under supervision | Full assembly and motion rendered in Isaac Sim | [CAD](../cad/gearboxes/two-gear/), [simulation](../software/simulation-demos/gearbox/) |
| Four-gear gearbox | Printed, enlarged parts assembled under supervision | Geometry and assembly strategy reused from the two-gear work | [CAD](../cad/gearboxes/four-gear/) |
| Inline-four engine | Large printed demonstrator partially assembled with supervised calibration and manual checkpoints | Twelve-part top-down sequence validated | [CAD](../cad/engine/), [simulation](../software/simulation-demos/engine-block/) |
| Brewers badge | Printable five-part regional demonstrator | Top-down five-part assembly validated | [CAD](../cad/brewers-badge/), [simulation](../software/simulation-demos/brewers-badge/) |
| Quick-change fingers | Push-push mechanism and finger concepts physically prototyped | Track, latch, rack, and tip geometry checked with scripts | [models/source](../cad/quick-change-tooling/) |
| Video-to-robot | Human demonstrations recorded and annotated; compiled instructions reviewed | Contact graph, behavior tree, task schema, and validation package generated | [results](../results/video-to-robot/), [method](video-to-robot.md) |
| RGB orbit reconstruction | A physical wrist-camera orbit was captured | Multiple reconstruction families compared; best retained point cloud scored after frozen output | [results](../results/reverse-engineering/), [method](reverse-engineering.md) |
| Drag-to-command twin | Used as a fast supervised interface to mirror and schedule moves | Reach checks and generated TCP commands available locally | [app](../apps/digital-twin/) |

## Evidence labels

- **Physical:** obtained from the CSI FANUC workcell or the team's printed hardware.
- **Simulation:** obtained from Isaac Sim/PhysX; useful for comparison and risk reduction, not a physical success rate.
- **Offline:** derived from recorded media or reconstruction data without controlling the robot live.
- **Prototype:** mechanically or digitally demonstrated but not qualified for production use.

The [five-minute demo](https://youtu.be/5eQ3-p7v9-A) provides the fastest visual tour. The [executive summary](../results/submission/executive-summary.pdf) is the one-page submission artifact.

[Back to documentation](README.md)

