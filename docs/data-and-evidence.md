# Data and evidence

## Evidence policy

The repository distinguishes four evidence types:

| Label | Meaning |
|---|---|
| Physical | captured from the CSI FANUC workcell or printed hardware |
| Simulation | generated in Isaac Sim/PhysX |
| Offline | computed from recorded observations without live robot control |
| Design | CAD, scripts, or concepts not yet physically qualified |

No simulation result is reported as a physical success rate. Human-assisted physical trials are labeled as supervised rather than autonomous.

## Physical archive

The private project archive documents ten approved access runs from October 20–31, 2026. Seven complete three-camera exports contain 23 recordings, 142,335 timestamped state/action rows, 237.5 minutes of robot operation, and 69 camera videos. Three additional runs retain partial evidence.

Raw exports are not public because they include facility imagery, people, audio, operational telemetry, camera calibration, and installation-specific control details. The public repository retains aggregate counts, selected consented frames, compact run reports, schemas, code, and the final submission artifacts.

## Simulation archive

The public truck package includes:

- generated final report;
- aggregate physics JSON;
- machine-readable ranking CSV;
- six final episode execution reports;
- frozen plans; and
- mid-task replan records.

Large observation images, frame caches, and duplicate generated runs are omitted. The code required to reproduce scene capture, policy inference, physics execution, and scoring remains under [`software/truck_benchmark`](../software/truck_benchmark/).

## Video-to-robot archive

The public package includes schemas, annotation configurations, the final task graph, behavior tree, instruction card, validation report, and hand-tracking benchmark summaries. Raw phone videos are omitted; the final public demo shows representative consented excerpts.

## Reconstruction archive

The public package includes source pipelines, selected metrics, a representative aligned point cloud, and compact previews. Multi-gigabyte workspaces, cloned upstream repositories, checkpoints, frame caches, and large intermediate meshes are omitted.

## Provenance expectations

Every new experiment should record:

- source and usage restrictions;
- real, simulated, or derived origin;
- software/model version;
- coordinate-frame and unit conventions;
- random seed and scene variation;
- inputs visible to the policy;
- evaluator-only information;
- thresholds and confidence gates; and
- checksums for frozen outputs.

## Submission artifacts

- [Five-minute demo](https://youtu.be/5eQ3-p7v9-A)
- [One-page executive summary](../results/submission/executive-summary.pdf)
- [Interactive tools](https://www.theateam.dev/)

[Back to documentation](README.md)

