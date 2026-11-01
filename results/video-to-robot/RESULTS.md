# Video-to-Robot results package

## Outcome

The six source clips were preserved and processed as **six repeated demonstrations**, not
as synchronized multi-view footage. Together they contain 24 verified cube placements.

The delivered system provides:

- 21-point hand tracking and translucent hand highlighting;
- target-bin masking and contextual action labels;
- reviewed action phases and transfer counts on every take;
- raw frame observations and automatic timelines for debugging;
- a typed robot task, behavior tree, symbolic contact graph, and safety gates;
- an evidence audit and an honest method-selection report.
- auxiliary speech/acoustic analysis where the available ElevenLabs quota permitted it.

## Best viewing files

- `grounded_annotated/grounded_all_six_takes.mp4` — all six annotated takes, original audio.
- `grounded_annotated/best_take_source_vs_grounded.mp4` — direct source/annotation comparison.
- `grounded_annotated/PXL_20261020_165201912_grounded.mp4` — best complete global view.
- `grounded_annotated/PXL_20261020_165414567_grounded.mp4` — best detail and final-state proof.
- `grounded_annotated/PXL_20261020_165355671_grounded.mp4` — best side-view motion evidence.

All six individual grounded videos are in `grounded_annotated/`.

## Machine-readable outputs

- `compiled_task/robot_task.json` — nine-step, evidence-linked task representation.
- `compiled_task/behavior_tree.json` — guarded execution plan.
- `compiled_task/contact_graph.json` — symbolic hand/part/bin contact-state graph.
- `compiled_task/INSTRUCTION_CARD.md` — human-readable procedure.
- `hybrid_system/tracks/*.json` — per-frame landmarks, masks, timestamps, and state evidence.
- `hybrid_system/timelines/*.json` — raw automatic action segmentation.

## Measured performance

| Measure | Result |
|---|---:|
| Source clips | 6 |
| Total duration | 84.078 s |
| Verified placements | 24/24 |
| Weighted hand-frame coverage | 90.8% |
| Bin-frame coverage | 100% |
| Automatic cycle count exact | 2/6 clips |
| Automatic cycle count MAE | 1.0 cycle/clip |

The clean grounded videos therefore use reviewed action windows (±0.25–0.5 s) plus
frame-by-frame model hand tracking. The uncorrected automatic outputs are retained so the
failure is measurable rather than hidden.

## Decision

For these RGB-only videos, the strongest deployable design is:

1. SAM 3.1 text + exemplar masks for hands, each part instance, and target fixtures;
2. MediaPipe hand landmarks as an independent kinematic signal;
3. persistent instance inventory and explicit occlusion state;
4. a hysteretic contact graph that verifies release from post-hand-clear state change;
5. a typed instruction compiler that refuses to invent unmeasured robot parameters.

For actual robot-frame execution, add synchronized calibrated RGB-D, CAD meshes,
gripper/joint telemetry, FoundationPose, and collision/force validation. The phone clips
alone cannot yield metric 6D poses, safe Z heights, forces, or millimetre tolerances.

## Supporting reports

- `audit/AUDIT_REPORT.md` — media integrity, content, exact repeat/view structure, and timelines.
- `METHOD_SELECTION_REPORT.md` — method benchmark, frontier-model roles, deployment architecture, and failure analysis.
- `method_tests/hand_tracking_benchmark/` — visual/runtime comparison of local trackers.
- `audio_transcripts/AUDIO_FINDINGS.md` — speech/acoustic evidence and its limits.

## Audio evidence

One take contains explicit narration to place four red cubes into the bin, agreeing with
the 24 visually reviewed placements across six repetitions. Three other completed
transcriptions contain only non-speech acoustic events. The remaining two audio tracks
are preserved locally but could not be submitted after the ElevenLabs account reached
zero remaining credits. Audio is therefore supporting context only; it is never used to
invent geometric or contact measurements.

## Reproduce

```powershell
python software\video_to_robot\run_pipeline.py `
  --input-dir "<path-to-demonstration-videos>" `
  --output-dir "outputs\video-to-robot\hybrid-system" `
  --width 960

python software\video_to_robot\render_verified.py `
  --input-dir "<path-to-demonstration-videos>" `
  --output-dir "outputs\video-to-robot\grounded-annotated" `
  --width 1280

.\.venv-video-ai\Scripts\python.exe video_to_robot_pipeline\compile_instructions.py `
  --output-dir "outputs\video_to_robot_pipeline_20261027\compiled_task"
```
