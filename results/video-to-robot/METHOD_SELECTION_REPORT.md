# Method Selection Report: Video-to-Robot Evidence Compiler

**Dataset:** six retained phone videos of the assembly demonstrations  
**Report date:** 2026-10-27  
**Decision scope:** hand highlighting, object/context tracking, action extraction, instruction compilation, and a path to robot-executable verification  
**Evidence standard:** measured local results are separated from proposed or untested components; no metric 3D pose is inferred from these clips.

## Executive decision

For the six supplied videos, the best defensible system is a **hybrid evidence pipeline**, not a monolithic Video-LLM:

1. normalize video timing and display orientation;
2. run a hand landmarker on every frame;
3. segment and track the known cubes and bin;
4. infer actions with a hysteretic contact/state graph;
5. confirm placement only from post-release state change;
6. compile typed, evidence-linked instructions in which unmeasured robot parameters remain explicitly unresolved.

The local implementation already follows that structure with MediaPipe hand landmarks, deterministic red-cube/green-bin masks, and a contact-state machine. It produced annotated videos, frame tracks, and timelines for all six takes. This is the correct **immediate visualization and evidence-extraction baseline**. It is not yet a robot-ready motion generator: automatic cycle count was exactly correct on only 2 of 6 clips, so the event compiler needs labeled calibration and stronger object identity/occupancy logic before unattended use.

The strongest next upgrade for these RGB-only clips is **SAM 3.1 concept/exemplar masks plus the same deterministic state graph**, while retaining MediaPipe landmarks for inexpensive kinematic cues. Meta officially released SAM 3.1 in March 2026; Object Multiplex improves multi-object video tracking efficiency, and the official system supports text, exemplar, and visual prompts. It should be benchmarked against corrected ground truth rather than assumed superior.

**FoundationPose is the right future 6D component, but it is not applicable to the supplied clips as recorded.** Its deployable path expects calibrated RGB-D plus a CAD mesh or RGB-D reference views. The six phone clips have no depth, intrinsics, camera-to-robot transform, or synchronized cross-view correspondence. HaMeR can add a camera-relative 3D hand mesh, but it cannot by itself recover world-metric contact or robot waypoints. VideoITG is useful for selecting instruction-relevant frames from long videos; it is not a substitute for dense action-boundary detection, and these clips are short enough to process every frame.

## 1. What the source data actually contains

The source audit established the following facts:

- Six independent HEVC phone recordings, 1920 x 1080, approximately 24 fps, totaling **84.078 seconds**.
- Each clip is a separate demonstration of the same task: place four red/brown cubes into a green open-top bin.
- The files do not overlap in time and the scene is reset between recordings. They are **complementary takes**, not simultaneous multi-camera streams.
- Two wide overhead takes, two side takes, and two close overhead takes.
- One file, `PXL_20261020_165240667.mp4`, carries a -180-degree display transform.
- Contact is frequently hidden by fingers or the opaque bin wall.
- The first pickup is absent in the longest deliberate take because the first cube is already in hand.
- The videos contain no robot execution telemetry, force, gripper state, depth, fiducial calibration, camera intrinsics, or camera-to-base extrinsics.

### What can be extracted now

| Claim | Support from these clips | Appropriate representation |
|---|---|---|
| Hands are present and move through the scene | Yes | 2D landmarks/masks, handedness, pixel trajectories |
| Four cubes are transferred to a green bin | Yes, from repeated takes and final states | object masks/tracks, symbolic inventory |
| Action order is reach/pick/transfer/insert/release/retract | Yes | temporal segments with source-frame evidence |
| A release succeeded | Usually, after the hand clears | post-release occupancy/state change |
| Exact contact frame | Often ambiguous | bounded uncertainty interval, not a single asserted frame |
| Metric object pose or insertion depth | No | unavailable |
| Camera-frame or robot-frame 6D trajectory | No | unavailable |
| Safe robot grasp depth, force, or collision clearance | No | unresolved parameter requiring calibrated sensing and planning |

The derived instructions must therefore encode **task semantics and observable postconditions**, not copy a human pixel trajectory into robot coordinates.

## 2. Tested local methods

### 2.1 End-to-end hybrid prototype

**Method:** MediaPipe Hand Landmarker + deterministic object masks + hysteretic contact graph.

The local run generated:

- annotated MP4s in `hybrid_system/annotated/`;
- per-frame observations in `hybrid_system/tracks/`;
- action timelines in `hybrid_system/timelines/`;
- a machine-readable summary in `hybrid_system/run_summary.json`.

Measured run summary:

| Clip | Hand-frame coverage | Bin-frame coverage | Predicted completed cycles | Audited true placements |
|---|---:|---:|---:|---:|
| `165201912` | 82.53% | 100% | 4 | 4 |
| `165226075` | 100% | 100% | 2 | 4 |
| `165240667` | 87.15% | 100% | 2 | 4 |
| `165311341` | 97.96% | 100% | 5 | 4 |
| `165355671` | 100% | 100% | 4 | 4 |
| `165414567` | 77.15% | 100% | 3 | 4 |

Across the six clips, hand-detection coverage was **90.80%** and bin coverage was **100%**. Exact placement count was correct for **2/6 clips (33.3%)**, with a mean absolute count error of **1.0 cycle per clip**. The system predicted 20 cycles against 24 audited placements.

These are coverage and count diagnostics, not landmark or temporal-segmentation accuracy. They show that the overlay layer is already useful, while the automatic action compiler is not yet reliable enough to command a robot without correction.

Observed error mechanisms:

- the side views hide the cube as soon as it passes the bin rim;
- the close views hide contact under the hand;
- the support hand can be selected instead of the manipulating hand;
- bin repositioning and presentation gestures look like additional task motion;
- color connected components can split one cube or include red/brown background marks;
- a missing first pickup breaks a state machine that assumes every cycle begins with an observed reach;
- a release transition based on hand motion alone can fire before the outcome is visible.

### 2.2 OpenCV hand-highlighting benchmark

Five inexpensive methods were compared on two representative 144-frame samples: a multi-hand overhead take and a close single-hand take. The overlap columns use a skin-color proxy rather than labeled masks; they are **consistency diagnostics, not accuracy**.

| Overall rank | Method | Approx. processing rate | Mean detection uptime | Main strength | Main failure |
|---:|---|---:|---:|---|---|
| 1 | temporal YCrCb/HSV skin mask | 96.1 fps | 99.3% | fast, smooth, stable visual hand region | skin/background confusion; no joints or identity semantics |
| 2 | raw YCrCb/HSV skin mask | 96.9 fps | 97.9% | fastest robust overlay baseline | noisier boundaries and no temporal memory |
| 3 | KCF with detector reinitialization | 30.9 fps | 95.1% | cheap persistent box after initialization | drift, eight dropout transitions in the close clip, poor mask agreement |
| 4 | CSRT with detector reinitialization | 10.2 fps | 100% | persistent boxes under slower appearance change | far slower, oversized regions, low proxy precision |
| 5 | motion-gated skin mask | 70.2 fps | 90.3% | suppresses stationary skin-like distractors | drops hands during pauses and produces the largest trajectory jumps |

The temporal skin method is a useful fallback or quality monitor, but it is not a replacement for landmark or mask tracking. KCF/CSRT are unsuitable as primary multi-object trackers because they do not understand hands, cubes, or the bin and can silently drift.

### 2.3 Ranked tested/local choices

1. **Hybrid MediaPipe + object masks + contact graph** — the only tested local method that produces hands, objects, events, and annotated videos together. Use now for review; repair event logic before control.
2. **Temporal skin mask** — best low-cost fallback for hand highlighting and detector-failure monitoring.
3. **Raw skin mask** — fast diagnostic baseline.
4. **KCF reinitialization** — acceptable only for short, already-initialized ROI tracking.
5. **CSRT reinitialization** — stable but too slow and spatially loose for this task.
6. **Motion-gated skin** — useful as an auxiliary motion signal, not as a primary hand detector.

## 3. Status of the proposed frontier components

### SAM 3.1: available and relevant, but not yet tested here

Meta's official SAM 3 system detects, segments, and tracks matching instances from short text prompts or visual exemplars. The official SAM 3.1 release notes, dated 2026-03-27, introduce Object Multiplex and new checkpoints, reporting roughly a 7x speedup at 128 objects on one H100 versus the original SAM 3 release. Meta also reports mixed per-benchmark changes rather than universal accuracy improvement, which is why an in-domain benchmark remains necessary.

**Best role here:** persistent masks for `human hand`, `red cube`, and `green open-top bin`, initialized with both concept prompts and one clean exemplar per class. A concept model is preferable to pure color thresholding when cube/bin color changes. The tracker must retain persistent instance IDs and expose mask confidence and occlusion state.

**Limits:** SAM 3.1 provides segmentation/tracking, not contact mechanics, metric depth, 6D pose, or an executable behavior tree. Meta documents that fine-grained out-of-domain terms may require adaptation and that short noun phrases are the intended prompt style. Checkpoint access, a Linux/CUDA environment, and model validation are deployment work. No SAM 3.1 result should be claimed for the current artifacts until it is actually run and scored.

Official sources: [Meta SAM 3 research page](https://ai.meta.com/research/sam3/), [Meta SAM 3 publication](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), [official SAM 3 repository](https://github.com/facebookresearch/sam3), and [official SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

### VideoITG: available, but optional for these short clips

NVIDIA's VideoITG is an instruction-conditioned **frame relevance and selection** system. The official repository describes scoring a larger decoded frame pool and selecting the top-K frames for a downstream Video-LLM. It was accepted at CVPR 2026 and ships code/checkpoints.

**Best role here:** find semantically relevant evidence in long untrimmed recordings, for example, “show every post-release state in which a cube is visible inside the bin.”

**Not its role:** it does not inherently produce precise action boundaries, contact timestamps, audio-event detection, hand masks, or trajectories. The user's six clips total only 84 seconds; dense 24 fps processing is feasible and avoids dropping a short contact event. VideoITG becomes valuable when the input grows to many minutes or hours, after which dense local analysis should still run around selected windows.

Official sources: [NVIDIA VideoITG project](https://nvlabs.github.io/VideoITG/), [official repository](https://github.com/NVlabs/VideoITG), and [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VideoITG_Multimodal_Video_Understanding_with_Instructed_Temporal_Grounding_CVPR_2026_paper.html).

### FoundationPose: available, but impossible to use correctly on these clips alone

FoundationPose is NVIDIA's model-based/model-free 6D pose estimator and tracker. The official implementation can apply to a novel object without fine-tuning when a CAD model is supplied; the deployable NVIDIA model card specifies RGB-D, an object mask, camera intrinsics, and a 3D model, or RGB-D reference views for model-free use.

**Best role in production:** estimate and track each known gearbox part or bin in calibrated RGB-D, then express relative part-to-housing transforms and verify assembly state.

**Why it cannot produce valid 6D results here:** the phone files have no aligned depth, calibrated intrinsics, CAD geometry with physical scale, or camera-to-world transforms. Because the takes are not simultaneous, cross-view bundle adjustment cannot turn them into one time-varying 3D trajectory. Inventing depth or treating the clips as synchronized views would create false metric evidence.

Official sources: [official FoundationPose repository](https://github.com/NVlabs/FoundationPose), [NVIDIA FoundationPose model card](https://catalog.ngc.nvidia.com/orgs/nvidia/isaac/models/foundationpose/-), and [NVIDIA research page](https://research.nvidia.com/labs/lpr/publication/wen2024fpose/).

### HaMeR: available and useful for contact cues, not world-metric manipulation

HaMeR reconstructs a 3D MANO-style hand mesh from a monocular crop. The official paper and project show improved robustness under many in-the-wild conditions. Its documented failure modes include spurious detections, wrong left/right classification, extreme poses, and extreme occlusion.

**Best role here:** on uncertain frames, estimate finger configuration and likely pinch region; compare the human contact patch with candidate robot grasp surfaces.

**Limits:** the released model is a per-image monocular hand-mesh method. “HaMeR + multi-view triangulation” is a separate engineering pipeline, not a native HaMeR output. With an uncalibrated moving phone, its mesh remains camera-relative and scale/translation ambiguous. It does not track object pose or prove physical contact. It should augment, not replace, part masks and postcondition checks.

Official sources: [HaMeR project page](https://geopavlakos.github.io/hamer/), [official HaMeR repository](https://github.com/geopavlakos/hamer), and [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html).

### MediaPipe Hand Landmarker: appropriate real-time baseline

Google's official Hand Landmarker returns 21 landmarks, handedness, and video/stream tracking. It is lightweight enough for every-frame processing and is the appropriate local baseline for annotation. Its so-called “world landmarks” are hand-centered model outputs; without camera calibration and registration they must not be treated as robot-base coordinates.

Official source: [Google MediaPipe Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

## 4. Recommended production architecture

### Stage 0 — evidence-preserving ingestion

- Decode by presentation timestamp, not `frame_index / 24` assumptions.
- Honor display matrices and record every applied transform.
- Preserve audio and original files.
- Assign each clip a distinct `episode_id`; never fuse the six files as a shared time axis.
- Store source hash, frame timestamp, model version, prompt, and calibration version with every derived record.

### Stage 1 — dense spatial perception

**Current RGB-only mode**

- MediaPipe: 21-point hand landmarks and handedness.
- SAM 3.1: hand, cube, and bin instance masks with persistent IDs.
- Lightweight color/geometry detector: independent cross-check for known red/green objects.
- Optical flow/Kalman smoothing: bridge short occlusions, with confidence decay rather than fabricated masks.

**Future robot mode**

- Synchronized, calibrated RGB-D streams.
- FoundationPose for every CAD-known part.
- Robot joint/TCP/gripper telemetry in the same clock domain.
- Optional HaMeR on human demonstration clips for grasp-region cues.

### Stage 2 — object-centric contact graph

Represent each frame as nodes (`left_hand`, `right_hand`, `cube_1..4`, `bin`) and probabilistic edges:

- `near(hand, cube)`;
- `overlap(hand_mask, cube_mask)`;
- `carried_by(cube, hand)`;
- `inside(cube, bin_opening)`;
- `supported_by(cube, table/bin)`;
- `occluded(cube, by=hand/bin_wall)`.

Transitions should use hysteresis and minimum duration. A successful deposit requires both:

1. a release/retract pattern; and
2. a delayed postcondition: outside-cube inventory decreases and inside-bin inventory increases after the hand clears.

Never force an exact contact frame when the object is hidden. Report `[earliest_possible, latest_possible]`, the evidence views, and uncertainty.

### Stage 3 — temporal compiler

For these short clips, analyze every frame. Use VideoITG only as a coarse retrieval front end for long recordings. Classify segments as:

- `idle/explanation`;
- `approach`;
- `contact/grasp`;
- `lift`;
- `transfer`;
- `insert`;
- `release`;
- `retract`;
- `verify`;
- `bin_reposition`;
- `goal_presentation`.

This explicit negative vocabulary prevents gestures, container movement, and final presentation from becoming false assembly cycles.

### Stage 4 — typed instruction synthesis

A VLM may summarize the evidence, but it must write into a strict schema and cite frame/time ranges. It may not invent metric values. Each step needs:

- object and target IDs;
- observed preconditions;
- source clip/frame evidence;
- symbolic primitive;
- qualitative approach axis when visually supported;
- verification rule;
- confidence and ambiguity;
- `requires_calibration: true` for all metric robot parameters not measured.

The evidence-supported task program is:

1. Detect the upright green bin and count four red/brown cubes outside it.
2. Repeat until the outside count is zero:
   - approach one exposed cube;
   - close a grasp on the cube;
   - lift it clear of the support surface;
   - transfer above the bin opening;
   - descend into the opening;
   - release;
   - retract;
   - verify that outside inventory decreased by one and bin occupancy increased by one.
3. Verify four cubes contained in the bin.

Safe robot approach height, jaw force, descent depth, drop height, and base-frame waypoints remain unresolved by these recordings.

### Stage 5 — robot execution and evaluation

Compile symbolic steps into skills only after live robot perception resolves object poses. Use a behavior tree with hard guards:

- perception confidence threshold;
- workspace and reachability checks;
- collision-free gripper corridor;
- proof lift after grasp;
- bounded descent and force/current stop;
- post-release inventory verification;
- retry budget and human stop.

Isaac Sim should be used to stress-test the controller, camera geometry, gripper collisions, and recovery logic under domain randomization. It cannot restore calibration or depth that the real phone recordings never contained.

## 5. Data required for metric 3D and robot reproduction

| Data | Minimum requirement | Why it matters |
|---|---|---|
| Camera intrinsics/distortion | per RGB and depth sensor | valid rays, undistortion, pose estimation |
| Camera extrinsics | sensor-to-robot-base transform with covariance | express perception in robot coordinates |
| Synchronization | common hardware clock or measured offset; target < one video frame, tighter for fast contact | cross-view and robot-state alignment |
| Aligned depth | metric depth, scale and invalid-pixel model | 3D translation and occlusion reasoning |
| CAD/mesh | watertight mesh, correct metric scale, explicit object frame | FoundationPose and collision geometry |
| Robot state | timestamped joints, TCP, controller mode | forward kinematics and execution comparison |
| Gripper state | jaw position/command/force or current | detect closure, contact, slip, and release |
| Workcell model | table, bin, fixtures, installed fingers | collision checks and safe approach corridor |
| Force/torque or motor-current data | optional but strongly recommended for insertion | distinguish seated part from jam/contact |
| Human labels | masks, IDs, event boundaries, success/failure on a stratified subset | honest validation and threshold tuning |

For synchronized multiview capture, use a calibration target and fixed cameras; for a wrist camera, log robot pose at image exposure. Record an explicit “empty table” frame and fiducial observations at the start and end to detect calibration drift.

## 6. Failure-mode register and mitigations

| Failure mode | Consequence | Detection | Mitigation |
|---|---|---|---|
| Ignored -180-degree display transform | inverted coordinates and wrong action geometry | metadata check + visual sentinel | normalize once and log transform |
| Sequential takes treated as synchronized | invalid triangulation/4D trajectory | timestamp overlap audit | separate episode IDs; consensus only at semantic level |
| Hand covers cube at release | false disappearance or early release | mask confidence drops while hand overlaps | bounded contact interval; verify after hand clears |
| Side view hides bin interior | false placement claim | no target-interior visibility | use side view for motion only; require top/final-state evidence |
| Support hand mistaken for manipulating hand | wrong contact graph | cube co-motion and pinch/contact score | role inference per cycle, not fixed handedness |
| Reposition/gesture counted as a transfer | extra cycle | object inventory unchanged | negative action states and postcondition gating |
| Same-color components split/merge | wrong cube count/IDs | area/shape jump and mask disagreement | SAM exemplar masks + track memory + inventory constraints |
| Tracker bridges long occlusion | fabricated trajectory | confidence decay/time since observation | retain “unknown” state; re-identify after occlusion |
| HaMeR left/right or extreme-pose failure | wrong grasp cue | reprojection error, mask disagreement | use as auxiliary estimate only |
| FoundationPose on missing/noisy depth | unstable or false 6D pose | depth-validity and score gates | do not run without calibrated RGB-D; re-register/reobserve |
| VLM invents tolerance or waypoint | unsafe execution | schema provenance and nullability checks | reject any numeric field without measured source |

## 7. Evaluation protocol

### 7.1 Build a real test set

Label a stratified set from all six clips, not just the clean overhead take:

- at least 100 frames per view family plus every transition window;
- hand masks or 21 landmarks;
- cube/bin masks and persistent IDs;
- action segments and boundary uncertainty;
- outside/inside cube inventory;
- occlusion and visibility flags;
- true cycle count and final success.

Keep one complete take from each view family as a held-out test episode. Tune thresholds only on the remaining takes.

### 7.2 Metrics

**Spatial perception**

- hand/cube/bin mask IoU and boundary F-score;
- hand keypoint PCK and landmark detection coverage;
- instance identity HOTA/IDF1 and ID switches;
- cube inventory count accuracy before and after release;
- calibration/runtime per frame.

**Temporal understanding**

- segment mAP at multiple temporal IoU thresholds;
- boundary F1 at +/-0.25 s and +/-0.50 s;
- sequence edit score;
- exact placement-count accuracy;
- false cycles from gestures/repositioning;
- release-success precision, with post-release verification.

**Instruction quality**

- object/target grounding accuracy;
- ordered-step exact match;
- evidence citation validity;
- unsupported numeric-claim rate, which should be zero;
- human engineer acceptance and required correction time.

**Future metric 3D**

- translation/rotation error and ADD-S for symmetric parts;
- relative part-to-housing seated-depth error;
- contact-time and contact-state accuracy;
- cross-view reprojection error and calibration drift.

**Robot execution**

- successful full assemblies / attempts across randomized layouts;
- collision, drop, jam, and human-intervention rates;
- proof-lift success, retry count, cycle time, and path length;
- final object-relative pose tolerance;
- worst-case performance, not only mean performance.

### 7.3 Required ablations

Compare, on identical held-out frames and transitions:

1. current color masks versus SAM 3.1 text prompts;
2. SAM 3.1 text-only versus exemplar-assisted prompts;
3. MediaPipe alone versus MediaPipe + SAM hand masks;
4. hand-motion-only release versus post-release inventory gating;
5. uniform full-frame processing versus VideoITG retrieval followed by dense local analysis;
6. RGB-only symbolic extraction versus calibrated RGB-D + FoundationPose on a newly captured dataset.

No method should be called “best” until it wins the relevant labeled metrics and passes replay on unseen takes.

## 8. Deployment recommendation

### Immediate deliverable

Use the generated hybrid annotated videos for communication and QA. Treat the frame tracks as useful 2D evidence. Mark the timelines as draft because only two clips currently have the correct automatic cycle count. Manually correct the event boundaries/counts for the six-video deliverable and store corrections as ground truth for the next run.

### Best next RGB-only experiment

Run SAM 3.1 on an NVIDIA GPU using these prompts/exemplars:

- `human hand` plus one clean hand-box exemplar;
- `small red cube` plus one clean cube exemplar;
- `green open-top bin` plus one clean bin exemplar.

Retain MediaPipe landmarks and the deterministic color model as independent voters. Rebuild events from persistent cube IDs and post-release occupancy. This is the most promising method for color changes and hand/object occlusion while preserving a clear evidence trail.

### Production robot system

Capture synchronized calibrated RGB-D and robot/gripper telemetry, add CAD with verified metric scale, run SAM 3.1 for instance masks and FoundationPose for 6D part pose, compile a relative object/contact graph, and dispatch only calibrated behavior-tree skills with verification after each node. Use HaMeR only for human grasp/contact interpretation and VideoITG only to retrieve evidence from long recordings.

### Final recommendation in one sentence

**For the current six videos, deploy the hybrid 2D evidence compiler with corrected ground truth; for autonomous reproduction, upgrade the capture system first, then add SAM 3.1 masks and FoundationPose 6D tracking—do not attempt to manufacture metric robot commands from the existing uncalibrated sequential phone footage.**

## Artifact index

- Source audit: `audit/AUDIT_REPORT.md`
- Video inventory: `audit/video_inventory.csv`
- Contact sheets: `audit/contact_sheets/`
- Local method metrics: `method_tests/hand_tracking_benchmark/benchmark_results.json`
- Local comparison videos: `method_tests/hand_tracking_benchmark/annotated/`
- Hybrid run summary: `hybrid_system/run_summary.json`
- Hybrid annotated videos: `hybrid_system/annotated/`
- Frame tracks: `hybrid_system/tracks/`
- Timelines: `hybrid_system/timelines/`
- Evidence schema: `../../video_to_robot_pipeline/schemas/robot_instruction.schema.json`

