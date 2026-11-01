# Hand-tracking method test

## Executive conclusion

The practical local options split into two very different classes:

1. **Fast but scene-specific:** the OpenCV `skin_temporal` prototype runs at 79–91 processed frames/s and visually follows one or two hands in the clean close-up view. It is not robust enough to drive the robot by itself: in the wide view, yellow/green bins at the image edge are sometimes classified as skin. The reported 98.6–100% detection uptime is therefore a continuity/proxy result, not task accuracy.
2. **Better contour semantics but far too slow on this machine:** cached SAM2.1 Hiera Small generally follows the prompted hand/forearm and does not select the red cubes in the reviewed sample, but CPU inference is 4.18 s/frame (0.239 fps). Its automatic maximum-visible-area prompt also misses frames before the hand is substantially visible.

For a real video-to-robot pipeline, use a **dedicated hand landmark/detector model (MediaPipe Hands or equivalent) as the detector**, then a short-lived temporal tracker for continuity. Treat the OpenCV skin method only as a confidence cue or fallback. Treat SAM2 as an offline mask/refinement tool unless a GPU-enabled inference stack is installed and measured.

## Scope and reproducibility

- Source: six retained phone videos of the assembly demonstrations
- Six 1920×1080 HEVC Main10 HLG phone videos, approximately 24 fps
- Total source duration: **84.078 s**
- Total source size: **157.07 MiB**
- Benchmark samples:
  - `PXL_20261020_165311341.mp4`: overhead, multi-hand, wide work area
  - `PXL_20261020_165414567.mp4`: overhead, single-hand, close-up
- OpenCV run: 12 s/sample, 144 frames at 12 fps, resized to 960 px width
- SAM2 run: representative 24-frame sample at 6 fps-equivalent sampling; CPU-only

Run:

```powershell
python benchmark_hand_tracking.py
python benchmark_hand_tracking.py --with-sam2
# Re-run only the expensive cached-model test:
python benchmark_hand_tracking.py --sam2-only
```

The benchmark writes `inventory.json`, `benchmark_results.json`, `benchmark_results.csv`, annotated videos, and QA contact sheets into this directory.

## Local environment inventory

| Capability | Local status | Consequence |
|---|---:|---|
| OpenCV | 4.13.0 | KCF, CSRT, and MOSSE APIs available |
| PyTorch | 2.7.0+cpu | No CUDA inference through this Python environment |
| ONNX Runtime | 1.20.1 CPU/Azure providers | CPU deployment possible; GPU provider not usable in this environment |
| Transformers | 4.57.1 | Cached SAM2 video runtime available |
| SAM2.1 Hiera Small cache | present, 175.8 MiB | Prompted segmentation could be tested without a download |
| MediaPipe | not installed in this environment | Not benchmarked here |
| Ultralytics | not installed in this environment | Not benchmarked here; generic YOLO also needs hand-specific weights/classes |
| Physical GPU | RTX 3060 6 GB | Present, but the installed PyTorch stack is CPU-only |

There are multiple OpenCV distributions installed (`opencv-python`, `opencv-contrib-python`, and headless variants at different versions). The import used for this run resolves to 4.13.0. Pinning one consistent OpenCV build is recommended before production use.

## Methods tested

### 1. Skin YCrCb + HSV

A conservative intersection/union of YCrCb and HSV skin ranges, morphology, connected components, and an edge-entry prior. The edge prior reflects these fixed-camera demonstrations, where a real hand/forearm normally enters from an image boundary. It removed many red-cube false positives but is deliberately scene-specific.

### 2. Temporal skin

The same proposals with nearest-center association and short persistence. This reduces small one-frame position changes and bridges brief threshold failures.

### 3. Motion-gated skin

MOG2 foreground motion intersected with skin proposals. It is intended to reject static colored objects, but a stationary or slowly moving hand disappears and moving hand regions fragment.

### 4. KCF with periodic reinitialization

KCF boxes are initialized and re-seeded from skin proposals every 36 frames. This is faster than CSRT but loses the hand in the close-up and requires repeated detector assistance.

### 5. CSRT with periodic reinitialization

CSRT is persistent, but the 960 px benchmark is approximately 10 fps. Its rectangular boxes include substantial table/background area and remain dependent on the initial skin proposal.

### 6. Prompted SAM2.1 Hiera Small

The automatic prompt is chosen at the sampled keyframe with the largest visible skin-proposal area, not frame zero. SAM2 propagates from that keyframe backward and forward. This change was necessary because frame zero contains only a clipped hand sliver. The run uses the locally cached model and excludes its 0.345 s model-load time from inference fps.

## Quantitative results

There is no hand-labeled ground truth in the six source videos. Consequently:

- `detection_uptime` measures whether a method returned at least one region, **not whether that region is actually a hand**.
- `skin_proxy_recall/precision` measures agreement with the skin proposal used by the harness, **not independent accuracy**.
- The visual QA findings below override misleading high proxy scores.

### Overhead multi-hand sample

| Method | Processing fps | Uptime | Dropout transitions | Mean boxes | Visual result |
|---|---:|---:|---:|---:|---|
| Skin YCrCb/HSV | 79.81 | 100% | 0 | 1.896 | Hands found, but colored bins can be false positives |
| Temporal skin | 79.22 | 100% | 0 | 1.924 | Smoothest fast method; inherits bin false positives |
| Motion-gated skin | 60.34 | 53.47% | 7 | 0.660 | Misses stationary hands; fragmented |
| KCF reinit | 49.07 | 100% | 0 | 1.722 | Stable boxes, but detector-seeded and may follow wrong object |
| CSRT reinit | 9.93 | 100% | 0 | 2.000 | Persistent but coarse and too slow for a lightweight fallback |

### Close-up single-hand sample

| Method | Processing fps | Uptime | Dropout transitions | Mean boxes | Visual result |
|---|---:|---:|---:|---:|---|
| Skin YCrCb/HSV | 90.04 | 95.83% | 1 | 0.979 | Clean in this controlled view |
| Temporal skin | 90.72 | 98.61% | 1 | 1.035 | Best local OpenCV result |
| Motion-gated skin | 65.22 | 95.83% | 1 | 1.257 | More jitter and fragmentation |
| KCF reinit | 15.31 | 90.28% | 8 | 0.903 | Loses/drifts from hand repeatedly |
| CSRT reinit | 9.85 | 100% | 0 | 1.000 | Stable, coarse box; high latency |

### Prompted SAM2 short sample

| Measure | Result |
|---|---:|
| Frames | 24 |
| Model load | 0.345 s |
| Inference | 100.217 s |
| Per-frame latency | 4.176 s |
| Processing fps | 0.239 |
| Detection uptime | 79.17% |
| Dropout transitions after valid entry | 0 |
| Automatic prompt | maximum-visible-area keyframe, frame 12 |

SAM2 visually follows the visible hand/forearm contour on the reviewed frames and leaves the red cubes unselected. It correctly has no mask before a meaningful hand region enters the image. The output still includes forearm as part of the prompted object and is not a 21-landmark hand-pose estimate.

## Visual findings and failure modes

1. **Color-only hand segmentation is not object invariant.** Red/orange cube false positives were reduced with a component/edge prior, but wide-view yellow/green bins still overlap the chosen chroma ranges and can be accepted when they touch an image edge.
2. **Motion is not identity.** MOG2 helps distinguish a moving hand from static parts, but cannot support pauses, slow manipulation, or occlusion.
3. **Classical trackers preserve mistakes.** KCF/CSRT can bridge detector dropouts, but they do not determine whether the seeded region is a hand. A false seed becomes a stable false track.
4. **A box is not articulation.** All OpenCV methods return coarse regions, not fingers, pinch state, hand orientation, or grasp event timing.
5. **Prompt timing matters for SAM2.** Prompting a clipped entry frame failed. Selecting the maximum-visible-area keyframe materially improved propagation.
6. **HDR phone video affects fixed color thresholds.** The HLG/Main10 sources are decoded into conventional frames for OpenCV; transfer/tone mapping and exposure changes can shift chroma. Production preprocessing should explicitly normalize HLG to SDR/BT.709 or use a model not dependent on fixed skin colors.
7. **Hands leave and re-enter.** A robust system must detect track termination/re-entry rather than extrapolate indefinitely.

## Ranked practical recommendation

### For the available local environment

1. **Temporal skin + strict geometric/scene constraints:** fastest usable prototype in the controlled close-up, but never the sole safety or robot-control signal.
2. **Detector + short KCF bridge:** acceptable only when an independent detector supplies trustworthy seeds; cap the bridge duration and force re-detection.
3. **Prompted SAM2 offline refinement:** best contour-style result in the reviewed sample, but unusable live at 0.239 fps on CPU.
4. **CSRT:** persistent but too slow/coarse for this role.
5. **Motion-gated skin:** supplementary motion confidence only.

### For the real video-to-robot pipeline

Recommended architecture:

1. **Dedicated hand detector/landmarks** (MediaPipe Hands or a validated hand-specific detector) produces left/right hand identity, wrist/palm/finger landmarks, visibility, and confidence.
2. **Short temporal association** smooths landmark trajectories and bridges only a few frames; it must expire quickly when confidence drops.
3. **Object segmentation/tracking runs separately.** Do not infer the manipulated part from skin masks. Associate hand-to-part through proximity, contact timing, and object motion.
4. **Event extraction** identifies approach, contact, lift, transport, placement, and release from hand/object trajectories rather than raw boxes.
5. **Camera calibration and table homography/depth** convert image coordinates into metric table coordinates. 2D hand tracking alone cannot yield safe robot coordinates.
6. **Human-review checkpoints** validate inferred object identity and target before generating a robot program.

Before selecting a production model, label at least 100–300 representative frames across all six videos with hand boxes/landmarks and occlusion flags. Then report actual precision/recall, ID switches, pixel/metric landmark error, and event-timing error. Without those labels, this test supports engineering selection and visual comparison, not an accuracy guarantee.

## Output guide

- `annotated/overhead_multihand_opencv_comparison_h264.mp4` — six-panel wide-view comparison
- `annotated/closeup_singlehand_opencv_comparison_h264.mp4` — six-panel close-up comparison
- `annotated/closeup_singlehand_sam2_prompted_h264.mp4` — source/SAM2 comparison
- `qa/*_contact.png` — representative visual QA frames
- `benchmark_results.csv` / `.json` — machine-readable metrics
- `inventory.json` — source and package inventory
- `benchmark_hand_tracking.py` — reproducible harness

