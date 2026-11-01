# Video-to-Robot Evidence Compiler

This project turns short human assembly demonstrations into four linked artifacts:

1. an annotated video with persistent hand/part overlays;
2. frame-level observations with timestamps and confidence;
3. an auditable manipulation timeline (`approach`, `contact`, `transfer`, `insert`, `release`);
4. robot-oriented instructions that explicitly distinguish measured evidence from assumptions.

The default implementation is deliberately hybrid. MediaPipe supplies 21-point hand
kinematics, deterministic color/geometry segmentation supplies the known red cubes and
green bin in this dataset, and a hysteretic contact state machine compiles manipulation
events. A mask-tracker backend (SAM 2.1/SAM 3.1) can replace the color segmenter without
changing the evidence or instruction schema.

## Run

```bash
python software/video_to_robot/run_pipeline.py \
  --input-dir /path/to/demonstrations \
  --output-dir runs/video_to_robot
```

The renderer preserves the original AAC audio when `ffmpeg` is available.
Download the official MediaPipe hand-landmarker task file separately and place
it at the path configured by the runner; the third-party binary is not vendored.

## Important scope boundary

The supplied phone clips are six sequential takes, not synchronized calibrated camera
streams. They support 2D hand/part tracking, temporal action extraction, and cross-take
consensus. They do **not** support metric 6D trajectories or millimetre insertion
tolerances. Those require synchronized cameras, intrinsics/extrinsics, depth, and CAD.

