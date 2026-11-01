# Reproduction guide

The repository supports three levels: hardware-free tests, offline data processing, and Isaac Sim execution. Physical robot execution is intentionally not a one-command public workflow.

## 1. Hardware-free tests

Requirements:

- Python 3.12+
- Git
- a virtual environment

```bash
git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-1-10439770d3.git
cd msoe-ai-club-rockwell-automation-nvidia-hackath-new-team-1-10439770d3
python -m venv .venv
```

Activate the environment, then:

```bash
python -m pip install --upgrade pip
python -m pip install -e "software/robot-control" pytest
python -m pytest software/robot-control/tests -q
python -m pytest software/truck_benchmark/test_benchmark.py -q
python scripts/check_public_release.py
```

The robot-control tests cover safety envelopes, phase control, sensor fusion, depth processing, side-camera reasoning, store behavior, MCP messages, and policy contracts. The truck test audits the observation-only boundary and geometry helpers.

## 2. Video-to-robot compiler

Additional requirements:

- FFmpeg/FFprobe on `PATH`
- OpenCV and NumPy
- MediaPipe
- the official MediaPipe hand-landmarker model downloaded separately

Example:

```bash
python software/video_to_robot/run_pipeline.py \
  --input-dir /path/to/demonstrations \
  --output-dir runs/video_to_robot
```

The source demonstrations are intentionally not included. Use your own short, consented videos. The pipeline README describes the expected files and output schema.

## 3. Observation-only benchmark

The policy/scoring code can be inspected without Isaac Sim. Full data generation and physics execution require a compatible NVIDIA Isaac Sim environment.

From the repository root inside that environment:

```bash
chmod +x software/truck_benchmark/run_suite.sh
SCENES=48 software/truck_benchmark/run_suite.sh runs/truck_observation_benchmark
```

Expected top-level outputs are `report/BENCHMARK_REPORT.md`, `report/ranking.csv`, and `report/benchmark_results.json`. Finalist physics execution uses `execute_physics_finalist.py` and the matrix scripts in the same directory.

## 4. Assembly simulation demos

Each folder under [`software/simulation-demos`](../software/simulation-demos/) contains:

- the Isaac Python script;
- an asset-export helper where required;
- a shell runner;
- a validation summary;
- a compact execution report; and
- representative rendered output.

Asset paths are environment-specific. Point the script at the corresponding folder under [`cad`](../cad/README.md) or update the runner variables before execution.

## 5. Reverse engineering

Start with the READMEs in:

- [`software/reverse-engineering/synthetic`](../software/reverse-engineering/synthetic/)
- [`software/reverse-engineering/physical-orbit`](../software/reverse-engineering/physical-orbit/)

Heavy methods have separate dependency gates. Do not report an unavailable dependency as a reconstruction failure. Freeze the candidate before comparing it with evaluator-only reference geometry.

## 6. Digital twin

The extracted planning core and MCP boundary are in [`apps/digital-twin`](../apps/digital-twin/). The live public tools are available at [theateam.dev](https://www.theateam.dev/). A physical-workcell adapter is intentionally not included in the public example.

## Physical robot warning

Do not replay repository coordinates on a robot. Physical operation requires site-specific calibration, an approved controller interface, verified frames/tooling, collision review, limits, and a qualified operator. Recorded positions are evidence, not universal motion programs.

[Back to documentation](README.md)

