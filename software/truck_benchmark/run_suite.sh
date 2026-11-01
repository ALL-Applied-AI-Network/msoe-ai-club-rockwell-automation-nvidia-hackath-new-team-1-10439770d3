#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_observation_benchmark}"
SCENES="${SCENES:-48}"
SEED="${SEED:-20261024}"

mkdir -p "$OUT"
cd "$ROOT"

/isaac-sim/python.sh -m unittest truck_benchmark.test_benchmark
/isaac-sim/python.sh truck_benchmark/capture_randomized_dataset.py \
  --out "$OUT/dataset" --scenes "$SCENES" --seed "$SEED" --headless
/isaac-sim/python.sh -m truck_benchmark.run_policies \
  --observations "$OUT/dataset/observations" --out "$OUT/predictions"
/isaac-sim/python.sh -m truck_benchmark.score_predictions \
  --predictions "$OUT/predictions" \
  --evaluator "$OUT/dataset/evaluator_only" \
  --out "$OUT/report"

echo "RESULTS=$OUT"

