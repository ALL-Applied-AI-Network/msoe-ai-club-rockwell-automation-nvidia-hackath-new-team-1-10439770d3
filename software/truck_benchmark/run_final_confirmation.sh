#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_final_confirmation_20261024}"
SCENES="${SCENES:-24}"
SEED="${SEED:-20261123}"
DATASET="$OUT/dataset"

POLICIES=(
  locked_calibration__depth_table_h6mm_s1_rawcube
  locked_calibration__hybrid_depth_anchor_hue_model_table_inventory_truckachromatic
  locked_calibration__task_hybrid_oblique_cube_recovery_achromatic_truck
)

mkdir -p "$OUT"
cd "$ROOT"
/isaac-sim/python.sh -m unittest truck_benchmark.test_benchmark
/isaac-sim/python.sh truck_benchmark/capture_randomized_dataset.py \
  --out "$DATASET" --scenes "$SCENES" --seed "$SEED" \
  --active-oblique-scan --headless
mkdir -p "$OUT/evaluation"
/isaac-sim/python.sh -m truck_benchmark.run_policies \
  --observations "$DATASET/observations" \
  --out "$OUT/evaluation/predictions" \
  --policies "${POLICIES[@]}"
/isaac-sim/python.sh -m truck_benchmark.score_predictions \
  --predictions "$OUT/evaluation/predictions" \
  --evaluator "$DATASET/evaluator_only" \
  --out "$OUT/evaluation/report"
echo "RESULTS=$OUT"
