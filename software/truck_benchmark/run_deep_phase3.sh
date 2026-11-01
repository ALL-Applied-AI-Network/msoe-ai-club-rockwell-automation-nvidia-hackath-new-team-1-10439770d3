#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_deep_phase3_20261024}"
SCENES="${SCENES:-24}"
SEED="${SEED:-20261113}"
DATASET="$OUT/dataset"

POLICIES=(
  locked_calibration__depth_table_h6mm_s1_rawcube
  locked_calibration__hybrid_depth_anchor_hue_model_table_inventory
  locked_calibration__active_wrist_selective_rgbd_robust_occlusion_recovery_rawcube
  locked_calibration__active_wrist_selective_rgbd_robust_occlusion_oblique_recovery_rawcube
)

mkdir -p "$OUT"
cd "$ROOT"
/isaac-sim/python.sh -m unittest truck_benchmark.test_benchmark

if [[ ! -f "$DATASET/dataset_manifest.json" ]]; then
  /isaac-sim/python.sh truck_benchmark/capture_randomized_dataset.py \
    --out "$DATASET" --scenes "$SCENES" --seed "$SEED" \
    --active-oblique-scan --headless
fi

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
