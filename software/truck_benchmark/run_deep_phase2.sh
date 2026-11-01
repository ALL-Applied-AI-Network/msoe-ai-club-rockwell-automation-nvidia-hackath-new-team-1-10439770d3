#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_deep_phase2_20261024}"
ACTIVE_SCENES="${ACTIVE_SCENES:-24}"
ACTIVE_SEED="${ACTIVE_SEED:-20261107}"
ORIGINAL_DATASET="${ORIGINAL_DATASET:-$ROOT/runs/truck_benchmark_20261023_v1/dataset}"
TRIVIEW_DATASET="${TRIVIEW_DATASET:-$ROOT/runs/truck_triview_benchmark_20261023_v1/dataset}"

mkdir -p "$OUT"
cd "$ROOT"

POLICIES=(
  locked_calibration__depth_table_h6mm_s1_rawcube
  locked_calibration__rgb_hue_nearest_r1.00_k13_model_fit_inventory
  locked_calibration__rgb_hue_nearest_r1.00_k21_model_fit_inventory
  locked_calibration__rgb_hue_nearest_r1.00_k31_model_fit_inventory
  locked_calibration__hybrid_depth_anchor_hue_model_table_inventory
  locked_calibration__active_wrist_selective_depth_recovery_rawcube
  locked_calibration__active_wrist_selective_rgbd_recovery_rawcube
  locked_calibration__active_wrist_selective_rgbd_robust_occlusion_recovery_rawcube
)

run_existing_dataset() {
  local label="$1"
  local dataset="$2"
  [[ -d "$dataset/observations" ]] || return 0
  mkdir -p "$OUT/$label"
  /isaac-sim/python.sh -m truck_benchmark.run_policies \
    --observations "$dataset/observations" \
    --out "$OUT/$label/predictions" \
    --policies "${POLICIES[@]}"
  /isaac-sim/python.sh -m truck_benchmark.score_predictions \
    --predictions "$OUT/$label/predictions" \
    --evaluator "$dataset/evaluator_only" \
    --out "$OUT/$label/report"
}

/isaac-sim/python.sh -m unittest truck_benchmark.test_benchmark
run_existing_dataset original48 "$ORIGINAL_DATASET"
run_existing_dataset heldout24 "$TRIVIEW_DATASET"

ACTIVE_DATASET="$OUT/active_wrist_dataset"
if [[ ! -f "$ACTIVE_DATASET/dataset_manifest.json" ]]; then
  /isaac-sim/python.sh truck_benchmark/capture_randomized_dataset.py \
    --out "$ACTIVE_DATASET" --scenes "$ACTIVE_SCENES" --seed "$ACTIVE_SEED" \
    --active-wrist-scan --headless
fi

mkdir -p "$OUT/active_wrist"
/isaac-sim/python.sh -m truck_benchmark.run_policies \
  --observations "$ACTIVE_DATASET/observations" \
  --out "$OUT/active_wrist/predictions" \
  --policies "${POLICIES[@]}"
/isaac-sim/python.sh -m truck_benchmark.score_predictions \
  --predictions "$OUT/active_wrist/predictions" \
  --evaluator "$ACTIVE_DATASET/evaluator_only" \
  --out "$OUT/active_wrist/report"

echo "RESULTS=$OUT"
