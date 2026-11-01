#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_replan_matrix_20261024_v1}"
POLICY="${POLICY:-locked_calibration__depth_table_h6mm_s1_rawcube}"
CONTROLLERS="${CONTROLLERS:-calibrated_guarded vision_replan}"
VARIATIONS="${VARIATIONS:-nominal depth_noise combined}"
SEEDS="${SEEDS:-20261107 20261108}"

mkdir -p "$OUT/episodes" "$OUT/report"
cd "$ROOT"

for controller in $CONTROLLERS; do
  for variation in $VARIATIONS; do
    for seed in $SEEDS; do
      episode="$OUT/episodes/${POLICY}__${controller}__${variation}__${seed}"
      mkdir -p "$episode"
      echo "=== $POLICY | $controller | $variation | $seed ==="
      /isaac-sim/python.sh truck_benchmark/execute_physics_finalist.py \
        --out-dir "$episode" --policy "$POLICY" --controller "$controller" \
        --variation "$variation" --seed "$seed" \
        >"$episode/run.log" 2>&1 || true
      tail -n 35 "$episode/run.log"
    done
  done
done

/isaac-sim/python.sh -m truck_benchmark.score_physics \
  --episodes "$OUT/episodes" --out "$OUT/report"
echo "REPLAN_MATRIX_DONE"
echo "RESULTS=$OUT"
