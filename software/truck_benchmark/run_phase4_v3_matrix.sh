#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_recovery_v3_matrix_20261024_v1}"
POLICY="${POLICY:-locked_calibration__depth_table_h6mm_s1_rawcube}"
VARIATIONS="${VARIATIONS:-nominal depth_noise combined}"
SEEDS="${SEEDS:-20261117 20261118}"

mkdir -p "$OUT/episodes" "$OUT/report"
cd "$ROOT"

for variation in $VARIATIONS; do
  for seed in $SEEDS; do
    episode="$OUT/episodes/${POLICY}__vision_recovery_v3__${variation}__${seed}"
    mkdir -p "$episode"
    echo "=== $POLICY | vision_recovery_v3 | $variation | $seed ==="
    /isaac-sim/python.sh truck_benchmark/execute_physics_finalist.py \
      --out-dir "$episode" --policy "$POLICY" --controller vision_recovery_v3 \
      --variation "$variation" --seed "$seed" \
      >"$episode/run.log" 2>&1 || true
    tail -n 35 "$episode/run.log"
  done
done

/isaac-sim/python.sh -m truck_benchmark.score_physics \
  --episodes "$OUT/episodes" --out "$OUT/report"
echo "PHASE4_V3_MATRIX_DONE"
echo "RESULTS=$OUT"
