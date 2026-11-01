#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_recovery_v2_targeted_20261024_v1}"
POLICY="${POLICY:-locked_calibration__depth_table_h6mm_s1_rawcube}"

mkdir -p "$OUT/episodes" "$OUT/report"
cd "$ROOT"

# These two deterministic stress cases defeated recovery v1 after its first
# camera-guided retry.  Run them first so an ineffective change is rejected
# before paying for the complete confirmation matrix.
for spec in "depth_noise 20261117" "combined 20261118" "nominal 20261117"; do
  read -r variation seed <<<"$spec"
  episode="$OUT/episodes/${POLICY}__vision_recovery_v2__${variation}__${seed}"
  mkdir -p "$episode"
  echo "=== $POLICY | vision_recovery_v2 | $variation | $seed ==="
  /isaac-sim/python.sh truck_benchmark/execute_physics_finalist.py \
    --out-dir "$episode" --policy "$POLICY" --controller vision_recovery_v2 \
    --variation "$variation" --seed "$seed" \
    >"$episode/run.log" 2>&1 || true
  tail -n 35 "$episode/run.log"
done

/isaac-sim/python.sh -m truck_benchmark.score_physics \
  --episodes "$OUT/episodes" --out "$OUT/report"
echo "PHASE4_TARGETED_DONE"
echo "RESULTS=$OUT"
