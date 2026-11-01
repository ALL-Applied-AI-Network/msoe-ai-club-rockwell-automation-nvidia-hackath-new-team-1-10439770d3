#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_recovery_v3_targeted_20261024_v1}"
POLICY="${POLICY:-locked_calibration__depth_table_h6mm_s1_rawcube}"

mkdir -p "$OUT/episodes" "$OUT/report"
cd "$ROOT"

# The loaded-bin loss is the discriminating case for v3.  Re-run the two
# camera-recovery failures after this passes so the transport guard cannot
# regress the v2 fixes.
for spec in "nominal 20261117" "depth_noise 20261117" "combined 20261118"; do
  read -r variation seed <<<"$spec"
  episode="$OUT/episodes/${POLICY}__vision_recovery_v3__${variation}__${seed}"
  mkdir -p "$episode"
  echo "=== $POLICY | vision_recovery_v3 | $variation | $seed ==="
  /isaac-sim/python.sh truck_benchmark/execute_physics_finalist.py \
    --out-dir "$episode" --policy "$POLICY" --controller vision_recovery_v3 \
    --variation "$variation" --seed "$seed" \
    >"$episode/run.log" 2>&1 || true
  tail -n 35 "$episode/run.log"
done

/isaac-sim/python.sh -m truck_benchmark.score_physics \
  --episodes "$OUT/episodes" --out "$OUT/report"
echo "PHASE4_V3_TARGETED_DONE"
echo "RESULTS=$OUT"
