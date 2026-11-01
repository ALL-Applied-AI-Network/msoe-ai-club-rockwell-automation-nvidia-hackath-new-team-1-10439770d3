#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/runs/truck_physics_finalists}"
POLICIES="${POLICIES:-depth_table_h6mm_s1 hybrid_depth_anchor_hue_model_table_inventory hybrid_depth_anchor_hue_model_multiview_inventory}"
CONTROLLERS="${CONTROLLERS:-direct_nominal geometry_lever calibrated_guarded}"
SEEDS="${SEEDS:-20261024}"
VARIATIONS="${VARIATIONS:-nominal combined}"
PERCEPTION_REPORT="${PERCEPTION_REPORT:-}"

mkdir -p "$OUT/episodes" "$OUT/report"
cd "$ROOT"

for policy in $POLICIES; do
  for controller in $CONTROLLERS; do
    for variation in $VARIATIONS; do
      for seed in $SEEDS; do
        episode="$OUT/episodes/${policy}__${controller}__${variation}__${seed}"
        mkdir -p "$episode"
        echo "=== $policy | $controller | $variation | $seed ==="
        /isaac-sim/python.sh truck_benchmark/execute_physics_finalist.py \
          --out-dir "$episode" --policy "$policy" --controller "$controller" \
          --variation "$variation" --seed "$seed" \
          >"$episode/run.log" 2>&1 || true
        tail -n 35 "$episode/run.log"
      done
    done
  done
done

score_args=(
  -m truck_benchmark.score_physics
  --episodes "$OUT/episodes"
  --out "$OUT/report"
)
if [[ -n "$PERCEPTION_REPORT" ]]; then
  score_args+=(--perception-report "$PERCEPTION_REPORT")
fi
/isaac-sim/python.sh "${score_args[@]}"
echo "RESULTS=$OUT"
