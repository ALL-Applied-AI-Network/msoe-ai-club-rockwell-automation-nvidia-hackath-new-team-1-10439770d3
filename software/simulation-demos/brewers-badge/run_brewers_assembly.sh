#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MODE="${1:-render}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/$STAMP-$MODE"
mkdir -p "$OUT"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ALLOW_ROOT=1
export PYTHONPATH="/workspace/fanuc_er4ia_sim:${PYTHONPATH:-}"

ARGS=(--out-dir "$OUT" --source-root /workspace/fanuc_er4ia_sim --fps 10)
if [[ "$MODE" == "physics" ]]; then
  ARGS+=(--physics-only)
elif [[ "$MODE" != "render" ]]; then
  echo "Usage: $0 [physics|render]" >&2
  exit 2
fi

set +e
timeout --signal=TERM --kill-after=30 2400 \
  /isaac-sim/python.sh "$ROOT/isaac_brewers_assembly.py" "${ARGS[@]}" \
  >"$OUT/run.log" 2>&1
CODE=$?
set -e

if [[ "$MODE" == "render" ]]; then
  for VIEW in hero overhead opposite; do
    if compgen -G "$OUT/frames/$VIEW/f*.png" >/dev/null; then
      ffmpeg -y -loglevel error -framerate 10 -i "$OUT/frames/$VIEW/f%05d.png" \
        -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
        "$OUT/brewers_assembly_${VIEW}.mp4"
    fi
  done
  if [[ -f "$OUT/brewers_assembly_hero.mp4" && -f "$OUT/brewers_assembly_overhead.mp4" && -f "$OUT/brewers_assembly_opposite.mp4" ]]; then
    ffmpeg -y -loglevel error \
      -i "$OUT/brewers_assembly_hero.mp4" \
      -i "$OUT/brewers_assembly_overhead.mp4" \
      -i "$OUT/brewers_assembly_opposite.mp4" \
      -filter_complex "[0:v]scale=1280:720[hero];[1:v]scale=640:360[top];[2:v]scale=640:360[opp];[top][opp]vstack=inputs=2[right];[hero][right]hstack=inputs=2[out]" \
      -map "[out]" -c:v libx264 -crf 19 -preset medium -pix_fmt yuv420p \
      "$OUT/brewers_assembly_three_view.mp4"
  fi
fi

echo "RESULTS=$OUT"
echo "PROCESS_CODE=$CODE"
exit "$CODE"
