#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MODE="${1:-render}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/runs/gearbox/$STAMP-$MODE"
mkdir -p "$OUT"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ALLOW_ROOT=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

ARGS=(--out-dir "$OUT")
if [[ "$MODE" == "physics" ]]; then
  ARGS+=(--physics-only)
elif [[ "$MODE" != "render" ]]; then
  echo "Usage: $0 [physics|render]" >&2
  exit 2
fi

set +e
timeout --signal=TERM --kill-after=30 1800 \
  /isaac-sim/python.sh "$ROOT/isaac_gearbox_assembly.py" "${ARGS[@]}" \
  >"$OUT/run.log" 2>&1
CODE=$?
set -e

if [[ "$MODE" == "render" ]] && compgen -G "$OUT/frames/f*.png" >/dev/null; then
  ffmpeg -y -loglevel error -framerate 12 -i "$OUT/frames/f%05d.png" \
    -c:v libx264 -crf 19 -preset medium -pix_fmt yuv420p \
    "$OUT/gearbox_assembly.mp4"
fi

echo "RESULTS=$OUT"
echo "PROCESS_CODE=$CODE"
exit "$CODE"
