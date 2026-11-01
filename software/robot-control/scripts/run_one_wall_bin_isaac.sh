#!/usr/bin/env bash
set -euo pipefail

# Run inside an Isaac Sim 6 container. The starter path is mounted read-only;
# evidence is written to /work/out so the caller can preserve it.
STARTER=${STARTER:-/starter}
WORK=${WORK:-/work}
OUT=${OUT:-/work/out}
mkdir -p "$OUT/frames"

export PYTHONPATH="$STARTER:${PYTHONPATH:-}"
cd "$WORK"

/isaac-sim/python.sh simulation/isaac_one_wall_bin_grip.py \
  --out "$OUT/report.json" \
  --frames-dir "$OUT/frames" \
  --mass-kg "${BIN_MASS_KG:-0.133}" \
  --friction "${GRIP_FRICTION:-0.90}" \
  --lift-mm "${LIFT_MM:-10}" \
  --insertion-mm "${INSERTION_MM:-14}"

python3 - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("/work/out/report.json").read_text())
if not report.get("task_complete"):
    raise SystemExit("Isaac process ended, but the measured acceptance gate failed")
print("VERIFIED_ONE_WALL_GRIP", json.dumps(report["measurements"], indent=2))
PY

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -loglevel error -framerate 30 -i "$OUT/frames/f%05d.png" \
    -c:v libx264 -pix_fmt yuv420p "$OUT/one_wall_bin_grip.mp4"
fi
