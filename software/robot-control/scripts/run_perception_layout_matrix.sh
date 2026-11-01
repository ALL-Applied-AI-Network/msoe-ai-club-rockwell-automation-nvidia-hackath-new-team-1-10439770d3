#!/usr/bin/env bash
set -uo pipefail

WORKSPACE_ROOT="${ATEAM_WORKSPACE_ROOT:-/home/yt-ateamdd25/workspace}"
STARTER="$WORKSPACE_ROOT/fanuc_er4ia_sim"
OUT="$STARTER/runs/perception_layout_matrix"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
mkdir -p "$OUT"

run_case() {
  local name="$1"
  shift
  local container="ateam-layout-$name"
  rm -f "$WORKSPACE_ROOT/isaac-cache/ov/_cache.lock"
  docker rm -f "$container" >/dev/null 2>&1 || true
  rm -rf "$OUT/$name"
  mkdir -p "$OUT/$name"
  echo "START $name $(date -Is)"
  docker run -d --name "$container" --gpus all -u 0 \
    --entrypoint /isaac-sim/python.sh \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
    -v "$WORKSPACE_ROOT:/workspace" \
    -v "$WORKSPACE_ROOT/isaac-cache/kit:/isaac-sim/kit/cache:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/ov:/root/.cache/ov:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/pip:/root/.cache/pip:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
    -w /workspace/fanuc_er4ia_sim \
    "$IMAGE" isaac_one_wall_bin_grip.py --physics-only --validate-perception \
    --out "runs/perception_layout_matrix/$name/report.json" \
    --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 \
    --jaw-target-mm 0 --jaw-max-force-n 34 "$@" >/dev/null

  local status
  for _ in $(seq 1 90); do
    status="$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)"
    [[ "$status" != "running" ]] && break
    sleep 1
  done
  status="$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)"
  if [[ "$status" == "running" ]]; then
    docker stop "$container" >/dev/null
    echo "$name INFRASTRUCTURE_TIMEOUT"
  fi
  docker logs "$container" >"$OUT/$name/run.log" 2>&1 || true
  docker rm "$container" >/dev/null 2>&1 || true

  python3 - "$OUT/$name/report.json" "$name" <<'PY'
import json
import sys
from pathlib import Path

path, name = Path(sys.argv[1]), sys.argv[2]
if not path.exists():
    print(name, "NO_REPORT")
    raise SystemExit
report = json.loads(path.read_text())
perception = report.get("perception") or {}
measurements = report.get("measurements") or {}
print(
    name,
    "PASS" if report.get("task_complete") else "FAIL",
    "perception_drives_motion", perception.get("drives_motion"),
    "wall_error_mm", round(perception.get("wall_error_mm", -999), 3),
    "lift_mm", round(measurements.get("measured_lift_mm", -999), 3),
    "slip_mm", round(measurements.get("max_hold_downward_slip_mm", -999), 3),
    "tilt_deg", round(measurements.get("max_lift_and_hold_tilt_deg", -999), 3),
    "error", report.get("error"),
)
PY
}

case "${1:-all}" in
  all)
    run_case center --bin-x-mm 450 --bin-y-mm 0 --bin-yaw-deg 0
    run_case near_left --bin-x-mm 380 --bin-y-mm -70 --bin-yaw-deg -12
    run_case far_right --bin-x-mm 520 --bin-y-mm 70 --bin-yaw-deg 12
    run_case left_back --bin-x-mm 400 --bin-y-mm 100 --bin-yaw-deg 20
    run_case right_front --bin-x-mm 500 --bin-y-mm -100 --bin-yaw-deg -20
    ;;
  rejected)
    run_case near_left_retry --bin-x-mm 380 --bin-y-mm -70 --bin-yaw-deg -12
    run_case far_right_retry --bin-x-mm 520 --bin-y-mm 70 --bin-yaw-deg 12
    run_case left_back_retry --bin-x-mm 400 --bin-y-mm 100 --bin-yaw-deg 20
    ;;
  *)
    echo "usage: $0 [all|rejected]" >&2
    exit 2
    ;;
esac

echo "DONE $(date -Is)"
