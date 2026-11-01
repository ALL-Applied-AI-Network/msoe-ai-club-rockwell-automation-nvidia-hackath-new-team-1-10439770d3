#!/usr/bin/env bash
set -uo pipefail

WORKSPACE_ROOT="${ATEAM_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
STARTER="${STARTER:-$WORKSPACE_ROOT/fanuc_er4ia_sim}"
OUT="${OUT:-$STARTER/runs/grasp_search}"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
mkdir -p "$OUT"

run_case() {
  local name="$1"
  shift
  echo "START $name $(date -Is)"
  # Isaac occasionally leaves an OmniHub cache lock after a container exits.
  # This sweep is strictly sequential, so no live process can own this file.
  rm -f "$WORKSPACE_ROOT/isaac-cache/ov/_cache.lock"
  timeout --signal=TERM --kill-after=5 120 docker run --rm --gpus all -u 0 \
    --entrypoint /isaac-sim/python.sh \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -v "$WORKSPACE_ROOT:/workspace" \
    -v "$WORKSPACE_ROOT/isaac-cache/kit:/isaac-sim/kit/cache:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/ov:/root/.cache/ov:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/pip:/root/.cache/pip:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "$WORKSPACE_ROOT/isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
    -w /workspace/fanuc_er4ia_sim \
    "$IMAGE" isaac_one_wall_bin_grip.py --physics-only \
    --out "runs/grasp_search/$name.json" "$@" >"$OUT/$name.log" 2>&1 || true

  if [[ ! -s "$OUT/$name.json" ]]; then
    echo "$name INFRASTRUCTURE_TIMEOUT (no report)"
    return 0
  fi

  python3 - "$OUT/$name.json" "$name" <<'PY'
import json
import sys

path, name = sys.argv[1:]
report = json.load(open(path, encoding="utf-8"))
measurements = report.get("measurements", {})
print(
    name,
    "PASS" if report.get("task_complete") else "FAIL",
    "lift", round(measurements.get("measured_lift_mm", -999), 3),
    "slip", round(measurements.get("max_hold_downward_slip_mm", -999), 3),
    "tilt", round(measurements.get("max_lift_and_hold_tilt_deg", -999), 3),
    "gap", round(measurements.get("contact_jaw_gap_mm", -999), 3),
    report.get("acceptance", {}).get("checks", {}),
)
PY
  sleep "${COOLDOWN_SECONDS:-0}"
}

case "${1:-stage1}" in
  stage1)
    run_case i14_j175 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 14 --jaw-target-mm 1.75
    run_case i18_j175 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75
    run_case i22_j175 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 22 --jaw-target-mm 1.75
    run_case i20_j125 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 20 --jaw-target-mm 1.25
    ;;
  stage2)
    # Robustness around the first passing seat: 18 mm insertion, 1.75 mm/jaw.
    run_case nominal_repeat --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75
    run_case mass_080 --mass-kg 0.080 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75
    run_case mass_200 --mass-kg 0.200 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75
    run_case friction_050 --mass-kg 0.133 --friction 0.50 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75
    run_case friction_150 --mass-kg 0.133 --friction 1.50 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75
    run_case xerr_m3 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75 --grip-x-error-mm -3
    run_case xerr_p3 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75 --grip-x-error-mm 3
    run_case yerr_m1 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75 --grip-y-error-mm -1
    run_case yerr_p1 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.75 --grip-y-error-mm 1
    ;;
  stage3)
    # Recover margin for heavier/slipperier bins by increasing seating force.
    run_case heavy_j150 --mass-kg 0.200 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.50
    run_case heavy_j125 --mass-kg 0.200 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.25
    run_case heavy_i20_j150 --mass-kg 0.200 --friction 0.90 --lift-mm 10 --insertion-mm 20 --jaw-target-mm 1.50
    run_case lowfric_j150 --mass-kg 0.133 --friction 0.50 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.50
    run_case xerr_m2 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.50 --grip-x-error-mm -2
    run_case xerr_p2 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 1.50 --grip-x-error-mm 2
    ;;
  stage4)
    # Binary-close tests with catalog-scale force caps. These are deployable
    # command semantics; no partial jaw target is assumed.
    run_case binary_f10 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 10
    run_case binary_f20 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 20
    run_case binary_f34 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34
    run_case binary_f50 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 50
    run_case binary_i20_f34 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 20 --jaw-target-mm 0 --jaw-max-force-n 34
    ;;
  stage5)
    # Robustness of the physically commandable, force-calibrated binary close.
    run_case robust_heavy --mass-kg 0.200 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34
    run_case robust_lowfric --mass-kg 0.133 --friction 0.50 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34
    run_case robust_xm2 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-x-error-mm -2
    run_case robust_xp2 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-x-error-mm 2
    run_case robust_ym1 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-y-error-mm -1
    run_case robust_yp1 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-y-error-mm 1
    run_case robust_yawm3 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-yaw-error-deg -3
    run_case robust_yawp3 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-yaw-error-deg 3
    run_case robust_worst --mass-kg 0.200 --friction 0.50 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-x-error-mm 2 --grip-y-error-mm 1 --grip-yaw-error-deg 3
    ;;
  stage6)
    # Retry only cases that suffered an Isaac startup timeout in stage5.
    run_case retry_heavy --mass-kg 0.200 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34
    run_case retry_xp2 --mass-kg 0.133 --friction 0.90 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-x-error-mm 2
    run_case retry_worst --mass-kg 0.200 --friction 0.50 --lift-mm 10 --insertion-mm 18 --jaw-target-mm 0 --jaw-max-force-n 34 --grip-x-error-mm 2 --grip-y-error-mm 1 --grip-yaw-error-deg 3
    ;;
  *)
    echo "unknown stage: $1" >&2
    exit 2
    ;;
esac
