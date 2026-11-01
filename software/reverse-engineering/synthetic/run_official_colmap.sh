#!/usr/bin/env bash
set -u
export QT_QPA_PLATFORM=offscreen

ROOT=/work/isaac_run

run_part() {
    name="$1"
    source="$ROOT/official_inputs/colmap/$name"
    result="$ROOT/official_results/colmap_mvs/$name"
    dense="$result/dense"
    mkdir -p "$dense"
    start=$(date +%s)
    colmap image_undistorter \
        --image_path "$source/images" --input_path "$source/sparse/0" \
        --output_path "$dense" --output_type COLMAP \
        > "$result/undistort.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$result/undistort.exit"
    if [ "$status" -eq 0 ]; then
        colmap patch_match_stereo \
            --workspace_path "$dense" --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true \
            > "$result/patchmatch.log" 2>&1
        status=$?
        printf '%s\n' "$status" > "$result/patchmatch.exit"
    fi
    if [ "$status" -eq 0 ]; then
        colmap stereo_fusion \
            --workspace_path "$dense" --workspace_format COLMAP \
            --input_type geometric --output_path "$result/fused.ply" \
            > "$result/fusion.log" 2>&1
        status=$?
        printf '%s\n' "$status" > "$result/fusion.exit"
    fi
    if [ "$status" -eq 0 ]; then
        colmap poisson_mesher --input_path "$result/fused.ply" \
            --output_path "$result/poisson.ply" \
            > "$result/poisson.log" 2>&1
        status=$?
        printf '%s\n' "$status" > "$result/poisson.exit"
    fi
    printf '%s\n' "$(( $(date +%s) - start ))" > "$result/wall_seconds.txt"
}

run_part gearboxopen
run_part gearboxclosed
printf 'complete\n' > "$ROOT/official_results/colmap_mvs/all.done"
