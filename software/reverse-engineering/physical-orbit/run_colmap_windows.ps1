param(
    [Parameter(Mandatory = $true)][string]$Images,
    [Parameter(Mandatory = $true)][string]$Workspace,
    [ValidateSet('SIFT', 'LOMA')][string]$Feature = 'LOMA',
    [ValidateSet('incremental', 'global')][string]$Mapper = 'incremental',
    [string]$Colmap = '',
    [switch]$CpuFeatures,
    [switch]$Dense
)

$ErrorActionPreference = 'Stop'
if (-not $Colmap) {
    $Colmap = (Get-ChildItem -LiteralPath 'tools\colmap-4.2.1' -Recurse -Filter 'colmap.exe' | Select-Object -First 1 -ExpandProperty FullName)
}
if (-not (Test-Path -LiteralPath $Colmap)) { throw "COLMAP executable not found: $Colmap" }
$Images = (Resolve-Path -LiteralPath $Images).Path
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
$Workspace = (Resolve-Path -LiteralPath $Workspace).Path
$Database = Join-Path $Workspace 'database.db'
$Sparse = Join-Path $Workspace 'sparse'
$DensePath = Join-Path $Workspace 'dense'
$Logs = Join-Path $Workspace 'logs'
New-Item -ItemType Directory -Force -Path $Sparse, $Logs | Out-Null

function Invoke-Colmap {
    param([string]$Stage, [string[]]$Arguments)
    $log = Join-Path $Logs "$Stage.log"
    $start = Get-Date
    "START=$($start.ToString('o'))" | Set-Content -LiteralPath $log
    "COMMAND=$Colmap $($Arguments -join ' ')" | Add-Content -LiteralPath $log
    & $Colmap @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "COLMAP stage $Stage failed with exit code $LASTEXITCODE; see $log" }
    $elapsed = ((Get-Date) - $start).TotalSeconds
    "ELAPSED_S=$elapsed" | Add-Content -LiteralPath $log
}

$cameraParams = '309.8199818460763,320,180,-0.019600611286561938'
$featureType = if ($Feature -eq 'LOMA') { 'LOMA_B' } else { 'SIFT' }
$featureGpu = if ($CpuFeatures) { '0' } else { '1' }
$featureArgs = @(
    'feature_extractor',
    '--database_path', $Database,
    '--image_path', $Images,
    '--ImageReader.camera_model', 'SIMPLE_RADIAL',
    '--ImageReader.single_camera', '1',
    '--ImageReader.camera_params', $cameraParams,
    '--FeatureExtraction.type', $featureType,
    '--FeatureExtraction.use_gpu', $featureGpu,
    '--FeatureExtraction.gpu_index', '0'
)
if ($Feature -eq 'LOMA') {
    $featureArgs += @('--LomaExtraction.max_num_features', '4096', '--LomaExtraction.use_bf16', '0')
} else {
    $featureArgs += @('--SiftExtraction.max_num_features', '8192', '--SiftExtraction.peak_threshold', '0.003')
}
Invoke-Colmap -Stage '01_feature_extractor' -Arguments $featureArgs

$matchingType = if ($Feature -eq 'LOMA' -and $CpuFeatures) { 'LOMA_BRUTEFORCE' } elseif ($Feature -eq 'LOMA') { 'LOMA_B' } else { 'SIFT_BRUTEFORCE' }
Invoke-Colmap -Stage '02_exhaustive_matcher' -Arguments @(
    'exhaustive_matcher',
    '--database_path', $Database,
    '--FeatureMatching.type', $matchingType,
    '--FeatureMatching.use_gpu', $featureGpu,
    '--FeatureMatching.gpu_index', '0',
    '--FeatureMatching.guided_matching', '1',
    '--TwoViewGeometry.min_num_inliers', '10'
)

if ($Mapper -eq 'global') {
    Invoke-Colmap -Stage '03_global_mapper' -Arguments @(
        'global_mapper', '--database_path', $Database, '--image_path', $Images, '--output_path', $Sparse,
        '--GlobalMapper.ba_refine_focal_length', '0', '--GlobalMapper.ba_refine_principal_point', '0', '--GlobalMapper.ba_refine_extra_params', '0'
    )
} else {
    Invoke-Colmap -Stage '03_mapper' -Arguments @(
        'mapper', '--database_path', $Database, '--image_path', $Images, '--output_path', $Sparse,
        '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_principal_point', '0', '--Mapper.ba_refine_extra_params', '0',
        '--Mapper.min_num_matches', '10', '--Mapper.init_min_num_inliers', '10', '--Mapper.abs_pose_min_num_inliers', '10'
    )
}

$models = Get-ChildItem -LiteralPath $Sparse -Directory | Sort-Object { (Get-Item (Join-Path $_.FullName 'images.bin') -ErrorAction SilentlyContinue).Length } -Descending
if (-not $models) { throw 'No sparse model was produced' }
$model = $models[0].FullName
Invoke-Colmap -Stage '04_model_analyzer' -Arguments @('model_analyzer', '--path', $model)
Invoke-Colmap -Stage '05_model_to_ply' -Arguments @('model_converter', '--input_path', $model, '--output_path', (Join-Path $Workspace 'sparse_points.ply'), '--output_type', 'PLY')

if ($Dense) {
    New-Item -ItemType Directory -Force -Path $DensePath | Out-Null
    Invoke-Colmap -Stage '06_undistort' -Arguments @('image_undistorter', '--image_path', $Images, '--input_path', $model, '--output_path', $DensePath, '--output_type', 'COLMAP', '--max_image_size', '640')
    Invoke-Colmap -Stage '07_patch_match' -Arguments @('patch_match_stereo', '--workspace_path', $DensePath, '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', '1', '--PatchMatchStereo.gpu_index', '0')
    Invoke-Colmap -Stage '08_fusion' -Arguments @('stereo_fusion', '--workspace_path', $DensePath, '--workspace_format', 'COLMAP', '--input_type', 'geometric', '--output_path', (Join-Path $DensePath 'fused.ply'))
    Invoke-Colmap -Stage '09_poisson' -Arguments @('poisson_mesher', '--input_path', (Join-Path $DensePath 'fused.ply'), '--output_path', (Join-Path $DensePath 'poisson.ply'), '--PoissonMeshing.trim', '8')
}

$manifest = [ordered]@{
    colmap_version = (& $Colmap version 2>&1 | Out-String).Trim()
    images = $Images
    feature = $Feature
    matcher = $matchingType
    mapper = $Mapper
    dense_requested = [bool]$Dense
    cpu_features = [bool]$CpuFeatures
    selected_sparse_model = $model
    completed_utc = (Get-Date).ToUniversalTime().ToString('o')
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Workspace 'run_manifest.json')
$manifest | ConvertTo-Json -Depth 5
