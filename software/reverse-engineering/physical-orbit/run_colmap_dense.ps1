param(
    [Parameter(Mandatory = $true)][string]$Images,
    [Parameter(Mandatory = $true)][string]$Workspace,
    [string]$Colmap = ''
)

$ErrorActionPreference = 'Stop'
if (-not $Colmap) {
    $Colmap = (Get-ChildItem -LiteralPath 'tools\colmap-4.2.1' -Recurse -Filter 'colmap.exe' | Select-Object -First 1 -ExpandProperty FullName)
}
$Images = (Resolve-Path -LiteralPath $Images).Path
$Workspace = (Resolve-Path -LiteralPath $Workspace).Path
$Sparse = Join-Path $Workspace 'sparse'
$Dense = Join-Path $Workspace 'dense'
$Logs = Join-Path $Workspace 'logs'
$models = Get-ChildItem -LiteralPath $Sparse -Directory | Sort-Object { (Get-Item (Join-Path $_.FullName 'images.bin') -ErrorAction SilentlyContinue).Length } -Descending
if (-not $models) { throw "No sparse model under $Sparse" }
$model = $models[0].FullName
New-Item -ItemType Directory -Force -Path $Dense, $Logs | Out-Null

function Invoke-Colmap {
    param([string]$Stage, [string[]]$Arguments)
    $log = Join-Path $Logs "$Stage.log"
    $start = Get-Date
    "START=$($start.ToString('o'))`nCOMMAND=$Colmap $($Arguments -join ' ')" | Set-Content -LiteralPath $log
    & $Colmap @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "COLMAP stage $Stage failed with exit code $LASTEXITCODE; see $log" }
    "ELAPSED_S=$(((Get-Date) - $start).TotalSeconds)" | Add-Content -LiteralPath $log
}

Invoke-Colmap '06_undistort' @('image_undistorter', '--image_path', $Images, '--input_path', $model, '--output_path', $Dense, '--output_type', 'COLMAP', '--max_image_size', '640')
Invoke-Colmap '07_patch_match' @('patch_match_stereo', '--workspace_path', $Dense, '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', '1', '--PatchMatchStereo.gpu_index', '0')
Invoke-Colmap '08_fusion' @('stereo_fusion', '--workspace_path', $Dense, '--workspace_format', 'COLMAP', '--input_type', 'geometric', '--output_path', (Join-Path $Dense 'fused.ply'))
Invoke-Colmap '09_poisson' @('poisson_mesher', '--input_path', (Join-Path $Dense 'fused.ply'), '--output_path', (Join-Path $Dense 'poisson.ply'), '--PoissonMeshing.trim', '8')
Get-ChildItem -LiteralPath $Dense -File | Select-Object Name, Length
