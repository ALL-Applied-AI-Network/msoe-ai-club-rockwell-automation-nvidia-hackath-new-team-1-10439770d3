param(
    [string]$Operator = "Abdallah Benelmadjat",
    [string]$OfficialClient = "E:\Hackathon data\Hackathon main event\site_crawl\sidclab_2026-10-19\PhysicalAI-Hackathon-2026"
)

$ErrorActionPreference = "Stop"
$MvpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:ATEAM_MVP_ROOT = $MvpRoot
$env:ATEAM_ROBOT_DB = Join-Path $MvpRoot "runtime\robot_plans.sqlite3"
$env:ATEAM_EXECUTION_MODE = "mock"
$env:PYTHONUTF8 = "1"

$PlanJson = & (Join-Path $MvpRoot ".venv\Scripts\ateam-robot.exe") draft-demo
$Plan = $PlanJson | ConvertFrom-Json
& (Join-Path $MvpRoot ".venv\Scripts\ateam-robot.exe") approve $Plan.id --mode mock --operator $Operator | Out-Null

Push-Location $OfficialClient
try {
    uv run run.py --mock --trials 1 --trial-s 3 --instruction "Move safely from point A to point B with the gripper open"
} finally {
    Pop-Location
}
