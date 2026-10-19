param(
    [string]$OfficialClient = "E:\Hackathon data\Hackathon main event\site_crawl\sidclab_2026-10-19\PhysicalAI-Hackathon-2026"
)

$ErrorActionPreference = "Stop"
$MvpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Push-Location $MvpRoot
try {
    uv sync --dev
    uv run pytest
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath (Join-Path $OfficialClient "run.py"))) {
    throw "Official client not found at $OfficialClient"
}

Copy-Item -LiteralPath (Join-Path $MvpRoot "policy.py") -Destination (Join-Path $OfficialClient "policy.py") -Force

$Python = Join-Path $MvpRoot ".venv\Scripts\python.exe"
$Existing = codex mcp list 2>$null | Select-String -SimpleMatch "ateam-robot"
if ($Existing) {
    codex mcp remove ateam-robot | Out-Null
}
codex mcp add ateam-robot --env "ATEAM_MVP_ROOT=$MvpRoot" -- $Python -m robot_control.mcp_server

Write-Host "Installed A-Team robot MVP and registered the ateam-robot MCP server."
Write-Host "Official client: $OfficialClient"
Write-Host "MCP database: $(Join-Path $MvpRoot 'runtime\robot_plans.sqlite3')"

