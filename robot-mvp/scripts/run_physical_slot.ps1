param(
    [Parameter(Mandatory = $true)]
    [string]$Operator,

    [ValidateRange(1, 10)]
    [int]$Rollouts = 3,

    [string]$OfficialClient = "E:\Hackathon data\Hackathon main event\site_crawl\sidclab_2026-10-19\PhysicalAI-Hackathon-2026"
)

$ErrorActionPreference = "Stop"
$MvpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Cli = Join-Path $MvpRoot ".venv\Scripts\ateam-robot.exe"
$Request = Join-Path $OfficialClient ".request.json"
$RequiredPhrase = "I AM AT THE CELL WITH THE OPERATOR AND E-STOP READY"

if (-not (Test-Path -LiteralPath $Cli)) {
    throw "Robot MVP is not installed. Run scripts\install_local.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $OfficialClient "run.py"))) {
    throw "Official SIDC client not found at $OfficialClient"
}
if (-not (Test-Path -LiteralPath $Request)) {
    throw "No private SIDC request file exists. Prepare, submit, and obtain approval before a physical slot."
}

Push-Location $OfficialClient
try {
    $Status = uv run run.py --status 2>&1 | Out-String
} finally {
    Pop-Location
}
Write-Host $Status.Trim()
if ($Status -notmatch "(?im)^request\s+\S+:\s+approved") {
    throw "The SIDC request is not approved. This script will not arm physical mode."
}

Write-Host ""
Write-Host "PHYSICAL MOTION CANDIDATE"
Write-Host "  Robot: FANUC ER-4iA"
Write-Host "  Frame: controller UF0; operator must confirm it matches the active physical setup"
Write-Host "  Tool: current installed gripper/fingers; open for the complete path"
Write-Host "  A: [450, -70, -230, -180, 0, 0, 0]"
Write-Host "  B: [450,  70, -230, -180, 0, 0, 0]"
Write-Host "  Scene: official default screws and yellow bin"
Write-Host ""
Write-Host "The nearby operator must have homed the arm, checked the active frame and clearance,"
Write-Host "and be holding responsibility for the cell stop controls."
$Confirmation = Read-Host "Type exactly: $RequiredPhrase"
if ($Confirmation -cne $RequiredPhrase) {
    throw "Confirmation did not match. Physical mode remains disarmed."
}

$env:ATEAM_MVP_ROOT = $MvpRoot
$env:ATEAM_ROBOT_DB = Join-Path $MvpRoot "runtime\robot_plans.sqlite3"
$env:ATEAM_ALLOW_PHYSICAL = "YES"
$env:ATEAM_EXECUTION_MODE = "physical"
$env:PYTHONUTF8 = "1"

try {
    $PlanIds = @()
    for ($Index = 1; $Index -le $Rollouts; $Index++) {
        $PlanJson = & $Cli draft-demo --label "A-Team supervised physical commissioning motion $Index/$Rollouts"
        $Plan = $PlanJson | ConvertFrom-Json
        & $Cli approve $Plan.id --mode physical --operator $Operator --confirm $RequiredPhrase | Out-Null
        $PlanIds += $Plan.id
    }
    Write-Host "Approved $Rollouts local physical plans: $($PlanIds -join ', ')"
    Write-Host "Waiting for the official cell operator..."

    Push-Location $OfficialClient
    try {
        uv run run.py
    } finally {
        Pop-Location
    }
} finally {
    if ($PlanIds) {
        $Recent = & $Cli status | ConvertFrom-Json
        foreach ($PlanId in $PlanIds) {
            $Current = $Recent | Where-Object { $_.id -eq $PlanId } | Select-Object -First 1
            if ($Current.status -in @("pending", "approved")) {
                & $Cli cancel $PlanId | Out-Null
            }
        }
    }
    Remove-Item Env:ATEAM_ALLOW_PHYSICAL -ErrorAction SilentlyContinue
    Remove-Item Env:ATEAM_EXECUTION_MODE -ErrorAction SilentlyContinue
    Write-Host "A-Team physical mode is disarmed."
}
