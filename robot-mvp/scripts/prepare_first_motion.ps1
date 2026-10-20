param(
    [Parameter(Mandatory = $true)]
    [string]$ContactName,

    [Parameter(Mandatory = $true)]
    [string]$ContactEmail,

    [string]$OfficialClient = "E:\Hackathon data\Hackathon main event\site_crawl\sidclab_2026-10-19\PhysicalAI-Hackathon-2026"
)

$ErrorActionPreference = "Stop"
$MvpRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TemplateRoot = Join-Path $MvpRoot "submission\first_motion"
$SubmissionRoot = Join-Path $OfficialClient "submission"

if ($ContactName -match "[`r`n]") {
    throw "ContactName must be one line."
}
if ($ContactEmail -notmatch "^[^\s@]+@[^\s@]+\.[^\s@]+$") {
    throw "ContactEmail must be a valid email address."
}
if (-not (Test-Path -LiteralPath (Join-Path $OfficialClient "run.py"))) {
    throw "Official SIDC client not found at $OfficialClient"
}

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackupRoot = Join-Path $OfficialClient "submission-backups\$Timestamp"
New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
foreach ($Name in @("submission.yaml", "placement.yaml")) {
    $Existing = Join-Path $SubmissionRoot $Name
    if (Test-Path -LiteralPath $Existing) {
        Copy-Item -LiteralPath $Existing -Destination (Join-Path $BackupRoot $Name)
    }
}

$Submission = Get-Content -Raw (Join-Path $TemplateRoot "submission.yaml")
$Submission = $Submission.Replace("__CONTACT_NAME__", $ContactName)
$Submission = $Submission.Replace("__CONTACT_EMAIL__", $ContactEmail)
[System.IO.File]::WriteAllText(
    (Join-Path $SubmissionRoot "submission.yaml"),
    $Submission,
    [System.Text.UTF8Encoding]::new($false)
)
Copy-Item -LiteralPath (Join-Path $TemplateRoot "placement.yaml") `
    -Destination (Join-Path $SubmissionRoot "placement.yaml") -Force
Copy-Item -LiteralPath (Join-Path $MvpRoot "policy.py") `
    -Destination (Join-Path $OfficialClient "policy.py") -Force
$ClientPackage = Join-Path $OfficialClient "robot_control"
New-Item -ItemType Directory -Path $ClientPackage -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $MvpRoot "robot_control") -Filter "*.py" -File |
    Copy-Item -Destination $ClientPackage -Force

Write-Host "Prepared the official SIDC client for ATeam-SafeMotion-v1."
Write-Host "Previous submission files were backed up under: $BackupRoot"
Write-Host ""
Write-Host "No external request was submitted and no robot was contacted."
Write-Host "Review these files with the assigned operator/mentor:"
Write-Host "  $(Join-Path $SubmissionRoot 'submission.yaml')"
Write-Host "  $(Join-Path $SubmissionRoot 'placement.yaml')"
Write-Host "  $ClientPackage"
Write-Host ""
Write-Host "Declared TCP targets (UF0, mm/deg, open gripper):"
Write-Host "  A = [450, -70, -230, -180, 0, 0, 0]"
Write-Host "  B = [450,  70, -230, -180, 0, 0, 0]"
Write-Host ""
Write-Host "After the organizer confirms the tool frame, installed fingers, default-scene clearance, and slot:"
Write-Host "  cd `"$OfficialClient`""
Write-Host "  uv run run.py --submit"
Write-Host "Then enter the printed request id in the SIDC website's Calendar tab and book a 30-minute slot."
