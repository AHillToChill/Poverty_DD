param(
    [string]$DataDir = "data/raw",
    [switch]$SkipInstall,
    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

$root = Get-ProjectRoot
Push-Location $root
try {
    $py = Ensure-Venv $root
    if (-not $SkipInstall) { Install-Project $py }
    if (-not $SkipTests) { Run-Tests $py }
    & $py -m poverty.validate --data-dir $DataDir
}
finally {
    Pop-Location
}
