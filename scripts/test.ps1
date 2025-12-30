Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
  [switch]$NoInstall
)

& "$PSScriptRoot\bootstrap.ps1" -NoInstall:$NoInstall
pytest -q
