Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. .\.venv\Scripts\Activate.ps1
py -m poverty.validate --data-dir data/raw
