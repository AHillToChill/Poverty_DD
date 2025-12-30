Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. .\.venv\Scripts\Activate.ps1
pytest -q
