Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    py -m venv .venv
}

try {
    . .\.venv\Scripts\Activate.ps1
} catch {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    . .\.venv\Scripts\Activate.ps1
}

py -m pip install -e ".[dev]" | Out-Null
pytest -q
py -m poverty.validate --data-dir data/raw
