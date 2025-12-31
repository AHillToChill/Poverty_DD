Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Ensure-Venv([string]$Root) {
    $pyExe = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $pyExe) {
        return $pyExe
    }

    Write-Host "Creating virtual environment (.venv)..."
    Push-Location $Root
    try {
        py -m venv .venv
    } finally {
        Pop-Location
    }

    if (-not (Test-Path $pyExe)) {
        throw "Virtual environment was created, but $pyExe was not found."
    }
    return $pyExe
}

function Install-Project([string]$PyExe) {
    & $PyExe -m pip install -U pip | Out-Null
    & $PyExe -m pip install -e ".[dev]" | Out-Null
}

function Run-Tests([string]$PyExe) {
    & $PyExe -m pytest -q
}

function Format-FloatTag([double]$x) {
    $s = $x.ToString("G", [System.Globalization.CultureInfo]::InvariantCulture)
    return $s.Replace(".", "p")
}
