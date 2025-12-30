param(
    [switch]$Viz,
    [string]$VizModel = "elasticnet",
    [string]$WeightMode = "none",
    [double]$Alpha = 0.01,
    [double]$L1Ratio = 0.2,
    [string]$DataDir = "data/raw"
)

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
py -m poverty.validate --data-dir $DataDir

if ($Viz) {
    $artifactRoot = "artifacts"
    $oofDir = Join-Path $artifactRoot "oof"
    $plotsDir = Join-Path $artifactRoot "plots"

    New-Item -ItemType Directory -Force -Path $oofDir | Out-Null
    New-Item -ItemType Directory -Force -Path $plotsDir | Out-Null

    $oofPath = Join-Path $oofDir ("{0}_{1}_a{2}_l1{3}.csv" -f $VizModel, $WeightMode, $Alpha, $L1Ratio)
    $plotOut = Join-Path $plotsDir ("{0}_{1}_a{2}_l1{3}" -f $VizModel, $WeightMode, $Alpha, $L1Ratio)

    py -m poverty.viz oof `
        --data-dir $DataDir `
        --model $VizModel `
        --weight-mode $WeightMode `
        --alpha $Alpha `
        --l1-ratio $L1Ratio `
        --out $oofPath

    py -m poverty.viz plots `
        --data-dir $DataDir `
        --oof $oofPath `
        --out-dir $plotOut `
        --model-name ("{0}({1})" -f $VizModel, $WeightMode)

    Write-Host "OOF:   $oofPath"
    Write-Host "Plots: $plotOut"
}
