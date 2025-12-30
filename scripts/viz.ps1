param(
    [string]$Model = "elasticnet",
    [string]$WeightMode = "none",
    [double]$Alpha = 0.01,
    [double]$L1Ratio = 0.2,
    [string]$DataDir = "data/raw"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Activate venv
. .\.venv\Scripts\Activate.ps1

# Output locations
$artifactRoot = "artifacts"
$oofDir = Join-Path $artifactRoot "oof"
$plotsDir = Join-Path $artifactRoot "plots"

New-Item -ItemType Directory -Force -Path $oofDir | Out-Null
New-Item -ItemType Directory -Force -Path $plotsDir | Out-Null

# Name outputs deterministically (easy to find)
$oofPath = Join-Path $oofDir ("{0}_{1}_a{2}_l1{3}.csv" -f $Model, $WeightMode, $Alpha, $L1Ratio)
$plotOut = Join-Path $plotsDir ("{0}_{1}_a{2}_l1{3}" -f $Model, $WeightMode, $Alpha, $L1Ratio)

# 1) Generate OOF predictions
py -m poverty.viz oof `
  --data-dir $DataDir `
  --model $Model `
  --weight-mode $WeightMode `
  --alpha $Alpha `
  --l1-ratio $L1Ratio `
  --out $oofPath

# 2) Generate plots + metrics report
py -m poverty.viz plots `
  --data-dir $DataDir `
  --oof $oofPath `
  --out-dir $plotOut `
  --model-name ("{0}({1})" -f $Model, $WeightMode)

Write-Host "OOF:   $oofPath"
Write-Host "Plots: $plotOut"
