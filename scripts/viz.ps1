param(
    [ValidateSet("elasticnet","hgb")]
    [string]$Model = "elasticnet",

    [ValidateSet("none","weight","sqrt_weight")]
    [string]$WeightMode = "none",

    [string]$DataDir = "data/raw",

    # ElasticNet params
    [double]$Alpha = 0.01,
    [double]$L1Ratio = 0.2,

    # HGB params
    [int]$HgbMaxIter = 400,
    [double]$HgbLearningRate = 0.05,
    [int]$HgbMaxDepth = 6,
    [int]$HgbMinSamplesLeaf = 20,
    [double]$HgbL2 = 0.0,
    [switch]$HgbEarlyStopping
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

. .\.venv\Scripts\Activate.ps1

$artifactRoot = "artifacts"
$oofDir = Join-Path $artifactRoot "oof"
$plotsDir = Join-Path $artifactRoot "plots"

New-Item -ItemType Directory -Force -Path $oofDir | Out-Null
New-Item -ItemType Directory -Force -Path $plotsDir | Out-Null

if ($Model -eq "elasticnet") {
    $tag = ("elasticnet_{0}_a{1}_l1{2}" -f $WeightMode, $Alpha, $L1Ratio)
}
elseif ($Model -eq "hgb") {
    $tag = ("hgb_{0}_i{1}_lr{2}_d{3}_leaf{4}_l2{5}" -f $WeightMode, $HgbMaxIter, $HgbLearningRate, $HgbMaxDepth, $HgbMinSamplesLeaf, $HgbL2)
    if ($HgbEarlyStopping) { $tag = $tag + "_es" }
}
else {
    throw "Unknown model: $Model"
}

$oofPath = Join-Path $oofDir ($tag + ".csv")
$plotOut = Join-Path $plotsDir $tag

$oofArgs = @(
    "-m","poverty.viz","oof",
    "--data-dir",$DataDir,
    "--model",$Model,
    "--weight-mode",$WeightMode,
    "--out",$oofPath
)

if ($Model -eq "elasticnet") {
    $oofArgs += @("--alpha",$Alpha,"--l1-ratio",$L1Ratio)
} else {
    $oofArgs += @(
        "--hgb-max-iter",$HgbMaxIter,
        "--hgb-learning-rate",$HgbLearningRate,
        "--hgb-max-depth",$HgbMaxDepth,
        "--hgb-min-samples-leaf",$HgbMinSamplesLeaf,
        "--hgb-l2",$HgbL2
    )
    if ($HgbEarlyStopping) { $oofArgs += @("--hgb-early-stopping") }
}

& py @oofArgs

$plotsArgs = @(
    "-m","poverty.viz","plots",
    "--data-dir",$DataDir,
    "--oof",$oofPath,
    "--out-dir",$plotOut,
    "--model-name",("{0}({1})" -f $Model, $WeightMode)
)
& py @plotsArgs

Write-Host "OOF:   $oofPath"
Write-Host "Plots: $plotOut"
