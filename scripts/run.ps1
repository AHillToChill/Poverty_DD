[CmdletBinding()]
param(
    [string]$DataDir = "data/raw",

    [ValidateSet("elasticnet","hgb")]
    [string]$Model = "elasticnet",

    [ValidateSet("none","weight","sqrt_weight")]
    [string]$WeightMode = "none",

    [switch]$Viz,
    [switch]$SkipOOF,
    [switch]$SkipInstall,
    [switch]$SkipTests,

    # ElasticNet
    [double]$Alpha = 0.01,
    [double]$L1Ratio = 0.2,

    # HGB (note: set -HgbMaxDepth 0 to mean None)
    [double]$HgbLearningRate = 0.05,
    [int]$HgbMaxIter = 400,
    [int]$HgbMaxDepth = 6,
    [int]$HgbMinSamplesLeaf = 20,
    [double]$HgbL2Regularization = 0.0,

    [string]$ArtifactsDir = "artifacts"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
    }
}

function Fmt([double]$x) {
    return $x.ToString("g", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-ElasticTag([string]$weightMode, [double]$alpha, [double]$l1) {
    return "elasticnet_{0}_a{1}_l1{2}" -f $weightMode, (Fmt $alpha), (Fmt $l1)
}

function Get-HgbTag([string]$weightMode, [int]$maxIter, [double]$lr, [int]$depth, [int]$leaf, [double]$l2) {
    # depth=0 means None; tag uses "None" just for readability
    $dtag = if ($depth -eq 0) { "None" } else { "$depth" }
    return "hgb_{0}_i{1}_lr{2}_d{3}_leaf{4}_l2{5}" -f $weightMode, $maxIter, (Fmt $lr), $dtag, $leaf, (Fmt $l2)
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $root

try {
    if (-not $SkipInstall) {
        $bootstrap = Join-Path $PSScriptRoot "bootstrap.ps1"
        if (Test-Path -LiteralPath $bootstrap) {
            & $bootstrap | Out-Host
        }
    }

    if (-not $SkipTests) {
        python -m pytest | Out-Host
    }

    $tag = switch ($Model) {
        "elasticnet" { Get-ElasticTag $WeightMode $Alpha $L1Ratio }
        "hgb"        { Get-HgbTag $WeightMode $HgbMaxIter $HgbLearningRate $HgbMaxDepth $HgbMinSamplesLeaf $HgbL2Regularization }
        default      { throw "Unknown model: $Model" }
    }

    $oofDir     = Join-Path $ArtifactsDir "oof"
    $metricsDir = Join-Path (Join-Path $ArtifactsDir "metrics") $tag
    $plotsDir   = Join-Path (Join-Path $ArtifactsDir "plots")   $tag

    Ensure-Dir $oofDir
    Ensure-Dir $metricsDir
    Ensure-Dir $plotsDir

    $oofPath = Join-Path $oofDir ("{0}.csv" -f $tag)

    if (-not $SkipOOF) {
        Write-Host ""
        Write-Host ("=== OOF: model={0} weight_mode={1} tag={2} ===" -f $Model, $WeightMode, $tag)

        $args = @(
            "-m","poverty.viz","oof",
            "--data-dir",$DataDir,
            "--model",$Model,
            "--weight-mode",$WeightMode,
            "--out",$oofPath
        )

        if ($Model -eq "elasticnet") {
            $args += @("--alpha",(Fmt $Alpha),"--l1-ratio",(Fmt $L1Ratio))
        }

        if ($Model -eq "hgb") {
            $args += @(
                "--hgb-learning-rate",(Fmt $HgbLearningRate),
                "--hgb-max-iter","$HgbMaxIter",
                "--hgb-max-depth","$HgbMaxDepth",
                "--hgb-min-samples-leaf","$HgbMinSamplesLeaf",
                "--hgb-l2-regularization",(Fmt $HgbL2Regularization)
            )
        }

        python @args | Out-Host
    }
    else {
        if (-not (Test-Path -LiteralPath $oofPath)) {
            throw "SkipOOF was specified but OOF file does not exist: $oofPath"
        }
        Write-Host ""
        Write-Host ("=== Reusing existing OOF: {0} ===" -f $oofPath)
    }

    # Always compute metrics (no plots) so tuning scripts can parse a stable line.
    Write-Host ""
    Write-Host ("=== Metrics (no plots): {0} ===" -f $tag)

    python -m poverty.viz metrics --data-dir $DataDir --oof $oofPath --out-dir $metricsDir | Out-Host

    if ($Viz) {
        Write-Host ""
        Write-Host ("=== Plots: {0} ===" -f $tag)
        python -m poverty.viz plots --data-dir $DataDir --oof $oofPath --out-dir $plotsDir --model-name $tag | Out-Host
        Write-Host ("OOF:   {0}" -f $oofPath)
        Write-Host ("Plots: {0}" -f $plotsDir)
    }
    else {
        Write-Host ("OOF:     {0}" -f $oofPath)
        Write-Host ("Metrics: {0}" -f $metricsDir)
    }
}
finally {
    Pop-Location
}
