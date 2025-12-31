param(
    [string]$DataDir = "data/raw",

    [ValidateSet("elasticnet", "hgb")]
    [string]$Model = "elasticnet",

    [ValidateSet("none", "weight", "sqrt_weight")]
    [string]$WeightMode = "none",

    [switch]$Viz,
    [switch]$Validate,

    [switch]$SkipInstall,
    [switch]$SkipTests,

    [string]$ArtifactsDir = "artifacts",
    [string]$Tag = "",

    [double]$Alpha = 0.01,
    [double]$L1Ratio = 0.2,
    [int]$MaxIter = 5000,

    [double]$HgbLearningRate = 0.05,
    [int]$HgbMaxIter = 600,
    [int]$HgbMaxDepth = 6,
    [int]$HgbMinSamplesLeaf = 20,
    [double]$HgbL2Regularization = 0.0,
    [switch]$HgbEarlyStopping
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

$root = Get-ProjectRoot
Push-Location $root
try {
    $py = Ensure-Venv $root

    if (-not $SkipInstall) {
        Write-Host "Installing project (editable, dev deps)..."
        Install-Project $py
    } else {
        Write-Host "Skipping install."
    }

    if (-not $SkipTests) {
        Write-Host "Running tests..."
        Run-Tests $py
    } else {
        Write-Host "Skipping tests."
    }

    if ($Validate) {
        Write-Host "Running sanity validation (poverty.validate)..."
        & $py -m poverty.validate --data-dir $DataDir
    }

    Ensure-Dir (Join-Path $ArtifactsDir "oof")
    Ensure-Dir (Join-Path $ArtifactsDir "metrics")
    Ensure-Dir (Join-Path $ArtifactsDir "plots")

    if ([string]::IsNullOrWhiteSpace($Tag)) {
        if ($Model -eq "elasticnet") {
            $a = Format-FloatTag $Alpha
            $l1 = Format-FloatTag $L1Ratio
            $Tag = "elasticnet_${WeightMode}_a${a}_l1${l1}"
        } elseif ($Model -eq "hgb") {
            $lr = Format-FloatTag $HgbLearningRate
            $l2 = Format-FloatTag $HgbL2Regularization
            $d = if ($HgbMaxDepth -in 0, -1) { "None" } else { "$HgbMaxDepth" }
            $Tag = "hgb_${WeightMode}_i${HgbMaxIter}_lr${lr}_d${d}_leaf${HgbMinSamplesLeaf}_l2${l2}"
        } else {
            throw "Unexpected Model=$Model"
        }
    }

    $oofPath = Join-Path $ArtifactsDir ("oof\{0}.csv" -f $Tag)
    $metricsPath = Join-Path $ArtifactsDir ("metrics\{0}.csv" -f $Tag)
    $plotsDir = Join-Path $ArtifactsDir ("plots\{0}" -f $Tag)

    Write-Host ""
    Write-Host ("=== OOF: model={0} weight_mode={1} tag={2} ===" -f $Model, $WeightMode, $Tag)

    $oofArgs = @(
        "-m", "poverty.viz", "oof",
        "--data-dir", $DataDir,
        "--model", $Model,
        "--weight-mode", $WeightMode,
        "--out", $oofPath
    )

    if ($Model -eq "elasticnet") {
        $oofArgs += @("--alpha", $Alpha, "--l1-ratio", $L1Ratio, "--max-iter", $MaxIter)
    } elseif ($Model -eq "hgb") {
        $oofArgs += @(
            "--hgb-learning-rate", $HgbLearningRate,
            "--hgb-max-iter", $HgbMaxIter,
            "--hgb-max-depth", $HgbMaxDepth,
            "--hgb-min-samples-leaf", $HgbMinSamplesLeaf,
            "--hgb-l2-regularization", $HgbL2Regularization
        )
        if ($HgbEarlyStopping) {
            $oofArgs += "--hgb-early-stopping"
        }
    }

    & $py @oofArgs

    if ($Viz) {
        Ensure-Dir $plotsDir
        Write-Host ""
        Write-Host ("=== PLOTS: {0} ===" -f $plotsDir)

        & $py -m poverty.viz plots `
            --data-dir $DataDir `
            --oof $oofPath `
            --out-dir $plotsDir `
            --model-name $Tag

        Write-Host ""
        Write-Host ("OOF:   {0}" -f $oofPath)
        Write-Host ("Plots: {0}" -f $plotsDir)
    } else {
        Write-Host ""
        Write-Host ("=== METRICS: {0} ===" -f $metricsPath)

        & $py -m poverty.viz metrics `
            --data-dir $DataDir `
            --oof $oofPath `
            --out $metricsPath

        Write-Host ""
        Write-Host ("OOF:     {0}" -f $oofPath)
        Write-Host ("Metrics: {0}" -f $metricsPath)
    }
}
finally {
    Pop-Location
}
