param(
    [string]$DataDir = "data/raw",

    [ValidateSet("none", "weight", "sqrt_weight")]
    [string]$WeightMode = "none",

    [double[]]$LearningRates = @(0.03, 0.05, 0.08),
    [int[]]$MaxIters = @(600, 1200),
    [int[]]$MaxDepths = @(4, 6),
    [int[]]$MinSamplesLeafs = @(10, 20, 50),
    [double[]]$L2Regs = @(0.0, 0.1),

    [int]$MaxRuns = 0,
    [switch]$VizTop,
    [int]$TopK = 3,

    [bool]$SkipInstall = $true,
    [bool]$SkipTests = $true,

    [string]$ArtifactsDir = "artifacts",
    [string]$OutCsv = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

$root = Get-ProjectRoot
Push-Location $root
try {
    $py = Ensure-Venv $root

    if (-not $SkipInstall) { Install-Project $py }
    if (-not $SkipTests) { Run-Tests $py }

    Ensure-Dir (Join-Path $ArtifactsDir "tuning")
    Ensure-Dir (Join-Path $ArtifactsDir "oof")
    Ensure-Dir (Join-Path $ArtifactsDir "metrics")
    Ensure-Dir (Join-Path $ArtifactsDir "plots")

    if ([string]::IsNullOrWhiteSpace($OutCsv)) {
        $stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
        $OutCsv = Join-Path $ArtifactsDir ("tuning\hgb_sweep_{0}.csv" -f $stamp)
    }

    $results = New-Object System.Collections.Generic.List[object]
    $n = 0

    foreach ($lr in $LearningRates) {
        foreach ($it in $MaxIters) {
            foreach ($d in $MaxDepths) {
                foreach ($leaf in $MinSamplesLeafs) {
                    foreach ($l2 in $L2Regs) {

                        $lrTag = Format-FloatTag $lr
                        $l2Tag = Format-FloatTag $l2
                        $dTag = if ($d -in 0, -1) { "None" } else { "$d" }
                        $tag = "hgb_${WeightMode}_i${it}_lr${lrTag}_d${dTag}_leaf${leaf}_l2${l2Tag}"

                        Write-Host ""
                        Write-Host ("=== HGB sweep: lr={0} it={1} depth={2} leaf={3} l2={4} ===" -f $lr, $it, $d, $leaf, $l2)

                        $metricsPath = Join-Path $ArtifactsDir ("metrics\{0}.csv" -f $tag)

                        try {
                            & "$PSScriptRoot\run.ps1" `
                                -DataDir $DataDir `
                                -Model hgb `
                                -WeightMode $WeightMode `
                                -HgbLearningRate $lr `
                                -HgbMaxIter $it `
                                -HgbMaxDepth $d `
                                -HgbMinSamplesLeaf $leaf `
                                -HgbL2Regularization $l2 `
                                -SkipInstall:([bool]$SkipInstall) `
                                -SkipTests:([bool]$SkipTests) `
                                -ArtifactsDir $ArtifactsDir `
                                -Tag $tag | Out-Null
                        }
                        catch {
                            Write-Warning ("Run failed for tag={0}. Error: {1}" -f $tag, $_.Exception.Message)
                        }

                        if (-not (Test-Path $metricsPath)) {
                            Write-Warning "Missing metrics file. Marking as +Inf."
                            $results.Add([pscustomobject]@{
                                tag = $tag
                                learning_rate = $lr
                                max_iter = $it
                                max_depth = $d
                                min_samples_leaf = $leaf
                                l2_regularization = $l2
                                blended_mean = [double]::PositiveInfinity
                                poverty_wmape_mean = [double]::PositiveInfinity
                                household_mape_mean = [double]::PositiveInfinity
                            }) | Out-Null
                        } else {
                            $rep = Import-Csv $metricsPath
                            $meanRow = $rep | Where-Object { $_.survey_id -eq "MEAN" } | Select-Object -First 1

                            if (-not $meanRow) {
                                Write-Warning "No MEAN row found. Marking as +Inf."
                                $blended = [double]::PositiveInfinity
                                $pwmape = [double]::PositiveInfinity
                                $hmape = [double]::PositiveInfinity
                            } else {
                                $blended = [double]$meanRow.blended
                                $pwmape = [double]$meanRow.poverty_wmape
                                $hmape = [double]$meanRow.household_mape
                            }

                            $results.Add([pscustomobject]@{
                                tag = $tag
                                learning_rate = $lr
                                max_iter = $it
                                max_depth = $d
                                min_samples_leaf = $leaf
                                l2_regularization = $l2
                                blended_mean = $blended
                                poverty_wmape_mean = $pwmape
                                household_mape_mean = $hmape
                            }) | Out-Null
                        }

                        $n++
                        if ($MaxRuns -gt 0 -and $n -ge $MaxRuns) { break }
                    }
                    if ($MaxRuns -gt 0 -and $n -ge $MaxRuns) { break }
                }
                if ($MaxRuns -gt 0 -and $n -ge $MaxRuns) { break }
            }
            if ($MaxRuns -gt 0 -and $n -ge $MaxRuns) { break }
        }
        if ($MaxRuns -gt 0 -and $n -ge $MaxRuns) { break }
    }

    $sorted = $results | Sort-Object blended_mean, poverty_wmape_mean, household_mape_mean
    $sorted | Export-Csv -NoTypeInformation $OutCsv

    Write-Host ""
    Write-Host ("Wrote sweep results: {0}" -f $OutCsv)
    Write-Host ""
    $sorted | Select-Object -First 15 | Format-Table -AutoSize

    if ($VizTop) {
        $k = [Math]::Max(1, $TopK)
        Write-Host ""
        Write-Host ("Generating plots for top {0} configurations..." -f $k)

        foreach ($r in ($sorted | Select-Object -First $k)) {
            $tag = $r.tag
            $oofPath = Join-Path $ArtifactsDir ("oof\{0}.csv" -f $tag)
            $plotsDir = Join-Path $ArtifactsDir ("plots\{0}" -f $tag)
            Ensure-Dir $plotsDir

            if (-not (Test-Path $oofPath)) {
                Write-Warning ("Missing OOF file for tag={0}; skipping plots." -f $tag)
                continue
            }

            Write-Host ""
            Write-Host ("=== PLOTS (top): {0} ===" -f $tag)
            & $py -m poverty.viz plots --data-dir $DataDir --oof $oofPath --out-dir $plotsDir --model-name $tag
        }
    }
}
finally {
    Pop-Location
}
