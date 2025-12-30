[CmdletBinding()]
param(
    [string]$DataDir = "data/raw",
    [ValidateSet("none","weight","sqrt_weight")]
    [string]$WeightMode = "none",

    [double[]]$LearningRates = @(0.03, 0.05, 0.08),
    [int[]]$MaxIters = @(600, 1200),
    [int[]]$MaxDepths = @(4, 6),
    [int[]]$MinSamplesLeafs = @(10, 20, 50),
    [double[]]$L2Regs = @(0.0, 0.1),

    [int]$MaxRuns = 0,     # 0 = all combos; otherwise stop after this many
    [switch]$VizTop,
    [int]$TopK = 3,

    [switch]$SkipInstall,
    [switch]$SkipTests,

    [string]$ArtifactsDir = "artifacts"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Fmt([double]$x) {
    return $x.ToString("g", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Ensure-Dir([string]$p) {
    if (-not (Test-Path -LiteralPath $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
    }
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $root

try {
    $resultsDir = Join-Path $ArtifactsDir "tuning"
    Ensure-Dir $resultsDir

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outCsv = Join-Path $resultsDir ("hgb_sweep_{0}.csv" -f $stamp)

    $rows = New-Object System.Collections.Generic.List[object]
    $n = 0

    foreach ($lr in $LearningRates) {
        foreach ($it in $MaxIters) {
            foreach ($d in $MaxDepths) {
                foreach ($leaf in $MinSamplesLeafs) {
                    foreach ($l2 in $L2Regs) {

                        $n++
                        if ($MaxRuns -gt 0 -and $n -gt $MaxRuns) { break }

                        Write-Host ""
                        Write-Host ("=== HGB sweep: lr={0} it={1} depth={2} leaf={3} l2={4} ===" -f (Fmt $lr), $it, $d, $leaf, (Fmt $l2))

                        $runArgs = @{
                            DataDir              = $DataDir
                            Model                = "hgb"
                            WeightMode           = $WeightMode
                            HgbLearningRate      = [double]$lr
                            HgbMaxIter           = [int]$it
                            HgbMaxDepth          = [int]$d
                            HgbMinSamplesLeaf    = [int]$leaf
                            HgbL2Regularization  = [double]$l2
                            SkipInstall          = $SkipInstall
                            SkipTests            = $SkipTests
                            ArtifactsDir         = $ArtifactsDir
                        }

                        $out = & "$PSScriptRoot\run.ps1" @runArgs 2>&1 | Out-String

                        $score = [double]::PositiveInfinity
                        if ($out -match "MEAN_BLENDED=([0-9eE\+\-\.]+)") {
                            $score = [double]$Matches[1]
                        }
                        else {
                            Write-Warning "Could not parse MEAN_BLENDED from run output. Marking as +Inf."
                        }

                        $rows.Add([pscustomobject]@{
                            blended_mean       = $score
                            learning_rate      = [double]$lr
                            max_iter           = [int]$it
                            max_depth          = [int]$d
                            min_samples_leaf   = [int]$leaf
                            l2_regularization  = [double]$l2
                            weight_mode        = $WeightMode
                        }) | Out-Null

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

    $sorted = $rows | Sort-Object blended_mean
    $sorted | Export-Csv -NoTypeInformation -Path $outCsv

    Write-Host ""
    Write-Host ("Wrote sweep results: {0}" -f $outCsv)

    Write-Host ""
    Write-Host ("Top {0} runs (lower is better):" -f $TopK)
    $sorted | Select-Object -First $TopK | Format-Table -AutoSize

    if ($VizTop) {
        Write-Host ""
        Write-Host ("Generating plots for TopK={0} (reusing existing OOF)..." -f $TopK)

        $top = $sorted | Select-Object -First $TopK
        foreach ($r in $top) {
            $runArgs2 = @{
                DataDir              = $DataDir
                Model                = "hgb"
                WeightMode           = $WeightMode
                HgbLearningRate      = [double]$r.learning_rate
                HgbMaxIter           = [int]$r.max_iter
                HgbMaxDepth          = [int]$r.max_depth
                HgbMinSamplesLeaf    = [int]$r.min_samples_leaf
                HgbL2Regularization  = [double]$r.l2_regularization
                SkipInstall          = $true
                SkipTests            = $true
                SkipOOF              = $true
                Viz                  = $true
                ArtifactsDir         = $ArtifactsDir
            }

            & "$PSScriptRoot\run.ps1" @runArgs2 | Out-Host
        }
    }
}
finally {
    Pop-Location
}
