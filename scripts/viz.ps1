param(
    [ValidateSet("plots", "metrics", "compare")]
    [string]$Cmd = "plots",

    [string]$DataDir = "data/raw",

    [string]$Oof = "",
    [string]$OutDir = "",
    [string]$Out = "",
    [string]$ModelName = "",

    [string[]]$OofList = @(),

    [switch]$SkipInstall,
    [switch]$SkipTests
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

    if ($Cmd -eq "plots") {
        if ([string]::IsNullOrWhiteSpace($Oof) -or [string]::IsNullOrWhiteSpace($OutDir)) {
            throw "For -Cmd plots, provide -Oof and -OutDir."
        }
        $args = @("-m","poverty.viz","plots","--data-dir",$DataDir,"--oof",$Oof,"--out-dir",$OutDir)
        if (-not [string]::IsNullOrWhiteSpace($ModelName)) { $args += @("--model-name",$ModelName) }
        & $py @args
    }
    elseif ($Cmd -eq "metrics") {
        if ([string]::IsNullOrWhiteSpace($Oof) -or [string]::IsNullOrWhiteSpace($Out)) {
            throw "For -Cmd metrics, provide -Oof and -Out."
        }
        & $py -m poverty.viz metrics --data-dir $DataDir --oof $Oof --out $Out
    }
    elseif ($Cmd -eq "compare") {
        if (-not $OofList -or [string]::IsNullOrWhiteSpace($OutDir)) {
            throw "For -Cmd compare, provide -OofList and -OutDir."
        }
        $args = @("-m","poverty.viz","compare","--data-dir",$DataDir,"--out-dir",$OutDir,"--oof")
        $args += $OofList
        & $py @args
    }
}
finally {
    Pop-Location
}
