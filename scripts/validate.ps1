Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
  [string]$DataDir = "data/raw",
  [ValidateSet("elasticnet","hgb","both")]
  [string]$Model = "both",
  [ValidateSet("none","weight","sqrt_weight")]
  [string]$WeightMode = "none",

  # ElasticNet
  [double]$Alpha = 0.01,
  [double]$L1Ratio = 0.2,

  # HGB
  [int]$HgbMaxIter = 400,
  [double]$HgbLearningRate = 0.05,
  [int]$HgbMaxDepth = 6,
  [int]$HgbMinSamplesLeaf = 20,
  [double]$HgbL2 = 0.0,
  [switch]$HgbEarlyStopping,

  [switch]$NoInstall
)

& "$PSScriptRoot\bootstrap.ps1" -NoInstall:$NoInstall

$runElastic = ($Model -eq "elasticnet" -or $Model -eq "both")
$runHgb     = ($Model -eq "hgb"       -or $Model -eq "both")

$cmd = @(
  "-m", "poverty.validate",
  "--data-dir", $DataDir,
  "--weight-mode", $WeightMode
)

if ($runElastic) { $cmd += "--run-elasticnet" }
if ($runHgb)     { $cmd += "--run-hgb" }

# ElasticNet args
$cmd += @("--alpha", "$Alpha", "--l1-ratio", "$L1Ratio")

# HGB args
$cmd += @(
  "--hgb-max-iter", "$HgbMaxIter",
  "--hgb-learning-rate", "$HgbLearningRate",
  "--hgb-max-depth", "$HgbMaxDepth",
  "--hgb-min-samples-leaf", "$HgbMinSamplesLeaf",
  "--hgb-l2", "$HgbL2"
)
if ($HgbEarlyStopping) { $cmd += "--hgb-early-stopping" }

py @cmd
