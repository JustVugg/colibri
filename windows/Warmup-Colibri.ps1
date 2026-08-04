<#
.SYNOPSIS
  Warm the expert-usage pin (.coli_usage) by running diverse prompts, so the engine
  pins the hottest experts into RAM/VRAM on the next start (raises hit rate). Thin
  wrapper over repo\c\warmup.ps1 with sane defaults + toolchain PATH.

.PARAMETER Model
  Model directory. Defaults to $env:COLI_MODEL.

.PARAMETER Rounds
  Passes over the 30-prompt set (default 1). More = deeper, more accurate pin.

.PARAMETER Ngen
  Tokens generated per prompt (default 32).

.PARAMETER Backend
  auto (default, launcher decides) | gpu (force device 0) | cpu (--gpu none).
  Warm on the SAME backend you infer with — routing differs slightly between them.

.PARAMETER CudaDense
  Put dense weights on the GPU during warmup (frees RAM for expert cache).

.EXAMPLE
  .\Warmup-Colibri.ps1                        # 1 round, GPU auto, default model
.EXAMPLE
  .\Warmup-Colibri.ps1 -Rounds 3 -CudaDense   # deeper warm, dense on GPU
#>
[CmdletBinding()]
param(
    [string]$Model,
    [int]$Rounds = 1,
    [int]$Ngen = 32,
    [ValidateSet('auto','gpu','cpu')][string]$Backend = 'auto',
    [switch]$CudaDense,
    # Optional file with one extra prompt per line, forwarded to warmup.ps1 —
    # lets you warm the cache on your own domain's prompts.
    [string]$PromptFile,
    # No $PSScriptRoot default here: PS 5.1 leaves it empty while evaluating param
    # defaults when the script is started via `powershell -File` from another
    # process, which made Join-Path throw. Resolved in the body instead.
    [string]$RepoRoot
)

$ErrorActionPreference = 'Stop'
if (-not $RepoRoot) {
    $RepoRoot = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..'
}
$env:PATH = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path','User') + ';' + $env:PATH

if (-not $Model) {
    if ($env:COLI_MODEL) { $Model = $env:COLI_MODEL }
    else { throw "No model given: pass -Model <dir> or set COLI_MODEL." }
}
if (-not (Test-Path $Model)) { throw "Model dir not found: $Model  (pass -Model <dir>)" }
$Model = (Resolve-Path $Model).Path

$warmup = Join-Path (Resolve-Path $RepoRoot) 'c\warmup.ps1'
if (-not (Test-Path $warmup)) { throw "warmup.ps1 not found at $warmup" }

# Scrub the stale env var that crashes the launcher; set CUDA_DENSE explicitly.
Remove-Item Env:CUDA_EXPERT_GB -ErrorAction SilentlyContinue
if ($CudaDense) { $env:CUDA_DENSE = '1'; Write-Host "CUDA_DENSE=1 (dense on GPU)" -ForegroundColor Green }
else { Remove-Item Env:CUDA_DENSE -ErrorAction SilentlyContinue }

Write-Host "Warmup: $Model  ·  $Rounds round(s) × $Ngen tok/prompt  ·  backend $Backend" -ForegroundColor Cyan
$wargs = @{ Model = $Model; Rounds = $Rounds; Ngen = $Ngen; Backend = $Backend }
if ($PromptFile) { $wargs.PromptFile = $PromptFile }
& $warmup @wargs
