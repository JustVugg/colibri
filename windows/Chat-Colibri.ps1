<#
.SYNOPSIS
  Start an interactive chat session with colibri (`coli chat`) on Windows/NVIDIA.
  Thin, chat-focused front door over Run-Colibri.ps1: sane model default, GPU auto,
  and the -CudaDense switch to fill VRAM with the dense weights.

.DESCRIPTION
  `Run-Colibri.ps1 -Action chat` already does this, but a named Chat script is the
  obvious thing to reach for. Defaults the model to $env:COLI_MODEL, else it falls
  through to Run-Colibri's interactive wizard, so a bare `.\Chat-Colibri.ps1` works.

.PARAMETER Model
  Model directory. Defaults to $env:COLI_MODEL, else the interactive wizard picks one.

.PARAMETER CudaDense
  Put the ~11 GB dense weights on the GPU. Fills VRAM (16 GB card lands ~14 GB) and
  frees that much RAM for a larger warm-expert cache. Recommended on a 16 GB box.

.PARAMETER VramGB
  Force the VRAM expert-tier budget in GB (0 = auto; launcher sizes from free VRAM).

.PARAMETER RamGB
  RAM budget in GB (0 = auto; engine uses ~88% of available RAM).

.EXAMPLE
  .\Chat-Colibri.ps1                         # chat, GPU auto, default model
.EXAMPLE
  .\Chat-Colibri.ps1 -CudaDense              # chat, dense on GPU (fills VRAM)
.EXAMPLE
  .\Chat-Colibri.ps1 -Model D:\Qwen3-30B -CudaDense
#>
[CmdletBinding()]
param(
    [string]$Model,
    [switch]$CudaDense,
    [double]$VramGB = 0,
    [int]$RamGB = 0,
    [Parameter(ValueFromRemainingArguments=$true)]$Passthrough
)

$ErrorActionPreference = 'Stop'

# No model? Honor COLI_MODEL, else fall through to Run-Colibri's interactive
# wizard (it lists your models, asks 3 questions, and explains every choice).
if (-not $Model -and $env:COLI_MODEL) { $Model = $env:COLI_MODEL }

# Delegate to Run-Colibri.ps1 with Action=chat. It handles PATH refresh, stale-env
# scrubbing, CUDA auto-enable, and the --vram/--ram flag mapping.
$run = Join-Path $PSScriptRoot 'Run-Colibri.ps1'
if (-not (Test-Path $run)) { throw "Run-Colibri.ps1 not found at $run" }

$fwd = @{ Action = 'chat'; VramGB = $VramGB; RamGB = $RamGB }
if ($Model) { $fwd.Model = $Model }          # absent -> Run-Colibri wizard takes over
if ($CudaDense) { $fwd.CudaDense = $true }
# Bind by NAME into the remaining-args parameter (positional would collide with
# Run-Colibri's [string]$Mirror). See Web-Colibri.ps1 for the same fix.
if ($Passthrough) { $fwd.Passthrough = $Passthrough }

Write-Host "Chat: $Model  ·  GPU auto$(if($CudaDense){' · CUDA_DENSE'})" -ForegroundColor Cyan
& $run @fwd
