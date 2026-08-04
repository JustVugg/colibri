<#
.SYNOPSIS
  Launch the colibri web dashboard (`coli web`): OpenAI-compatible API + live metrics
  UI on one port, auto-opened in your browser. Builds the frontend on first run.

.DESCRIPTION
  `coli chat` is the TERMINAL REPL; the web DASHBOARD is `coli web` (see README).
  The dashboard is a Vite/React app in repo\web that must be compiled to repo\web\dist
  once (npm install && npm run build). This script does that build automatically if
  dist is missing, then delegates to Run-Colibri.ps1 -Action web for GPU auto-enable,
  stale-env scrubbing, and --vram/--ram mapping.

.PARAMETER Model
  Model directory. Defaults to $env:COLI_MODEL.

.PARAMETER CudaDense
  Put the ~11 GB dense weights on the GPU (fills VRAM, frees RAM for experts).

.PARAMETER Port
  Dashboard/API port (default 8000).

.PARAMETER Rebuild
  Force a fresh `npm run build` of the web UI even if dist already exists.

.PARAMETER VramGB
  Force VRAM expert-tier budget in GB (0 = auto).

.EXAMPLE
  .\Web-Colibri.ps1 -CudaDense          # build UI if needed, serve dashboard, dense on GPU
.EXAMPLE
  .\Web-Colibri.ps1 -Port 8080 -Rebuild
#>
[CmdletBinding()]
param(
    [string]$Model,
    [switch]$CudaDense,
    [int]$Port = 8000,
    [switch]$Rebuild,
    [double]$VramGB = 0,
    [int]$RamGB = 0,
    [Parameter(ValueFromRemainingArguments=$true)]$Passthrough
)

$ErrorActionPreference = 'Stop'

if (-not $Model) {
    if ($env:COLI_MODEL) { $Model = $env:COLI_MODEL }
    else { throw "No model given: pass -Model <dir> or set COLI_MODEL." }
}

# The engine serves static files from web\dist (coli -> dirname(HERE)/web/dist,
# HERE = <repo>\c). Build it once if it's not there, or on -Rebuild.
$webDir  = Join-Path $PSScriptRoot '..\web'
$distIdx = Join-Path $webDir 'dist\index.html'
if (-not (Test-Path $webDir)) { throw "web source not found at $webDir" }

if ($Rebuild -or -not (Test-Path $distIdx)) {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) { throw "npm not found on PATH; install Node.js to build the dashboard (or run 'coli serve' for API-only)." }
    Write-Host "Building web dashboard (one-time)  ->  $webDir\dist" -ForegroundColor Cyan
    Push-Location $webDir
    try {
        if (-not (Test-Path (Join-Path $webDir 'node_modules'))) {
            Write-Host "  npm install ..." -ForegroundColor DarkGray
            & $npm.Source install
            if ($LASTEXITCODE -ne 0) { throw "npm install failed ($LASTEXITCODE)" }
        }
        Write-Host "  npm run build ..." -ForegroundColor DarkGray
        & $npm.Source run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed ($LASTEXITCODE)" }
    } finally { Pop-Location }
    if (-not (Test-Path $distIdx)) { throw "build finished but $distIdx is missing" }
    Write-Host "Dashboard built." -ForegroundColor Green
}

# Delegate to Run-Colibri.ps1 -Action web. It forwards --port via -Passthrough.
$run = Join-Path $PSScriptRoot 'Run-Colibri.ps1'
if (-not (Test-Path $run)) { throw "Run-Colibri.ps1 not found at $run" }

$fwd = @{ Model = $Model; Action = 'web'; VramGB = $VramGB; RamGB = $RamGB }
if ($CudaDense) { $fwd.CudaDense = $true }
# Bind extra coli args by NAME into Run-Colibri's remaining-args parameter. Passing
# them positionally would bind the array to Run-Colibri's positional [string]$Mirror
# ("cannot convert value to type System.String").
$rest = @('--port', "$Port")
if ($Passthrough) { $rest += $Passthrough }
$fwd.Passthrough = $rest

Write-Host "Dashboard: http://127.0.0.1:$Port/  (opens automatically when the engine is ready)" -ForegroundColor Cyan
& $run @fwd
