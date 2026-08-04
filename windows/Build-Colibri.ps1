<#
.SYNOPSIS
  Detect the Windows CUDA toolchain, guide any missing installs, and build
  colibri's GPU DLL + host engine for an NVIDIA GPU. Replaces the hardcoded
  c/build_cuda.bat (which pointed at C:\Users\Mark\...).

.DESCRIPTION
  Two artifacts are produced in <repo>\c :
    coli_cuda.dll  - GPU expert kernels (nvcc + MSVC cl.exe)
    colibri.exe    - host engine that loads the DLL at runtime (MinGW gcc)

  This mirrors c\Makefile (the source of truth for flags) but invokes nvcc and
  gcc directly, so you do NOT need GNU make or an MSYS2 POSIX shell installed.

.PARAMETER CheckOnly
  Run detection only (a "doctor"): report what's present/missing and how to
  install it, then exit without building.

.PARAMETER CudaArch
  'portable' (default) emits SASS for sm_80/86/89/90/120 + a compute_120 PTX
  fallback -> one DLL runs on any Ampere..Blackwell card. 'native' builds only
  for the detected GPU (smaller/faster to compile). Or pass an explicit sm_XX.

.PARAMETER RepoRoot
  Path to the colibri repo root. Defaults to the parent of this script's directory.

.EXAMPLE
  .\Build-Colibri.ps1 -CheckOnly
.EXAMPLE
  .\Build-Colibri.ps1                 # detect + build (portable DLL)
.EXAMPLE
  .\Build-Colibri.ps1 -CudaArch native
#>
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [string]$CudaArch = 'portable',
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..')
)

$ErrorActionPreference = 'Stop'
function Say($m,$c='Gray'){ Write-Host $m -ForegroundColor $c }
function Ok($m){  Say "  [ok]   $m" 'Green' }
function Warn($m){ Say "  [MISS] $m" 'Yellow' }
function Head($m){ Say "`n=== $m ===" 'Cyan' }

$CDir = Join-Path (Resolve-Path $RepoRoot) 'c'
if(-not (Test-Path (Join-Path $CDir 'colibri.c'))){
    throw "colibri.c not found under $CDir - pass -RepoRoot <path to cloned colibri>"
}

# --- collected state ---
$state = [ordered]@{ gpu=$null; smArch=$null; cudaHome=$null; cudaVer=$null; nvcc=$null; vcvars=$null; gcc=$null }
$missing = New-Object System.Collections.Generic.List[string]

# ---------------------------------------------------------------- GPU + arch
Head 'NVIDIA GPU'
$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if($smi){
    $line = (& nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if($line){
        $parts = $line -split '\s*,\s*'
        $state.gpu = $parts[0]
        $cc = $parts[1]                                  # e.g. "12.0"
        $vram = $parts[2]
        if($cc -match '^(\d+)\.(\d+)$'){ $state.smArch = "sm_$($matches[1])$($matches[2])" }
        Ok "$($state.gpu)  (compute $cc -> $($state.smArch), ${vram} MiB VRAM)"
    }
}
if(-not $state.gpu){ Warn "no NVIDIA GPU detected (nvidia-smi missing) - install the NVIDIA driver"; $missing.Add('driver') }

# ------------------------------------------------------------- CUDA Toolkit
Head 'CUDA Toolkit (provides nvcc)'
$cudaCandidates = @()
if($env:CUDA_PATH){ $cudaCandidates += $env:CUDA_PATH }
$cudaCandidates += (Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*' -Directory -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending | Select-Object -ExpandProperty FullName)
foreach($cand in $cudaCandidates){
    $nv = Join-Path $cand 'bin\nvcc.exe'
    if(Test-Path $nv){
        $state.cudaHome = $cand; $state.nvcc = $nv
        $vtxt = (& $nv --version 2>$null | Select-String 'release ([\d.]+)').Matches.Groups[1].Value
        $state.cudaVer = $vtxt
        Ok "nvcc $vtxt  ($cand)"
        break
    }
}
if(-not $state.nvcc){ Warn "CUDA Toolkit not found"; $missing.Add('cuda') }
elseif($state.smArch -eq 'sm_120' -or $CudaArch -eq 'portable'){
    # RTX 50-series (Blackwell sm_120) requires CUDA >= 12.8 to emit sm_120 SASS/PTX.
    $v = 0.0; [double]::TryParse($state.cudaVer,[ref]$v) | Out-Null
    if($v -gt 0 -and $v -lt 12.8){
        Warn "CUDA $($state.cudaVer) is < 12.8 - cannot target sm_120 (RTX 50-series). Upgrade the toolkit."
        $missing.Add('cuda-old')
    }
}

# ------------------------------------------------------------- MSVC (cl.exe)
Head 'MSVC build tools (nvcc host compiler)'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if(Test-Path $vswhere){
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1
    if($vsPath){
        $vc = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
        if(Test-Path $vc){ $state.vcvars = $vc; Ok "vcvars64.bat  ($vsPath)" }
    }
}
if(-not $state.vcvars){ Warn "MSVC x64 build tools not found"; $missing.Add('msvc') }

# ------------------------------------------------------------- MinGW gcc
Head 'MinGW-w64 GCC (host engine)'
$g = Get-Command gcc -ErrorAction SilentlyContinue
if(-not $g){
    # Fresh shells often lack the WinGet WinLibs bin dir on PATH — locate it
    # directly instead of failing.
    $wl = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT*\mingw64\bin\gcc.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if($wl){ $env:PATH = $wl.DirectoryName + ';' + $env:PATH; $g = Get-Command gcc -ErrorAction SilentlyContinue }
}
if($g){
    $triple = (& gcc -dumpmachine 2>$null)
    if($triple -match 'mingw'){ $state.gcc = $g.Source; Ok "gcc -> $triple  ($($g.Source))" }
    else { Warn "gcc found but targets '$triple', not mingw - install MinGW-w64 (UCRT)"; $missing.Add('gcc') }
} else { Warn "gcc not on PATH"; $missing.Add('gcc') }

# ---------------------------------------------------------------- summary
Head 'Summary'
if($missing.Count -eq 0){ Say "  All required toolchain components present." 'Green' }
else {
    Say "  Missing components - install these, then re-run:" 'Yellow'
    $guide = @{
        driver   = "NVIDIA driver:        https://www.nvidia.com/Download/index.aspx"
        cuda     = "CUDA Toolkit 12.8+:   winget install Nvidia.CUDA   (or developer.nvidia.com/cuda-downloads)"
        'cuda-old'="CUDA Toolkit upgrade: need >= 12.8 for your RTX 50-series (developer.nvidia.com/cuda-downloads)"
        msvc     = "VS Build Tools:       winget install Microsoft.VisualStudio.2022.BuildTools  then add 'Desktop development with C++'"
        gcc      = "MinGW-w64 (UCRT):     winget install BrechtSanders.WinLibs.POSIX.UCRT   (or via MSYS2: pacman -S mingw-w64-ucrt-x86_64-gcc)"
    }
    foreach($k in $missing){ Say "    - $($guide[$k])" 'Yellow' }
}

if($CheckOnly){ Say "`n(-CheckOnly: stopping before build)" 'DarkGray'; return }
if($missing.Count -gt 0){ throw "Cannot build until the components above are installed." }

# ================================================================ BUILD
Head 'Importing MSVC environment (vcvars64)'
# nvcc on Windows needs cl.exe on PATH. Pull vcvars64's env into this session.
$before = @{}; Get-ChildItem env: | ForEach-Object { $before[$_.Name] = $_.Value }
cmd /c "`"$($state.vcvars)`" >nul 2>&1 && set" | ForEach-Object {
    if($_ -match '^([^=]+)=(.*)$'){ Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
Ok "cl.exe: $((Get-Command cl -ErrorAction SilentlyContinue).Source)"

# Ensure CUDA on PATH + CUDA_HOME for this session.
$env:CUDA_PATH = $state.cudaHome
$env:PATH = (Join-Path $state.cudaHome 'bin') + ';' + $env:PATH

# --- pick gencode ---
if($CudaArch -eq 'portable'){
    $gencode = @(
        '-gencode','arch=compute_80,code=sm_80',
        '-gencode','arch=compute_86,code=sm_86',
        '-gencode','arch=compute_89,code=sm_89',
        '-gencode','arch=compute_90,code=sm_90',
        '-gencode','arch=compute_120,code=sm_120',
        '-gencode','arch=compute_120,code=compute_120')
} elseif($CudaArch -eq 'native'){
    $a = if($state.smArch){ $state.smArch } else { 'sm_120' }
    $gencode = @("-arch=$a")
} else {
    $gencode = @("-arch=$CudaArch")
}

Push-Location $CDir
try {
    Head 'Building coli_cuda.dll (nvcc)'
    # -cudart static: link the CUDA runtime INTO the DLL. Without it the DLL needs
    # cudart64_XX.dll at load time, which on CUDA 13.x lives in bin\x64 (not bin, not
    # on PATH) -> LoadLibrary fails and the engine silently falls back to CPU.
    $nvccArgs = @('-O3','-std=c++17') + $gencode + @(
        '-Xcompiler=-W3','-shared','-DCOLI_CUDA_BUILDING_DLL','-cudart','static',
        "-L`"$($state.cudaHome)\lib\x64`"",
        'backend_cuda.cu','-o','coli_cuda.dll')
    Say "  nvcc $($nvccArgs -join ' ')" 'DarkGray'
    & $state.nvcc @nvccArgs
    if($LASTEXITCODE -ne 0){ throw "nvcc failed (exit $LASTEXITCODE)" }
    Ok "coli_cuda.dll ($([math]::Round((Get-Item coli_cuda.dll).Length/1MB,1)) MB)"

    Head 'Building colibri.exe (gcc, CUDA_DLL loader)'
    # Mirrors c\Makefile Windows branch + CUDA_DLL path: -DCOLI_CUDA links the
    # runtime loader (backend_loader.c) instead of cudart.
    # -march=native: on this i5-13400 it enables AVX-VNNI (vpdpbusd in the int8 IDOT
    # kernels), measured ~+6% decode over -march=x86-64-v3. Portable builds: set
    # COLI_MARCH=x86-64-v3 before running the script.
    $march = if ($env:COLI_MARCH) { "-march=$($env:COLI_MARCH)" } else { '-march=native' }
    # Compile every registered model family: arch_*.c is the registry on disk.
    $archSrcs = @(Get-ChildItem 'arch_*.c' | Sort-Object Name | Select-Object -ExpandProperty Name)
    $gccArgs = @(
        '-D_FILE_OFFSET_BITS=64','-O3',$march,'-fopenmp',
        '-Wall','-Wextra','-Wno-unused-parameter','-Wno-misleading-indentation','-Wno-unused-function',
        '-DCOLI_CUDA','colibri.c') + $archSrcs + @('backend_loader.c','-o','colibri.exe',
        '-lm','-fopenmp','-static','-lpsapi')
    Say "  gcc $($gccArgs -join ' ')" 'DarkGray'
    & $state.gcc @gccArgs
    if($LASTEXITCODE -ne 0){ throw "gcc failed (exit $LASTEXITCODE)" }
    Ok "colibri.exe ($([math]::Round((Get-Item colibri.exe).Length/1MB,1)) MB)"

    Head 'Done'
    Say "  Built in: $CDir" 'Green'
    Say "  Next:  ..\windows\Run-Colibri.ps1 -Model <path to glm52_i4>" 'Green'
}
finally { Pop-Location }
