# build_win.ps1 — build nativa Windows del motore colibrì (clang, target MSVC).
# Uso:  .\build_win.ps1            # motore CPU-only + test C
#       .\build_win.ps1 -Cuda     # + backend CUDA (richiede nvcc e toolset MSVC v143)
#       .\build_win.ps1 -Arch x86-64-v3   # binario portabile (default: native)
param(
    [switch]$Cuda,
    [string]$Arch = "native",
    [string]$CudaArch = "sm_120"        # RTX 5090 (Blackwell). Cambiare per altre GPU.
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$cflags = @("-O3", "-march=$Arch", "-fopenmp",
            "-D_FILE_OFFSET_BITS=64",   # richiesto dal guard di compat.h (belt-and-braces)
            "-D_CRT_SECURE_NO_WARNINGS", "-D_CRT_NONSTDC_NO_DEPRECATE",
            "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-misleading-indentation",
            "-Wno-unused-function", "-Wno-deprecated-declarations")

$objs = @()
if ($Cuda) {
    # nvcc su Windows esige cl.exe (MSVC) come host compiler. CUDA 13.x supporta
    # i toolset fino a VS 2022 (v143): con MSVC piu' nuovi cudafe++ crasha.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $msvc = Get-ChildItem "$vs\VC\Tools\MSVC" | Where-Object { $_.Name -lt "14.50" } |
            Sort-Object Name -Descending | Select-Object -First 1
    if (-not $msvc) { throw "Nessun toolset MSVC <= v143 trovato: installare 'MSVC v143' dal Visual Studio Installer" }
    # nvcc va invocato DENTRO vcvars64 (INCLUDE/LIB di MSVC) e SENZA -std:
    # con -std=c++17 forwardato a MSVC 14.44+, cudafe++ crasha (0xC0000409, CUDA 13.1).
    $vcvars = "$vs\VC\Auxiliary\Build\vcvars64.bat"
    $ver = ($msvc.Name -split "\.")[0..1] -join "."
    cmd /c "`"$vcvars`" -vcvars_ver=$ver >nul 2>&1 && set CL= && nvcc -O3 -arch=$CudaArch -c backend_cuda.cu -o backend_cuda.obj"
    if ($LASTEXITCODE) { throw "nvcc fallito" }
    $cflags += "-DCOLI_CUDA"
    $objs += "backend_cuda.obj"
    $cudaLib = Join-Path $env:CUDA_PATH "lib\x64"
    $ldflags = @("-L$cudaLib", "-lcudart")
} else {
    $ldflags = @()
}

clang @cflags glm.c @objs -o glm.exe @ldflags
if ($LASTEXITCODE) { throw "build glm.exe fallita" }
Write-Host "glm.exe ok ($Arch$(if($Cuda){" + CUDA $CudaArch"}))"

foreach ($t in "test_json", "test_st", "test_tier", "test_tok") {
    clang -O2 -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -Wno-deprecated-declarations `
        "tests\$t.c" -o "tests\$t.exe"
    if ($LASTEXITCODE) { throw "build $t fallita" }
}
foreach ($t in "test_json", "test_st", "test_tier") {   # test_tok richiede tokenizer.json + casi
    & ".\tests\$t.exe"
    if ($LASTEXITCODE) { throw "$t FALLITO" }
}
Write-Host "test C: ok"

if (Test-Path glm_tiny) {
    $env:SNAP = ".\glm_tiny"; $env:TF = "1"
    $out = .\glm.exe 64 16 16 2>&1 | Select-String "posizioni"
    Write-Host "self-test oracolo: $out"
    Remove-Item Env:SNAP, Env:TF
}
