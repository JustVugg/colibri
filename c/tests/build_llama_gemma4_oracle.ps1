param(
    [Parameter(Mandatory = $true)]
    [string]$LlamaRoot,
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$LlamaRoot = (Resolve-Path $LlamaRoot).Path
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $ScriptDirectory "llama_gemma4_oracle.cpp"
if (-not $Output) {
    $Output = Join-Path $ScriptDirectory "..\build-gemma4\llama-gemma4-oracle.exe"
}
$Output = [IO.Path]::GetFullPath($Output)
$OutputDirectory = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    throw "vswhere.exe was not found; install Visual Studio C++ build tools"
}
$VisualStudio = & $VsWhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $VisualStudio) {
    throw "a Visual Studio installation with x64 C++ tools was not found"
}
$VcVars = Join-Path $VisualStudio "VC\Auxiliary\Build\vcvars64.bat"
$EnvironmentDump = & cmd.exe /d /s /c "call `"$VcVars`" >nul && set"
function Get-EnvironmentValue([string]$Name) {
    $Prefix = "$Name="
    $Line = $EnvironmentDump | Where-Object {
        $_.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -First 1
    if (-not $Line) {
        throw "vcvars64.bat did not define $Name"
    }
    return $Line.Substring($Prefix.Length)
}
$env:Path = Get-EnvironmentValue "PATH"
$env:INCLUDE = Get-EnvironmentValue "INCLUDE"
$env:LIB = Get-EnvironmentValue "LIB"

$LlamaLibrary = Join-Path $LlamaRoot "build\src\$Configuration\llama.lib"
$GgmlLibraryDirectory = Join-Path $LlamaRoot "build\ggml\src\$Configuration"
foreach ($Required in @(
    (Join-Path $LlamaRoot "include\llama.h"),
    (Join-Path $LlamaRoot "ggml\include\ggml.h"),
    $LlamaLibrary,
    (Join-Path $GgmlLibraryDirectory "ggml.lib"),
    (Join-Path $GgmlLibraryDirectory "ggml-base.lib"),
    (Join-Path $GgmlLibraryDirectory "ggml-cpu.lib")
)) {
    if (-not (Test-Path $Required)) {
        throw "required llama.cpp build artifact was not found: $Required"
    }
}

$Arguments = @(
    "/nologo", "/O2", "/EHsc", "/std:c++17", "/DNDEBUG",
    "/I", (Join-Path $LlamaRoot "include"),
    "/I", (Join-Path $LlamaRoot "ggml\include"),
    "/Fo$OutputDirectory\llama_gemma4_oracle.obj",
    $Source,
    "/link",
    "/LIBPATH:$(Split-Path -Parent $LlamaLibrary)",
    "/LIBPATH:$GgmlLibraryDirectory",
    "llama.lib", "ggml.lib", "ggml-base.lib", "ggml-cpu.lib",
    "/OUT:$Output"
)
& cl.exe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$RuntimeDirectory = Join-Path $LlamaRoot "build\bin\$Configuration"
Write-Host "built: $Output"
Write-Host "llama.cpp DLL directory: $RuntimeDirectory"
