@echo off
rem ---------------------------------------------------------------------------
rem build_cuda.bat - one-shot build of coli_cuda.dll (Windows, nvcc + MSVC).
rem
rem Mirrors the Makefile's `cuda-dll` recipe so it can be double-clicked or run
rem from a plain shell: it locates vcvars64.bat and the CUDA toolkit itself
rem rather than hardcoding install paths.
rem
rem Overrides (set before calling):
rem   CUDA_ARCH   GPU compute capability (sm_NNN). Auto-detected from the
rem               installed GPU via nvidia-smi; set to override or to
rem               cross-compile. sm_120 = Blackwell / RTX 50xx, sm_89 = Ada /
rem               RTX 40xx, sm_86 = Ampere / RTX 30xx.
rem   CUDA_PATH   CUDA toolkit root. Normally set by the CUDA installer.
rem   CCBIN       Directory containing the cl.exe nvcc should use as host
rem               compiler. Set this to force a specific toolset.
rem   ALLOW_UNSUPPORTED  =1 adds -allow-unsupported-compiler.
rem   DSTORAGE_HOME  Extracted Microsoft.Direct3D.DirectStorage nuget
rem               (include/ + lib/x64) - adds the disk->VRAM DMA transport.
rem               Optional: without it the DLL builds and works unchanged
rem               (the DirectStorage entry points are optional exports).
rem ---------------------------------------------------------------------------
setlocal

rem --- GPU arch: detect from the installed GPU unless overridden ------------
rem nvidia-smi reports compute capability as e.g. "12.0"; strip the dot for the
rem sm_NNN suffix (12.0 -> sm_120, 8.9 -> sm_89). The value is validated as
rem <digits>.<digits>; anything else falls back to sm_120.
if not "%CUDA_ARCH%"=="" goto have_arch
for /f "usebackq skip=1 delims=" %%c in (`nvidia-smi -i 0 --query-gpu=compute_cap --format=csv 2^>nul`) do set "CC=%%c"
echo(%CC%| findstr /r "^[0-9][0-9]*\.[0-9][0-9]*$" >nul 2>&1
if errorlevel 1 goto arch_default
for /f "tokens=1,2 delims=." %%a in ("%CC%") do set "CUDA_ARCH=sm_%%a%%b"
if not "%CUDA_ARCH%"=="" goto have_arch
:arch_default
echo WARNING: could not detect GPU compute capability; defaulting to sm_120.>&2
set "CUDA_ARCH=sm_120"
:have_arch
set "CC="
set "VSINSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
set "VSWHERE=%VSINSTALLER%\vswhere.exe"

rem --- MSVC: nvcc needs cl.exe as its host compiler -------------------------
rem Prefer the VS 2022 toolset (range [17.0,18.0)); fall back to -latest if no
rem 2022 install exists.
if not "%CCBIN%"=="" goto have_msvc
where cl.exe >nul 2>&1
if %ERRORLEVEL%==0 goto have_msvc

if not exist "%VSWHERE%" goto no_msvc

rem Invoke vswhere by bare name from its own directory.
set "VSDIR="
pushd "%VSINSTALLER%"
for /f "usebackq delims=" %%i in (`vswhere.exe -products * -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not "%VSDIR%"=="" goto found_vs
echo WARNING: no VS 2017-2022 found; falling back to the newest install.>&2
echo          If nvcc reports "unsupported Microsoft Visual Studio version",>&2
echo          install the 2022 toolset - see the error text at the end.>&2
for /f "usebackq delims=" %%i in (`vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
:found_vs
popd
if "%VSDIR%"=="" goto no_msvc
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" goto no_msvc
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl.exe >nul 2>&1
if not %ERRORLEVEL%==0 goto no_msvc
:have_msvc

rem -ccbin pins the host compiler explicitly.
set "CCBIN_FLAG="
if not "%CCBIN%"=="" set "CCBIN_FLAG=-ccbin "%CCBIN%""
set "UNSUPPORTED_FLAG="
if "%ALLOW_UNSUPPORTED%"=="1" set "UNSUPPORTED_FLAG=-allow-unsupported-compiler"

rem --- CUDA -----------------------------------------------------------------
set "NVCC=nvcc"
where nvcc.exe >nul 2>&1
if %ERRORLEVEL%==0 goto have_nvcc
if "%CUDA_PATH%"=="" goto no_cuda
if not exist "%CUDA_PATH%\bin\nvcc.exe" goto no_cuda
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
:have_nvcc

rem Add -L<cuda>\lib\x64 for cudart only when CUDA_PATH is known.
set "CUDA_LIB="
if not "%CUDA_PATH%"=="" set "CUDA_LIB=-L"%CUDA_PATH%\lib\x64""

rem --- DirectStorage (optional): disk->VRAM DMA transport -------------------
set "DS_SRC="
set "DS_FLAGS="
set "DS_NOTE="
if "%DSTORAGE_HOME%"=="" goto ds_done
if not exist "%~dp0backend_dstorage.cpp" (
    echo NOTE: DSTORAGE_HOME is set but this tree has no backend_dstorage.cpp - ignoring.
    goto ds_done
)
if not exist "%DSTORAGE_HOME%\include\dstorage.h" (
    echo WARNING: DSTORAGE_HOME is set but include\dstorage.h is missing under it - ignoring.>&2
    goto ds_done
)
set "DS_SRC=backend_dstorage.cpp"
set "DS_FLAGS=-I"%DSTORAGE_HOME%\include" -L"%DSTORAGE_HOME%\lib\x64" -ldstorage -ld3d12 -ldxgi"
set "DS_NOTE=with DirectStorage "
:ds_done

rem --- Build ----------------------------------------------------------------
rem Build in the script's own directory so backend_cuda.cu resolves and the DLL
rem lands next to colibri.exe.
pushd "%~dp0"

echo Building coli_cuda.dll %DS_NOTE%for %CUDA_ARCH% ...
rem -Xcompiler=-W3 sets the MSVC host-compiler warning level.
rem -cudart static links the CUDA runtime INTO the DLL. Without it the DLL
rem needs cudart64_XX.dll at load time, which on CUDA 13.x lives in bin\x64
rem (not on PATH), so LoadLibrary failed silently and every Windows run fell
rem back to CPU while looking healthy.
"%NVCC%" -O3 -std=c++17 -arch=%CUDA_ARCH% -Xcompiler=-W3 -shared ^
    %CCBIN_FLAG% %UNSUPPORTED_FLAG% ^
    -DCOLI_CUDA_BUILDING_DLL %CUDA_LIB% -cudart static ^
    %DS_FLAGS% ^
    backend_cuda.cu %DS_SRC% -o coli_cuda.dll
set "RC=%ERRORLEVEL%"

popd
if not "%RC%"=="0" goto failed
echo.
echo OK: %~dp0coli_cuda.dll
echo Now build the host with the runtime loader:
echo     make colibri.exe CUDA_DLL=1 ARCH=native
exit /b 0

:no_msvc
echo ERROR: cl.exe (MSVC) not found.>&2
echo Install "Desktop development with C++":>&2
echo     winget install Microsoft.VisualStudio.2022.BuildTools>&2
echo ...or run this from the "x64 Native Tools Command Prompt for VS 2022".>&2
exit /b 1

:no_cuda
echo ERROR: nvcc not found, and CUDA_PATH is unset or has no bin\nvcc.exe.>&2
echo CUDA 12.8+ is required for sm_120 (Blackwell / RTX 50xx):>&2
echo     winget install Nvidia.CUDA>&2
exit /b 1

:failed
echo.>&2
echo FAILED: nvcc exited with %RC%.>&2
echo.>&2
echo "unsupported gpu architecture" - CUDA older than the target.>&2
echo     sm_120 needs CUDA 12.8+.  Override: set CUDA_ARCH=sm_89>&2
echo.>&2
echo "unsupported Microsoft Visual Studio version" - your VS is NEWER than>&2
echo CUDA supports (it accepts 2017-2022 only). Install the 2022 toolset>&2
echo side by side; it coexists with newer VS and this script prefers it:>&2
echo     winget install Microsoft.VisualStudio.2022.BuildTools>&2
echo     (then add the "Desktop development with C++" workload)>&2
echo Or point at a specific cl.exe directory:>&2
echo     set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\<ver>\bin\Hostx64\x64>&2
exit /b %RC%
