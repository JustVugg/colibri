@echo off
rem Build coli_cuda_k3.dll: Kimi K3's prompt-step experts on the GPU, the CPU's bytes.
rem CUDA_ARCH defaults to RTX 4090 (sm_89); override for other GPUs.
setlocal
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_89"
where cl.exe >nul 2>&1
if not errorlevel 1 goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo Visual Studio C++ tools not found.>&2 & exit /b 1)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "K3_VS=%%i"
if "%K3_VS%"=="" (echo Visual Studio C++ tools not found.>&2 & exit /b 1)
call "%K3_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
:compiler_ready
set "K3_NVCC=nvcc"
if not "%CUDA_PATH%"=="" set "K3_NVCC=%CUDA_PATH%\bin\nvcc.exe"
pushd "%~dp0"
"%K3_NVCC%" -O3 -std=c++17 -arch=%CUDA_ARCH% -ftz=false -Xcompiler=-W3 -shared --cudart=static -DK3CU_BUILDING_DLL backend_cuda_k3.cu -o coli_cuda_k3.dll
set "RC=%ERRORLEVEL%"

popd
exit /b %RC%
