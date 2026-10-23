@echo off
rem colibri: one-step setup and start for Windows. Double-click this file.
rem
rem First run: finds your hardware, recommends a model that fits, gets the
rem engine (prebuilt, or built with MSYS2 when it is installed), downloads the
rem model (safe to interrupt: run this file again to continue), then opens the
rem dashboard in your browser.
rem Later runs: start colibri straight away.
rem
rem Options are passed through to `coli setup`, for example:
rem     START-HERE.bat --model qwen3-coder-30b --dir D:\colibri-models
rem     START-HERE.bat --reconfigure          choose another model
rem     START-HERE.bat --no-gpu               CPU only
rem Stop: close this window, or press Ctrl+C in it.
setlocal
title colibri
cd /d "%~dp0"

set "COLI_LAUNCHER="
if exist "%~dp0c\coli" set "COLI_LAUNCHER=%~dp0c\coli"
if not defined COLI_LAUNCHER if exist "%~dp0coli" set "COLI_LAUNCHER=%~dp0coli"
if not defined COLI_LAUNCHER (
    echo This file belongs at the top of the colibri folder, next to the c folder.
    goto :hold_fail
)

rem A Python 3.10 or newer. The py launcher first: it stays right when several
rem versions are installed. `python` alone can be the Microsoft Store alias,
rem which prints nothing useful, so every candidate is tried with a version check.
set "COLI_PY_EXE="
set "COLI_PY_ARG="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && (set "COLI_PY_EXE=py" & set "COLI_PY_ARG=-3")
if not defined COLI_PY_EXE python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "COLI_PY_EXE=python"
if not defined COLI_PY_EXE call :find_user_python
if not defined COLI_PY_EXE goto :no_python

:run
"%COLI_PY_EXE%" %COLI_PY_ARG% "%COLI_LAUNCHER%" setup %*
set "COLI_RC=%ERRORLEVEL%"
if "%COLI_RC%"=="0" exit /b 0
echo.
echo colibri stopped (code %COLI_RC%). Read the message above; running START-HERE.bat
echo again continues where it stopped.
goto :hold_fail

:find_user_python
for %%V in (314 313 312 311 310) do (
    if not defined COLI_PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "COLI_PY_EXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
)
exit /b 0

:no_python
echo colibri needs Python 3.10 or newer, and it was not found.
echo The engine itself is plain C; Python runs the setup, the launcher and the API.
echo.
where winget >nul 2>&1
if errorlevel 1 goto :manual_python
choice /c YN /m "Install Python 3.12 now with winget, for your user only"
if errorlevel 2 goto :manual_python
winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
call :find_user_python
if defined COLI_PY_EXE goto :run
echo.
echo Python was installed, but this window cannot see it yet. Close it and
echo double-click START-HERE.bat again.
goto :hold_fail

:manual_python
echo Install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH" in the installer, then double-click START-HERE.bat again.

:hold_fail
echo.
pause
exit /b 1
