@echo off
rem Windows entry point for a release archive (#1241).
rem
rem The archive ships the engines as .exe files and the launcher as `coli`, a
rem Python script with no extension: on Windows that leaves nothing to click,
rem and double-clicking an engine opens a console that closes immediately
rem because the engine has no model. This file is the thing to click, and the
rem thing to type: `coli chat --model D:\models\glm52_i4` works from cmd and
rem PowerShell exactly as `./coli chat` does elsewhere.
setlocal enabledelayedexpansion
set "COLI_HERE=%~dp0"

rem Find a Python 3. The py launcher ships with python.org installers and is
rem the only one that stays correct when several versions are installed.
set "COLI_PY="
py -3 -c "import sys" >nul 2>&1 && set "COLI_PY=py -3"
if not defined COLI_PY python -c "import sys; sys.exit(0 if sys.version_info[0]==3 else 1)" >nul 2>&1 && set "COLI_PY=python"
if not defined COLI_PY python3 -c "import sys" >nul 2>&1 && set "COLI_PY=python3"

if not defined COLI_PY (
    echo colibri: Python 3 was not found on this machine.
    echo.
    echo The engines are pure C and need nothing, but the launcher, the API
    echo gateway and the model tools are Python. Install Python 3 from
    echo    https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then run this again.
    call :hold
    exit /b 1
)

if "%~1"=="" (
    echo colibri — run frontier models from your own storage.
    echo.
    echo You are one argument away: this launcher needs a model directory.
    echo.
    echo     coli.cmd chat   --model D:\models\glm52_i4     interactive chat
    echo     coli.cmd serve  --model D:\models\glm52_i4     OpenAI-compatible API
    echo     coli.cmd web    --model D:\models\glm52_i4     API plus dashboard
    echo     coli.cmd doctor --model D:\models\glm52_i4     check a model is usable
    echo     coli.cmd info                                  what this build supports
    echo.
    echo The .exe files next to this script are the ENGINES. They are not meant
    echo to be started directly: without a model they exit at once, which is why
    echo double-clicking one only flashes a window.
    echo.
    echo Getting a model, step by step:
    echo     https://github.com/JustVugg/colibri/blob/main/docs/quickstart.md
    call :hold
    exit /b 0
)

%COLI_PY% "%COLI_HERE%coli" %*
exit /b %ERRORLEVEL%

rem Keep the window readable when this script was double-clicked from Explorer.
rem cmd.exe leaves CMDCMDLINE containing /c when it was started to run us and
rem will close on exit; a shell the user already had open does not.
:hold
echo %CMDCMDLINE% | find /i "/c" >nul && pause
exit /b 0
