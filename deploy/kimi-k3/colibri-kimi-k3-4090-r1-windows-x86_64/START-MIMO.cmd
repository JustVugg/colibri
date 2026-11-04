@echo off
rem MiMo-V2.6-Pro on an RTX 4090: the measured profile (tools\start_mimo_cuda_4090.ps1).
rem Arguments go to the script, e.g.  START-MIMO.cmd -ModelDir D:\models\MiMo-V2.6-Pro -ShardDir ""
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_mimo_cuda_4090.ps1" %*
if errorlevel 1 pause
