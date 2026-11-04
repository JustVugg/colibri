@echo off
rem Kimi K3 on an RTX 4090: the measured profile (tools\start_kimi_k3_4090.ps1), API model kimi-k3.
rem Arguments go to the script, e.g.  START-KIMI-K3.cmd -Context 65536 -Port 18081
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_kimi_k3_4090.ps1" %*
if errorlevel 1 pause
