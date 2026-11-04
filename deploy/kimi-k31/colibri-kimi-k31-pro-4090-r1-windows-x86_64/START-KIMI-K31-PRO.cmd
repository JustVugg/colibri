@echo off
rem Kimi K3.1 Pro on an RTX 4090: Kimi K3's measured profile (tools\start_kimi_k31_pro_4090.ps1), API model kimi-k3.1-pro.
rem Weights first: powershell -ExecutionPolicy Bypass -File tools\download_kimi_k31_pro_4090.ps1
rem Arguments go to the script, e.g.  START-KIMI-K31-PRO.cmd -Context 65536 -Port 18081
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_kimi_k31_pro_4090.ps1" %*
if errorlevel 1 pause
