@echo off
rem MiMo-V2.6-Pro-MOPD-heretic on an RTX 4090: the same profile with its own o_proj shard and the
rem base model's other shards (tools\start_mimo_cuda_4090.ps1 -Model heretic), API model
rem mimo-v2.6-pro-mopd-heretic. Arguments go to the script, e.g.  START-MIMO-HERETIC.cmd -ModelId mimo-v2.6-pro-mopd
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_mimo_cuda_4090.ps1" -Model heretic %*
if errorlevel 1 pause
