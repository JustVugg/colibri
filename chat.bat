@echo off
title Colibri GLM-5.2 Chat
cd /d E:\colibri\c
set PATH=C:\Users\Up1t3\.local\bin;C:\Users\Up1t3\bin;%PATH%

if exist C:\Users\Up1t3\.local\bin\uv.exe (
    C:\Users\Up1t3\.local\bin\uv.exe run --with requests python cli_chat.py
    goto done
)

py -3 cli_chat.py
if %errorlevel% equ 0 goto done

python cli_chat.py
if %errorlevel% equ 0 goto done

echo [ERROR] Python not found.
pause

:done