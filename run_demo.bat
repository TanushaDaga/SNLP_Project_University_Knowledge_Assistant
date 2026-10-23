@echo off
rem Double-click to start the API and the REST demo page. Add the word "original" to also start the original UI.
cd /d "%~dp0"
if /i "%1"=="original" (
    powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Original
) else (
    powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
)
