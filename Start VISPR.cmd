@echo off
cd /d "%~dp0"
if exist .venv-app\Scripts\python.exe (
  .venv-app\Scripts\python.exe scripts\start_vispr.py
) else (
  python scripts\start_vispr.py
)
pause
