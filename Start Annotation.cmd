@echo off
"%~dp0.venv-annotation\Scripts\python.exe" "%~dp0scripts\start_label_studio.py"
if errorlevel 1 pause
