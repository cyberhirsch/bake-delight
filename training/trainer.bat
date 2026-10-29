@echo off
REM Trainer console: start, watch and stop the training pipeline.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" trainer_gui.py
