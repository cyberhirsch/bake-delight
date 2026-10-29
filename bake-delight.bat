@echo off
REM bake-delight CLI; runs in the training environment (OpenCV with EXR, ONNX Runtime, MCP SDK).
set "BD_PY=%~dp0training\.venv\Scripts\python.exe"
if not exist "%BD_PY%" set "BD_PY=python"
"%BD_PY%" "%~dp0bake_delight.py" %*
