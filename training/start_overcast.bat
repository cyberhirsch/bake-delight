@echo off
REM Overnight run: bake -> post-process -> train the overcast de-lighting model.
REM Safe to re-run after an interruption; every stage resumes.
cd /d "%~dp0"
".venv\Scripts\python.exe" run_pipeline.py --profile overcast >> "F:\DelighterData\logs\pipeline_overcast.log" 2>&1
