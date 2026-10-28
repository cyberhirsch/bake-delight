@echo off
REM Overnight bake only for the overcast model. Safe to re-run; finished assets are skipped.
cd /d "%~dp0"
".venv\Scripts\python.exe" run_pipeline.py --profile overcast --stop-after bake >> "F:\DelighterData\logs\pipeline_overcast.log" 2>&1
