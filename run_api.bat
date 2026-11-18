@echo off
REM Laptop 1: start the PathSnap API on all interfaces, port 8000.
REM Optional: set PATHSNAP_EVAL_DIR=C:\path\to\eval_trips
cd /d %~dp0
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
