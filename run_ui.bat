@echo off
REM Laptop 2: serve the built evaluator UI on port 5173 (needs only Python).
REM Build once with:  cd ui && npm install && npm run build
cd /d %~dp0
python -m http.server 5173 --directory ui\dist
