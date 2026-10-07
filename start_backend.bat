@echo off
cd /d "%~dp0"

echo ==========================================
echo       PE-FedHealth Backend
echo ==========================================
echo Starting FastAPI on port 8001...
echo.

py -3.10 -m uvicorn backend.main:app --host 127.0.0.1 --port 8001

pause