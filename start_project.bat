@echo off
cd /d "%~dp0"

echo ==========================================
echo          PE-FedHealth Project
echo ==========================================
echo.
echo Starting Backend...
start "PE-FedHealth Backend" cmd /k "py -3.10 -m uvicorn backend.main:app --host 127.0.0.1 --port 8001"

timeout /t 3 /nobreak >nul

echo Starting Frontend...
start "PE-FedHealth Frontend" cmd /k "cd /d "%~dp0frontend-ui" && npm run dev"

echo.
echo ==========================================
echo Backend:  http://127.0.0.1:8001
echo Frontend: http://localhost:5174
echo ==========================================
echo.
pause