@echo off
echo ===================================================
echo             STARTING SATQUERY AI
echo   Smart India Hackathon 2026 (SIH26167)
echo ===================================================

start "SatQuery AI Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8008"
timeout /t 3 /nobreak > nul

start "SatQuery AI Frontend" cmd /k "cd /d %~dp0frontend && npm run dev -- --port 5174"

echo Application launched!
echo Backend:  http://127.0.0.1:8008
echo Frontend: http://localhost:5174
pause
