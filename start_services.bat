@echo off
echo ======================================================================
echo Starting Financial Document Risk Analysis API and Streamlit UI...
echo ======================================================================

start "FastAPI Backend (Port 8000)" cmd /k "python run.py"
timeout /t 3 /nobreak >nul
start "Streamlit UI (Port 8501)" cmd /k "python run_ui.py"

echo.
echo Both servers have been launched!
echo - Swagger UI:   http://localhost:8000/docs
echo - Streamlit UI: http://localhost:8501
echo ======================================================================
pause
