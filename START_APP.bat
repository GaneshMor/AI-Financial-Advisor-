@echo off
REM Starts the dashboard with demo user U02. Run RUN_EVERYTHING.bat once before this.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist venv (
    echo Run RUN_EVERYTHING.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
start "" "http://localhost:8501/?demo=U02"
python -m streamlit run app.py --server.headless true
pause
