@echo off
REM ===================================================================
REM  AI Personal Financial Advisor: one click setup and full run
REM  Double click this file. It will:
REM    1. create a Python environment and install everything
REM    2. ask you to paste your OpenAI key (only the first time)
REM    3. run the tests, build the knowledge index, run the smoke test
REM       and the full evaluation
REM    4. save all output in the run_outputs folder
REM ===================================================================
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo [1/6] Finding Python...
set "PY=python"
py -3.11 --version >nul 2>nul && set "PY=py -3.11"
%PY% --version
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 from https://www.python.org/downloads/
    echo During install, tick "Add Python to PATH". Then run this file again.
    pause
    exit /b 1
)

echo.
echo [2/6] Setting up the environment (first time takes 3 to 5 minutes)...
if not exist venv (
    %PY% -m venv venv
    if errorlevel 1 goto :fail
)
call venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo [3/6] Checking your OpenAI key...
if not exist .env copy .env.example .env >nul
findstr /r /c:"^OPENAI_API_KEY=..........." .env >nul
if errorlevel 1 (
    echo.
    echo Notepad will open the .env file.
    echo Paste your key right after OPENAI_API_KEY=  with no spaces, save, close Notepad,
    echo then double click RUN_EVERYTHING.bat again.
    notepad .env
    pause
    exit /b 1
)

if not exist run_outputs mkdir run_outputs

echo.
echo [4/6] Running the 338 automated tests...
python -m pytest -q > run_outputs\1_tests.txt 2>&1
type run_outputs\1_tests.txt | findstr /c:"passed" /c:"failed" /c:"error"

echo.
echo [5/6] Building the knowledge index and running the smoke test...
python -m rag.ingest > run_outputs\2_ingest.txt 2>&1
type run_outputs\2_ingest.txt
python -m scripts.smoke_test_llm > run_outputs\3_smoke_test.txt 2>&1
type run_outputs\3_smoke_test.txt | findstr /c:"Goal parsing:" /c:"Tool selection:" /c:"Routing:" /c:"RAG:" /c:"Safety status"

echo.
echo [6/6] Running the evaluation (12 users; with a free tier this can take 20 to 40 minutes)...
python -m evaluation.evaluate --users 12 > run_outputs\4_evaluation.txt 2>&1
type run_outputs\4_evaluation.txt | findstr /c:"Saved to"

echo.
echo ===================================================================
echo  DONE. Send these files to Claude:
echo    run_outputs\1_tests.txt
echo    run_outputs\2_ingest.txt
echo    run_outputs\3_smoke_test.txt
echo    run_outputs\4_evaluation.txt
echo  and the summary.md inside evaluation\results\run_...\
echo.
echo  To open the app:  double click START_APP.bat
echo ===================================================================
explorer run_outputs
pause
exit /b 0

:fail
echo.
echo Something failed. Take a screenshot of this window and send it to Claude.
pause
exit /b 1
