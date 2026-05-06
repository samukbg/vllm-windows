@echo off
REM =====================================================================
REM  PP=2 + ngram spec-decode variant.
REM  Uses GPU0 + GPU1 to enable ctx=96k. Trades MTP (and some tok/s) for
REM  a much bigger context window. Listens on :5002 — can run alongside
REM  the 72 tok/s baseline on :5001.
REM =====================================================================
setlocal
cd /d "%~dp0"
set "REPO_ROOT=%~dp0.."
set CUDA_VISIBLE_DEVICES=0,1
set "PY="
if defined VLLM_WINDOWS_VENV if exist "%VLLM_WINDOWS_VENV%\Scripts\python.exe" set "PY=%VLLM_WINDOWS_VENV%\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\venv\Scripts\python.exe" set "PY=%REPO_ROOT%\venv\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\python\python.exe" set "PY=%REPO_ROOT%\python\python.exe"
if not defined PY (
    echo [start_pp2_160k.bat] no python interpreter found under %REPO_ROOT%
    pause
    exit /b 1
)
"%PY%" -u "%~dp0start_pp2.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [start_pp2_160k.bat] exited with code %RC%
    pause
)
endlocal & exit /b %RC%
