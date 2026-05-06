@echo off
REM =====================================================================
REM  Preserve 72 tok/s baseline: Qwen3.6-27B Lorbus AutoRound INT4
REM  TP=1 PP=1 MTP n=3 fp8_e4m3 TRITON_ATTN ctx=32k GPU1 only
REM  Do not edit start_72tps.py's knobs above without updating this file.
REM =====================================================================
setlocal
cd /d "%~dp0"
set "REPO_ROOT=%~dp0.."
set "PY="
if defined VLLM_WINDOWS_VENV if exist "%VLLM_WINDOWS_VENV%\Scripts\python.exe" set "PY=%VLLM_WINDOWS_VENV%\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\venv\Scripts\python.exe" set "PY=%REPO_ROOT%\venv\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\python\python.exe" set "PY=%REPO_ROOT%\python\python.exe"
if not defined PY (
    echo [start_72tps.bat] no python interpreter found under %REPO_ROOT%
    pause
    exit /b 1
)
"%PY%" -u "%~dp0start_72tps.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [start_72tps.bat] exited with code %RC%
    pause
)
endlocal & exit /b %RC%
