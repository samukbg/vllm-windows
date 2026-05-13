@echo off
REM start_speed.bat, GPU1 MTP n=6 ctx=90k mem=0.948 350W, peak decode ~64.5 tok/s
setlocal
cd /d "%~dp0"
set "REPO_ROOT=%~dp0.."
set "PY="
if defined VLLM_WINDOWS_VENV if exist "%VLLM_WINDOWS_VENV%\Scripts\python.exe" set "PY=%VLLM_WINDOWS_VENV%\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\venv\Scripts\python.exe" set "PY=%REPO_ROOT%\venv\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\python\python.exe" set "PY=%REPO_ROOT%\python\python.exe"
if not defined PY (
    echo [start_speed.bat] no python interpreter found under %REPO_ROOT%
    pause
    exit /b 1
)
"%PY%" -u "%~dp0start_speed.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [start_speed.bat] exited with code %RC%
    pause
)
endlocal & exit /b %RC%
