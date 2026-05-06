@echo off
REM start_thinking_coding.bat - Thinking mode tuned for coding, 127k ctx
setlocal
cd /d "%~dp0"
set "REPO_ROOT=%~dp0.."
set "PY="
if defined VLLM_WINDOWS_VENV if exist "%VLLM_WINDOWS_VENV%\Scripts\python.exe" set "PY=%VLLM_WINDOWS_VENV%\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\venv\Scripts\python.exe" set "PY=%REPO_ROOT%\venv\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\python\python.exe" set "PY=%REPO_ROOT%\python\python.exe"
if not defined PY (
    echo [start_thinking_coding.bat] no python interpreter found under %REPO_ROOT%
    pause
    exit /b 1
)
"%PY%" -u "%~dp0start_thinking_coding.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [start_thinking_coding.bat] exited with code %RC%
    pause
)
endlocal & exit /b %RC%
