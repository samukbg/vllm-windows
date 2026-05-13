@echo off
REM stop_vllm.bat, kill any vLLM server on ports 5000-5010 + sweep EngineCore/APIServer children
setlocal
cd /d "%~dp0"
set "REPO_ROOT=%~dp0.."
set "PY="
if defined VLLM_WINDOWS_VENV if exist "%VLLM_WINDOWS_VENV%\Scripts\python.exe" set "PY=%VLLM_WINDOWS_VENV%\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\venv\Scripts\python.exe" set "PY=%REPO_ROOT%\venv\Scripts\python.exe"
if not defined PY if exist "%REPO_ROOT%\python\python.exe" set "PY=%REPO_ROOT%\python\python.exe"
if not defined PY (
    echo [stop_vllm.bat] no python interpreter found under %REPO_ROOT%
    pause
    exit /b 1
)
"%PY%" -u "%~dp0stop_vllm.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo [stop_vllm.bat] exited with code %RC%
    pause
)
endlocal & exit /b %RC%
