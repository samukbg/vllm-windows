@echo off
setlocal
cd /d "%~dp0"
set "REPO_ROOT=%~dp0.."
set "PY="
if exist "%REPO_ROOT%\python\python.exe" set "PY=%REPO_ROOT%\python\python.exe"
if not defined PY if exist "%REPO_ROOT%\venv\Scripts\python.exe" set "PY=%REPO_ROOT%\venv\Scripts\python.exe"
if not defined PY (
    echo [start_5090.bat] no python interpreter found
    pause & exit /b 1
)
"%PY%" "%~dp0start_5090.py" %*
