@echo off
setlocal
cd /d "%~dp0"

REM Resolve install root: when this file lives at <install>\update.bat, %~dp0 IS the
REM install root. When it lives in launcher\update.bat (dev tree), the launcher source
REM doesn't have its own python\python.exe, so fall through to system python.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PY=%ROOT%\python\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" "%ROOT%\windows_tools\update.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo update.py exited with code %RC%.
    pause
)
endlocal & exit /b %RC%
