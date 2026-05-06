@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Resolve install root: when this file lives at <install>\update.bat, %~dp0 IS the
REM install root. When it lives in launcher\update.bat (dev tree), the launcher source
REM doesn't have its own python\python.exe, so fall through to system python.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PY=%ROOT%\python\python.exe"
set "STAGED_PY="

if exist "%PY%" (
    REM The embedded interpreter holds a lock on its own DLLs while it runs,
    REM so doing rmtree on python\ during the update fails with WinError 5.
    REM Stage a throwaway copy of the embedded python tree under %TEMP% and
    REM run update.py from there so the install copy is unlocked.
    set "STAGED_PY=%TEMP%\qwen36-update-py"
    if exist "!STAGED_PY!" rmdir /s /q "!STAGED_PY!" 2>nul
    echo [update.bat] staging embedded python under !STAGED_PY! ...
    xcopy /e /i /q /y /h "%ROOT%\python" "!STAGED_PY!" >nul
    if errorlevel 1 (
        echo [update.bat] WARNING: could not stage python to TEMP, running in-place.
        set "STAGED_PY="
    ) else (
        set "PY=!STAGED_PY!\python.exe"
    )
) else (
    set "PY=python"
)

"!PY!" "%ROOT%\windows_tools\update.py" %*
set "RC=!ERRORLEVEL!"

if defined STAGED_PY (
    rmdir /s /q "!STAGED_PY!" 2>nul
)

if not "!RC!"=="0" (
    echo.
    echo update.py exited with code !RC!.
    pause
)
endlocal & exit /b %RC%
