@echo off
REM ===================================================================
REM  vllm-windows launcher, portable Windows TUI
REM
REM  This .bat is the only thing you need to run. It uses the embeddable
REM  Python that ships next to it (..\python\), no pip install, no
REM  conda, no admin, no internet required.
REM
REM  Note: avoids parenthesized IF blocks because the install path may
REM  contain unbalanced parens (e.g. C:\Program Files (x86)\vllm\)
REM  which break cmd.exe's parser inside (...).
REM ===================================================================

setlocal EnableDelayedExpansion
title vLLM Launcher

cd /d "%~dp0"
set "APP_ROOT=%~dp0"
if "%APP_ROOT:~-1%"=="\" set "APP_ROOT=%APP_ROOT:~0,-1%"
set "REPO_ROOT=%APP_ROOT%\.."

REM Resolve PYTHON via flat IF chain (parens-safe).
set "PYTHON="
set "PYTHONHOME="
if exist "%REPO_ROOT%\python\python.exe" set "PYTHON=%REPO_ROOT%\python\python.exe" & set "PYTHONHOME=%REPO_ROOT%\python"
if not defined PYTHON if exist "%APP_ROOT%\python\python.exe" set "PYTHON=%APP_ROOT%\python\python.exe" & set "PYTHONHOME=%APP_ROOT%\python"
if not defined PYTHON if exist "%REPO_ROOT%\venv\Scripts\python.exe" set "PYTHON=%REPO_ROOT%\venv\Scripts\python.exe"
if not defined PYTHON goto :no_python

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
REM Embedded python ignores PYTHONPATH (python312._pth gates sys.path),
REM so the build script also writes "..\launcher" into the _pth file.
REM We still set PYTHONPATH for non-embedded fallback (developer venv).
set "PYTHONPATH=%APP_ROOT%"

REM Open inside Windows Terminal if available, the launcher's TUI looks
REM much better there than in legacy cmd. Skip if already inside WT, or
REM if the caller is running in headless / scripted mode (--headless,
REM --snapshot, --auto-download, --setup-only). Relaunching into a new WT
REM window would detach from the parent shell and break automation ,
REM the parent process exits 0 with no captured output and the user is
REM left wondering what happened. Detect common non-interactive parents
REM (CI runners, git-bash, MSYS, anywhere TERM is set) and stay in-place.
if defined WT_SESSION goto :run
if defined VLLM_NO_WT goto :run
if defined CI goto :run
if defined GITHUB_ACTIONS goto :run
if defined MSYSTEM goto :run
if defined TERM goto :run
echo.%*| findstr /I /C:"--headless" /C:"--snapshot" /C:"--auto-download" /C:"--model-dir" /C:"--setup-only" /C:" -y " /C:"--yes" >nul && goto :run
REM Prefer the bundled portable Windows Terminal that ships in the release
REM zip (..\terminal\WindowsTerminal.exe). Fall back to a system install
REM under Program Files if the user happens to have one. If neither is
REM present, just run in the current cmd window.
set "WT_EXE="
if exist "%REPO_ROOT%\terminal\WindowsTerminal.exe" set "WT_EXE=%REPO_ROOT%\terminal\WindowsTerminal.exe"
if not defined WT_EXE if exist "%APP_ROOT%\terminal\WindowsTerminal.exe" set "WT_EXE=%APP_ROOT%\terminal\WindowsTerminal.exe"
if not defined WT_EXE if exist "C:\Program Files\WindowsTerminal\wt.exe" set "WT_EXE=C:\Program Files\WindowsTerminal\wt.exe"
if not defined WT_EXE if exist "C:\Program Files\WindowsTerminal\WindowsTerminal.exe" set "WT_EXE=C:\Program Files\WindowsTerminal\WindowsTerminal.exe"
if not defined WT_EXE goto :run
REM Hide this cmd window before launching WT (matches portable-launcher pattern).
powershell -NoProfile -Command "Add-Type -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern IntPtr GetConsoleWindow(); [DllImport(\"user32.dll\")] public static extern bool ShowWindow(IntPtr h,int c);' -Name W -Namespace V; [V.W]::ShowWindow([V.W]::GetConsoleWindow(),0)" >nul 2>&1
"!WT_EXE!" -w vllm-launcher new-tab -d "!APP_ROOT!" --title "vLLM Launcher" cmd /c """%~f0""" %*
start /b "" "!PYTHON!" "!APP_ROOT!\activate_wt.py" "vLLM Launcher" >nul 2>&1
exit /b 0

:run
"%PYTHON%" -m app %*
set "ERR=%ERRORLEVEL%"
if "%ERR%"=="0" exit /b 0
echo.
echo [start.bat] launcher exited with code %ERR%.
echo Scroll up for the traceback. Press a key to close this window.
pause >nul
exit /b %ERR%

:no_python
echo [start.bat] could not find a Python interpreter. Expected:
echo    %REPO_ROOT%\python\python.exe   (portable release zip)
echo    %REPO_ROOT%\venv\Scripts\python.exe   (developer checkout)
pause
exit /b 1
