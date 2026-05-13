@echo off
REM ===================================================================
REM  qwen3.6-windows-server, top-level entrypoint.
REM
REM  Delegates to launcher\start.bat. Both files are intentionally
REM  available so a user can double-click either one and get the same
REM  result. Kept thin so the two never drift apart.
REM
REM  Note: avoids parenthesized IF blocks because the install path may
REM  contain unbalanced parens (e.g. "C:\Program Files (x86)\vllm\")
REM  which break cmd.exe's parser inside (...).
REM ===================================================================
setlocal
set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"

if not exist "%HERE%\launcher\start.bat" goto :missing

call "%HERE%\launcher\start.bat" %*
exit /b %ERRORLEVEL%

:missing
echo [start.bat] launcher\start.bat not found.
echo This zip looks incomplete. Re-extract from the GitHub Release.
pause
exit /b 1
