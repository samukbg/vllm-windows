@echo off
REM ===================================================================
REM  qwen3.6-windows-server, top-level entrypoint.
REM
REM  Modified to run the vLLM server at port 11434 with 'speed' profile.
REM ===================================================================
setlocal
set "HERE=%~dp0"
if not exist "%HERE%\launcher\start.bat" goto :missing

call "%HERE%\launcher\start.bat" --auto-download --snapshot start_speed %*
exit /b %ERRORLEVEL%

:missing
echo [start.bat] launcher\start.bat not found.
echo This zip looks incomplete. Re-extract from the GitHub Release.
pause
exit /b 1
