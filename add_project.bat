@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Run setup_and_run.bat first.
  pause
  exit /b 1
)
echo.
echo === Give the AI workforce access to one of your project folders ===
echo (You can also drag a folder onto this file.)
echo.
if "%~1"=="" (
  venv\Scripts\python.exe server.py --add-project
) else (
  venv\Scripts\python.exe server.py --add-project "%~1"
)
echo.
pause
