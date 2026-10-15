@echo off
cd /d "%~dp0"
echo.
echo === Unstoppable AI Workforce v3 - install and setup ===
echo.
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this file again.
  pause
  exit /b 1
)
if not exist venv\Scripts\python.exe (
  echo Creating a private Python environment ...
  %PY% -m venv venv
)
echo Installing packages (1-2 minutes the first time) ...
venv\Scripts\python.exe -m pip install --quiet --upgrade pip
venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo Package install failed. Check your internet connection and run this file again.
  pause
  exit /b 1
)
echo.
venv\Scripts\python.exe server.py --setup
echo.
pause
