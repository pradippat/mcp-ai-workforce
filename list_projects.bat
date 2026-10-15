@echo off
cd /d "%~dp0"
venv\Scripts\python.exe server.py --list-projects
echo.
pause
