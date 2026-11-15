@echo off
setlocal
cd /d "%~dp0"
title Silero TTS Discord Bridge

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if not exist venv\Scripts\python.exe goto CREATE_VENV
goto RUN_APP

:CREATE_VENV
echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 goto PYTHON_ERROR

:RUN_APP
call venv\Scripts\activate.bat

echo [2/3] Checking dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 goto PIP_ERROR

echo.
echo [3/3] Starting Silero TTS Server...
echo.
python app.py
if errorlevel 1 goto APP_ERROR
goto END

:PYTHON_ERROR
echo.
echo [ERROR] Python is not installed or not added to PATH!
echo Download Python 3.10+ from https://www.python.org and check "Add Python to PATH".
pause
exit /b 1

:PIP_ERROR
echo.
echo [ERROR] Failed to install packages from requirements.txt.
pause
exit /b 1

:APP_ERROR
echo.
echo [!] Server stopped unexpectedly.
pause
exit /b 1

:END
