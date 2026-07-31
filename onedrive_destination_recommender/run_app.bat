@echo off
setlocal

set "APP_ROOT=%~dp0"
set "APP_PYTHON=%APP_ROOT%.venv\Scripts\python.exe"

if not exist "%APP_PYTHON%" (
    echo Python virtual environment was not found.
    echo Run: py -3.12 -m venv .venv
    echo Then: .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
    exit /b 1
)

"%APP_PYTHON%" -m onedrive_destination_recommender
exit /b %errorlevel%
