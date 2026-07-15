@echo off
setlocal EnableExtensions
rem 常に outlook_google_sync/ をカレントにして起動する（ダブルクリックでも同じ）
cd /d "%~dp0.."
if errorlevel 1 (
  echo [ERROR] プロジェクトルートへ移動できませんでした: "%~dp0.."
  pause
  exit /b 1
)

set "ROOT=%CD%"
set "PYTHONPATH=%ROOT%\src"
set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] 仮想環境が見つかりません。
  echo   期待パス: %VENV_PY%
  echo   先に次を実行してください:
  echo     .\scripts\setup_env.bat
  echo   （カレントは %ROOT% ）
  pause
  exit /b 1
)

"%VENV_PY%" -m outlook_google_sync.main
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo.
  echo [ERROR] GUI が異常終了しました。終了コード: %ERR%
  pause
)
exit /b %ERR%
