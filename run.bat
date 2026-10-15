@echo off
chcp 949 >nul
cd /d "%~dp0"
title 챔피언스 배틀 AI

rem ---- 파이썬 찾기 (py 우선)
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

rem ---- 가상환경 (처음 1회)
if not exist ".venv\Scripts\python.exe" (
    echo 가상환경을 만드는 중입니다. 잠시만 기다려 주세요...
    %PY% -m venv .venv
    if errorlevel 1 goto NOPY
)

rem ---- 업데이트 확인 후 실행 (updater.py 가 app.py 를 띄움)
".venv\Scripts\python.exe" updater.py
goto END

:NOPY
echo Python 이 없습니다. https://www.python.org/downloads/ 에서 설치한 뒤 다시 실행하세요.
echo (설치할 때 "Add python.exe to PATH" 를 꼭 체크하세요)

:END
pause
