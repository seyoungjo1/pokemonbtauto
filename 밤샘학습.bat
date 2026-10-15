@echo off
chcp 949 >nul
cd /d "%~dp0"
title 챔피언스 배틀 AI - 밤샘 학습

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")
if not exist ".venv\Scripts\python.exe" (
    echo 가상환경을 만드는 중입니다...
    %PY% -m venv .venv
    if errorlevel 1 goto NOPY
)
set "VPY=.venv\Scripts\python.exe"

rem ---- 최신 프로그램·데이터·모델 받기
"%VPY%" -c "import updater; updater.apply()"
"%VPY%" -m pip install -q -r requirements.txt

rem ---- Node.js 확인 (쇼다운 시뮬레이터용)
where node >nul 2>nul
if errorlevel 1 goto NONODE

rem ---- 쇼다운 최신판 준비 (7일마다 새로 받음)
"%VPY%" tools\setup_showdown.py
if errorlevel 1 goto FAIL

rem ---- GPU 학습용 PyTorch (CUDA 12.4) - 처음 1회, 약 2.5GB
"%VPY%" -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo GPU 학습용 PyTorch 를 설치합니다. 처음 한 번만, 몇 분 걸립니다...
    "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
)
"%VPY%" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음 (CPU 로 학습)')"

rem ---- 셀프테스트 (데이터·데미지·신경망 GPU 경로)
"%VPY%" tests\selftest.py data stats damage features sim torch

echo.
set "HOURS="
set /p HOURS=몇 시간 학습할까요? [기본 8] : 
if "%HOURS%"=="" set "HOURS=8"
set "TYPE="
set /p TYPE=형식 - singles / doubles / both [기본 both] : 
if "%TYPE%"=="" set "TYPE=both"

echo.
echo 학습을 시작합니다. 멈추려면 이 폴더에 '학습중지.txt' 파일을 만드세요 (이번 회차를 마치고 멈춥니다).
echo 진행 기록: userdata\overnight\log.txt   리포트: reports\insights.md
echo.
if exist "학습중지.txt" del "학습중지.txt"
"%VPY%" tools\overnight.py --hours %HOURS% --type %TYPE%
goto END

:NONODE
echo Node.js 가 없습니다. https://nodejs.org 에서 LTS 버전을 설치한 뒤 다시 실행하세요.
goto END

:NOPY
echo Python 이 없습니다. https://www.python.org/downloads/ 에서 설치한 뒤 다시 실행하세요.
goto END

:FAIL
echo 준비 중 오류가 났습니다. 위 메시지를 확인하세요.

:END
pause
