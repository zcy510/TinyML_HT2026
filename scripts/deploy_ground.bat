@echo off
REM ===========================================================================
REM One-click deploy + start of the ground station on the Windows lab PC.
REM Double-click it, or from the repo root:
REM     scripts\deploy_ground.bat              (yolov8s.pt)
REM     scripts\deploy_ground.bat yolov8m.pt   (other model)
REM     scripts\deploy_ground.bat yolov8s.pt http://192.168.1.20:8100   (report to the live demo dashboard)
REM     set SKIP_TESTS=1 ^& scripts\deploy_ground.bat
REM Steps: .venv -> PyTorch (CUDA build if an NVIDIA GPU is present) -> deps
REM        (only when requirements change) -> quick tests -> firewall -> server
REM ===========================================================================
setlocal
cd /d "%~dp0\.."

set "WEIGHTS=%~1"
if "%WEIGHTS%"=="" set "WEIGHTS=yolov8s.pt"
set "DEMO_ARG="
if not "%~2"=="" set "DEMO_ARG=--demo %~2"
REM CUDA 12.6 wheels: Python 3.9-3.13, NVIDIA driver 528+ (change if needed)
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"

REM ---------------------------------------------------------------- Python
set "PY=python"
where py >nul 2>&1 && set "PY=py -3"
%PY% -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [deploy-ground] ERROR: Python 3.9-3.13 not found.
  echo Install Python 3.12 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run again.
  goto :fail
)

REM ------------------------------------------------------------ 1. venv
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [deploy-ground] 1/5 Creating virtual environment .venv
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
) else (
  echo [deploy-ground] 1/5 Using existing .venv
)
call ".venv\Scripts\activate.bat"

REM ---------------------------------------------- 2. deps (only if changed)
fc /b requirements-ground.txt ".venv\installed-requirements-ground.txt" >nul 2>&1
if not errorlevel 1 (
  echo [deploy-ground] 2/5 Python deps up to date
  goto :cuda_check
)
echo.
echo [deploy-ground] 2/5 Installing Python deps ^(first time downloads PyTorch, several minutes^)
python -m pip install --upgrade pip
nvidia-smi >nul 2>&1
if not errorlevel 1 (
  echo [deploy-ground] NVIDIA GPU found: installing CUDA build of PyTorch
  python -m pip install torch torchvision --index-url %TORCH_INDEX%
)
python -m pip install -r requirements-ground.txt
if errorlevel 1 goto :fail
copy /y requirements-ground.txt ".venv\installed-requirements-ground.txt" >nul

:cuda_check
REM If there is a GPU but torch is the CPU build, swap it for the CUDA build once.
nvidia-smi >nul 2>&1
if errorlevel 1 goto :torch_info
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 goto :torch_info
echo [deploy-ground] GPU present but PyTorch is CPU-only: reinstalling CUDA build
python -m pip install --force-reinstall --no-deps torch torchvision --index-url %TORCH_INDEX%

:torch_info
python -c "import torch; print('[deploy-ground] torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'running on CPU')"

REM ------------------------------------------------------------ 3. tests
if "%SKIP_TESTS%"=="1" (
  echo [deploy-ground] 3/5 Tests skipped
) else (
  echo.
  echo [deploy-ground] 3/5 Running quick tests
  python -m pytest tests -q
  if errorlevel 1 echo [deploy-ground] WARNING: tests failed - starting the server anyway
)

REM --------------------------------------------------------- 4. firewall
netsh advfirewall firewall show rule name="UAV Ground Station 8000" >nul 2>&1
if errorlevel 1 (
  netsh advfirewall firewall add rule name="UAV Ground Station 8000" dir=in action=allow protocol=TCP localport=8000 >nul 2>&1
  if errorlevel 1 (
    echo [deploy-ground] 4/5 NOTE: could not add a firewall rule ^(needs admin^).
    echo                If Windows asks, click "Allow access" for Python.
  ) else (
    echo [deploy-ground] 4/5 Firewall rule for TCP 8000 added
  )
) else (
  echo [deploy-ground] 4/5 Firewall rule already present
)

REM ------------------------------------------------------------ 5. server
echo.
echo [deploy-ground] 5/5 Starting ground station with %WEIGHTS%
echo Give the Raspberry Pi one of these addresses ^(http://IP:8000^):
ipconfig | findstr /i "IPv4"
echo ^(Ctrl+C to stop^)
if not exist logs mkdir logs
python -m ground_station.server --host 0.0.0.0 --port 8000 --weights %WEIGHTS% --log logs\server_log.csv %DEMO_ARG%
goto :end

:fail
echo.
echo [deploy-ground] FAILED - see the messages above.
:end
pause
endlocal
