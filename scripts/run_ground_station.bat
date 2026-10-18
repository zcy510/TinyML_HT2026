@echo off
REM Start the ground-station detection server. Run from the repo root.
REM Usage: scripts\run_ground_station.bat [weights] [demo-url]   (default yolov8s.pt)
REM   e.g. scripts\run_ground_station.bat yolov8s.pt http://192.168.1.20:8100   (live demo dashboard)
call .venv\Scripts\activate.bat
set WEIGHTS=%1
if "%WEIGHTS%"=="" set WEIGHTS=yolov8s.pt
set DEMO_ARG=
if not "%~2"=="" set DEMO_ARG=--demo %~2
echo Local IP addresses (give one of these to the Raspberry Pi):
ipconfig | findstr /i "IPv4"
python -m ground_station.server --host 0.0.0.0 --port 8000 --weights %WEIGHTS% --log logs\server_log.csv %DEMO_ARG%
