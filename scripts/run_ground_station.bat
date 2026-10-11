@echo off
REM Start the ground-station detection server. Run from the repo root.
REM Usage: scripts\run_ground_station.bat [weights]   (default yolov8s.pt)
call .venv\Scripts\activate.bat
set WEIGHTS=%1
if "%WEIGHTS%"=="" set WEIGHTS=yolov8s.pt
echo Local IP addresses (give one of these to the Raspberry Pi):
ipconfig | findstr /i "IPv4"
python -m ground_station.server --host 0.0.0.0 --port 8000 --weights %WEIGHTS% --log logs\server_log.csv
