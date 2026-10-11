@echo off
REM One-time setup of the ground station on Windows. Run from the repo root.
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
REM If the PC has an NVIDIA GPU, install the CUDA build of PyTorch first (cu121 works for most drivers):
nvidia-smi >nul 2>&1 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-ground.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -m pytest tests -q
echo.
echo Setup done. Start the server with scripts\run_ground_station.bat
