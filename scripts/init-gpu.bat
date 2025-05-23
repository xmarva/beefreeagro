@echo off
echo Setting up BeeFreeAgro environment with GPU support...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.x and add it to your PATH.
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy tqdm matplotlib seaborn pyyaml scikit-learn pillow ultralytics

:: Create necessary directories
echo Creating directory structure...
if not exist data\raw\stickers mkdir data\raw\stickers
if not exist data\raw\background mkdir data\raw\background
if not exist data\synth\imgs mkdir data\synth\imgs
if not exist models mkdir models

echo.
echo Environment setup complete with GPU support!
echo.
echo To activate this environment in the future, run:
echo call .venv\Scripts\activate.bat
echo.
echo To verify GPU availability:
.venv\Scripts\python.exe -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
echo.