@echo off
echo Setting up BeeFreeAgro environment for CPU...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.x and add it to your PATH.
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy tqdm matplotlib seaborn pyyaml scikit-learn pillow ultralytics

echo Creating directory structure...
if not exist models mkdir models

echo.
echo Environment setup complete for CPU!
echo.
echo To activate this environment in the future, run:
echo call .venv\Scripts\activate.bat
echo.