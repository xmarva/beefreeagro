@echo off
echo Setting up virtual environment for GPU...

:: Create virtual environment
python -m venv .venv

:: Activate virtual environment
call .\.venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install other dependencies
pip install opencv-python
pip install numpy
pip install tqdm
pip install matplotlib
pip install seaborn
pip install pyyaml
pip install scikit-learn
pip install ultralytics

echo Virtual environment setup complete for GPU.
echo To activate, run: .\.venv\Scripts\activate