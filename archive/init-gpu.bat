@echo off
setlocal enabledelayedexpansion

echo === BeeFreePro GPU Initialization Script ===
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python and add it to PATH.
    pause
    exit /b 1
)

:: Create temp directory for downloads
set "TEMP_DIR=%TEMP%\beefreeagro_temp"
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

:: Check Docker installation
echo Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Docker not installed. Attempting to install Docker...
    
    echo Downloading Docker Desktop for Windows...
    python -c "import urllib.request; urllib.request.urlretrieve('https://desktop.docker.com/win/stable/Docker Desktop Installer.exe', r'%TEMP_DIR%\DockerDesktopInstaller.exe')"
    
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download Docker installer.
        echo Please install Docker Desktop manually: https://www.docker.com/products/docker-desktop/
        pause
        exit /b 1
    )
    
    echo Starting Docker Desktop installer...
    echo NOTE: Follow the Docker Desktop installer instructions.
    start /wait "" "%TEMP_DIR%\DockerDesktopInstaller.exe"
    
    echo.
    echo After Docker Desktop installation completes, start it and wait for full initialization.
    echo Then press any key to continue...
    pause >nul
    
    docker --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Docker still not available. Make sure Docker Desktop is running.
        pause
        exit /b 1
    )
) else (
    echo [OK] Docker already installed.
)

:: Check NVIDIA Container Toolkit
echo Checking NVIDIA support in Docker...
docker info | findstr "Runtimes: nvidia" >nul
if %errorlevel% neq 0 (
    echo [WARNING] NVIDIA Container Toolkit not detected.
    echo Make sure you have:
    echo 1. NVIDIA drivers installed
    echo 2. NVIDIA Container Toolkit installed
    echo 3. GPU support enabled in Docker Desktop (Settings -^> General -^> Use GPU acceleration)
    echo.
    echo Instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
    echo.
    echo Press any key to continue (installation will proceed, but GPU may not be available)...
    pause >nul
)

echo.
echo Downloading required files from repository...

:: Download Dockerfile
echo Downloading Dockerfile...
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/xmarva/beefreeagro/main/Dockerfile', 'Dockerfile')"
if %errorlevel% neq 0 (
    echo [WARNING] Failed to download Dockerfile from repository. Using local Dockerfile.
    if not exist "Dockerfile" (
        echo [ERROR] Local Dockerfile not found.
        pause
        exit /b 1
    )
)

:: Download stickers_env.yml
echo Downloading stickers_env.yml...
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/xmarva/beefreeagro/main/stickers_env.yml', 'stickers_env.yml')"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download stickers_env.yml from repository.
    pause
    exit /b 1
)

echo.
echo Building Docker image for GPU...
docker build --build-arg BUILD_TYPE=gpu -t beefreeagro:gpu .

if %errorlevel% neq 0 (
    echo [ERROR] Failed to build Docker image.
    pause
    exit /b 1
)

echo.
echo Starting Docker container with GPU support...
docker run -it --name beefreeagro-gpu --rm --gpus all --shm-size=1g -v "%cd%:/beefreeagro" beefreeagro:gpu

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container.
    echo NVIDIA support might not be configured in Docker or GPU unavailable.
    echo Try running the CPU version with init-cpu.bat
    pause
    exit /b 1
)

echo.
echo Container started with GPU support!
echo To verify GPU within container run:
echo python -c "import torch; print('GPU available:', torch.cuda.is_available())"
echo.
echo Done!
pause