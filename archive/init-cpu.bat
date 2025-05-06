@echo off
setlocal enabledelayedexpansion

echo === BeeFreePro CPU Initialization Script ===
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
echo Building Docker image for CPU...
docker build --build-arg BUILD_TYPE=cpu -t beefreeagro:cpu .

if %errorlevel% neq 0 (
    echo [ERROR] Failed to build Docker image.
    pause
    exit /b 1
)

echo.
echo Starting Docker container...
docker run -it --name beefreeagro-cpu --rm -v "%cd%:/beefreeagro" beefreeagro:cpu

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container.
    pause
    exit /b 1
)

echo.
echo Done!
pause