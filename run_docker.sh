#!/bin/bash

# Скрипт для сборки и запуска Docker-контейнера
# с поддержкой GPU или CPU в зависимости от параметров

# Имя образа
IMAGE_NAME="stickers_env"

# Проверяем доступность NVIDIA Container Toolkit
if command -v nvidia-smi &> /dev/null
then
    USE_GPU=True
    echo "NVIDIA GPU обнаружена, запускаем с поддержкой GPU"
    
    # Собираем Docker образ
    docker build -t $IMAGE_NAME .
    
    # Запускаем контейнер с поддержкой GPU
    docker run --gpus all -it \
        -e USE_GPU=True \
        -v $(pwd):/app \
        $IMAGE_NAME
else
    USE_GPU=False
    echo "NVIDIA GPU не обнаружена, запускаем на CPU"
    
    # Собираем Docker образ
    docker build -t $IMAGE_NAME .
    
    # Запускаем контейнер на CPU
    docker run -it \
        -e USE_GPU=False \
        -v $(pwd):/app \
        $IMAGE_NAME
fi