ARG BUILD_TYPE=gpu

# Базовый образ: с CUDA для GPU или ubuntu для CPU
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS gpu-base
FROM ubuntu:22.04 AS cpu-base

# Выбираем нужный базовый образ в зависимости от BUILD_TYPE
FROM ${BUILD_TYPE}-base

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем символьную ссылку python на python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Установка Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Добавление conda в PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Копирование файла environment.yml
COPY stickers_env.yml /tmp/

# Создание conda-окружения из environment.yml
RUN conda env create -f /tmp/stickers_env.yml

# Делаем conda-окружение активным по умолчанию
RUN echo "source activate stickers_env" > ~/.bashrc
ENV PATH /opt/conda/envs/stickers_env/bin:$PATH

# Устанавливаем PyTorch с поддержкой CUDA для GPU или без для CPU
RUN if [ "$BUILD_TYPE" = "gpu" ] ; then \
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 ; \
    else \
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Устанавливаем дополнительные зависимости
RUN pip install pillow

# Клонирование репозитория
RUN git clone https://github.com/xmarva/beefreeagro.git /beefreeagro

# Рабочая директория
WORKDIR /beefreeagro

# Переменная окружения для определения, использовать ли GPU
ENV USE_GPU=${BUILD_TYPE}

# Команда для запуска при старте контейнера
CMD ["bash"]