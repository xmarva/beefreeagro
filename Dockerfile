ARG BUILD_TYPE=gpu

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS gpu-base
FROM ubuntu:22.04 AS cpu-base

FROM ${BUILD_TYPE}-base

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

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

COPY stickers_env.yml /tmp/

RUN conda env create -f /tmp/stickers_env.yml

RUN echo "source activate stickers_env" > ~/.bashrc
ENV PATH /opt/conda/envs/stickers_env/bin:$PATH

RUN if [ "$BUILD_TYPE" = "gpu" ] ; then \
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 ; \
    else \
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu ; \
    fi

RUN pip install pillow

RUN git clone https://github.com/xmarva/beefreeagro.git /beefreeagro

WORKDIR /beefreeagro

ENV USE_GPU=${BUILD_TYPE}

CMD ["bash"]