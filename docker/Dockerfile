FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    sudo \
    build-essential \
    curl \
    wget \
    git \
    vim \
    tar \
    unzip \
    openssh-server \
    python3-pip \
    python3-venv \
    ninja-build \
    cmake \
    libopenblas-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# symlink for python3
RUN ln -s /usr/bin/python3 /usr/bin/python

ARG PIP_VERSION=23.3.1
ARG SETUPTOOLS_VERSION=59.5.0
RUN python -m pip install pip==${PIP_VERSION} setuptools==${SETUPTOOLS_VERSION}

RUN pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install opencv-python ninja

ENV CUDA_HOME=/usr/local/cuda-11.7
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout 017abbf && \
    pip install .

COPY ./requirements.txt .
RUN pip install git+https://github.com/cocodataset/panopticapi.git && \
    pip install git+https://github.com/mcordts/cityscapesScripts.git && \
    pip install -r requirements.txt

COPY ./oneformer/modeling/pixel_decoder/ops ops
RUN cd ops && sh make.sh

# add user and his password
ENV USER=docker_oneformer
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}

RUN mkdir OneFormer && chown -R ${UID}:${GID} /home/${USER}
RUN mkdir data && chown -R ${UID}:${GID} /home/${USER}

ENV PYTHONPATH=/home/${USER}/OneFormer

USER ${UID}:${GID}
