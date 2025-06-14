# ---------- base image ----------
FROM python:3.11-slim

# ---------- system packages ----------
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# ---------- python packages ----------
RUN pip install --upgrade pip \
 && pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0 \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir \
        numpy \
        h5py \
        scikit-image \
        pandas \
        pyyaml \
        torchmetrics \
        runstats \
        fastmri \
        requests \
        flake8 \
        pylint
