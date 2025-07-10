# ---------- base image ----------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# ---------- system packages ----------
RUN export DEBIAN_FRONTEND=noninteractive \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python-is-python3 \
        git \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && rm -rf /var/lib/apt/lists/*

# ---------- python packages ----------
RUN pip3 install --no-cache-dir --upgrade pip \
        && rm -rf /root/.cache/pip \
        && pip3 install --no-cache-dir \
        numpy \
        h5py \
        matplotlib \
        scikit-image \
        pandas \
        pyyaml \
        torchmetrics \
        runstats \
        fastmri \
        requests \
        flake8 \
        black \
        && pip3 install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
