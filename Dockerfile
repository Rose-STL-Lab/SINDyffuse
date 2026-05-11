FROM nvidia/cuda:12.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge based)
RUN wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    && conda config --system --set auto_activate_base false \
    && conda clean -afy

WORKDIR /workspace

# Copy environment definition and create conda environment
COPY environment.yaml /workspace/environment.yaml
RUN conda env create -f /workspace/environment.yaml \
    && conda clean -afy

# Install GPU PyTorch instead of CPU-only version
RUN conda run -n sindyffuse python -m pip install --upgrade pip \
    && conda run -n sindyffuse python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Copy project
COPY . /workspace

ENV CONDA_DEFAULT_ENV=sindyffuse
ENV PATH=${CONDA_DIR}/envs/sindyffuse/bin:${PATH}

SHELL ["conda", "run", "-n", "sindyffuse", "/bin/bash", "-c"]

# Default command: interactive shell in prepared environment
CMD ["/bin/bash"]
