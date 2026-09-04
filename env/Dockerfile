FROM nvidia/cuda:13.1.2-cudnn-runtime-ubuntu24.04

LABEL org.opencontainers.image.description="SINDyffuse: diffusion + SINDy + OpenSim / OpenSimAD"

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Use bash for RUN (dash lacks pipefail; keep scripts portable either way).
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    build-essential \
    cmake \
    g++ \
    libegl1 \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    liblapack3 \
    libblas3 \
    libopenblas0 \
    libquadmath0 \
    && rm -rf /var/lib/apt/lists/*

# OpenSim 4.5.2 .so links libgfortran.so.3 (GCC 6); Ubuntu 24.04 only ships .so.5.
RUN cd /tmp && \
    wget -q http://archive.ubuntu.com/ubuntu/pool/universe/g/gcc-6/libgfortran3_6.4.0-17ubuntu1_amd64.deb && \
    dpkg-deb -x libgfortran3_6.4.0-17ubuntu1_amd64.deb /tmp/libgfortran3 && \
    cp -a /tmp/libgfortran3/usr/lib/x86_64-linux-gnu/libgfortran.so.3* /usr/lib/x86_64-linux-gnu/ && \
    ldconfig && \
    test -e /usr/lib/x86_64-linux-gnu/libgfortran.so.3 && \
    rm -rf /tmp/libgfortran3 /tmp/libgfortran3_*.deb

RUN wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    && conda config --system --set auto_activate_base false \
    && conda clean -afy

WORKDIR /mnt

COPY env/environment.yaml /mnt/environment.yaml
COPY env/constraints.txt /mnt/constraints.txt
COPY env/patch_opensim_moco.py /mnt/patch_opensim_moco.py

# Build a self-consistent env in-image (no runtime package repair in jobs).
# OpenSim may leave conda libcasadi.so* in $ENV/lib; those ABI-clash with pip
# casadi==3.7.1 (undefined symbol copy_elision_min_size). Remove them, then pin pip.
RUN mamba env create -f /mnt/environment.yaml && \
    ENV_DIR=${CONDA_DIR}/envs/sindyffuse && \
    PY=${ENV_DIR}/bin/python && \
    export PIP_CONSTRAINT=/mnt/constraints.txt && \
    mamba remove -n sindyffuse -y --force casadi || true && \
    rm -f ${ENV_DIR}/lib/libcasadi.so* ${ENV_DIR}/lib/libcasadi_* && \
    rm -rf ${ENV_DIR}/casadi && \
    ${PY} -m pip install --no-deps --force-reinstall 'casadi==3.7.1' && \
    ${PY} -m pip install --force-reinstall 'numpy>=1.25,<1.26' && \
    python3 /mnt/patch_opensim_moco.py && \
    mamba clean --all --yes && \
    conda clean --all --yes && \
    rm -rf /opt/conda/pkgs/* /root/.cache/pip

# Append conda libs (do not prepend): prepending $ENV/lib makes the dynamic linker
# prefer OpenSim leftovers over pip casadi's bundled libcasadi.
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_DIR}/envs/sindyffuse/lib

# Verify OpenSim alone, CasADi alone, then CasADi-before-OpenSim (OpenSimAD order).
RUN ENV_DIR=${CONDA_DIR}/envs/sindyffuse && PY=${ENV_DIR}/bin/python && \
    test -z "$(ls ${ENV_DIR}/lib/libcasadi.so* 2>/dev/null)" && echo 'no conda libcasadi leftovers' && \
    ${PY} -c "\
import numpy as np; \
import scipy, sklearn, joblib, torch, torchvision, clip; \
import opensim; \
assert np.__version__.startswith('1.25'), np.__version__; \
assert hasattr(opensim, 'Model'); \
print('numpy', np.__version__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('opensim', opensim.GetVersion()); \
print('opensim.moco', hasattr(opensim, 'MocoStudy')); \
import importlib.util; \
assert importlib.util.find_spec('nimblephysics') is None, 'nimblephysics must not be installed'" && \
    ${PY} -c "\
import casadi; \
assert casadi.__version__.startswith('3.7.1'), casadi.__version__; \
print('casadi', casadi.__version__)" && \
    ${PY} -c "\
import casadi, opensim; \
assert hasattr(opensim, 'Model'); \
print('combined Model+casadi ok'); \
print('combined moco available', hasattr(opensim, 'MocoStudy'))"

ENV CONDA_DEFAULT_ENV=sindyffuse
ENV PATH=${CONDA_DIR}/envs/sindyffuse/bin:${PATH}

SHELL ["conda", "run", "-n", "sindyffuse", "/bin/bash", "-c"]

CMD ["/bin/bash"]
