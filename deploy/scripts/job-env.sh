#!/usr/bin/env bash
# Source from Kubernetes job manifests for conda + runtime env only.
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate sindyffuse
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-/mnt/SINDyffuse}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CLIP_DOWNLOAD_ROOT="${CLIP_DOWNLOAD_ROOT:-/mnt/SINDyffuse/.cache/clip}"
