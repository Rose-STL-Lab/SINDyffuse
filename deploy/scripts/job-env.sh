#!/usr/bin/env bash
# Source from Kubernetes job manifests for conda + runtime env only.
set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate sindyffuse

# Append conda libs so NVIDIA container-runtime paths stay first (prepend breaks GPU
# kernel launch on some nodes when torch.cuda.is_available() is true but kernels fail).
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  case ":${LD_LIBRARY_PATH}:" in
    *":${CONDA_PREFIX}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib" ;;
  esac
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
fi

# K8s NVIDIA device plugin often sets NVIDIA_VISIBLE_DEVICES but not CUDA_VISIBLE_DEVICES.
# Must be exported before Python imports torch.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

export PYTHONPATH="${PYTHONPATH:-/mnt/SINDyffuse}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export CLIP_DOWNLOAD_ROOT="${CLIP_DOWNLOAD_ROOT:-/mnt/SINDyffuse/.cache/clip}"
