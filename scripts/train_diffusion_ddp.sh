#!/usr/bin/env bash
# Multi-GPU diffusion training (single node).
# train.batch_size in config is PER GPU; global batch = batch_size * NPROC.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC="${NPROC:-1}"
CONFIG="${CONFIG:-configs/train_diffusion.json}"
OUT_DIR="${OUT_DIR:-results/diffusion_ddp_$(date +%Y%m%d_%H%M%S)}"
PRELOAD="${PRELOAD:-}"

EXTRA=()
if [[ -n "${PRELOAD}" ]]; then
  EXTRA+=(--preload)
fi

echo "torchrun nproc_per_node=${NPROC} config=${CONFIG} out_dir=${OUT_DIR}"

exec torchrun --standalone --nproc_per_node="${NPROC}" \
  train_diffusion.py \
  --config "${CONFIG}" \
  --out_dir "${OUT_DIR}" \
  "${EXTRA[@]}"
