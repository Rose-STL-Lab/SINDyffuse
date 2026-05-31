#!/usr/bin/env bash
# Multi-GPU SINDy training (single node).
# batch_size in train_sindy.json is PER GPU; global batch = batch_size * NPROC.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC="${NPROC:-1}"
CONFIG="${CONFIG:-configs/train_sindy.json}"
OUTPUT="${OUTPUT:-results/sindy_ddp_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-${HUMANML3D_ROOT:-datasets/HumanML3D}}"
PRELOAD="${PRELOAD:-}"

EXTRA=()
if [[ -n "${PRELOAD}" ]]; then
  EXTRA+=(--preload)
fi

echo "torchrun nproc_per_node=${NPROC} config=${CONFIG} output=${OUTPUT}"

exec torchrun --standalone --nproc_per_node="${NPROC}" \
  -m sindy.train \
  --config "${CONFIG}" \
  --data_root "${DATA_ROOT}" \
  --output "${OUTPUT}" \
  "${EXTRA[@]}"
