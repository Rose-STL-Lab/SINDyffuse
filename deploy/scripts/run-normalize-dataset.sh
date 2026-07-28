#!/usr/bin/env bash
# Compute Mean.npy / Std.npy after preprocess-dataset workers finish.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-k8s}"

if [[ "$MODE" == "local" ]]; then
  shift || true
  exec python scripts/compute_normalization.py "$@"
fi

if [[ "$MODE" != "k8s" ]]; then
  echo "Usage: $0 [k8s|local] [extra compute_normalization.py args...]" >&2
  exit 1
fi

echo "Applying normalize-dataset job..."
kubectl apply -k deploy/jobs/normalize-dataset

echo "Waiting for normalize-dataset to complete (timeout 24h)..."
kubectl wait --for=condition=complete job/sindyffuse-normalize-dataset --timeout=24h
