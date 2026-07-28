#!/usr/bin/env bash
# Run dataset preprocess workers only (indexed shards).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PREPROCESS_NUM_SHARDS="${PREPROCESS_NUM_SHARDS:-128}"
MODE="${1:-k8s}"

if [[ "$MODE" == "local" ]]; then
  shift || true
  exec python scripts/preprocess_mint.py "$@"
fi

if [[ "$MODE" != "k8s" ]]; then
  echo "Usage: $0 [k8s|local] [extra preprocess_mint.py args...]" >&2
  exit 1
fi

echo "Applying preprocess-dataset workers (PREPROCESS_NUM_SHARDS=${PREPROCESS_NUM_SHARDS})..."
kubectl apply -k deploy/jobs/preprocess-dataset

echo "Waiting for preprocess-dataset workers to complete (timeout 168h)..."
kubectl wait --for=condition=complete job/sindyffuse-preprocess-dataset --timeout=168h
