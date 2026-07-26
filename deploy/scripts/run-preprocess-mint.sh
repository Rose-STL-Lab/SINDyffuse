#!/usr/bin/env bash
# Run MinT preprocess: indexed worker job(s) then normalization (mirrors preprocess-nimble).
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

echo "Applying preprocess-mint workers (PREPROCESS_NUM_SHARDS=${PREPROCESS_NUM_SHARDS})..."
kubectl apply -k deploy/jobs/preprocess-mint/workers

echo "Waiting for preprocess-mint workers to complete (timeout 168h)..."
if ! kubectl wait --for=condition=complete job/sindyffuse-preprocess-mint --timeout=168h; then
  echo "Worker job did not complete successfully; skipping normalization." >&2
  exit 1
fi

echo "Applying MinT normalization job..."
kubectl apply -k deploy/jobs/preprocess-mint/normalization
