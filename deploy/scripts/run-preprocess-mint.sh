#!/usr/bin/env bash
# Run MinT preprocess on Kubernetes/local (mirrors run-preprocess-nimble.sh).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
python scripts/preprocess_mint.py "$@"
