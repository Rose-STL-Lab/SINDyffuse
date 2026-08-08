#!/usr/bin/env bash
# Local driver: moco-track workers → normalization (kubectl wait; no sleep).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

BASE="${ROOT}/deploy/jobs/preprocess-dataset"

run_phase sindyffuse-preprocess-moco-track "${BASE}/moco-track" 24h "moco-track"
run_phase sindyffuse-compute-normalization "${BASE}/normalization" 2h "normalization"

echo "MocoTrack + normalization pipeline complete."
