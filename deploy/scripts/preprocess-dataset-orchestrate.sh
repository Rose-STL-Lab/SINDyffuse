#!/usr/bin/env bash
# In-cluster orchestrator: IK → path-fit → moco+normalization (kubectl wait; no sleep).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

BASE="${ROOT}/deploy/jobs/preprocess-dataset"

run_phase sindyffuse-preprocess-ik "${BASE}/ik" 12h "preprocess-dataset"
run_phase sindyffuse-fit-function-paths "${BASE}/fit-function-paths" 24h "path-fit"
run_phase sindyffuse-moco-track-orchestrator "${BASE}/moco-track/orchestrator" 26h "moco-track-orchestrator"

echo "Preprocess-dataset pipeline complete."
