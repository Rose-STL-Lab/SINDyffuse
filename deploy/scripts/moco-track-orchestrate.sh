#!/usr/bin/env bash
# Local driver: compiled F → polynomials → canary → workers → normalization.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

BASE="${ROOT}/deploy/jobs/preprocess-dataset"

run_phase sindyffuse-build-opensimad-ext "${BASE}/build-opensimad-ext" 6h "opensimad-ext"
run_polynomial_phases
run_phase sindyffuse-opensimad-canary "${BASE}/opensimad-canary" 4h "opensimad-canary"
run_phase sindyffuse-prepare-moco-tasks "${BASE}/prepare-moco-tasks" 2h "motion-task-prepare"
run_phase sindyffuse-preprocess-moco-track "${BASE}/moco-track" "${MOCO_WAIT_TIMEOUT:-720h}" "opensimad-motion-tasks"
run_phase sindyffuse-compute-normalization "${BASE}/normalization" 2h "normalization"

echo "LaiUhlrich2022 OpenSimAD (MinT) + normalization pipeline complete."
