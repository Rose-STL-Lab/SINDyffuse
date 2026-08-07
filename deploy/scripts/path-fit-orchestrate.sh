#!/usr/bin/env bash
# In-cluster orchestrator: prepare → convert → fit (kubectl wait; no sleep).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

BASE="${ROOT}/deploy/jobs/preprocess-dataset/fit-function-paths"

run_phase sindyffuse-path-fit-prepare "${BASE}/prepare" 10m "path-fit"
run_phase sindyffuse-path-fit-convert "${BASE}/convert" 2h "path-fit"
run_phase sindyffuse-path-fit-fit "${BASE}/fit" 6h "path-fit"

echo "Path-fit pipeline complete."
