#!/usr/bin/env bash
# Local driver: path-fit prepare → convert → fit (kubectl wait; no sleep).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

BASE="${ROOT}/deploy/jobs/preprocess-dataset/fit-function-paths"

run_phase sindyffuse-path-fit-prepare "${BASE}/prepare" 2h "path-fit-prepare"
run_phase sindyffuse-path-fit-convert "${BASE}/convert" 24h "path-fit-convert"
run_phase sindyffuse-path-fit-fit "${BASE}/fit" 48h "path-fit-fit"

echo "Path-fit pipeline complete."
