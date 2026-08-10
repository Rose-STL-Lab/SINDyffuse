#!/usr/bin/env bash
# Local driver: single path-fit Job (sample → B3D→.mot → OpenSim fit).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

BASE="${ROOT}/deploy/jobs/preprocess-dataset/fit-function-paths"

run_phase sindyffuse-fit-function-paths "${BASE}" 48h "path-fit"

echo "Path-fit complete."
