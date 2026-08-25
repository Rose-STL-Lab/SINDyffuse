#!/usr/bin/env bash
# Local driver: single path-fit Job (sample → B3D→.mot → OpenSim fit).
# Skips when FunctionBasedPathSet.xml already exists unless PATH_FIT_FORCE=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

PATH_SET="${ROOT}/models/rajagopal/Rajagopal2015_FunctionBasedPathSet.xml"
FORCE="${PATH_FIT_FORCE:-0}"
if [[ -f "${PATH_SET}" && ! "${FORCE}" =~ ^(1|true|yes|on)$ ]]; then
  echo "=== path-fit: skipped (existing ${PATH_SET}); set PATH_FIT_FORCE=1 to re-fit ==="
  exit 0
fi

BASE="${ROOT}/deploy/jobs/preprocess-dataset/fit-function-paths"

run_phase sindyffuse-fit-function-paths "${BASE}" 48h "path-fit"

echo "Path-fit complete."
