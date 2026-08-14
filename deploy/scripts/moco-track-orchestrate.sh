#!/usr/bin/env bash
# Local driver: OpenSimAD ext build → activation workers → normalization (kubectl wait).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"
k8s_orchestrate_init

BASE="${ROOT}/deploy/jobs/preprocess-dataset"

run_phase sindyffuse-build-opensimad-ext "${BASE}/build-opensimad-ext" 6h "opensimad-ext"
# MinT-scale OpenSimAD on HumanML3D is long-running; 7d wait budget for 180 shards.
run_phase sindyffuse-preprocess-moco-track "${BASE}/moco-track" 168h "opensimad-track"
run_phase sindyffuse-compute-normalization "${BASE}/normalization" 2h "normalization"

echo "OpenSimAD (MinT) + normalization pipeline complete."
