#!/usr/bin/env bash
# Local/laptop driver for preprocess-dataset pipeline (kubectl wait; no sleep).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"

ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export SINDYFFUSE_ROOT="${ROOT}"
NS="${KUBE_NAMESPACE:-${2:-default}}"

BASE="${ROOT}/deploy/jobs/preprocess-dataset"
STAGE="${1:-full}"

case "${STAGE}" in
  full)
    run_phase sindyffuse-preprocess-dataset-orchestrator "${BASE}/orchestrator" 48h "preprocess-dataset"
    ;;
  ik)
    run_phase sindyffuse-preprocess-ik "${BASE}/ik" 12h "ik"
    ;;
  path-fit)
    run_phase sindyffuse-path-fit-orchestrator "${BASE}/fit-function-paths/orchestrator" 8h "path-fit"
    ;;
  moco)
    run_phase sindyffuse-moco-track-orchestrator "${BASE}/moco-track/orchestrator" 26h "moco-track"
    ;;
  *)
    echo "Usage: $0 [full|ik|path-fit|moco] [namespace]" >&2
    exit 2
    ;;
esac

echo "Done (${STAGE})."
