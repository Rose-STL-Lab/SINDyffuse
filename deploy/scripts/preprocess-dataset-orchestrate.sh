#!/usr/bin/env bash
# Local driver for preprocess-dataset pipeline (kubectl wait; no sleep).
# Usage: preprocess-dataset-orchestrate.sh [full|ik|path-fit|moco] [namespace]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"

export KUBE_NAMESPACE="${KUBE_NAMESPACE:-${2:-default}}"
STAGE="${1:-full}"

_run_full_pipeline() {
  k8s_orchestrate_init
  BASE="${ROOT}/deploy/jobs/preprocess-dataset"
  run_phase sindyffuse-preprocess-ik "${BASE}/inverse_kinematics" 12h "inverse-kinematics"
  bash "${SCRIPT_DIR}/path-fit-orchestrate.sh"
  bash "${SCRIPT_DIR}/moco-track-orchestrate.sh"
  echo "Preprocess-dataset pipeline complete."
}

case "${STAGE}" in
  full)
    _run_full_pipeline
    ;;
  ik|inverse-kinematics)
    k8s_orchestrate_init
    run_phase sindyffuse-preprocess-ik "${ROOT}/deploy/jobs/preprocess-dataset/inverse_kinematics" 12h "inverse-kinematics"
    echo "Done (${STAGE})."
    ;;
  path-fit)
    exec "${SCRIPT_DIR}/path-fit-orchestrate.sh"
    ;;
  moco)
    exec "${SCRIPT_DIR}/moco-track-orchestrate.sh"
    ;;
  *)
    echo "Usage: $0 [full|ik|path-fit|moco] [namespace]" >&2
    exit 2
    ;;
esac
