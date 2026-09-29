#!/usr/bin/env bash
# Local driver for preprocess-dataset pipeline (kubectl wait; no sleep).
# Usage: preprocess-dataset-orchestrate.sh [full|ik|opensimad|moco|build-ext|build-polynomials|canary] [namespace]
# Note: path-fit is not required for OpenSimAD (polynomial MT paths); kept as legacy alias no-op.
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
  bash "${SCRIPT_DIR}/moco-track-orchestrate.sh"
  echo "Preprocess-dataset pipeline complete (LaiUhlrich2022 + OpenSimAD)."
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
  build-ext|opensimad-ext)
    k8s_orchestrate_init
    run_phase sindyffuse-build-opensimad-ext "${ROOT}/deploy/jobs/preprocess-dataset/build-opensimad-ext" 6h "opensimad-ext"
    echo "Done (${STAGE})."
    ;;
  build-polynomials|opensimad-polynomials)
    k8s_orchestrate_init
    run_phase sindyffuse-build-opensimad-polynomials "${ROOT}/deploy/jobs/preprocess-dataset/build-opensimad-polynomials" 12h "opensimad-polynomials"
    echo "Done (${STAGE})."
    ;;
  canary|opensimad-canary)
    k8s_orchestrate_init
    run_phase sindyffuse-opensimad-canary "${ROOT}/deploy/jobs/preprocess-dataset/opensimad-canary" 4h "opensimad-canary"
    echo "Done (${STAGE})."
    ;;
  opensimad|moco)
    exec "${SCRIPT_DIR}/moco-track-orchestrate.sh"
    ;;
  path-fit)
    echo "path-fit skipped on uhlrich branch (OpenSimAD uses polynomial MT paths)."
    ;;
  *)
    echo "Usage: $0 [full|ik|build-ext|build-polynomials|canary|opensimad|moco] [namespace]" >&2
    exit 2
    ;;
esac
