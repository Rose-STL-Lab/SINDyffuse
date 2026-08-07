#!/usr/bin/env bash
# Local/laptop driver for path-fit Job (kubectl wait; no sleep).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=deploy/scripts/k8s-orchestrate-lib.sh
source "${SCRIPT_DIR}/k8s-orchestrate-lib.sh"

ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export SINDYFFUSE_ROOT="${ROOT}"
NS="${KUBE_NAMESPACE:-${1:-default}}"

BASE="${ROOT}/deploy/jobs/preprocess-dataset/fit-function-paths"

run_phase sindyffuse-fit-function-paths "${BASE}" 24h "path-fit"

echo "Path-fit complete."
