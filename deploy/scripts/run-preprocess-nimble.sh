#!/usr/bin/env bash
# Run distributed preprocess on Kubernetes: 180 worker pods, then compute normalization.
#
# Usage:
#   ./deploy/scripts/run-preprocess-nimble.sh [none|static-optimization|moco-track]
#   ACTIVATION_METHOD=static_optimization ./deploy/scripts/run-preprocess-nimble.sh
#
# Environment overrides:
#   PREPROCESS_NUM_SHARDS    shard count / Job parallelism+completions (default: 64)
#   SKIP_EXISTING            1 to skip motions that already have .b3d (default: leave yaml)
#   WORKER_TIMEOUT           kubectl wait timeout for worker Job (default: 48h)
#   NORMALIZATION_TIMEOUT    kubectl wait timeout for normalization Job (default: 1h)
#   KUSTOMIZE_DIR            override worker job kustomization path
#   NORMALIZATION_JOB        override normalization job yaml path
#   SKIP_DELETE              set to 1 to skip deleting stale jobs before apply

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/deploy/jobs/preprocess-nimble"

resolve_method() {
  local raw="${1:-${ACTIVATION_METHOD:-static_optimization}}"
  case "${raw}" in
    none)
      METHOD_FOLDER="none"
      ACTIVATION_METHOD="none"
      WORKER_JOB_NAME="sindyffuse-preprocess-none"
      NORMALIZATION_JOB_NAME="sindyffuse-compute-normalization-none"
      ;;
    static_optimization|static-optimization)
      METHOD_FOLDER="static-optimization"
      ACTIVATION_METHOD="static_optimization"
      WORKER_JOB_NAME="sindyffuse-preprocess-static-optimization"
      NORMALIZATION_JOB_NAME="sindyffuse-compute-normalization-static-optimization"
      ;;
    moco_track|moco-track)
      METHOD_FOLDER="moco-track"
      ACTIVATION_METHOD="moco_track"
      WORKER_JOB_NAME="sindyffuse-preprocess-moco-track"
      NORMALIZATION_JOB_NAME="sindyffuse-compute-normalization-moco-track"
      ;;
    *)
      echo "Unknown preprocess method: ${raw}" >&2
      echo "Use: none, static-optimization, or moco-track" >&2
      exit 1
      ;;
  esac
}

METHOD_ARG="${1:-}"
resolve_method "${METHOD_ARG}"

KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-${JOBS_ROOT}/${METHOD_FOLDER}}"
NORMALIZATION_JOB="${NORMALIZATION_JOB:-${KUSTOMIZE_DIR}/normalization-job.yaml}"

WORKER_TIMEOUT="${WORKER_TIMEOUT:-48h}"
NORMALIZATION_TIMEOUT="${NORMALIZATION_TIMEOUT:-${FINALIZE_TIMEOUT:-1h}}"
export PREPROCESS_NUM_SHARDS="${PREPROCESS_NUM_SHARDS:-64}"
RUN_LOG_ID="${RUN_LOG_ID:-preprocess_${METHOD_FOLDER}_$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_LOG_ID
# Optional override; empty keeps the value from job.yaml.
SKIP_EXISTING_OVERRIDE="${SKIP_EXISTING:-}"

echo "=== preprocess nimble (distributed) ==="
echo "method=${METHOD_FOLDER} activation_method=${ACTIVATION_METHOD}"
echo "num_shards=${PREPROCESS_NUM_SHARDS}"
echo "run_log_id=${RUN_LOG_ID}"
echo "skip_existing_override=${SKIP_EXISTING_OVERRIDE:-<job.yaml>}"
echo "worker kustomize:    ${KUSTOMIZE_DIR}"
echo "normalization job:   ${NORMALIZATION_JOB}"
echo "worker_timeout=${WORKER_TIMEOUT}"

if [[ "${SKIP_DELETE:-0}" != "1" ]]; then
  echo "Deleting stale jobs (if any)..."
  kubectl delete job "${WORKER_JOB_NAME}" "${NORMALIZATION_JOB_NAME}" --ignore-not-found
fi

echo "Applying worker Indexed Job..."
manifest="$(kubectl kustomize "${KUSTOMIZE_DIR}")"
# Keep Indexed Job pod count and --num_shards in lockstep. Otherwise setting
# PREPROCESS_NUM_SHARDS=1 still launches 64 pods that each process the full dataset.
sed_script="s|RUN_LOG_ID_PLACEHOLDER|${RUN_LOG_ID}|g; \
  /name: PREPROCESS_NUM_SHARDS/{n;s/value: .*/value: \"${PREPROCESS_NUM_SHARDS}\"/;}; \
  s|^  parallelism: .*|  parallelism: ${PREPROCESS_NUM_SHARDS}|; \
  s|^  completions: .*|  completions: ${PREPROCESS_NUM_SHARDS}|"
if [[ -n "${SKIP_EXISTING_OVERRIDE}" ]]; then
  sed_script="${sed_script}; /name: SKIP_EXISTING/{n;s/value: .*/value: \"${SKIP_EXISTING_OVERRIDE}\"/;}"
fi
manifest="$(printf '%s\n' "${manifest}" | sed "${sed_script}")"
printf '%s\n' "${manifest}" | kubectl apply -f -

echo "Waiting for worker Job to complete (timeout=${WORKER_TIMEOUT})..."
kubectl wait --for=condition=complete "job/${WORKER_JOB_NAME}" --timeout="${WORKER_TIMEOUT}"

echo "Applying normalization Job..."
normalization_manifest="$(cat "${NORMALIZATION_JOB}")"
normalization_manifest="$(printf '%s\n' "${normalization_manifest}" | sed \
  "s|RUN_LOG_ID_PLACEHOLDER|${RUN_LOG_ID}|g; \
  /name: PREPROCESS_NUM_SHARDS/{n;s/value: .*/value: \"${PREPROCESS_NUM_SHARDS}\"/;}")"
printf '%s\n' "${normalization_manifest}" | kubectl apply -f -

echo "Waiting for normalization Job to complete (timeout=${NORMALIZATION_TIMEOUT})..."
kubectl wait --for=condition=complete "job/${NORMALIZATION_JOB_NAME}" --timeout="${NORMALIZATION_TIMEOUT}"

echo "=== preprocess complete ==="
echo "shared log: ${REPO_ROOT}/logs/preprocess_nimble.log"
kubectl get job "${WORKER_JOB_NAME}" "${NORMALIZATION_JOB_NAME}"
