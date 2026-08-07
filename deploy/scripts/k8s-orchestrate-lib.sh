#!/usr/bin/env bash
# Shared kubectl wait orchestration (no sleep). Source from pipeline scripts.
# Requires: set -euo pipefail in caller.

k8s_orchestrate_init() {
  ROOT="${SINDYFFUSE_ROOT:-/mnt/SINDyffuse}"
  if [[ -f /var/run/secrets/kubernetes.io/serviceaccount/namespace ]]; then
    NS="${KUBE_NAMESPACE:-$(tr -d '\n' < /var/run/secrets/kubernetes.io/serviceaccount/namespace)}"
  else
    NS="${KUBE_NAMESPACE:-default}"
  fi
}

_report_job_failure() {
  local job_name=$1
  echo "ERROR: ${job_name} failed or did not complete" >&2
  kubectl get job "${job_name}" -n "${NS}" -o wide >&2 || true
  kubectl describe job "${job_name}" -n "${NS}" >&2 || true
  local pod
  pod="$(kubectl get pods -n "${NS}" -l "job-name=${job_name}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  if [[ -n "${pod}" ]]; then
    echo "--- logs: ${pod} ---" >&2
    kubectl logs "${pod}" -n "${NS}" --tail=200 >&2 || true
  fi
}

run_phase() {
  local job_name=$1
  local kustomize_dir=$2
  local timeout=$3
  local label="${4:-${job_name}}"
  echo "=== ${label}: ${job_name} (${timeout}) namespace=${NS} ==="
  kubectl delete job "${job_name}" --ignore-not-found -n "${NS}"
  kubectl apply -k "${kustomize_dir}" -n "${NS}"

  kubectl wait --for=condition=complete "job/${job_name}" -n "${NS}" --timeout="${timeout}" &
  local complete_pid=$!
  kubectl wait --for=condition=failed "job/${job_name}" -n "${NS}" --timeout="${timeout}" &
  local failed_pid=$!

  wait -n "${complete_pid}" "${failed_pid}"

  if kill -0 "${complete_pid}" 2>/dev/null; then
    wait "${failed_pid}" || true
    kill "${complete_pid}" 2>/dev/null || true
    wait "${complete_pid}" 2>/dev/null || true
    _report_job_failure "${job_name}"
    exit 1
  fi

  if wait "${complete_pid}"; then
    kill "${failed_pid}" 2>/dev/null || true
    wait "${failed_pid}" 2>/dev/null || true
    echo "=== completed: ${job_name} ==="
    return 0
  fi

  kill "${failed_pid}" 2>/dev/null || true
  wait "${failed_pid}" 2>/dev/null || true
  _report_job_failure "${job_name}"
  exit 1
}
