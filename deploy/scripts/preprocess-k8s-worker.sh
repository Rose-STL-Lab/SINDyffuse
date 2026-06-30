#!/usr/bin/env bash
# Kubernetes Indexed Job worker for preprocess_nimble.py.
# Writes per-shard diagnostics to the PVC so failures survive pod deletion.
set -euo pipefail

REPO_ROOT="/mnt/SINDyffuse"
HML="${HML:-${REPO_ROOT}/datasets/HumanML3D}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
RUN_LOG_ID="${SINDYFFUSE_RUN_LOG_ID:-unknown_run}"
SHARD_INDEX="${JOB_COMPLETION_INDEX:-0}"
SHARD_TAG="$(printf '%04d' "${SHARD_INDEX}")"
NUM_SHARDS="${PREPROCESS_NUM_SHARDS:-1}"
ACTIVATION_METHOD="${ACTIVATION_METHOD:-static_optimization}"
MAX_MOTIONS="${MAX_MOTIONS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

DIAG_DIR="${LOG_ROOT}/k8s_diagnostics/${RUN_LOG_ID}"
mkdir -p "${DIAG_DIR}"
DIAG_FILE="${DIAG_DIR}/shard_${SHARD_TAG}.txt"
POD_LOG="${DIAG_DIR}/shard_${SHARD_TAG}.pod.log"
MANIFEST="${HML}/preprocess_manifest.${SHARD_TAG}.jsonl"

HEARTBEAT_PID=""
_TERM_RECEIVED=0
_PYTHON_PID=""

diag() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "${DIAG_FILE}"
}

_mem_snapshot() {
  local rss_kb="" cgroup_bytes=""
  if [[ -r /proc/self/status ]]; then
    rss_kb="$(awk '/^VmRSS:/ {print $2}' /proc/self/status 2>/dev/null || true)"
  fi
  if [[ -r /sys/fs/cgroup/memory.current ]]; then
    cgroup_bytes="$(tr -d '[:space:]' < /sys/fs/cgroup/memory.current 2>/dev/null || true)"
  elif [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
    cgroup_bytes="$(tr -d '[:space:]' < /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || true)"
  fi
  echo "vmrss_kb=${rss_kb:-na} cgroup_bytes=${cgroup_bytes:-na}"
}

_manifest_stats() {
  if [[ -f "${MANIFEST}" ]]; then
  stat -c 'manifest_bytes=%s manifest_mtime=%y' "${MANIFEST}" 2>/dev/null || echo "manifest_bytes=unknown"
  else
    echo "manifest_bytes=missing"
  fi
}

_stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  HEARTBEAT_PID=""
}

_on_term() {
  _TERM_RECEIVED=1
  diag "signal=SIGTERM"
  if [[ -n "${_PYTHON_PID}" ]] && kill -0 "${_PYTHON_PID}" 2>/dev/null; then
    diag "forwarding SIGTERM to python pid=${_PYTHON_PID}"
    kill -TERM "${_PYTHON_PID}" 2>/dev/null || true
  fi
}

_on_exit() {
  local ec=$?
  _stop_heartbeat
  if [[ "${_TERM_RECEIVED}" -eq 1 ]]; then
    diag "terminated=1"
  fi
  diag "exit_code=${ec}"
  diag "hostname=$(hostname)"
  diag "pod=${HOSTNAME:-unknown}"
  diag "num_shards=${NUM_SHARDS} activation_method=${ACTIVATION_METHOD}"
  diag "skip_existing=${SKIP_EXISTING} max_motions=${MAX_MOTIONS}"
  diag "$(_mem_snapshot)"
  diag "$(_manifest_stats)"
  if [[ "${ec}" -eq 137 ]]; then
    diag "hint=exit_137_often_OOM_or_SIGKILL"
  fi
  sync "${DIAG_FILE}" 2>/dev/null || true
  exit "${ec}"
}

trap _on_exit EXIT
trap _on_term TERM
trap 'diag signal=INT; exit 130' INT

(
  while true; do
    sleep 300
    diag "heartbeat $(_mem_snapshot) $(_manifest_stats)"
    sync "${DIAG_FILE}" 2>/dev/null || true
  done
) &
HEARTBEAT_PID=$!

cd "${REPO_ROOT}"
eval "$(conda shell.bash hook)"
conda activate sindyffuse
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}"

diag "start run_log_id=${RUN_LOG_ID} shard=${SHARD_TAG}/${NUM_SHARDS}"
diag "diag_file=${DIAG_FILE}"
diag "pod_log=${POD_LOG}"

ARGS=(
  scripts/preprocess_nimble.py
  --hml_root "${HML}"
  --out_root "${HML}"
  --log_dir "${LOG_ROOT}"
  --num_shards "${NUM_SHARDS}"
  --shard_index "${SHARD_INDEX}"
  --num_workers 1
  --activation_method "${ACTIVATION_METHOD}"
  --skip_normalization
  --opensim_log_level Off
  --run_log_id "${RUN_LOG_ID}"
)
[[ "${MAX_MOTIONS}" != "0" ]] && ARGS+=(--max_motions "${MAX_MOTIONS}")
[[ "${SKIP_EXISTING}" == "1" ]] && ARGS+=(--skip_existing)

diag "argv=${ARGS[*]}"

set +e
python "${ARGS[@]}" > >(tee -a "${POD_LOG}") 2>&1 &
_PYTHON_PID=$!
wait "${_PYTHON_PID}"
PY_EXIT=$?
set -e
_PYTHON_PID=""

diag "python_exit=${PY_EXIT}"
diag "$(_manifest_stats)"

if [[ "${PY_EXIT}" -ne 0 ]]; then
  diag "status=python_failed"
  exit "${PY_EXIT}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
  diag "status=manifest_missing path=${MANIFEST}"
  exit 1
fi

diag "status=ok"
exit 0
